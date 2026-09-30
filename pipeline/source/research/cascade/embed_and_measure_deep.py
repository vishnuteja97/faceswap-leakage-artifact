#!/usr/bin/env python3
"""Measure the deep (K=25) FaceFusion cascade: per-pass leakage AND attack strength.

Answers the interactive-rebuttal request "What does the AUC look like for larger number of
steps?" by extending the submitted 5-pass series to 25 passes on the same chains.

Per pass k we report
  median target-gallery cosine        (the paper's leakage measure)
  non-member baseline                 (the predicted floor)
  excess = leakage - floor, and its pass-to-pass ratio (predicted -> rho(B) = 0.894)
  membership-inference AUC and TPR at 1% / 0.1% FPR
  raw embedding norm                  (checks the constant-norm assumption at depth)

Chains truncate when a donor image fails detection, so every statistic is reported twice:
  ALL      all chains that reached depth >= k (largest N, depth-varying sample)
  BALANCED only chains that reached the full depth (fixed sample, no attrition bias)

Run in the facefusion env (insightface + cv2 + GPU onnxruntime).
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CHAINS_DIR = OUT / "chains"
MANIFEST = OUT / "chains_manifest_deep.jsonl"
RAW_NPZ = Path("/opt/reproduction/S&P2027-Artifact/data/D2/embeddings/vggface2_raw_embeddings.npz")
RHO_B = 0.894


def l2(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv)
    r = np.empty(len(allv), float)
    r[order] = np.arange(1, len(allv) + 1)
    sv = allv[order]
    i = 0
    while i < len(sv):                      # average ranks within ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = r[order[i:j + 1]].mean()
        i = j + 1
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def tpr_at_fpr(pos, neg, fpr):
    neg = np.asarray(neg, float)
    if len(neg) == 0:
        return float("nan")
    return float(np.mean(np.asarray(pos, float) >= np.quantile(neg, 1.0 - fpr)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-max", type=int, default=25)
    args = ap.parse_args()
    K = args.k_max

    chains = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    z = np.load(RAW_NPZ, allow_pickle=True)
    raw = np.asarray(z["embeddings"], np.float32)

    from insightface.model_zoo import get_model
    model = get_model("buffalo_l")
    try:
        model.prepare(ctx_id=0)
    except Exception:
        model.prepare(ctx_id=-1)

    def embed(path: Path):
        img = cv2.imread(str(path))
        if img is None:
            return None
        e = np.asarray(model.get_feat(cv2.resize(img, (112, 112))))
        return (e[0] if e.ndim > 1 else e).astype(np.float32)

    # Non-member control. The submitted 5-pass measurement gave each chain ONE unrelated
    # gallery (the chain half the corpus away). At depth the excess over the floor becomes
    # small, so a floor estimated from a single non-member per chain is too noisy to take
    # ratios of. We therefore keep that control for comparability AND pool N_NONMEM
    # unrelated galleries per chain, which shrinks the floor's variance and enlarges the
    # negative class of the ROC by the same factor.
    n_ch = len(chains)
    N_NONMEM = 10
    nonmem = {c["chain_id"]: chains[(i + n_ch // 2) % n_ch]["t0_gallery_idx"]
              for i, c in enumerate(chains)}
    nonmem_pool = {c["chain_id"]: [chains[(i + 1 + j * (n_ch // (N_NONMEM + 1))) % n_ch]
                                   ["t0_gallery_idx"] for j in range(N_NONMEM)]
                   for i, c in enumerate(chains)}

    rows = []
    emb_store: dict[str, np.ndarray] = {}   # chain_id -> (K, 512) raw, NaN past its depth
    for ci, c in enumerate(chains):
        cdir = CHAINS_DIR / c["chain_id"]
        gal = l2(raw[np.asarray(c["t0_gallery_idx"], np.int64)])
        nmg = l2(raw[np.asarray(nonmem[c["chain_id"]], np.int64)])
        nmp = [l2(raw[np.asarray(g, np.int64)]) for g in nonmem_pool[c["chain_id"]]]
        t0u = l2(raw[c["t0_image_idx"]])
        rec = {"chain_id": c["chain_id"], "t0_id": c["t0_id"]}
        chain_emb = np.full((K, raw.shape[1]), np.nan, np.float32)
        depth = 0
        for k in range(1, K + 1):
            f = cdir / f"s{k}.jpg"
            if not (f.is_file() and f.stat().st_size > 0):
                break
            e = embed(f)
            if e is None:
                break
            chain_emb[k - 1] = e
            su = l2(e)
            rec[f"gal_med_s{k}"] = float(np.median(gal @ su))
            rec[f"gal_max_s{k}"] = float(np.max(gal @ su))
            rec[f"nonmem_med_s{k}"] = float(np.median(nmg @ su))
            pooled = [float(np.median(g @ su)) for g in nmp]
            rec[f"nonmem_pool_med_s{k}"] = float(np.median(pooled))
            rec[f"nonmem_pool_all_s{k}"] = pooled
            rec[f"cos_s{k}_t0"] = float(su @ t0u)
            rec[f"norm_s{k}"] = float(np.linalg.norm(e))
            depth = k
        rec["depth"] = depth
        rows.append(rec)
        emb_store[c["chain_id"]] = chain_emb
        if (ci + 1) % 25 == 0:
            print(f"[deep-measure] {ci + 1}/{n_ch} chains", flush=True)

    # Raw per-pass embeddings, kept so the operator can be refit on the cascade's own
    # outputs (Reviewer B's iterative-refit question) without regenerating images.
    ids = list(emb_store)
    np.savez_compressed(
        OUT / "cascade_embeddings_deep.npz",
        chain_ids=np.asarray(ids, dtype=object),
        embeddings=np.stack([emb_store[i] for i in ids]),          # (n_chains, K, 512)
        donor_emb_idx=np.asarray([c["donor_emb_idx"][:K] for c in chains
                                  if c["chain_id"] in emb_store], np.int64),
        t0_image_idx=np.asarray([c["t0_image_idx"] for c in chains
                                 if c["chain_id"] in emb_store], np.int64),
        depths=np.asarray([r["depth"] for r in rows], np.int64),
    )

    flat = [{k: v for k, v in r.items() if not k.startswith("nonmem_pool_all")}
            for r in rows]
    keys = sorted({k for r in flat for k in r})
    with (OUT / "cascade_measurements_deep.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)

    full = [r for r in rows if r["depth"] >= K]
    print(f"[deep-measure] {len(rows)} chains, {len(full)} reached depth {K}")

    def series(sample, label):
        out = []
        for k in range(1, K + 1):
            rs = [r for r in sample if r.get(f"gal_med_s{k}") is not None]
            if not rs:
                continue
            pos = [r[f"gal_med_s{k}"] for r in rs]
            neg1 = [r[f"nonmem_med_s{k}"] for r in rs]
            negp = [v for r in rs for v in r[f"nonmem_pool_all_s{k}"]]
            out.append({
                "pass": k, "n": len(rs), "n_negatives_pooled": len(negp),
                "leak_median": float(np.median(pos)),
                "leak_mean": float(np.mean(pos)),
                "nonmember_median_1control": float(np.median(neg1)),
                "nonmember_median_pooled": float(np.median(negp)),
                "excess_median": float(np.median(pos) - np.median(negp)),
                "auc_1control": auc(pos, neg1),
                "auc": auc(pos, negp),
                "tpr_at_1pct_fpr": tpr_at_fpr(pos, negp, 0.01),
                "tpr_at_0p1pct_fpr": tpr_at_fpr(pos, negp, 0.001),
                "norm_mean": float(np.mean([r[f"norm_s{k}"] for r in rs])),
            })
        for i in range(1, len(out)):
            prev, cur = out[i - 1]["excess_median"], out[i]["excess_median"]
            out[i]["excess_ratio"] = float(cur / prev) if prev > 0 else None
        print(f"\n=== {label} (n varies: {out[0]['n']} -> {out[-1]['n']}) ===")
        print(f"{'k':>3} {'n':>5} {'leak':>7} {'floor':>7} {'excess':>8} {'ratio':>7} "
              f"{'AUC':>6} {'AUC1':>6} {'TPR@1%':>7} {'norm':>6}")
        for o in out:
            print(f"{o['pass']:>3} {o['n']:>5} {o['leak_median']:>7.4f} "
                  f"{o['nonmember_median_pooled']:>7.4f} {o['excess_median']:>8.4f} "
                  f"{(o.get('excess_ratio') or float('nan')):>7.3f} {o['auc']:>6.3f} "
                  f"{o['auc_1control']:>6.3f} "
                  f"{o['tpr_at_1pct_fpr'] * 100:>6.1f}% {o['norm_mean']:>6.2f}")
        return out

    s_all = series(rows, "ALL chains reaching depth k")
    s_bal = series(full, f"BALANCED panel (depth >= {K})")

    # geometric fit to the tail: how deep until the attack is effectively chance?
    def tail_fit(s, k_from=8):
        pts = [(o["pass"], o["excess_median"]) for o in s
               if o["pass"] >= k_from and o["excess_median"] > 0]
        if len(pts) < 3:
            return None
        ks = np.array([p[0] for p in pts], float)
        ys = np.log(np.array([p[1] for p in pts], float))
        slope, icpt = np.polyfit(ks, ys, 1)
        rate = float(np.exp(slope))
        return {"fit_from_pass": k_from, "geometric_rate": rate,
                "rho_B_for_comparison": RHO_B,
                "passes_to_excess_below_0.01": (
                    float((np.log(0.01) - icpt) / slope) if slope < 0 else None),
                "passes_to_excess_below_0.005": (
                    float((np.log(0.005) - icpt) / slope) if slope < 0 else None)}

    summary = {
        "n_chains": len(rows), "n_chains_full_depth": len(full), "K": K,
        "depth_histogram": {str(k): sum(1 for r in rows if r["depth"] == k)
                            for k in sorted({r["depth"] for r in rows})},
        "per_pass_all": s_all,
        "per_pass_balanced": s_bal,
        "tail_fit_all": tail_fit(s_all),
        "tail_fit_balanced": tail_fit(s_bal),
    }
    (OUT / "cascade_summary_deep.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({k: summary[k] for k in
                             ["n_chains", "n_chains_full_depth", "tail_fit_all",
                              "tail_fit_balanced"]}, indent=2))


if __name__ == "__main__":
    main()
