#!/usr/bin/env python3
"""Measure the deep (K=25) BlendFace / CanonSwap cascades: per-pass leakage + attack.

Mirrors embed_and_measure_deep.py (FaceFusion). Passes 1..3 come from D1/D3, passes
4..25 from cascade/outputs/chains_deep_aux/<tool>/. Per pass k we report the median
target-gallery cosine, a pooled non-member floor, the excess and its pass-to-pass ratio,
identity-inference AUC, and TPR at 1%/0.1% FPR.

Gallery protocol matches the D3 three-pass measurement: every buffalo_l embedding of the
target's folder EXCEPT the pass-1 target image.

Run in the facefusion env (insightface + cv2 + GPU onnxruntime):
  CUDA_VISIBLE_DEVICES=... python cascade/embed_and_measure_deep_aux.py --tool blendface
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
ART = Path("/opt/reproduction/S&P2027-Artifact/data")
MANIFEST = OUT / "chains_manifest_deep_aux.jsonl"
RAW_NPZ = ART / "D2/embeddings/vggface2_raw_embeddings.npz"
K = 25
N_NONMEM = 10


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
    while i < len(sv):
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


def pass_path(c: dict, tool: str, k: int) -> Path:
    if k == 1:
        return ART / f"D1/swaps/{tool}/vggface2_seed_42" / c["swap_basenames"]["1"]
    if k == 2:
        return ART / f"D3/swaps/{tool}/swap2/vggface2_seed_42" / c["swap_basenames"]["2"]
    if k == 3:
        return ART / f"D3/swaps/{tool}/swap3/vggface2_seed_42" / c["swap_basenames"]["3"]
    return OUT / "chains_deep_aux" / tool / c["pair_id"] / f"s{k}.jpg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tool", required=True, choices=["blendface", "canonswap"])
    args = ap.parse_args()
    tool = args.tool

    chains = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    z = np.load(RAW_NPZ, allow_pickle=True)
    raw = np.asarray(z["embeddings"], np.float32)
    uids = np.asarray(z["unique_identity_ids"], dtype=object)
    s_idx = np.asarray(z["identity_start_idx"], np.int64)
    e_idx = np.asarray(z["identity_end_idx"], np.int64)
    id_range = {str(u): (int(a), int(b)) for u, a, b in zip(uids, s_idx, e_idx)}

    def gallery_idx(c: dict) -> np.ndarray:
        a, b = id_range[str(c["target_id"])]
        rows = np.arange(a, b)
        return rows[rows != int(c["target_image_idx"])]

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

    n_ch = len(chains)
    gal_rows = {c["pair_id"]: gallery_idx(c) for c in chains}
    nonmem_pool = {c["pair_id"]: [gal_rows[chains[(i + 1 + j * (n_ch // (N_NONMEM + 1))) % n_ch]
                                           ["pair_id"]] for j in range(N_NONMEM)]
                   for i, c in enumerate(chains)}

    rows = []
    emb_store: dict[str, np.ndarray] = {}
    for ci, c in enumerate(chains):
        gal = l2(raw[gal_rows[c["pair_id"]]])
        nmp = [l2(raw[g]) for g in nonmem_pool[c["pair_id"]]]
        rec = {"pair_id": c["pair_id"], "target_id": c["target_id"]}
        chain_emb = np.full((K, raw.shape[1]), np.nan, np.float32)
        depth = 0
        for k in range(1, K + 1):
            f = pass_path(c, tool, k)
            if not (f.is_file() and f.stat().st_size > 0):
                break
            e = embed(f)
            if e is None:
                break
            chain_emb[k - 1] = e
            su = l2(e)
            rec[f"gal_med_s{k}"] = float(np.median(gal @ su))
            pooled = [float(np.median(g @ su)) for g in nmp]
            rec[f"nonmem_pool_med_s{k}"] = float(np.median(pooled))
            rec[f"nonmem_pool_all_s{k}"] = pooled
            rec[f"norm_s{k}"] = float(np.linalg.norm(e))
            depth = k
        rec["depth"] = depth
        rows.append(rec)
        emb_store[c["pair_id"]] = chain_emb
        if (ci + 1) % 50 == 0:
            print(f"[{tool}] {ci + 1}/{n_ch} chains", flush=True)

    ids = list(emb_store)
    np.savez_compressed(
        OUT / f"cascade_embeddings_deep_{tool}.npz",
        pair_ids=np.asarray(ids, dtype=object),
        embeddings=np.stack([emb_store[i] for i in ids]),
        depths=np.asarray([r["depth"] for r in rows], np.int64),
    )

    flat = [{k: v for k, v in r.items() if not k.startswith("nonmem_pool_all")}
            for r in rows]
    keys = sorted({k for r in flat for k in r})
    with (OUT / f"cascade_measurements_deep_{tool}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(flat)

    full = [r for r in rows if r["depth"] >= K]
    print(f"[{tool}] {len(rows)} chains, {len(full)} reached depth {K}")

    def series(sample, label):
        out = []
        for k in range(1, K + 1):
            rs = [r for r in sample if r.get(f"gal_med_s{k}") is not None]
            if not rs:
                continue
            pos = [r[f"gal_med_s{k}"] for r in rs]
            negp = [v for r in rs for v in r[f"nonmem_pool_all_s{k}"]]
            out.append({
                "pass": k, "n": len(rs),
                "leak_median": float(np.median(pos)),
                "nonmember_median_pooled": float(np.median(negp)),
                "excess_median": float(np.median(pos) - np.median(negp)),
                "auc": auc(pos, negp),
                "tpr_at_1pct_fpr": tpr_at_fpr(pos, negp, 0.01),
                "tpr_at_0p1pct_fpr": tpr_at_fpr(pos, negp, 0.001),
                "norm_mean": float(np.mean([r[f"norm_s{k}"] for r in rs])),
            })
        for i in range(1, len(out)):
            prev, cur = out[i - 1]["excess_median"], out[i]["excess_median"]
            out[i]["excess_ratio"] = float(cur / prev) if prev > 0 else None
        print(f"\n=== {tool} {label} ===")
        print(f"{'k':>3} {'n':>5} {'leak':>7} {'floor':>7} {'excess':>8} {'ratio':>7} "
              f"{'AUC':>6} {'TPR@1%':>7}")
        for o in out:
            print(f"{o['pass']:>3} {o['n']:>5} {o['leak_median']:>7.4f} "
                  f"{o['nonmember_median_pooled']:>7.4f} {o['excess_median']:>8.4f} "
                  f"{(o.get('excess_ratio') or float('nan')):>7.3f} {o['auc']:>6.3f} "
                  f"{o['tpr_at_1pct_fpr'] * 100:>6.1f}%")
        return out

    s_all = series(rows, "ALL chains reaching depth k")
    s_bal = series(full, f"BALANCED (depth >= {K})")

    def tail_fit(s, k_from=8):
        pts = [(o["pass"], o["excess_median"]) for o in s
               if o["pass"] >= k_from and o["excess_median"] > 0]
        if len(pts) < 3:
            return None
        ks = np.array([p[0] for p in pts], float)
        ys = np.log(np.array([p[1] for p in pts], float))
        slope, icpt = np.polyfit(ks, ys, 1)
        return {"fit_from_pass": k_from, "geometric_rate": float(np.exp(slope)),
                "passes_to_excess_below_0.01": (
                    float((np.log(0.01) - icpt) / slope) if slope < 0 else None),
                "passes_to_excess_below_0.005": (
                    float((np.log(0.005) - icpt) / slope) if slope < 0 else None)}

    summary = {
        "tool": tool, "n_chains": len(rows), "n_chains_full_depth": len(full), "K": K,
        "depth_histogram": {str(k): sum(1 for r in rows if r["depth"] == k)
                            for k in sorted({r["depth"] for r in rows})},
        "per_pass_all": s_all,
        "per_pass_balanced": s_bal,
        "tail_fit_all": tail_fit(s_all),
        "tail_fit_balanced": tail_fit(s_bal),
    }
    (OUT / f"cascade_summary_deep_{tool}.json").write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps({k: summary[k] for k in
                             ["tool", "n_chains", "n_chains_full_depth",
                              "tail_fit_all", "tail_fit_balanced"]}, indent=2))


if __name__ == "__main__":
    main()
