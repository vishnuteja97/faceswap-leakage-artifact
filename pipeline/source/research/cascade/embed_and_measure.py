#!/usr/bin/env python3
"""Embed the generated cascade swaps s1..sK with buffalo_l and measure original-target
leakage per pass. Target/donor/gallery embeddings are pulled directly from the raw store
(no re-embedding). Run in the facefusion env (has insightface + cv2).

Outputs cascade/outputs/:
  cascade_embeddings.npz   per-chain s_k embeddings + indices
  cascade_measurements.csv per-chain per-pass leakage metrics
  cascade_summary.json     per-pass aggregate decay curve + ratios + floor
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import cv2

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CHAINS_DIR = OUT / "chains"
MANIFEST = OUT / "chains_manifest.jsonl"
RAW_NPZ = Path("/opt/reproduction/S&P2027-Artifact/data/D2/embeddings/vggface2_raw_embeddings.npz")
K = 5


def l2(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def main():
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
        img = cv2.resize(img, (112, 112))
        e = np.asarray(model.get_feat(img))
        return (e[0] if e.ndim > 1 else e).astype(np.float32)

    # Non-member gallery assignment: each chain is scored against ANOTHER chain's
    # target gallery (a different identity, guaranteed distinct by the no-reuse design).
    # This measures the empirical cross-identity floor in-distribution.
    nonmem_gallery_idx = {}
    n_ch = len(chains)
    for i, c in enumerate(chains):
        donor_chain = chains[(i + n_ch // 2) % n_ch]
        nonmem_gallery_idx[c["chain_id"]] = donor_chain["t0_gallery_idx"]

    rows = []
    all_s = {k: [] for k in range(1, K + 1)}
    kept_chain_ids = []
    n_done = 0
    for c in chains:
        cdir = CHAINS_DIR / c["chain_id"]
        s_paths = [cdir / f"s{k+1}.jpg" for k in range(K)]
        if not all(p.is_file() and p.stat().st_size > 0 for p in s_paths):
            continue
        s_emb = [embed(p) for p in s_paths]
        if any(e is None for e in s_emb):
            continue
        t0 = raw[c["t0_image_idx"]]
        t0u = l2(t0)
        gal = l2(raw[np.asarray(c["t0_gallery_idx"], np.int64)])  # (G, D)
        nmgal = l2(raw[np.asarray(nonmem_gallery_idx[c["chain_id"]], np.int64)])  # (G', D)
        donors = l2(raw[np.asarray(c["donor_emb_idx"], np.int64)])  # (K, D)
        rec = {"chain_id": c["chain_id"], "t0_id": c["t0_id"]}
        for k in range(K):
            su = l2(s_emb[k])
            galcos = gal @ su
            rec[f"cos_s{k+1}_t0"] = float(su @ t0u)
            rec[f"gal_med_s{k+1}"] = float(np.median(galcos))
            rec[f"gal_max_s{k+1}"] = float(np.max(galcos))
            rec[f"gal_mean_s{k+1}"] = float(np.mean(galcos))
            rec[f"nonmem_med_s{k+1}"] = float(np.median(nmgal @ su))
            rec[f"cos_s{k+1}_donor{k+1}"] = float(su @ donors[k])
            rec[f"norm_s{k+1}"] = float(np.linalg.norm(s_emb[k]))
            all_s[k + 1].append(su)
        rows.append(rec)
        kept_chain_ids.append(c["chain_id"])
        n_done += 1
        if n_done % 50 == 0:
            print(f"[measure] {n_done} chains embedded", flush=True)

    # write per-chain CSV
    if rows:
        keys = list(rows[0].keys())
        with (OUT / "cascade_measurements.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(keys)
            for r in rows:
                w.writerow([r[k] for k in keys])

    # aggregate per-pass decay
    def agg(prefix):
        return {f"pass{k}": {
            "mean": float(np.mean([r[f"{prefix}{k}"] for r in rows])),
            "median": float(np.median([r[f"{prefix}{k}"] for r in rows])),
            "std": float(np.std([r[f"{prefix}{k}"] for r in rows])),
        } for k in range(1, K + 1)}

    cos_t0 = {f"pass{k}": {
        "mean": float(np.mean([r[f"cos_s{k}_t0"] for r in rows])),
        "median": float(np.median([r[f"cos_s{k}_t0"] for r in rows])),
        "std": float(np.std([r[f"cos_s{k}_t0"] for r in rows])),
    } for k in range(1, K + 1)}
    gal_med = agg("gal_med_s")
    gal_max = agg("gal_max_s")
    nonmem_med = agg("nonmem_med_s")
    norms = {f"pass{k}": float(np.mean([r[f"norm_s{k}"] for r in rows])) for k in range(1, K + 1)}

    def ratios(series_mean):
        v = [series_mean[f"pass{k}"]["mean"] for k in range(1, K + 1)]
        return [v[k] / v[k - 1] for k in range(1, K)]

    summary = {
        "n_chains_measured": len(rows), "K": K,
        "cos_to_t0_image": cos_t0,
        "gallery_median": gal_med,
        "gallery_max": gal_max,
        "nonmember_median": nonmem_med,
        "floor_nonmember_median_overall": float(np.median(
            [r[f"nonmem_med_s{k}"] for r in rows for k in range(1, K + 1)])),
        "mean_norm_per_pass": norms,
        "cos_to_t0_per_pass_ratios": ratios(cos_t0),
        "gallery_median_per_pass_ratios": ratios(gal_med),
    }
    (OUT / "cascade_summary.json").write_text(json.dumps(summary, indent=2))

    if all_s[1]:
        np.savez_compressed(
            OUT / "cascade_embeddings.npz",
            chain_ids=np.asarray(kept_chain_ids, dtype=object),
            **{f"s{k}": np.stack(all_s[k]) for k in range(1, K + 1)},
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
