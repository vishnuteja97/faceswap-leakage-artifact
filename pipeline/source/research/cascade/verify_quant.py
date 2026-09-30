#!/usr/bin/env python3
"""Quantitative verification of the linear-dynamics predictions against the deep
FaceFusion cascade (cascade_summary.json + cascade_measurements.csv + cascade_embeddings.npz).

Tests P1 (two-rate fade), P2 (spectral rates b_u -> rho(B)), P3 (first pass removes most),
P4 (floor = non-member baseline; MIA AUC -> 0.5), P5 (persistent directions).
Writes cascade/outputs/verify_quant.json and prints a readable report. CPU-only.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
FITS = HERE.parent / "fits/outputs"
MANIFEST = OUT / "chains_manifest.jsonl"
RAW_NPZ = Path("/opt/reproduction/S&P2027-Artifact/data/D2/embeddings/vggface2_raw_embeddings.npz")
K = 5


def l2(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def auc_mannwhitney(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    avg = np.zeros(len(cnt)); pos_acc = 0
    sums = np.zeros(len(cnt))
    for r, i in zip(ranks, inv):
        sums[i] += r
    avg = sums / cnt
    ranks = avg[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    summ = json.loads((OUT / "cascade_summary.json").read_text())
    pred = json.loads((FITS / "predicted_curve.json").read_text())
    op = np.load(FITS / "operator_raw_vgg.npz")
    B = op["B"].astype(np.float64)
    rhoB = float(np.abs(op["eigB"]).max())
    smaxB = float(op["svB"].max())

    # measured per-pass gallery-median leakage and floor
    g = [summ["gallery_median"][f"pass{k}"]["median"] for k in range(1, K + 1)]
    # pass-0 baseline: s_0 = t_0, so leakage_0 = median cos(t0_image, t0 held-out gallery)
    z0 = np.load(RAW_NPZ, allow_pickle=True)
    raw0 = np.asarray(z0["embeddings"], np.float32)
    chains0 = [json.loads(l) for l in MANIFEST.read_text().splitlines() if l.strip()]
    c0_list = []
    for c in chains0:
        u = l2(raw0[c["t0_image_idx"]].astype(np.float64))
        gal = l2(raw0[np.asarray(c["t0_gallery_idx"], np.int64)].astype(np.float64))
        c0_list.append(float(np.median(gal @ u)))
    c0 = float(np.median(c0_list))
    gmean = [summ["gallery_median"][f"pass{k}"]["mean"] for k in range(1, K + 1)]
    floor = summ.get("floor_nonmember_median_overall")
    nm = [summ["nonmember_median"][f"pass{k}"]["median"] for k in range(1, K + 1)]
    norms = [summ["mean_norm_per_pass"][f"pass{k}"] for k in range(1, K + 1)]
    pred_noisy = pred["predicted_leakage_cos_to_t0"]["noisy"][:K]

    # full series including pass 0 (the target itself)
    gfull = [c0] + g
    # P1: monotone decrease + convex on log (excess over floor)
    excess = [max(gi - floor, 1e-6) for gi in gfull]
    monotone = all(gfull[i + 1] <= gfull[i] + 1e-4 for i in range(len(gfull) - 1))
    log_slopes = [np.log(excess[i + 1]) - np.log(excess[i]) for i in range(len(excess) - 1)]
    convex = all(log_slopes[i + 1] >= log_slopes[i] - 1e-3 for i in range(len(log_slopes) - 1))

    # P2/P3: per-pass ratios (excess over floor), step k->k+1 starting at 0->1
    ratios = [excess[i + 1] / excess[i] for i in range(len(excess) - 1)]
    first_drop = gfull[0] - gfull[1]
    later_drops = [gfull[i] - gfull[i + 1] for i in range(1, len(gfull) - 1)]

    # P4: MIA AUC per pass (member gal_med vs nonmember nonmem_med)
    rows = list(csv.DictReader((OUT / "cascade_measurements.csv").open()))
    auc = {}
    for k in range(1, K + 1):
        pos = [float(r[f"gal_med_s{k}"]) for r in rows]
        neg = [float(r[f"nonmem_med_s{k}"]) for r in rows]
        auc[f"pass{k}"] = auc_mannwhitney(pos, neg)

    # P5: persistent directions. Build real basis of top-m eigenspace of B; for each
    # chain measure energy of u=t0_dir in that subspace, correlate with measured decay
    # (s5/s1 leakage ratio: higher = slower = more persistent).
    eigval, eigvec = np.linalg.eig(B)
    idx = np.argsort(-np.abs(eigval))
    m = 20
    top = eigvec[:, idx[:m]]
    basis = np.concatenate([top.real, top.imag], axis=1)
    q, _ = np.linalg.qr(basis)  # (512, <=2m) orthonormal real basis for P
    z = np.load(RAW_NPZ, allow_pickle=True)
    raw = np.asarray(z["embeddings"], np.float32)
    chains = {json.loads(l)["chain_id"]: json.loads(l)
              for l in MANIFEST.read_text().splitlines() if l.strip()}
    aligns, decay_ratio, lk1 = [], [], []
    for r in rows:
        c = chains[r["chain_id"]]
        u = l2(raw[c["t0_image_idx"]].astype(np.float64))
        proj = q.T @ u
        aligns.append(float(np.dot(proj, proj)))  # fraction of energy in P (0..1)
        s1 = max(float(r["gal_med_s1"]) - floor, 1e-6)
        s5 = max(float(r["gal_med_s5"]) - floor, 1e-6)
        decay_ratio.append((s5 / s1) ** (1 / 4))  # geo per-pass survival
        lk1.append(float(r["gal_med_s1"]))

    def spearman(a, b):
        a = np.asarray(a); b = np.asarray(b)
        ra = a.argsort().argsort().astype(float)
        rb = b.argsort().argsort().astype(float)
        ra = (ra - ra.mean()) / (ra.std() + 1e-12)
        rb = (rb - rb.mean()) / (rb.std() + 1e-12)
        return float((ra * rb).mean())

    rho_align_decay = spearman(aligns, decay_ratio)
    rho_align_leak1 = spearman(aligns, lk1)

    report = {
        "n_chains": len(rows),
        "pass0_self_similarity": c0,
        "measured_gallery_median_per_pass": g,
        "measured_series_with_pass0": gfull,
        "measured_gallery_mean_per_pass": gmean,
        "predicted_noisy_per_pass": pred_noisy,
        "measured_nonmember_per_pass": nm,
        "floor_nonmember_median": floor,
        "predicted_plateau": pred["predicted_floor_unrelated_identity"]["plateau"],
        "norms_per_pass": norms,
        "rho_B": rhoB, "sigma_max_B": smaxB, "b_u_expected": 0.18,
        "P1_monotone_fade": bool(monotone),
        "P1_log_convex_two_rate": bool(convex),
        "P1_log_slopes": [float(s) for s in log_slopes],
        "P2_excess_ratios_per_pass": [float(x) for x in ratios],
        "P2_first_ratio_0to1_vs_bu": {"ratio_0to1": float(ratios[0]), "b_u_expected": 0.18},
        "P2_late_ratio_vs_rhoB": {"late_ratio": float(ratios[-1]), "rho_B": rhoB},
        "P3_first_pass_drop": float(first_drop),
        "P3_later_pass_drops": [float(x) for x in later_drops],
        "P3_first_removes_most": bool(first_drop > max(later_drops) if later_drops else True),
        "P4_mia_auc_per_pass": auc,
        "P4_auc_trend_to_chance": bool(auc[f"pass{K}"] < auc["pass1"]),
        "P5_spearman_align_vs_survival": rho_align_decay,
        "P5_spearman_align_vs_pass1_leak": rho_align_leak1,
        "P5_note": "positive corr => targets aligned with B's dominant eigenspace decay slower",
    }
    (OUT / "verify_quant.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
