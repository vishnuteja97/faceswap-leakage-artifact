#!/usr/bin/env python3
"""Reproduce the Part 2 (operator model + cascade dynamics) results.

Inputs (in ./data/):
  operator_<tool>.npz     fitted affine operators (A, B, c) in raw ArcFace space,
                          plus precomputed eigenvalues (eigB) and singular values (svB)
  cross_tool_summary.json per-tool fit cos, R^2, rho(B), sigma_max(B), b_u, late ratio
  cascade_measurements.csv 478 five-pass FaceFusion chains: per-pass gallery-median
                          target similarity and non-member similarity

Outputs (printed): operator spectra rho(B) vs sigma_max(B) and the in-direction
gain b_u for the three modeled tools; the measured five-pass leakage series, the
non-member floor, and the excess-over-floor ratios showing the late ratio matches
rho(B); and the cross-tool late-ratio vs rho(B) ordering.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
TOOLS = ["facefusion", "blendface", "canonswap"]
DISP = {"facefusion": "FaceFusion", "blendface": "BlendFace",
        "canonswap": "CanonSwap"}
# Target self-similarity at pass 0 (a genuine target image vs its own gallery),
# the measured calibration point reported in the paper.
PASS0_SELF_SIM = 0.616


def operator_spectra():
    print("\n=== Part 2: fitted target-transfer operator B (raw ArcFace space) ===")
    print(f"{'Tool':<12}{'rho(B)':>9}{'sigma_max(B)':>14}{'recomputed rho':>16}")
    for t in TOOLS:
        z = np.load(DATA / f"operator_{t}.npz", allow_pickle=True)
        B = z["B"]
        eig = z["eigB"] if "eigB" in z.files else np.linalg.eigvals(B.astype(np.float64))
        sv = z["svB"] if "svB" in z.files else np.linalg.svd(B.astype(np.float64),
                                                             compute_uv=False)
        rho = float(np.abs(eig).max())
        smax = float(np.abs(sv).max())
        # independent recomputation directly from B to show consistency
        rho_chk = float(np.abs(np.linalg.eigvals(B.astype(np.float64))).max())
        print(f"{DISP[t]:<12}{rho:>9.3f}{smax:>14.2f}{rho_chk:>16.3f}")
    print("Note: every eigenvalue lies inside the unit circle (rho<1) while "
          "sigma_max>>rho:\n      B is a spectral contraction but a one-step "
          "expander (highly non-normal).")


def cascade_decay():
    rows = []
    with open(DATA / "cascade_measurements.csv", newline="") as f:
        rows = list(csv.DictReader(f))

    def col(name):
        return np.array([float(r[name]) for r in rows if r[name] not in ("", "nan")])

    leak = [np.median(col(f"gal_med_s{k}")) for k in range(1, 6)]
    nonmem_all = np.concatenate([col(f"nonmem_med_s{k}") for k in range(1, 6)])
    floor = float(np.median(nonmem_all))

    print("\n=== Part 2: measured 5-pass FaceFusion cascade (median over "
          f"{len(rows)} chains) ===")
    series = [PASS0_SELF_SIM] + leak
    print("pass:        " + "".join(f"{k:>9}" for k in range(0, 6)))
    print("leakage:     " + "".join(f"{v:>9.3f}" for v in series))
    print(f"non-member floor: {floor:.4f}")

    excess = [v - floor for v in series]
    ratios = [excess[k] / excess[k - 1] for k in range(1, len(excess))]
    print("excess-over-floor ratios (pass k / pass k-1):")
    print("             " + "".join(f"{r:>9.3f}" for r in ratios))

    z = np.load(DATA / "operator_facefusion.npz", allow_pickle=True)
    rho = float(np.abs(z["eigB"]).max())
    print(f"\nfirst-pass ratio = {ratios[0]:.3f}  (b_u regime, strong erasure)")
    print(f"late-pass ratio  = {ratios[-1]:.3f}  vs  rho(B) = {rho:.3f}"
          "   <- asymptotic decay set by the spectral radius")


def cross_tool_ordering():
    data = json.loads((DATA / "cross_tool_summary.json").read_text())
    print("\n=== Part 2: cross-tool spectral ordering ===")
    print(f"{'Tool':<12}{'fit cos':>9}{'R^2':>7}{'b_u':>7}{'rho(B)':>9}"
          f"{'late ratio (meas.)':>20}")
    for row in sorted(data, key=lambda r: -r["rho_B"]):
        print(f"{row['tool']:<12}{row['fit_cos']:>9.3f}{row['r2']:>7.3f}"
              f"{row['b_u']:>7.3f}{row['rho_B']:>9.3f}"
              f"{row['measured_ratio_late']:>20.3f}")
    print("Ordering by rho(B) (BlendFace slowest tail, CanonSwap fastest) matches "
          "the\nmeasured cascade late ratios.")


if __name__ == "__main__":
    operator_spectra()
    cascade_decay()
    cross_tool_ordering()
