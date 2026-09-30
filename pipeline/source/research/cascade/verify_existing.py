#!/usr/bin/env python3
"""Initial verification of the cascade model against ALREADY-EXISTING measured cascades:
  - FaceFusion 2-pass gallery leakage (exp21, N=2000)
  - BlendFace / CanonSwap 3-pass gallery leakage (exp22, N~1000 each)

Checks the qualitative predictions (fade; two-rate slowing decay; first-pass-dominant;
diminishing returns) and compares per-pass decay ratios to the predicted FaceFusion shape.
Outputs cascade/outputs/verify_existing.json and .png.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
PRED = Path("/opt/reproduction/faceswap_experiments/Cascade_Privacy_Dynamics/fits/outputs/predicted_curve.json")

# Measured median-gallery leakage (mean over samples) from prior experiments.
MEASURED = {
    "facefusion": {"passes": [1, 2], "median_gallery": [0.08166504884127061, 0.057327197397942654],
                   "source": "exp21 (N=2000)"},
    "blendface":  {"passes": [1, 2, 3], "median_gallery": [0.13668529374641367, 0.10356120563670992, 0.09130508655961603],
                   "source": "exp22 (N=1000)"},
    "canonswap":  {"passes": [1, 2, 3], "median_gallery": [0.12395186320663215, 0.07811131929881773, 0.06236817059177044],
                   "source": "exp22 (N=999)"},
}
FLOOR_EST = 0.02  # rough gallery chance baseline; nailed down in quantitative verification


def ratios(vals):
    return [vals[k] / vals[k - 1] for k in range(1, len(vals))]


def excess_ratios(vals, floor):
    e = [max(v - floor, 1e-6) for v in vals]
    return [e[k] / e[k - 1] for k in range(1, len(e))]


def main():
    pred = json.loads(PRED.read_text())
    out = {"floor_estimate_used": FLOOR_EST, "tools": {}}
    for tool, d in MEASURED.items():
        v = d["median_gallery"]
        out["tools"][tool] = {
            "source": d["source"],
            "median_gallery": v,
            "raw_per_pass_ratios": ratios(v),
            "excess_per_pass_ratios": excess_ratios(v, FLOOR_EST),
            "pass1_to_pass2_drop": v[0] - v[1],
            "total_drop": v[0] - v[-1],
        }
    # qualitative checks
    checks = {}
    for tool, r in out["tools"].items():
        rr = r["raw_per_pass_ratios"]
        checks[tool] = {
            "fade_all_passes_decrease": all(r["median_gallery"][k] < r["median_gallery"][k - 1]
                                            for k in range(1, len(r["median_gallery"]))),
            "two_rate_slowing_ratio_increases": (len(rr) >= 2 and rr[-1] > rr[0]),
        }
    out["qualitative_checks"] = checks
    out["predicted_facefusion_carrier_ratios"] = pred["predicted_carrier_proj_ratios_per_pass"][:4]
    out["predicted_facefusion_cos_to_t0"] = pred["predicted_leakage_cos_to_t0"]["noisy"][:4]
    (OUT / "verify_existing.json").write_text(json.dumps(out, indent=2))

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    colors = {"facefusion": "C0", "blendface": "C1", "canonswap": "C2"}
    for tool, d in MEASURED.items():
        ax[0].plot(d["passes"], d["median_gallery"], "o-", color=colors[tool],
                   label=f"{tool} (measured)")
    ax[0].axhline(FLOOR_EST, color="k", ls=":", lw=1, label="floor est.")
    ax[0].set_xlabel("cascade pass k"); ax[0].set_ylabel("median gallery cos to t0")
    ax[0].set_title("Measured cascade leakage (existing data)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[0].set_xticks([1, 2, 3])

    # per-pass ratio: measured vs predicted shape
    for tool, d in MEASURED.items():
        rr = excess_ratios(d["median_gallery"], FLOOR_EST)
        ax[1].plot(range(1, len(rr) + 1), rr, "o-", color=colors[tool],
                   label=f"{tool} excess ratio")
    pr = pred["predicted_carrier_proj_ratios_per_pass"][:3]
    ax[1].plot(range(1, len(pr) + 1), pr, "k^--", label="predicted FF carrier ratio")
    ax[1].axhline(pred["spectrum"]["rho_B"], color="r", ls=":", label=f"rho(B)={pred['spectrum']['rho_B']:.2f}")
    ax[1].set_xlabel("pass transition k-1 -> k"); ax[1].set_ylabel("per-pass leakage ratio")
    ax[1].set_title("Two-rate decay: ratios increase toward asymptote")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "verify_existing.png")

    print(json.dumps(out["qualitative_checks"], indent=2))
    for tool, r in out["tools"].items():
        print(tool, "raw ratios", [round(x, 3) for x in r["raw_per_pass_ratios"]],
              "excess ratios", [round(x, 3) for x in r["excess_per_pass_ratios"]])


if __name__ == "__main__":
    main()
