#!/usr/bin/env python3
"""Predict the cascade leakage trajectory from the fitted operator (BEFORE measuring real
deep cascades). Monte-Carlo simulates the recursion

    s_k = A d_k + B s_{k-1} + c + e_k,   s_0 = t0,

with fresh donors d_k drawn from a held-out donor pool, optional noise e_k ~ N(0, diag),
and reports:
  - predicted leakage curve cos(s_k, u_t0) vs k (with/without noise),
  - the pure carrier curve cos(B^k t0, u_t0) (t0-only, no donors/noise/floor),
  - the predicted floor (cos of stationary swap to an UNRELATED identity),
  - norm stability ||s_k|| vs k,
  - persistent-direction overlap and the predicted two-rate decay (b_u early, rho(B) late),
  - passes-to-threshold from the predicted curve.

Outputs: fits/outputs/predicted_curve.json and fits/outputs/predicted_curve.png
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
EMB = Path("/opt/reproduction/faceswap_experiments/outputs/embeddings")
VGG_NPZ = EMB / "facefusion_vggface2_train_metadata_embeddings.npz"
DIM = 512
K = 10
M = 800            # number of simulated cascades
SEED = 1
rng = np.random.default_rng(SEED)


def l2(X, eps=1e-12):
    return X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), eps)


def main():
    op = np.load(OUT / "operator_raw_vgg.npz", allow_pickle=True)
    A = op["A"].astype(np.float64)
    B = op["B"].astype(np.float64)
    c = op["c"].astype(np.float64)
    sig_e = op["Sigma_e_diag"].astype(np.float64)
    sig_e = np.maximum(sig_e, 0.0)

    z = np.load(VGG_NPZ, allow_pickle=True)
    donors = np.asarray(z["donor_embeddings"], np.float64)
    targets = np.asarray(z["target_embeddings"], np.float64)

    # disjoint pools: targets for t0, a separate donor bank for fresh donors per pass
    idx = rng.permutation(len(targets))
    t0_idx = idx[:M]
    donor_bank = donors[idx[M:M + 20000]] if len(idx) > M + 100 else donors
    T0 = targets[t0_idx]                       # (M, D) original targets (raw)
    u = l2(T0)                                 # target identity directions

    eigB = np.linalg.eigvals(B)
    rhoB = float(np.abs(eigB).max())
    # in-direction gain b_u along these target dirs
    Bu = (B @ u.T).T
    b_u = float(np.einsum("ij,ij->i", u, Bu).mean())

    def simulate(noise: bool):
        s = T0.copy()
        cos_curve = []
        norm_curve = []
        # also track cos to an unrelated identity (floor): shuffle u
        u_other = u[rng.permutation(M)]
        cos_floor = []
        for k in range(1, K + 1):
            dk = donor_bank[rng.integers(0, len(donor_bank), size=M)]
            e = rng.normal(size=(M, DIM)) * np.sqrt(sig_e)[None, :] if noise else 0.0
            s = (A @ dk.T).T + (B @ s.T).T + c[None, :] + e
            sn = l2(s)
            cos_curve.append(float(np.mean(np.einsum("ij,ij->i", sn, u))))
            cos_floor.append(float(np.mean(np.einsum("ij,ij->i", sn, u_other))))
            norm_curve.append(float(np.mean(np.linalg.norm(s, axis=1))))
        return cos_curve, cos_floor, norm_curve

    cos_noiseless, floor_noiseless, norm_noiseless = simulate(noise=False)
    cos_noisy, floor_noisy, norm_noisy = simulate(noise=True)

    # pure carrier B^k t0 projected on u (t0-only, isolates carrier decay)
    carrier = []
    h = T0.copy()
    for k in range(1, K + 1):
        h = (B @ h.T).T
        hn = l2(h)
        carrier.append(float(np.mean(np.einsum("ij,ij->i", hn, u))))
    # carrier magnitude ratios (per pass)
    carrier_proj = []
    h = T0.copy()
    base = np.einsum("ij,ij->i", T0, u)  # <t0,u> = ||t0||
    for k in range(1, K + 1):
        h = (B @ h.T).T
        carrier_proj.append(float(np.mean(np.einsum("ij,ij->i", h, u) / base)))
    ratios = [carrier_proj[0]] + [carrier_proj[k] / carrier_proj[k - 1]
                                  for k in range(1, K)]

    floor = float(np.mean(floor_noisy[-3:]))  # plateau of unrelated-identity cosine
    # passes-to-threshold using the noisy curve excess over floor
    excess = np.array(cos_noisy) - floor
    excess0 = excess[0]
    pass_to = {}
    for tau_frac in (0.5, 0.25, 0.1):
        target_excess = tau_frac * excess0
        k_star = next((k + 1 for k in range(K) if excess[k] <= target_excess), None)
        pass_to[f"reach_{int(tau_frac*100)}pct_of_pass1_excess"] = k_star

    result = {
        "config": {"K": K, "M": M, "seed": SEED, "operator": "raw_vgg"},
        "spectrum": {"rho_B": rhoB, "b_u_target_dir": b_u},
        "predicted_leakage_cos_to_t0": {
            "noiseless": cos_noiseless, "noisy": cos_noisy,
        },
        "predicted_carrier_cos": carrier,
        "predicted_carrier_proj_ratios_per_pass": ratios,
        "predicted_floor_unrelated_identity": {
            "noiseless": floor_noiseless, "noisy": floor_noisy, "plateau": floor,
        },
        "norm_stability_mean_norm": {"noiseless": norm_noiseless, "noisy": norm_noisy},
        "passes_to_threshold": pass_to,
        "note": "u_t0 = t0/||t0|| (target identity direction). Real cascades will be "
                "measured the same way plus the gallery variant.",
    }
    (OUT / "predicted_curve.json").write_text(json.dumps(result, indent=2))

    # plot
    ks = np.arange(1, K + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=150)
    ax[0].plot(ks, cos_noisy, "o-", label=r"model cos$(s_k, t_0)$ (with noise)")
    ax[0].plot(ks, cos_noiseless, "s--", label="model (noiseless, expected)")
    ax[0].plot(ks, carrier, "^:", label=r"pure carrier cos$(B^k t_0, t_0)$")
    ax[0].axhline(floor, color="k", ls=":", lw=1, label=f"predicted floor={floor:.3f}")
    ax[0].set_xlabel("cascade pass k"); ax[0].set_ylabel(r"cos to original target $t_0$")
    ax[0].set_title("Predicted target-leakage decay (FaceFusion fit)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].semilogy(ks, np.maximum(np.array(cos_noisy) - floor, 1e-4), "o-",
                   label="excess leakage (noisy)")
    ax[1].semilogy(ks, np.maximum(np.array(carrier), 1e-4), "^:", label="carrier")
    # reference slopes
    ax[1].semilogy(ks, carrier[0] * (rhoB ** (ks - 1)), "k--", lw=1,
                   label=fr"$\rho(B)^k$ slope ({rhoB:.2f})")
    ax[1].semilogy(ks, carrier[0] * (b_u ** (ks - 1)), "r--", lw=1,
                   label=fr"$b_u^k$ slope ({b_u:.2f})")
    ax[1].set_xlabel("cascade pass k"); ax[1].set_ylabel("excess leakage (log)")
    ax[1].set_title("Two-rate decay: early $b_u$, asymptotic $\\rho(B)$")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "predicted_curve.png")
    print("rho(B)=", rhoB, "b_u=", b_u, "floor=", floor)
    print("predicted noisy cos curve:", [round(x, 4) for x in cos_noisy])
    print("carrier proj ratios:", [round(x, 3) for x in ratios])
    print("passes_to_threshold:", pass_to)


if __name__ == "__main__":
    main()
