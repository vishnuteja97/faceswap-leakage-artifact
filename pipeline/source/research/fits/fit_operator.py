#!/usr/bin/env python3
"""Fit the single-swap affine operator s = A d + B t + c (+ noise) for FaceFusion and
extract the spectral quantities that govern the cascade.

Outputs (in fits/outputs/):
  operator_<space>_<dataset>.npz   A, B, c, plus spectra and residual covariance summary
  fit_summary.json                 human/paper-readable summary

Two "spaces":
  raw         operators fit on raw buffalo_l embeddings (cascade recursion is affine here)
  normalized  operators fit on L2-normalized embeddings (robustness; matches prior M5)

Key spectral objects (the paper's contribution):
  rho(B)        spectral radius (max |eigenvalue|)  -> asymptotic cascade decay rate
  sigma_max(B)  operator norm (max singular value)  -> one-step transfer bound / transient
  Rayleigh gain E_t[u^T B u], u = t/||t||           -> AR(1) in-direction decay rate
  two-step empirical carrier ratio                  -> direct multiplicative decay predictor
  Lyapunov Sigma_inf (stationary covariance)        -> noise-floor structure
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
EMB = Path("/opt/reproduction/faceswap_experiments/outputs/embeddings")
DIM = 512
SEED = 0
TRAIN_FRAC = 0.7
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
CV_FOLDS = 5

VGG_NPZ = EMB / "facefusion_vggface2_train_metadata_embeddings.npz"
CELEBA_NPZS = [EMB / f"facefusion_celeba_150k_part{i}_metadata_embeddings.npz" for i in (1, 2, 3, 4)]


def log(m: str) -> None:
    print(f"[fit] {m}", flush=True)


def l2(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), eps)


def vgg_id(p: str) -> str:
    return Path(p).parent.name


def celeba_id(p: str) -> str:
    return Path(p).stem


def load(npzs, idfn) -> Dict[str, np.ndarray]:
    if not isinstance(npzs, (list, tuple)):
        npzs = [npzs]
    parts = [np.load(p, allow_pickle=True) for p in npzs]
    d = np.concatenate([np.asarray(p["donor_embeddings"], np.float32) for p in parts])
    t = np.concatenate([np.asarray(p["target_embeddings"], np.float32) for p in parts])
    s = np.concatenate([np.asarray(p["swap_embeddings"], np.float32) for p in parts])
    dp = np.concatenate([np.asarray(p["donor_paths"]) for p in parts])
    tp = np.concatenate([np.asarray(p["target_paths"]) for p in parts])
    return {
        "donor": d, "target": t, "swap": s,
        "donor_ids": np.array([idfn(x) for x in dp]),
        "target_ids": np.array([idfn(x) for x in tp]),
    }


def id_disjoint_split(donor_ids, target_ids, frac=TRAIN_FRAC, seed=SEED):
    all_ids = np.unique(np.concatenate([donor_ids, target_ids]))
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(all_ids))
    ntr = int(round(frac * len(all_ids)))
    tr = set(all_ids[perm[:ntr]].tolist())
    te = set(all_ids[perm[ntr:]].tolist())
    is_tr = np.array([(a in tr) and (b in tr) for a, b in zip(donor_ids, target_ids)])
    is_te = np.array([(a in te) and (b in te) for a, b in zip(donor_ids, target_ids)])
    return np.where(is_tr)[0], np.where(is_te)[0]


def fit_ridge(X, Y, lam):
    p = X.shape[1]
    XtX = (X.T @ X).astype(np.float64)
    XtY = (X.T @ Y).astype(np.float64)
    M = XtX + lam * np.eye(p)
    M[-1, -1] -= lam  # don't regularize bias
    return np.linalg.solve(M, XtY)


def design(d, t):
    return np.concatenate([d, t, np.ones((d.shape[0], 1), d.dtype)], axis=1)


def cv_lambda(d, t, s, did, tid):
    rng = np.random.default_rng(SEED + 1)
    ids = np.unique(np.concatenate([did, tid]))
    perm = rng.permutation(len(ids))
    fold = dict(zip(ids[perm], perm % CV_FOLDS))
    tf = np.array([fold[a] if fold[a] == fold[b] else -1 for a, b in zip(did, tid)])
    best, best_mse = RIDGE_LAMBDAS[0], np.inf
    for lam in RIDGE_LAMBDAS:
        mses = []
        for f in range(CV_FOLDS):
            va = tf == f
            tr = (tf != f) & (tf != -1)
            if va.sum() == 0 or tr.sum() == 0:
                continue
            W = fit_ridge(design(d[tr], t[tr]), s[tr], lam)
            pred = design(d[va], t[va]) @ W
            mses.append(float(np.mean((pred - s[va]) ** 2)))
        m = float(np.mean(mses)) if mses else np.inf
        if m < best_mse:
            best_mse, best = m, lam
    return best


def spectral_block(M: np.ndarray) -> Dict[str, object]:
    Md = M.astype(np.float64)
    eig = np.linalg.eigvals(Md)
    sv = np.linalg.svd(Md, compute_uv=False)
    mod = np.abs(eig)
    order = np.argsort(mod)[::-1]
    eig_sorted = eig[order]
    return {
        "spectral_radius": float(mod.max()),
        "operator_norm_sigma_max": float(sv[0]),
        "frobenius_norm": float(np.linalg.norm(Md)),
        "trace": float(np.trace(Md)),
        "mean_abs_eig": float(mod.mean()),
        "n_eig_mod_gt_0p9": int((mod > 0.9).sum()),
        "n_eig_mod_gt_0p95": int((mod > 0.95).sum()),
        "n_eig_mod_gt_0p99": int((mod > 0.99).sum()),
        "top20_eig_modulus": [float(x) for x in mod[order][:20]],
        "top20_eig_real": [float(x.real) for x in eig_sorted[:20]],
        "top20_eig_imag": [float(x.imag) for x in eig_sorted[:20]],
        "top20_singular_values": [float(x) for x in sv[:20]],
        "stable_rank": float((sv ** 2).sum() / max(sv[0] ** 2, 1e-12)),
    }


def rayleigh_gains(B: np.ndarray, t_unit: np.ndarray) -> Dict[str, float]:
    """u^T B u over real target directions u = t/||t||: the AR(1) in-direction gain."""
    Bu = t_unit @ B.T.astype(np.float32)        # (N, D) = u B^T -> rows are (B u)^T? careful
    # we want u^T B u for each row u: = sum_j u_ij (B u_i)_j ; (B u)_i = B @ u_i
    Bu = (B.astype(np.float32) @ t_unit.T).T     # (N, D), row i = B u_i
    g = np.einsum("ij,ij->i", t_unit, Bu)        # u_i^T B u_i
    return {
        "mean": float(g.mean()), "median": float(np.median(g)),
        "std": float(g.std()), "p10": float(np.quantile(g, 0.1)),
        "p90": float(np.quantile(g, 0.9)),
    }


def carrier_ratios(B: np.ndarray, t_unit: np.ndarray, t_raw: np.ndarray) -> Dict[str, float]:
    """Empirical multiplicative decay of the target carrier projected on its own direction.
    ratio1 = <B t0, u>/<t0,u>;  ratio2 = <B^2 t0, u>/<B t0, u>  (k=1->2)."""
    Bf = B.astype(np.float32)
    u = t_unit
    t0 = t_raw
    Bt = (Bf @ t0.T).T            # B t0
    B2t = (Bf @ Bt.T).T           # B^2 t0
    c0 = np.einsum("ij,ij->i", u, t0)     # <t0,u> = ||t0||
    c1 = np.einsum("ij,ij->i", u, Bt)     # <B t0, u>
    c2 = np.einsum("ij,ij->i", u, B2t)    # <B^2 t0, u>
    r1 = c1 / np.where(np.abs(c0) < 1e-9, np.nan, c0)
    r2 = c2 / np.where(np.abs(c1) < 1e-9, np.nan, c1)
    return {
        "ratio_step0to1_mean": float(np.nanmean(r1)),
        "ratio_step1to2_mean": float(np.nanmean(r2)),
        "ratio_step1to2_median": float(np.nanmedian(r2)),
    }


def metrics(pred, s, d, t) -> Dict[str, float]:
    pu = l2(pred)
    return {
        "mean_cos_pred_swap": float((pu * l2(s)).sum(1).mean()),
        "mean_r2": float(np.mean(1.0 - ((pred - s) ** 2).mean(0) / np.maximum(s.var(0), 1e-12))),
        "mse": float(np.mean((pred - s) ** 2)),
        "mean_cos_pred_donor": float((pu * l2(d)).sum(1).mean()),
        "mean_cos_pred_target": float((pu * l2(t)).sum(1).mean()),
        "mean_cos_real_donor": float((l2(s) * l2(d)).sum(1).mean()),
        "mean_cos_real_target": float((l2(s) * l2(t)).sum(1).mean()),
    }


def fit_space(data, tr, te, space: str, name: str) -> Dict[str, object]:
    d, t, s = data["donor"], data["target"], data["swap"]
    if space == "normalized":
        d, t, s = l2(d), l2(t), l2(s)
    d_tr, t_tr, s_tr = d[tr], t[tr], s[tr]
    d_te, t_te, s_te = d[te], t[te], s[te]
    lam = cv_lambda(d_tr, t_tr, s_tr, data["donor_ids"][tr], data["target_ids"][tr])
    log(f"[{name}/{space}] lambda*={lam}")
    W = fit_ridge(design(d_tr, t_tr), s_tr, lam)
    A = W[:DIM].T.copy()          # s = A d + B t + c ;  W rows are [d|t|1], so A = (W[:DIM])^T
    B = W[DIM:2 * DIM].T.copy()
    c = W[2 * DIM:].squeeze(0).copy()
    # NOTE: with design rows [d, t, 1] and pred = X@W, pred_n = sum_j d_j W[j] + ...
    # => A[n,:] = W[:DIM, n], i.e. A = W[:DIM].T. Confirmed.
    pred_te = design(d_te, t_te) @ W
    met = metrics(pred_te, s_te, d_te, t_te)
    # residual covariance on test
    resid = (s_te - pred_te).astype(np.float64)
    Sigma_e = np.cov(resid, rowvar=False)
    tr_Sig_e = float(np.trace(Sigma_e))
    tr_cov_s = float(np.trace(np.cov(s_te.astype(np.float64), rowvar=False)))
    # spectra
    spB = spectral_block(B)
    spA = spectral_block(A)
    t_unit_te = l2(data["target"][te])  # target directions in *raw* identity space (privacy-relevant)
    rg = rayleigh_gains(B, t_unit_te.astype(np.float32))
    cr = carrier_ratios(B, t_unit_te.astype(np.float32), data["target"][te].astype(np.float32))
    # Lyapunov stationary covariance (only meaningful if rho(B)<1)
    lyap = {}
    if spB["spectral_radius"] < 1.0:
        Cov_d = np.cov(d_tr.astype(np.float64), rowvar=False)
        Sigma_w = A.astype(np.float64) @ Cov_d @ A.T.astype(np.float64) + Sigma_e
        try:
            Sig_inf = solve_discrete_lyapunov(B.astype(np.float64), Sigma_w)
            mu_inf = np.linalg.solve(np.eye(DIM) - B.astype(np.float64),
                                     A.astype(np.float64) @ d_tr.mean(0).astype(np.float64) + c.astype(np.float64))
            lyap = {
                "trace_Sigma_inf": float(np.trace(Sig_inf)),
                "trace_Sigma_w": float(np.trace(Sigma_w)),
                "mu_inf_norm": float(np.linalg.norm(mu_inf)),
            }
        except Exception as ex:
            lyap = {"error": repr(ex)}
    # save operators
    np.savez_compressed(
        OUT / f"operator_{space}_{name}.npz",
        A=A.astype(np.float32), B=B.astype(np.float32), c=c.astype(np.float32),
        lam=np.float32(lam),
        eigB=np.linalg.eigvals(B.astype(np.float64)).astype(np.complex64),
        svB=np.linalg.svd(B.astype(np.float64), compute_uv=False).astype(np.float32),
        d_mean=d_tr.mean(0).astype(np.float32),
        Sigma_e_diag=np.diag(Sigma_e).astype(np.float32),
    )
    return {
        "space": space, "dataset": name, "lambda": float(lam),
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "fit_metrics": met,
        "residual": {
            "trace_Sigma_e": tr_Sig_e, "trace_cov_s": tr_cov_s,
            "stochastic_fraction": tr_Sig_e / max(tr_cov_s, 1e-12),
        },
        "B_spectrum": spB,
        "A_spectrum": spA,
        "rayleigh_gain_B_target_dir": rg,
        "carrier_decay_ratios": cr,
        "lyapunov": lyap,
    }


def run():
    t0 = time.time()
    summary = {"config": {"dim": DIM, "seed": SEED, "train_frac": TRAIN_FRAC,
                          "ridge_lambdas": list(RIDGE_LAMBDAS), "cv_folds": CV_FOLDS,
                          "vgg_npz": str(VGG_NPZ), "celeba_npzs": [str(p) for p in CELEBA_NPZS],
                          "model": "facefusion hyperswap_1a_256", "embedding": "buffalo_l"}}
    log("loading VGG")
    vgg = load(VGG_NPZ, vgg_id)
    tr, te = id_disjoint_split(vgg["donor_ids"], vgg["target_ids"])
    log(f"VGG split train={len(tr)} test={len(te)}")
    summary["vgg_raw"] = fit_space(vgg, tr, te, "raw", "vgg")
    summary["vgg_normalized"] = fit_space(vgg, tr, te, "normalized", "vgg")

    log("loading CelebA")
    cel = load(CELEBA_NPZS, celeba_id)
    trc, tec = id_disjoint_split(cel["donor_ids"], cel["target_ids"])
    log(f"CelebA split train={len(trc)} test={len(tec)}")
    summary["celeba_raw"] = fit_space(cel, trc, tec, "raw", "celeba")

    summary["wall_time_s"] = float(time.time() - t0)
    (OUT / "fit_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log(f"done in {summary['wall_time_s']:.1f}s -> {OUT/'fit_summary.json'}")


if __name__ == "__main__":
    run()
