#!/usr/bin/env python3
"""Fit large-N affine operators for auxiliary swappers.

Inputs are raw buffalo_l embeddings from embed_aux_swaps.py:
    s = A d + B t + c
The train/test split is the explicit identity-disjoint split used when the
FaceFusion-derived pair manifest was generated.

Outputs:
  generality/outputs/aux_operators/<tool>_operator_raw.npz
  generality/outputs/aux_operators/aux_operator_summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/opt/reproduction/faceswap_experiments/Cascade_Privacy_Dynamics")
EMB = ROOT / "generality/outputs/aux_embeddings"
OUT = ROOT / "generality/outputs/aux_operators"
OUT.mkdir(parents=True, exist_ok=True)
TOOLS = ("blendface", "canonswap")
DIM = 512
LAMBDAS = (0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def design(d: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.concatenate([d, t, np.ones((len(d), 1), dtype=d.dtype)], axis=1)


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    p = X.shape[1]
    xtx = (X.T @ X).astype(np.float64)
    xty = (X.T @ Y).astype(np.float64)
    mat = xtx + lam * np.eye(p, dtype=np.float64)
    mat[-1, -1] -= lam
    return np.linalg.solve(mat, xty)


def choose_lambda(X: np.ndarray, Y: np.ndarray) -> tuple[float, dict[str, float]]:
    rng = np.random.default_rng(11)
    perm = rng.permutation(len(X))
    n_val = max(1, int(round(0.2 * len(perm))))
    va = perm[:n_val]
    tr = perm[n_val:]
    scores: dict[str, float] = {}
    best_lam = LAMBDAS[0]
    best_cos = -1.0
    for lam in LAMBDAS:
        W = fit_ridge(X[tr], Y[tr], lam)
        pred = X[va] @ W
        cos = float((l2(pred) * l2(Y[va])).sum(1).mean())
        scores[str(lam)] = cos
        if cos > best_cos:
            best_cos = cos
            best_lam = lam
    return best_lam, scores


def metrics(pred: np.ndarray, s: np.ndarray, d: np.ndarray, t: np.ndarray) -> dict[str, float]:
    pred_u = l2(pred)
    s_u = l2(s)
    return {
        "mean_cos_pred_swap": float((pred_u * s_u).sum(1).mean()),
        "median_cos_pred_swap": float(np.median((pred_u * s_u).sum(1))),
        "mean_r2": float(np.mean(1.0 - ((pred - s) ** 2).mean(0) / np.maximum(s.var(0), 1e-12))),
        "mse": float(np.mean((pred - s) ** 2)),
        "mean_cos_real_target_source": float((s_u * l2(t)).sum(1).mean()),
        "mean_cos_real_donor_source": float((s_u * l2(d)).sum(1).mean()),
    }


def spectrum(M: np.ndarray) -> dict[str, object]:
    Md = M.astype(np.float64)
    eig = np.linalg.eigvals(Md)
    sv = np.linalg.svd(Md, compute_uv=False)
    mod = np.abs(eig)
    order = np.argsort(mod)[::-1]
    return {
        "spectral_radius": float(mod.max()),
        "operator_norm_sigma_max": float(sv[0]),
        "frobenius_norm": float(np.linalg.norm(Md)),
        "stable_rank": float((sv**2).sum() / max(sv[0] ** 2, 1e-12)),
        "mean_abs_eig": float(mod.mean()),
        "n_eig_mod_gt_0p9": int((mod > 0.9).sum()),
        "top20_eig_modulus": [float(x) for x in mod[order][:20]],
        "top20_singular_values": [float(x) for x in sv[:20]],
    }


def rayleigh_gain(B: np.ndarray, t: np.ndarray) -> dict[str, float]:
    u = l2(t).astype(np.float32)
    Bu = (B.astype(np.float32) @ u.T).T
    g = np.einsum("ij,ij->i", u, Bu)
    return {
        "mean": float(g.mean()),
        "median": float(np.median(g)),
        "std": float(g.std()),
        "p10": float(np.quantile(g, 0.1)),
        "p90": float(np.quantile(g, 0.9)),
    }


def fit_tool(tool: str) -> dict[str, object]:
    path = EMB / f"{tool}_vggface2_ffpairs_buffalo_l_raw.npz"
    z = np.load(path, allow_pickle=True)
    d = np.asarray(z["donor_embeddings"], np.float32)
    t = np.asarray(z["target_embeddings"], np.float32)
    s = np.asarray(z["swap_embeddings"], np.float32)
    splits = np.asarray(z["splits"], dtype=object)
    tr = np.where(splits == "train")[0]
    te = np.where(splits == "test")[0]
    Xtr = design(d[tr], t[tr])
    Xte = design(d[te], t[te])
    lam, cv = choose_lambda(Xtr, s[tr])
    print(f"[fit_aux] {tool}: n_train={len(tr)} n_test={len(te)} lambda={lam}", flush=True)
    W = fit_ridge(Xtr, s[tr], lam)
    A = W[:DIM].T.copy()
    B = W[DIM : 2 * DIM].T.copy()
    c = W[2 * DIM :].squeeze(0).copy()
    pred = Xte @ W
    resid = (s[te] - pred).astype(np.float64)
    tr_sig_e = float(np.trace(np.cov(resid, rowvar=False)))
    tr_cov_s = float(np.trace(np.cov(s[te].astype(np.float64), rowvar=False)))
    met = metrics(pred, s[te], d[te], t[te])
    spB = spectrum(B)
    spA = spectrum(A)
    rg = rayleigh_gain(B, t[te])
    out_npz = OUT / f"{tool}_operator_raw.npz"
    np.savez_compressed(
        out_npz,
        A=A.astype(np.float32),
        B=B.astype(np.float32),
        c=c.astype(np.float32),
        lam=np.float32(lam),
        eigB=np.linalg.eigvals(B.astype(np.float64)).astype(np.complex64),
        svB=np.linalg.svd(B.astype(np.float64), compute_uv=False).astype(np.float32),
        source_npz=str(path),
    )
    return {
        "tool": tool,
        "source_npz": str(path),
        "operator_npz": str(out_npz),
        "lambda": float(lam),
        "cv_cos_by_lambda": cv,
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "fit_metrics": met,
        "residual": {
            "trace_Sigma_e": tr_sig_e,
            "trace_cov_s": tr_cov_s,
            "stochastic_fraction": tr_sig_e / max(tr_cov_s, 1e-12),
        },
        "B_spectrum": spB,
        "A_spectrum": spA,
        "rayleigh_gain_B_target_dir": rg,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", nargs="+", default=list(TOOLS), choices=TOOLS)
    args = ap.parse_args()
    summary_path = OUT / "aux_operator_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for tool in args.tools:
        summary[tool] = fit_tool(tool)
    (OUT / "aux_operator_summary.json").write_text(json.dumps(summary, indent=2))
    for tool in args.tools:
        row = summary[tool]
        print(
            f"{tool}: cos={row['fit_metrics']['mean_cos_pred_swap']:.3f} "
            f"R2={row['fit_metrics']['mean_r2']:.3f} "
            f"rhoB={row['B_spectrum']['spectral_radius']:.3f} "
            f"bu={row['rayleigh_gain_B_target_dir']['mean']:.3f}",
            flush=True,
        )
    print(f"[fit_aux] wrote {OUT/'aux_operator_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
