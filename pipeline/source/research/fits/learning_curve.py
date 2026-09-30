#!/usr/bin/env python3
"""Learning curve for identifying the FaceFusion affine target-transfer operator.

This answers the paper limitation "how large a corpus is needed?" by fitting
    s = A d + B t + c
on increasing identity-disjoint training subsets of the existing FaceFusion
VGGFace2 triplet corpus, while evaluating every fit on the same held-out
identity-disjoint test set.

Outputs:
  fits/outputs/learning_curve.json
  fits/outputs/learning_curve.pdf
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

EMB = Path("/opt/reproduction/faceswap_experiments/outputs/embeddings")
VGG_NPZ = EMB / "facefusion_vggface2_train_metadata_embeddings.npz"
DIM = 512
SEED = 17
TRAIN_FRAC = 0.70
RIDGE_LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
SIZES = (500, 1000, 2000, 5000, 10000, 20000, 40000, 57000, 80000, 116000)
VAL_FRAC = 0.20


def l2(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)


def vgg_id(path: object) -> str:
    return Path(str(path)).parent.name


def design(d: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.concatenate([d, t, np.ones((len(d), 1), dtype=d.dtype)], axis=1)


def fit_ridge(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    p = X.shape[1]
    xtx = (X.T @ X).astype(np.float64)
    xty = (X.T @ Y).astype(np.float64)
    mat = xtx + lam * np.eye(p, dtype=np.float64)
    mat[-1, -1] -= lam
    return np.linalg.solve(mat, xty)


def metrics(pred: np.ndarray, s: np.ndarray, d: np.ndarray, t: np.ndarray) -> dict[str, float]:
    pred_u = l2(pred)
    s_u = l2(s)
    return {
        "mean_cos_pred_swap": float((pred_u * s_u).sum(1).mean()),
        "mean_r2": float(np.mean(1.0 - ((pred - s) ** 2).mean(0) / np.maximum(s.var(0), 1e-12))),
        "mse": float(np.mean((pred - s) ** 2)),
        "mean_cos_real_target": float((s_u * l2(t)).sum(1).mean()),
        "mean_cos_real_donor": float((s_u * l2(d)).sum(1).mean()),
    }


def spectrum(B: np.ndarray) -> dict[str, float]:
    Bd = B.astype(np.float64)
    eig = np.linalg.eigvals(Bd)
    sv = np.linalg.svd(Bd, compute_uv=False)
    return {
        "rho_B": float(np.abs(eig).max()),
        "sigma_max_B": float(sv[0]),
        "frob_B": float(np.linalg.norm(Bd)),
        "stable_rank_B": float((sv**2).sum() / max(sv[0] ** 2, 1e-12)),
    }


def rayleigh_gain(B: np.ndarray, t: np.ndarray) -> dict[str, float]:
    u = l2(t).astype(np.float32)
    Bu = (B.astype(np.float32) @ u.T).T
    g = np.einsum("ij,ij->i", u, Bu)
    return {
        "b_u_mean": float(g.mean()),
        "b_u_median": float(np.median(g)),
        "b_u_std": float(g.std()),
    }


def id_disjoint_indices(donor_ids: np.ndarray, target_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    all_ids = np.unique(np.concatenate([donor_ids, target_ids]))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(all_ids))
    n_train = int(round(TRAIN_FRAC * len(all_ids)))
    train_ids = set(all_ids[perm[:n_train]].tolist())
    test_ids = set(all_ids[perm[n_train:]].tolist())
    is_train = np.array([(d in train_ids) and (t in train_ids) for d, t in zip(donor_ids, target_ids)])
    is_test = np.array([(d in test_ids) and (t in test_ids) for d, t in zip(donor_ids, target_ids)])
    return np.where(is_train)[0], np.where(is_test)[0]


def cv_lambda(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator) -> tuple[float, dict[str, float]]:
    perm = rng.permutation(len(X))
    n_val = max(1, int(round(VAL_FRAC * len(perm))))
    val = perm[:n_val]
    tr = perm[n_val:]
    scores: dict[str, float] = {}
    best_lam = RIDGE_LAMBDAS[0]
    best_mse = float("inf")
    for lam in RIDGE_LAMBDAS:
        W = fit_ridge(X[tr], Y[tr], lam)
        pred = X[val] @ W
        mse = float(np.mean((pred - Y[val]) ** 2))
        scores[str(lam)] = mse
        if mse < best_mse:
            best_mse = mse
            best_lam = lam
    return best_lam, scores


def stabilization_threshold(rows: list[dict[str, object]], key: str, tol: float = 0.01) -> int | None:
    final = float(rows[-1][key])
    for row in rows:
        if abs(float(row[key]) - final) <= tol:
            return int(row["n_train"])
    return None


def main() -> None:
    t0 = time.time()
    print(f"[learning] loading {VGG_NPZ}", flush=True)
    z = np.load(VGG_NPZ, allow_pickle=True)
    d = np.asarray(z["donor_embeddings"], np.float32)
    t = np.asarray(z["target_embeddings"], np.float32)
    s = np.asarray(z["swap_embeddings"], np.float32)
    donor_ids = np.array([vgg_id(p) for p in z["donor_paths"]])
    target_ids = np.array([vgg_id(p) for p in z["target_paths"]])
    train_pool, test_idx = id_disjoint_indices(donor_ids, target_ids)
    rng = np.random.default_rng(SEED + 1)
    train_pool = rng.permutation(train_pool)

    # Keep a fixed held-out test set for all learning-curve points.
    max_test = min(12000, len(test_idx))
    test_idx = rng.choice(test_idx, size=max_test, replace=False)
    Xte = design(d[test_idx], t[test_idx])
    ste, dte, tte = s[test_idx], d[test_idx], t[test_idx]

    actual_sizes: list[int] = []
    for requested in SIZES:
        n = min(requested, len(train_pool))
        if not actual_sizes or actual_sizes[-1] != n:
            actual_sizes.append(n)

    rows: list[dict[str, object]] = []
    for n in actual_sizes:
        idx = train_pool[:n]
        X = design(d[idx], t[idx])
        Y = s[idx]
        lam, cv_scores = cv_lambda(X, Y, rng)
        print(f"[learning] n={n} lambda={lam}", flush=True)
        start = time.time()
        W = fit_ridge(X, Y, lam)
        A = W[:DIM].T.copy()
        B = W[DIM : 2 * DIM].T.copy()
        pred = Xte @ W
        met = metrics(pred, ste, dte, tte)
        sp = spectrum(B)
        rg = rayleigh_gain(B, tte)
        row = {
            "requested_n": int(requested),
            "n_train": int(n),
            "n_test": int(len(test_idx)),
            "lambda": float(lam),
            "cv_mse_by_lambda": cv_scores,
            "fit_seconds": float(time.time() - start),
            **met,
            **sp,
            **rg,
            "rho_A": float(np.abs(np.linalg.eigvals(A.astype(np.float64))).max()),
        }
        rows.append(row)
        print(
            "[learning] n={n} cos={cos:.3f} r2={r2:.3f} rhoB={rho:.3f} bu={bu:.3f}".format(
                n=n,
                cos=row["mean_cos_pred_swap"],
                r2=row["mean_r2"],
                rho=row["rho_B"],
                bu=row["b_u_mean"],
            ),
            flush=True,
        )

    rho_thr = stabilization_threshold(rows, "rho_B", tol=0.01)
    bu_thr = stabilization_threshold(rows, "b_u_mean", tol=0.01)
    cos_thr = stabilization_threshold(rows, "mean_cos_pred_swap", tol=0.01)
    summary = {
        "config": {
            "vgg_npz": str(VGG_NPZ),
            "requested_sizes": list(SIZES),
            "actual_sizes": actual_sizes,
            "max_identity_disjoint_train_n": int(len(train_pool)),
            "ridge_lambdas": list(RIDGE_LAMBDAS),
            "seed": SEED,
            "train_frac": TRAIN_FRAC,
            "fixed_test_n": int(len(test_idx)),
        },
        "rows": rows,
        "stabilization_thresholds_tol_0p01": {
            "rho_B": rho_thr,
            "b_u_mean": bu_thr,
            "mean_cos_pred_swap": cos_thr,
        },
        "wall_time_s": float(time.time() - t0),
    }
    (OUT / "learning_curve.json").write_text(json.dumps(summary, indent=2))

    ns = np.array([r["n_train"] for r in rows], dtype=float)
    fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.0), sharex=True)
    axs = axs.ravel()
    series = [
        ("mean_cos_pred_swap", "Held-out cos(pred, swap)"),
        ("mean_r2", "Held-out $R^2$"),
        ("rho_B", r"$\rho(B)$"),
        ("b_u_mean", r"$E[u^TBu]$"),
    ]
    for ax, (key, label) in zip(axs, series):
        ax.plot(ns, [float(r[key]) for r in rows], "o-")
        ax.set_xscale("log")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axs[-2].set_xlabel("training triplets")
    axs[-1].set_xlabel("training triplets")
    fig.suptitle("Operator identification learning curve (FaceFusion VGGFace2)")
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.pdf")
    print(json.dumps(summary["stabilization_thresholds_tol_0p01"], indent=2), flush=True)
    print(f"[learning] wrote {OUT/'learning_curve.json'} and {OUT/'learning_curve.pdf'}", flush=True)


if __name__ == "__main__":
    main()
