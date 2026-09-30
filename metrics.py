"""Shared CPU-only score statistics. Fractions, not percentages, unless printed."""
import numpy as np


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg) or not np.isfinite(np.r_[pos, neg]).all():
        raise ValueError("AUC requires nonempty finite score arrays")
    # Equivalent to Mann-Whitney ranks, with half credit for ties.
    ordered = np.sort(neg)
    return float(np.mean((np.searchsorted(ordered, pos, side="left") +
                          np.searchsorted(ordered, pos, side="right")) / (2 * len(neg))))


def operating_points(pos, neg):
    """Empirical FPR budgets, including the reject-all threshold."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg) or not np.isfinite(np.r_[pos, neg]).all():
        raise ValueError("Operating points require nonempty finite score arrays")
    thresholds = np.r_[np.unique(np.r_[pos, neg]), np.inf]
    tpr = 1 - np.searchsorted(np.sort(pos), thresholds, side="left") / len(pos)
    fpr = 1 - np.searchsorted(np.sort(neg), thresholds, side="left") / len(neg)
    return {"tpr_1pct": float(tpr[fpr <= .01 + 1e-15].max()),
            "tpr_0p1pct": float(tpr[fpr <= .001 + 1e-15].max()),
            "balanced_accuracy": float(((tpr + 1 - fpr) / 2).max())}


def quantile_tpr(pos, neg, budget):
    """Historical cascade convention; also return achieved empirical FPR."""
    threshold = float(np.quantile(neg, 1 - budget))
    return float(np.mean(pos >= threshold)), float(np.mean(neg >= threshold))


def cascade_series(target, nonmember):
    """target: chains x passes; nonmember: chains x passes x 10 controls."""
    target, nonmember = np.asarray(target), np.asarray(nonmember)
    if nonmember.shape != (*target.shape, 10):
        raise ValueError("Expected ten non-member scores per chain and pass")
    if not np.isfinite(target).all() or not np.isfinite(nonmember).all():
        raise ValueError("Incomplete cascade: scores must all be finite")
    result = []
    for k in range(target.shape[1]):
        pos, neg = target[:, k], nonmember[:, k, :].ravel()
        op = operating_points(pos, neg)
        t1, f1 = quantile_tpr(pos, neg, .01)
        t01, f01 = quantile_tpr(pos, neg, .001)
        floor = float(np.median(neg))
        row = {"pass": k + 1, "n": len(pos), "n_negatives_pooled": len(neg),
               "leak_median": float(np.median(pos)), "nonmember_median_pooled": floor,
               "excess_median": float(np.median(pos) - floor), "auc": auc(pos, neg),
               "tpr_at_1pct_fpr": t1, "tpr_at_0p1pct_fpr": t01,
               "achieved_fpr_1pct": f1, "achieved_fpr_0p1pct": f01,
               "strict_tpr_1pct": op["tpr_1pct"], "strict_tpr_0p1pct": op["tpr_0p1pct"]}
        if result:
            row["excess_ratio"] = row["excess_median"] / result[-1]["excess_median"]
        result.append(row)
    return result


def tail_fit(series):
    rows = [r for r in series if r["pass"] >= 8 and r["excess_median"] > 0]
    slope, intercept = np.polyfit([r["pass"] for r in rows],
                                 np.log([r["excess_median"] for r in rows]), 1)
    return {"fit_from_pass": 8, "geometric_rate": float(np.exp(slope)),
            "passes_to_excess_below_0.01": float((np.log(.01) - intercept) / slope)}
