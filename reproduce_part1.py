#!/usr/bin/env python3
"""Reproduce the Part 1 (single-swap leakage) results from released score data.

Inputs (in ./data/):
  similarity_scores.csv   per-(pair, tool, recognizer) target/non-member/donor
                          cosine similarities (median and max gallery aggregation)
  reid_ranks.npz          closed-set rank of the true target among 948 candidates

Outputs (printed): per-tool donor/target/non-member similarity and D>T fraction;
membership-inference AUC and TPR at FPR <= 1% and <= 0.1%; closed-set CMC
rank-k identification rates. Recognizer buffalo_l, median aggregation, matching
the paper's main-text numbers. Pure numpy; no scikit-learn required.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
TOOL_ORDER = ["FaceFusion", "DiffFace", "BlendFace", "E4S", "CanonSwap",
              "FaceShifter", "DiffSwap"]
KEY2DISP = {"facefusion": "FaceFusion", "diffface": "DiffFace",
            "blendface": "BlendFace", "e4s": "E4S", "canonswap": "CanonSwap",
            "faceshifter": "FaceShifter", "diffswap": "DiffSwap"}


def load_similarity():
    rows = []
    with open(DATA / "similarity_scores.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["embedding_model"] != "buffalo_l" or r["status"] != "ok":
                continue
            rows.append(r)
    return rows


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def tpr_at_fpr(pos, neg, fpr_budget):
    """Highest TPR achievable with empirical FPR <= fpr_budget (score >= tau)."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    thresholds = np.unique(np.concatenate([pos, neg]))
    best = 0.0
    for tau in thresholds:
        fpr = np.mean(neg >= tau)
        if fpr <= fpr_budget:
            best = max(best, np.mean(pos >= tau))
    return 100.0 * best


def auc(pos, neg):
    """Mann-Whitney AUC (probability a random positive outranks a random negative)."""
    pos = np.asarray(pos, float)
    neg = np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    start = 0
    for i, c in enumerate(counts):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def part1_leakage(rows):
    print("\n=== Part 1: single-swap leakage (VGGFace2, buffalo_l, median) ===")
    print(f"{'Tool':<12}{'donor':>8}{'target':>8}{'nonmem':>8}{'D>T %':>8}"
          f"{'AUC':>7}{'TPR@1%':>9}{'TPR@.1%':>9}")
    by_tool = {}
    for r in rows:
        by_tool.setdefault(r["tool"], []).append(r)
    for key in [k for k, _ in sorted(KEY2DISP.items(),
                                     key=lambda kv: TOOL_ORDER.index(kv[1]))]:
        rs = by_tool.get(key)
        if not rs:
            continue
        tgt = np.array([fnum(r["target_sim_median"]) for r in rs])
        non = np.array([fnum(r["nonmember_sim_median"]) for r in rs])
        don = np.array([fnum(r["donor_sim_median"]) for r in rs])
        dgt = 100.0 * np.mean(don > tgt)
        a = auc(tgt, non)
        t1 = tpr_at_fpr(tgt, non, 0.01)
        t01 = tpr_at_fpr(tgt, non, 0.001)
        print(f"{KEY2DISP[key]:<12}{np.nanmean(don):>8.3f}{np.nanmean(tgt):>8.3f}"
              f"{np.nanmean(non):>8.3f}{dgt:>8.1f}{a:>7.3f}{t1:>9.1f}{t01:>9.1f}")


def part1_cmc():
    z = np.load(DATA / "reid_ranks.npz", allow_pickle=True)
    tool = z["tool"].astype(str)
    emb = z["embedding"].astype(str)
    met = z["metric"].astype(str)
    rank = z["target_rank"].astype(float)
    status = z["status"].astype(str)
    mask = (emb == "buffalo_l") & (met == "median") & (status == "ok")
    print("\n=== Part 1: closed-set re-identification (rank-k %, gallery=948) ===")
    print(f"{'Tool':<12}{'rank-1':>8}{'rank-5':>8}{'rank-10':>9}{'rank-50':>9}")
    for key in [k for k, _ in sorted(KEY2DISP.items(),
                                     key=lambda kv: TOOL_ORDER.index(kv[1]))]:
        m = mask & (tool == key)
        if m.sum() == 0:
            continue
        r = rank[m]
        print(f"{KEY2DISP[key]:<12}{100*np.mean(r<=1):>8.1f}{100*np.mean(r<=5):>8.1f}"
              f"{100*np.mean(r<=10):>9.1f}{100*np.mean(r<=50):>9.1f}")


if __name__ == "__main__":
    rows = load_similarity()
    part1_leakage(rows)
    part1_cmc()
    print("\n(Compare against the paper's single-swap leakage discussion and the"
          " membership-inference / CMC tables.)")
