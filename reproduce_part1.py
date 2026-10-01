"""Camera-ready single-swap table from released scores; NumPy only."""
import argparse
import csv
from pathlib import Path
import numpy as np
from metrics import auc, operating_points

DATA = Path(__file__).resolve().parent / 'data'
TOOLS = ['blendface', 'canonswap', 'diffface', 'e4s', 'facefusion', 'diffswap', 'faceshifter']
EMBEDDINGS = ['buffalo_l', 'Facenet512']


def load_rows(common=False):
    with (DATA / 'similarity_scores.csv').open(newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['status'] == 'ok']
    if common:
        panels = [{r['pair_id'] for r in rows if r['tool'] == t and r['embedding_model'] == e}
                  for t in TOOLS for e in EMBEDDINGS]
        panel = set.intersection(*panels)
        rows = [r for r in rows if r['pair_id'] in panel]
    return rows


def results(common=False):
    rows = load_rows(common)
    result = []
    for tool in TOOLS:
        for emb in EMBEDDINGS:
            selected = [r for r in rows if r['tool'] == tool and r['embedding_model'] == emb]
            for agg in ['median', 'max']:
                pos = np.array([float(r[f'target_sim_{agg}']) for r in selected])
                neg = np.array([float(r[f'nonmember_sim_{agg}']) for r in selected])
                donor = np.array([float(r[f'donor_sim_{agg}']) for r in selected])
                result.append({'tool': tool, 'embedding': emb, 'aggregation': agg,
                               'n': len(pos), 'auc': auc(pos, neg),
                               'target_mean': float(pos.mean()), 'donor_mean': float(donor.mean()),
                               'nonmember_mean': float(neg.mean()), 'donor_gt_target': float(np.mean(donor > pos)),
                               **operating_points(pos, neg)})
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--common-cohort', action='store_true', help='947-pair sensitivity analysis, not the published table')
    args = ap.parse_args()
    print('Cohort:', 'common 947 pairs (sensitivity)' if args.common_cohort else
          'per-tool valid pairs: 947 DiffFace; 948 other tools (published table)')
    print('Tool         Recognizer  Agg       N     AUC   TPR<=1%  TPR<=.1%   Max BA')
    for r in results(args.common_cohort):
        print(f"{r['tool']:<12} {r['embedding']:<11} {r['aggregation']:<6} {r['n']:4d} "
              f"{r['auc']:7.3f} {100*r['tpr_1pct']:9.1f} {100*r['tpr_0p1pct']:9.1f} "
              f"{100*r['balanced_accuracy']:8.1f}")
    print('\nBuffalo_L median: mean donor / target / nonmember; donor > target (%)')
    for r in results(args.common_cohort):
        if r['embedding'] == 'buffalo_l' and r['aggregation'] == 'median':
            print(f"{r['tool']:<12} {r['donor_mean']:.6f} {r['target_mean']:.6f} "
                  f"{r['nonmember_mean']:.6f} {100*r['donor_gt_target']:.1f}")


if __name__ == '__main__':
    main()
