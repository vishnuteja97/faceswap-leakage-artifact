"""Recompute alignment correlations; inspect recorded model/learning diagnostics."""
import json
from pathlib import Path
import numpy as np

DATA = Path(__file__).resolve().parent / 'data'


def correlations():
    result = {}
    for tool in ('facefusion', 'blendface', 'canonswap'):
        z = np.load(DATA / f'alignment_{tool}.npz', allow_pickle=False)
        floor = .004 if tool == 'canonswap' else .007
        survival = (np.maximum(z['target_pass25'] - floor, 1e-6) /
                    np.maximum(z['target_pass1'] - floor, 1e-6)) ** (1 / 24)
        def corr(a, b):
            # Original figure diagnostic used argsort ranks (no tie averaging).
            a = a.argsort().argsort().astype(float)
            b = b.argsort().argsort().astype(float)
            return float(np.corrcoef(a, b)[0, 1])
        result[tool] = {'n': len(z['alignment']),
                        'eigenspace_rank': int(z['eigenspace_rank']),
                        'align_vs_pass1': corr(z['alignment'], z['target_pass1']),
                        'align_vs_survival': corr(z['alignment'], survival),
                        'historical_qr_columns': int(z['historical_qr_columns']),
                        'historical_qr_vs_pass1': corr(z['historical_qr_alignment'], z['target_pass1']),
                        'historical_qr_vs_survival': corr(z['historical_qr_alignment'], survival)}
    return result


def main():
    print('Recomputed diagnostic: rank-aware eigenspace alignment; historical QR also shown (see AUDIT.txt):')
    for tool, row in correlations().items():
        print(tool, row)
    validation = json.loads((DATA / 'recorded/model_validation.json').read_text())
    print('\nRECORDED raw-space baselines: cosine / R^2 (not refitted)')
    for name, row in validation['single_swap']['baselines'].items():
        print(f"{name:<24} {row['mean_cos_pred_swap']:.3f} {row['mean_r2']:.3f}")
    knn = json.loads((DATA / 'recorded/knn_baseline.json').read_text())
    print('RECORDED kNN:', knn['note'])
    print(knn['metrics'])
    curve = json.loads((DATA / 'recorded/learning_curve.json').read_text())
    print('\nRECORDED learning curve; train / test / rho(B) / fit cosine / R^2')
    for row in sorted(curve['rows'], key=lambda r: r['n_train']):
        print(row['n_train'], row['n_test'], row['rho_B'], row['mean_cos_pred_swap'], row['mean_r2'])
    pred = json.loads((DATA / 'recorded/figure5_monte_carlo.json').read_text())
    print('\nRECORDED replay of figure-source Monte Carlo: first-pass mean cosine / final-three-pass unrelated mean')
    for tool, row in pred['tools'].items():
        print(tool, row['predicted_mean_cosine_to_source'][0], row['mean_unrelated_cosine_last3_passes'])
    print('These simulation outputs are not Lyapunov plateaus or measured median-gallery scores.')


if __name__ == '__main__':
    main()
