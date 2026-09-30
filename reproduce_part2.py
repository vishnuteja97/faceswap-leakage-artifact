"""Recompute 25-pass cascades and spectra; display labelled recorded fit statistics."""
import json
from pathlib import Path
import numpy as np
from metrics import cascade_series, tail_fit

DATA = Path(__file__).resolve().parent / 'data'
TOOLS = ['facefusion', 'blendface', 'canonswap']


def results():
    output = {}
    for tool in TOOLS:
        with np.load(DATA / f'cascade_{tool}_25pass.npz', allow_pickle=False) as z:
            series = cascade_series(z['target'], z['nonmember'])
            pass0 = float(np.median(z['pass0_target']))
        with np.load(DATA / f'operator_{tool}.npz', allow_pickle=False) as op:
            B = op['B'].astype(float)
            rho = float(np.abs(np.linalg.eigvals(B)).max())
            sigma = float(np.linalg.svd(B, compute_uv=False)[0])
        figure_pass0 = .616 if tool == 'facefusion' else .617
        first_ratio = series[0]['excess_median'] / (figure_pass0 - series[0]['nonmember_median_pooled'])
        output[tool] = {'series': series, 'tail': tail_fit(series), 'rho_B': rho,
                        'sigma_max_B': sigma, 'pass0_median': pass0,
                        'figure_pass0_calibration': figure_pass0, 'first_excess_ratio': first_ratio}
    return output


def main():
    values = results()
    fits = json.loads((DATA / 'cross_tool_summary.json').read_text())
    print('Spectra recomputed from B; R^2 and b_u below are recorded fit summaries.')
    print('Tool          rho(B)  sigma_max(B)    R^2    b_u    deep tail (8..25)')
    for tool, fit in zip(TOOLS, fits):
        assert fit['tool'].lower() == tool
        r = values[tool]
        print(f"{tool:<12} {r['rho_B']:7.3f} {r['sigma_max_B']:13.3f} "
              f"{fit['r2']:6.3f} {fit['b_u']:6.3f} {r['tail']['geometric_rate']:12.3f}")
    for tool, r in values.items():
        print(f"\n{tool}: {r['series'][0]['n']} complete chains, 10 controls per chain, 25 passes")
        print(f"Pass-zero median {r['pass0_median']:.6f}; figure calibration {r['figure_pass0_calibration']:.3f}; first ratio {r['first_excess_ratio']:.3f}")
        print('Pass  Target   Floor  Excess    AUC  TPRq@1%  FPRq(%)  TPR<=1%')
        for p in r['series']:
            print(f"{p['pass']:4d} {p['leak_median']:7.4f} {p['nonmember_median_pooled']:7.4f} "
                  f"{p['excess_median']:7.4f} {p['auc']:6.3f} {100*p['tpr_at_1pct_fpr']:8.2f} "
                  f"{100*p['achieved_fpr_1pct']:8.3f} {100*p['strict_tpr_1pct']:8.2f}")
        print(f"Tail rate: {r['tail']['geometric_rate']:.6f}; fitted crossing of excess .01: "
              f"{r['tail']['passes_to_excess_below_0.01']:.1f} (extrapolation, not measured)")
    print('\nTPRq uses the saved analysis quantile threshold; TPR<= uses a strict empirical FPR budget.')
    print('Mid-depth rates and whole-image deep-tail rates are different comparisons.')
    print('The measured deep-tail ordering does not follow the fitted spectral-radius ordering.')


if __name__ == '__main__':
    main()
