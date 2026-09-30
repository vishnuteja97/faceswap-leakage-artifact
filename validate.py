"""Validate released evidence against frozen camera-ready values and source summaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from reproduce_part1 import results as part1, load_rows, TOOLS, EMBEDDINGS
from reproduce_part2 import results as part2
from reproduce_controls import transplant_results, hard_results
from refit_operators import run as refit

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manuscript', type=Path, help='Also verify that this manuscript matches the frozen reference SHA-256')
    args = ap.parse_args()
    reference = json.loads((DATA / 'camera_ready_reference.json').read_text())
    if args.manuscript and sha(args.manuscript) != reference['sha256']:
        raise AssertionError('Manuscript changed since export: review and regenerate reference')
    count = 0
    for tool, expected in reference['mia_table_percent'].items():
        group = [r for r in part1() if r['tool'] == tool]
        actual = [100*r[key] for key in ['tpr_1pct', 'tpr_0p1pct', 'balanced_accuracy'] for r in group]
        assert len(actual) == len(expected) == 12
        for a, b in zip(actual, expected):
            assert f'{a:.1f}' == f'{b:.1f}', (tool, a, b)
            count += 1
    print(f'PASS: {count} single-swap table entries independently recomputed')
    for row in part1():
        assert row['n'] == (947 if row['tool'] == 'diffface' else 948)
    assert all(r['n'] == 947 for r in part1(True))
    ranks = np.load(DATA / 'reid_ranks.npz', allow_pickle=False)
    with (DATA / 'provenance/d1_pairs_manifest.csv').open() as f:
        pair_ids = {r['pair_id'] for r in csv.DictReader(f)}
    assert set(ranks['pair_id']) <= pair_ids
    assert set(ranks['gallery_n']) == {948}
    print('PASS: cohort counts, common-cohort option, and restored ranking provenance join')

    all_results = part2()
    macros = reference['numeric_macros']
    spec_macros = {'facefusion': ('rhoB', 'sigmaxB', 1), 'blendface': ('bfRhoB', 'bfSigma', 1),
                   'canonswap': ('csRhoB', 'csSigma', 2)}
    latest = {'facefusion': ('maucdeep', 'mtaildeep'), 'blendface': ('bfAucDeep', 'bfTailDeep'),
              'canonswap': ('csAucDeep', 'csTailDeep')}
    for tool, r in all_results.items():
        rho, sigma, digits = spec_macros[tool]
        assert f"{r['rho_B']:.3f}" == f"{macros[rho]:.3f}"
        assert round(r['sigma_max_B'], digits) == macros[sigma]
        am, tm = latest[tool]
        assert f"{r['series'][-1]['auc']:.3f}" == f'{macros[am]:.3f}'
        assert f"{r['tail']['geometric_rate']:.3f}" == f'{macros[tm]:.3f}'
        source = json.loads((DATA / f'recorded/cascade_{tool}.json').read_text())
        assert len(r['series']) == 25
        assert r['series'][0]['n'] == (478 if tool == 'facefusion' else 999)
        for a, b in zip(r['series'], source['per_pass_all']):
            for key in ['leak_median', 'nonmember_median_pooled', 'auc', 'tpr_at_1pct_fpr', 'tpr_at_0p1pct_fpr']:
                assert abs(a[key] - b[key]) < 2e-6, (tool, key, a[key], b[key])
            assert a['n_negatives_pooled'] == 10 * a['n']
            # Quantile thresholds are not automatically <= the nominal FPR budget.
            assert a['achieved_fpr_1pct'] < .011
        z = np.load(DATA / f'cascade_{tool}_25pass.npz', allow_pickle=False)
        assert len(set(z['chain_id'])) == len(z['chain_id'])
        assert not np.any(z['nonmember_chain_id'] == z['chain_id'][:, None])
        provenance = json.loads((DATA / f'provenance/cascade_{tool}_25pass.json').read_text())
        assert [p['chain_id'] for p in provenance] == z['chain_id'].tolist()
        assert [p['nonmember_chain_ids'] for p in provenance] == z['nonmember_chain_id'].tolist()
        by_id = {p['chain_id']: p for p in provenance}
        for p in provenance:
            assert p['gallery_images'] and p['target_image'] not in p['gallery_images']
            identity = p['target_image'].split('/')[0]
            assert all(s.split('/')[0] == identity for s in p['gallery_images'])
            assert all(by_id[c]['target_image'].split('/')[0] != identity for c in p['nonmember_chain_ids'])
        align = np.load(DATA / f'alignment_{tool}.npz', allow_pickle=False)
        assert np.array_equal(align['chain_id'], z['chain_id'])
        assert np.isfinite(align['alignment']).all()
        assert np.all((align['alignment'] >= 0) & (align['alignment'] <= 1 + 1e-12))
        assert int(align['eigenspace_rank']) < int(align['historical_qr_columns'])
    print('PASS: all 75 cascade passes, pooled AUC/TPR/floors, tail fits, spectra, latest AUC macros and manifest joins')

    recorded = transplant_results()
    cells = 0
    for label, expected in reference['transplant_table_auc'].items():
        k = 0 if label == 'No swap' else int(label)
        conditions = ['full', 'face_transplant', 'foreign_face'] if k else ['noswap_full', 'noswap_transplant', 'noswap_foreign']
        actual = [recorded[f'{tool}|{cond}|pass{k}']['auc']
                  for tool in ['facefusion', 'blendface', 'canonswap'] for cond in conditions]
        assert len(expected) == len(actual) == 9
        for a, b in zip(actual, expected):
            assert f'{a:.3f}' == f'{b:.3f}', (label, a, b)
            cells += 1
    print(f'PASS: {cells} face/context table entries independently recomputed from pooled scores')
    for tool, keys in {'facefusion': ('fitRtwo', 'buGain'), 'blendface': ('bfRtwo', 'bfBu'),
                       'canonswap': ('csRtwo', 'csBu')}.items():
        r, w = refit(tool)
        assert r['max_B_error_vs_original'] < 3e-5
        original = np.load(DATA / f'operator_{tool}.npz', allow_pickle=False)
        assert np.max(np.abs(w[:512].T - original['A'])) < 3e-5
        assert np.max(np.abs(w[-1] - original['c'])) < 3e-5
        for metric, key in zip(['R2', 'b_u'], keys):
            assert f'{r[metric]:.3f}' == f'{macros[key]:.3f}', (tool, metric, r[metric], macros[key])
        with (DATA / f'operator_fit/{tool}/split_manifest.csv').open() as f:
            splits = list(csv.DictReader(f))
        identities = {s: {x[k] for x in splits if x['split'] == s for k in ['donor_id', 'target_id']}
                      for s in ['train', 'test']}
        assert not identities['train'] & identities['test']
        for split in ['train', 'test']:
            assert sum(x['split'] == split for x in splits) == r['n_' + split]
        executed = json.loads((DATA / f'provenance/executed_donors_{tool}.json').read_text())
        assert len(executed) == (478 if tool == 'facefusion' else 999)
        assert all(len(x['donor_images']) == 25 and x['depth'] == 25 for x in executed)
    print('PASS: three A/B/c refits, independent R2 and b_u, identity-disjoint train/test manifests, donor schedule lengths')
    hard = hard_results()
    expected_hard = json.loads((DATA / 'hard_nonmember_recovered_summary.json').read_text())
    for tool, r in hard.items():
        assert abs(r['auc'] - expected_hard[tool]['auc_hard']) < 1e-12
        assert abs(100*r['tpr_at_1pct_fpr'] - expected_hard[tool]['tpr_hard']) < 1e-12
    print('PASS: reconstructed harder-negative scores reproduce their OWN summary; historical paper agreement is not asserted')
    hashes_path = DATA / 'provenance/release_hashes.json'
    hashes = json.loads(hashes_path.read_text())
    for name, expected in hashes.items():
        assert sha(ROOT / name) == expected, ('Release file changed', name)
    print(f'PASS: {len(hashes)} released-file SHA-256 hashes')
    print('Known manuscript discrepancies and unrecomputed claims remain explicitly documented in AUDIT.txt.')


if __name__ == '__main__':
    main()
