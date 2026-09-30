"""Maintainer export from saved research outputs (no swaps, inference, or fitting).

This needs the private/local research tree and original embedding store; reviewers
only need the exported files and reproduce_*.py. Paths are supplied via CLI.
"""
import argparse
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from metrics import cascade_series, tail_fit

DEST = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative_image(path):
    return "/".join(Path(str(path)).parts[-2:])


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("/"):
        return "source:" + Path(obj).name
    return obj


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research-root", type=Path, required=True)
    ap.add_argument("--source-data-root", type=Path, required=True,
                    help="Directory containing D1 and D2")
    args = ap.parse_args()
    root, source = args.research_root.resolve(), args.source_data_root.resolve()
    sources = {}

    def record(path, name=None):
        name = name or str(path.relative_to(root))
        sources[name] = digest(path)

    manuscript = root / "paper/camera_ready.tex"
    record(manuscript)
    text = manuscript.read_text()
    # Numeric table expectations are extracted from the reference, never from results.
    mia = {}
    start = text.index("\\label{tab:mia-metrics}")
    block = text[start:text.index("\\end{tabular}", start)]
    for line in block.splitlines():
        if "&" in line and line.split("&")[0].strip().lower() in (
                "blendface", "canonswap", "diffface", "e4s", "facefusion", "faceshifter", "diffswap"):
            mia[line.split("&")[0].strip().lower()] = [float(x) for x in re.findall(r"\d+\.\d+", line)]
    transplant = {}
    start = text.index("\\label{tab:transplant}")
    block = text[start:text.index("\\end{tabular}", start)]
    for line in block.splitlines():
        key = line.split("&")[0].strip()
        if key in ("1", "5", "10", "25", "No swap"):
            transplant[key] = [float(x) for x in re.findall(r"\d+\.\d+", line)]
    macros = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^\n]*)\}", text))
    dump(DEST / "data/camera_ready_reference.json", {
        "source": "paper/camera_ready.tex", "sha256": digest(manuscript),
        "mia_table_percent": mia, "transplant_table_auc": transplant,
        "numeric_macros": {k: float(v) for k, v in macros.items()
                           if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", v)}})

    print("Loading saved original-image embeddings for score recovery...", flush=True)
    raw_path = source / "D2/embeddings/vggface2_raw_embeddings.npz"
    record(raw_path, "source-data/D2/embeddings/vggface2_raw_embeddings.npz")
    z = np.load(raw_path, allow_pickle=True)
    raw, paths = z["embeddings"], z["paths"]
    ranges = {str(u): (int(a), int(b)) for u, a, b in
              zip(z["unique_identity_ids"], z["identity_start_idx"], z["identity_end_idx"])}
    for tool in ("facefusion", "blendface", "canonswap"):
        ff = tool == "facefusion"
        suffix = "" if ff else "_" + tool
        mp = root / "cascade/outputs" / ("chains_manifest_deep.jsonl" if ff else "chains_manifest_deep_aux.jsonl")
        ep = root / f"cascade/outputs/cascade_embeddings_deep{suffix}.npz"
        cp = root / f"cascade/outputs/cascade_measurements_deep{suffix}.csv"
        sp = root / f"cascade/outputs/cascade_summary_deep{suffix}.json"
        for p in (mp, ep, cp, sp):
            record(p)
        chains = lines(mp)
        idkey, imgkey = ("chain_id", "t0_image_idx") if ff else ("pair_id", "target_image_idx")
        ids = [c[idkey] for c in chains]
        ez = np.load(ep, allow_pickle=True)
        assert list(ez["chain_ids" if ff else "pair_ids"].astype(str)) == ids
        assert np.all(ez["depths"] == 25)
        emb = ez["embeddings"]
        rows = {r[idkey]: r for r in csv_rows(cp)}
        target = np.array([[float(rows[cid][f"gal_med_s{k}"]) for k in range(1, 26)] for cid in ids])
        galleries = []
        for c in chains:
            if ff:
                g = np.asarray(c["t0_gallery_idx"])
            else:
                a, b = ranges[str(c["target_id"])]
                g = np.arange(a, b)
                g = g[g != int(c[imgkey])]
            galleries.append(g)
        normalized = [unit(raw[g]) for g in galleries]
        nonmember = np.empty((len(ids), 25, 10), np.float64)
        assignments = []
        pass0 = []
        max_target_error = 0.0
        provenance = []
        for i, c in enumerate(chains):
            controls = [(i + 1 + j * (len(chains) // 11)) % len(chains) for j in range(10)]
            assert all(j != i for j in controls)
            assignments.append([ids[j] for j in controls])
            # Keep float32 matvec order used in the measurement scripts.
            for k in range(25):
                su = unit(emb[i, k])
                actual = float(np.median(normalized[i] @ su))
                max_target_error = max(max_target_error, abs(actual - target[i, k]))
                nonmember[i, k] = [float(np.median(normalized[j] @ su)) for j in controls]
            pass0.append(float(np.median(normalized[i] @ unit(raw[c[imgkey]]))))
            provenance.append({"chain_id": ids[i], "target_image": relative_image(paths[c[imgkey]]),
                               "gallery_images": [relative_image(paths[j]) for j in galleries[i]],
                               "nonmember_chain_ids": assignments[-1],
                               "planned_donor_images": [relative_image(p) for p in c.get("donor_images", c.get("donor_images_deep", []))],
                               "planned_donor_first_pass": 1 if ff else 4,
                               "source_swap_basenames_pass1_to3": c.get("swap_basenames", {}),
                               "donor_note": "Planned inputs from source manifest; fallback/retry donor execution is not independently attested here."})
            if (i + 1) % 250 == 0:
                print(f"{tool}: recovered {i + 1}/{len(ids)} chains", flush=True)
        assert max_target_error < 2e-7, (tool, max_target_error)
        recovered = cascade_series(target, nonmember)
        original = load(sp)
        errors = {}
        for key in ("leak_median", "nonmember_median_pooled", "auc", "tpr_at_1pct_fpr", "tpr_at_0p1pct_fpr"):
            errors[key] = max(abs(a[key] - b[key]) for a, b in zip(recovered, original["per_pass_all"]))
            assert errors[key] < 2e-6, (tool, key, errors[key])
        assert abs(tail_fit(recovered)["geometric_rate"] - original["tail_fit_all"]["geometric_rate"]) < 2e-6
        np.savez_compressed(DEST / f"data/cascade_{tool}_25pass.npz", chain_id=np.array(ids),
                            target=target, nonmember=nonmember, pass0_target=np.array(pass0),
                            nonmember_chain_id=np.array(assignments))
        dump(DEST / f"data/recorded/cascade_{tool}.json", original)
        dump(DEST / f"data/provenance/cascade_{tool}_25pass.json", provenance)
        dump(DEST / f"data/recorded/recovery_{tool}.json", {
            "max_target_score_error": max_target_error, "max_summary_errors": errors,
            "method": "Saved swap embeddings scored against saved original-image galleries; no image generation or inference."})
        print(tool, "verified", errors, flush=True)

        # Release scalar alignments so the persistent-direction diagnostic can be
        # recomputed without publishing original-image face embeddings.
        op_path = root / ("fits/outputs/operator_raw_vgg.npz" if ff else
                         f"generality/outputs/aux_operators/{tool}_operator_raw.npz")
        record(op_path)
        B = np.load(op_path, allow_pickle=True)['B'].astype(float)
        ev, vectors = np.linalg.eig(B)
        top = vectors[:, np.argsort(-np.abs(ev))[:20]]
        basis = np.concatenate([top.real, top.imag], axis=1)
        historical_q, _ = np.linalg.qr(basis)
        u_basis, singular, _ = np.linalg.svd(basis, full_matrices=False)
        rank = int(np.sum(singular > singular[0] * max(basis.shape) * np.finfo(float).eps))
        q = u_basis[:, :rank]
        align, historical_align = [], []
        for c in chains:
            u = unit(raw[int(c[imgkey])].astype(float))
            projection = q.T @ u
            align.append(float(projection @ projection))
            historical_projection = historical_q.T @ u
            historical_align.append(float(historical_projection @ historical_projection))
        np.savez_compressed(DEST / f"data/alignment_{tool}.npz", chain_id=np.array(ids),
                            alignment=np.array(align), historical_qr_alignment=np.array(historical_align),
                            eigenspace_rank=np.array(rank), historical_qr_columns=np.array(historical_q.shape[1]),
                            target_pass1=target[:, 0], target_pass25=target[:, -1])

    # Keep original rankings, adding the missing join key from their source.
    rp = source / "D1/analysis/ReID/outputs/reid_ranks_vggface2_seed_42.npz"
    record(rp, "source-data/D1/analysis/ReID/outputs/reid_ranks_vggface2_seed_42.npz")
    old = np.load(DEST / "data/reid_ranks.npz", allow_pickle=False)
    current = np.load(rp, allow_pickle=True)
    arrays = {k: old[k] for k in old.files}
    for k in arrays:
        if k != "pair_id":
            assert np.array_equal(arrays[k], current[k])
    arrays["pair_id"] = current["pair_id"].astype(str)
    np.savez_compressed(DEST / "data/reid_ranks.npz", **arrays)

    summary_paths = ["cascade/outputs/transplant_deep.json", "fits/outputs/model_validation.json",
                     "fits/outputs/learning_curve.json", "fits/outputs/predicted_curve.json",
                     "fits/outputs/aux_predicted_floors.json", "fits/outputs/fit_summary.json",
                     "generality/outputs/aux_operators/aux_operator_summary.json",
                     "rebuttal/outputs/hard_nonmember_paper_protocol.json"]
    for rel in summary_paths:
        p = root / rel
        record(p)
        dump(DEST / "data/recorded" / p.name, sanitize(load(p)))
    # Per-chain medians cannot reconstruct pooled-control ROC: retain with that limitation.
    p = root / "cascade/outputs/transplant_deep.csv"
    record(p)
    (DEST / "data/transplant_per_chain_medians.csv").write_bytes(p.read_bytes())
    # Replace the historical mixed-depth late-ratio field in the default summary.
    p = root / "generality/outputs/cross_tool/cross_tool_summary.json"
    record(p)
    summary = load(p)
    for row in summary:
        tool = row["tool"].lower()
        saved = load(DEST / f"data/recorded/cascade_{tool}.json")
        row.pop("measured_ratio_late", None)
        row.pop("cascade_measurement", None)
        row["deep_tail_rate_pass8_to25"] = saved["tail_fit_all"]["geometric_rate"]
    dump(DEST / "data/cross_tool_summary.json", summary)

    # Replay exactly the figure-source Monte Carlo, recording its provenance and
    # keeping it distinct from both the old rollout and the Lyapunov floor.
    fig = root / "paper/revision/figtools/make_fig5_and_ratio.py"
    record(fig)
    tree = ast.parse(fig.read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('l2', 'predict')]
    train = root.parent / 'outputs/embeddings/facefusion_vggface2_train_metadata_embeddings.npz'
    record(train, 'source-embeddings/facefusion_vggface2_train_metadata_embeddings.npz')
    for tool in ('blendface', 'canonswap'):
        record(root / f'generality/outputs/aux_embeddings/{tool}_vggface2_ffpairs_buffalo_l_raw.npz')
    namespace = {'np': np, 'json': json, 'ROOT': root,
                 'AUX_SUM': root / 'generality/outputs/aux_operators/aux_operator_summary.json',
                 'FF_EMB': train, 'SEED': 1, 'M_MC': 400, 'K_PRED': 25}
    exec(compile(tree, str(fig), 'exec'), namespace)
    predictions = {}
    for tool in ('facefusion', 'blendface', 'canonswap'):
        curve, floor = namespace['predict'](tool)
        predictions[tool] = {'predicted_mean_cosine_to_source': curve.tolist(),
                             'mean_unrelated_cosine_last3_passes': floor}
    dump(DEST / 'data/recorded/figure5_monte_carlo.json', {
        'status': 'Replayed from figure source; stored summary, not a refit',
        'source': 'paper/revision/figtools/make_fig5_and_ratio.py',
        'seed': 1, 'n_rollouts': 400, 'tools': predictions})
    knn_source = root.parent / 'Analytical Experiments/03_linear_swap_model/outputs/summary.json'
    record(knn_source, 'analytical/03_linear_swap_model/outputs/summary.json')
    original_knn = load(knn_source)['vgg']
    dump(DEST / 'data/recorded/knn_baseline.json', {
        'n_train': original_knn['n_train'], 'n_test_available': original_knn['n_test'],
        'n_test_evaluated': original_knn['n_test_knn'],
        'metrics': original_knn['metrics']['kNN_k25'],
        'note': 'Historical normalized-space baseline; 2000 test samples. Not the full raw-space 10667-test protocol claimed in the table caption.'})
    dump(DEST / "data/provenance/source_hashes.json", sources)


if __name__ == "__main__":
    main()
