# Face-swapping privacy artifact — camera-ready update

This release is cross-checked against `paper/camera_ready.tex`, identified by its
SHA-256 in `data/camera_ready_reference.json`. It updates the original five-pass
artifact to the camera-ready paper's **25-pass, three-tool evaluation**. It uses
saved derived data; no images, face-recognition inference, swap generation, GPU,
or model training are needed to run the score-analysis scripts. An additional CPU
command refits the three linear operators from aggregate training statistics.
Start with [WALKTHROUGH.md](WALKTHROUGH.md) for a paper-to-command map and expected outputs.
For fresh image experiments, [EXTERNAL_RESOURCES.md](EXTERNAL_RESOURCES.md) links
the datasets, all seven tools, recognition models, weights and setup instructions,
with expected filenames and known access/provenance gaps.

The original June release is preserved byte-for-byte under
`legacy/original_submission/` (Git commit `adb3941cbd2ac842215fbff37f8f9edb1475ae10`).
Its historical README and scripts contain the limitations identified in
[AUDIT.txt](AUDIT.txt); use the scripts at this directory's top level for current results.

## Run

Python 3.9+ and NumPy are sufficient. Tested with Python 3.10.19 and NumPy 2.2.6.
Use an existing suitable environment or install `requirements.txt` in your own
isolated environment. Optional thread limits avoid excessive BLAS CPU use:

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python reproduce_part1.py
python reproduce_part2.py
python reproduce_controls.py
python reproduce_diagnostics.py
python refit_operators.py --cv
python validate.py
python -m unittest test_metrics.py test_refit.py
```

To verify the exact manuscript used for this release when its source is available:

```sh
python validate.py --manuscript ../paper/camera_ready.tex
```

## What is reproduced

| Evidence | Coverage | Computation |
|---|---|---|
| Single-swap identity-inference table | All seven tools, two recognizers, median/max; all 84 entries | Recomputed from released individual scores |
| Donor/target/non-member means and donor dominance | All seven tools | Recomputed; cohort counts printed explicitly |
| Operator fits and spectra | Three operators, identity-disjoint train/test manifests | A/B/c refitted from normal equations; R², b_u and spectra recomputed |
| 25-pass cascades | FaceFusion 478 chains; BlendFace and CanonSwap 999 each | Target medians, pooled-control AUC, TPR, floors, excess ratios and pass-8..25 tail fits recomputed from scores |
| Face/context separation table | Passes 1, 5, 10, 25 and no-swap controls; all 45 entries | Recomputed from 37,140 target scores and 371,400 individual negative scores; exact agreement with saved AUC |
| Harder-non-member control | All seven tools | Fully specified reconstruction and scores, compared with historical summaries; **not exact historical assignments** |
| Fit cosine | Three operators | Recomputed from scalar dot/norm components for the recovered solution; not arbitrary new predictions |
| Model baselines and learning curves | Saved diagnostics | Recorded summaries; baseline refitting inputs are not included |
| Figure-source Monte Carlo | 25 simulated passes for three tools | Recorded replay of the figure source, with seed and source hash; distinct from measured gallery scores |
| Persistent-direction diagnostic | Three tools | Correlations recomputed from scalar alignments; corrected rank-aware and historical QR variants distinguished in AUDIT.txt |

These files do **not** reproduce every statement in the manuscript. Remaining
manuscript discrepancies, unavailable inputs, and protocol qualifications are
listed in [AUDIT.txt](AUDIT.txt). A successful validation means the listed numerical
checks pass; it does not validate every scientific interpretation or rerun image generation.

Optional figures (Matplotlib required; tested with 3.10.8):

```sh
python make_figures.py --output ../artifact_figures
```

Ten generated PDF figures are included. These visualize the released measurements;
recorded model predictions and learning curves are labelled as such.

## Latest cascade results

| Tool | Pass-25 full-image AUC | Pass-8..25 geometric rate | Fitted rho(B) |
|---|---:|---:|---:|
| FaceFusion | 0.602 | 0.966 | 0.894 |
| BlendFace | 0.599 | 0.935 | 0.920 |
| CanonSwap | 0.553 | 0.946 | 0.854 |

The measured deep-tail ordering does not follow the spectral-radius ordering.
Mid-depth excess ratios are a separate comparison. The original fifth-pass
FaceFusion ratio 0.893 is not a universal or deep-tail rate.

## Protocol and score schema

`reproduce_part1.py` uses the per-tool valid cohort that reproduces the published
table: 947 pairs for DiffFace, 948 for the other six tools. The paper's claim of a
common 947-pair cohort is inconsistent with those table inputs. Run
`python reproduce_part1.py --common-cohort` for the actual common-cohort sensitivity
analysis; those are explicitly different results, not replacements for the table.
Identity-inference AUC compares target against unrelated-person scores; it is not
training-set membership inference or person-identification accuracy.

Each `data/cascade_<tool>_25pass.npz` contains:

- `chain_id`: N chain identifiers, in fixed order.
- `target`: N x 25 median target-gallery scores.
- `nonmember`: N x 25 x 10 individual non-member-gallery scores.
- `nonmember_chain_id`: N x 10 scoring-control chain identifiers.
- `pass0_target`: N original-target/gallery scores recovered from saved embeddings.

At each pass the **10N individual negative scores** are pooled. Taking one median
per chain before constructing the ROC is a different analysis and must not be used
as a substitute. The individual negative scores were recovered from the retained
swap and original-image embeddings, then checked against all 75 saved per-pass
summaries. No swaps or embeddings were regenerated. Recovery errors are recorded
under `data/recorded/recovery_<tool>.json`.

The saved cascade evaluator used a NumPy quantile threshold. This can yield an
empirical FPR slightly above the nominal 1%/0.1% value. `reproduce_part2.py` prints
the historical TPR, its actual FPR, and a strict empirical-budget TPR separately.
The single-swap table uses the strict empirical-budget convention.

The cascade revision changed the negative controls, not the retained first-five-pass
FaceFusion target scores: it uses ten pooled controls, and its single-control
assignment also changed for 430/478 chains when the assignment list shrank from
500 planned chains to 478 completed chains. Thus original pass-5 AUC 0.709 and
revised pooled AUC 0.716 refer to different controls.

## Provenance and limitations

`data/provenance/cascade_<tool>_25pass.json` records exact target-gallery image
lists and non-member assignments, plus planned donor paths relative to the public
dataset. FaceFusion galleries contain at most 30 images; auxiliary-tool galleries
use all available stored images of the target except its source photograph.
`executed_donors_<tool>.json` incorporates the substitutions recorded in retained
completion logs: 46 for FaceFusion and none recorded for the auxiliary tools.
These logs do not independently attest every previously skipped output.
Current checkout commits, local patches and checkpoint hashes are recorded in
`data/provenance/model_versions.json`; historical generation versions were not
independently frozen. Reference orchestration sources are under `pipeline/source/`.
See [pipeline/README.md](pipeline/README.md) for inputs, recovery commands and limits.

`data/provenance/d1_pairs_manifest.csv` joins to `similarity_scores.csv` and the
restored `pair_id` field in `reid_ranks.npz`. Historical closed-set identification
is available with `python reproduce_part1.py --legacy-reid`; it is not a table in
the camera-ready manuscript. The root-level five-pass `cascade_measurements.csv`
and `cascade_chains_manifest.jsonl` are historical compatibility files; their
original incomplete gallery annotation is not the current 25-pass protocol.

`data/transplant_scores.npz` contains the complete ten-control scores. They were
recovered by rerunning the original fixed-mask composites and recognizer on saved
cascade images; no face swaps were regenerated. All target scores match the old
CSV exactly and all 45 pooled AUCs agree to numerical precision. Context/control
assignments, image hashes and recovery runtime are retained in provenance.
The older `transplant_per_chain_medians.csv` alone cannot reproduce pooled AUC.

Operator exports include normal equations, held-out moments and split identities,
not individual face embeddings. FaceFusion supports independent five-fold MSE
selection. The auxiliary runs actually used one row-based holdout and validation
cosine, unlike the paper's general five-fold description. Their exported scalar
components reproduce selection for the original candidates; they cannot evaluate
arbitrary new models. Full details are in the walkthrough and audit.

`data/provenance/source_hashes.json` fingerprints research inputs using project
relative labels. `release_hashes.json` fingerprints release files. The frozen
manuscript reference includes numeric expectations and a hash, not author details.
No source images or face embeddings are distributed. Public dataset identifiers
in the manifests are retained for provenance; recognizers and swappers have their
own licenses, and obtaining the image dataset remains subject to its terms.

For maintainers with the research inputs, `export_camera_ready.py --help` describes
the base export; [pipeline/README.md](pipeline/README.md) covers the additional
recovery exports. They require retained research inputs and are not required for
reviewer reproduction. After a reviewed change,
regenerate `release_hashes.json` with `python freeze_release.py` and run validation.

## License

Code and released derived data retain the MIT license in `LICENSE`. Dataset
references do not convey rights to redistribute the underlying source images.
