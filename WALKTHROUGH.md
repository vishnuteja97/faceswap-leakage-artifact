# Reviewer walkthrough

Run commands from the artifact directory. The main reproduction needs Python 3.9+
and NumPy, a CPU and approximately 1 GB of available memory. It does not require
the image dataset or model downloads. Tested with Python 3.10.19 / NumPy 2.2.6;
figures additionally use Matplotlib (tested with 3.10.8).

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python reproduce_part1.py
python reproduce_part2.py
python reproduce_controls.py
python refit_operators.py --cv
python reproduce_diagnostics.py
python validate.py
python -m unittest test_metrics.py test_refit.py
```

The scripts print measurements, cohort counts and the distinction between
recomputed and recorded results. Validation fails with a nonzero exit status
if a checked numerical result or release hash changes. `--cv` is optional for
refitting but is needed to check the released regularization-selection evidence.
For plots, run `python make_figures.py --output ../artifact_figures`. The included PDFs can
also be inspected directly. Plot styling is new; it is not a byte-for-byte
reconstruction of the manuscript artwork. Regenerating PDFs may change their
metadata and therefore their release hashes; the default output directory is
outside the artifact to keep the frozen release hash check intact.

## Paper-to-command map

LaTeX labels refer to the frozen `camera_ready.tex` identified in
`data/camera_ready_reference.json`.

| Paper result | Command | Expected result / coverage |
|---|---|---|
| `tab:mia-metrics`, single-swap inference | `python reproduce_part1.py` | 84 entries, seven tools, two recognizers, median/max scoring; exact at reported precision |
| Donor/target/non-member distributions, `fig:histgrid` | `python reproduce_part1.py`; `python make_figures.py` | Means and seven histogram panels; cohort qualification in AUDIT.txt |
| `tab:crossop`, main linear operators | `python refit_operators.py --cv` | Independent A/B/c solve and R², b_u, spectra; cosine from saved scalar components |
| `fig:spectrum` | `python reproduce_part2.py`; `python make_figures.py` | Eigenvalues and singular values from released B; `operator_spectra.pdf` |
| `fig:predicted` | `python reproduce_diagnostics.py`; `python make_figures.py` | Recorded model Monte Carlo curves versus recomputed measured gallery medians; these are different summary metrics |
| `fig:ratio` | `python reproduce_part2.py`; `python make_figures.py` | Excess ratios and pass-8..25 fitted rates; `cascade_ratios.pdf` |
| `fig:deepauc`, `fig:deeptpr` | `python reproduce_part2.py`; `python make_figures.py` | 75 measured passes; historical quantile thresholds and strict empirical-budget TPR both printed |
| `tab:transplant` | `python reproduce_controls.py` | All 45 AUC cells recomputed from individual scores; `face_context_auc.pdf` also provided |
| Harder-non-member paragraph | `python reproduce_controls.py` | New specified reconstruction versus historical values; not exact historical reproduction |
| `tab:ladder`, `fig:modelval` | `python reproduce_diagnostics.py` | Recorded baseline/residual summaries; no independent baseline refitting or recreated residual figure |
| `fig:learning` | `python reproduce_diagnostics.py`; `python make_figures.py` | Recorded learning-curve points, labelled as recorded |
| Persistent-direction diagnostic | `python reproduce_diagnostics.py` | Historical QR and corrected rank-aware scalar correlations distinguished |

Illustrative photographs, synthetic illustrations and t-SNE are outside the
released numerical reproduction. No image or individual face embedding is bundled.

## Numerical checkpoints

| Tool | Train / test triplets | Lambda | R² | b_u | rho(B) | Pass-25 AUC | Pass-8..25 rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| FaceFusion | 57,048 / 10,667 | 10 | .585 | .183 | .894 | .602 | .966 |
| BlendFace | 56,848 / 10,397 | 3 | .399 | .219 | .920 | .599 | .935 |
| CanonSwap | 16,849 / 7,239 | 10 | .338 | .198 | .854 | .553 | .946 |

The cascade cohorts contain 478, 999 and 999 chains respectively. Each pass pools
ten individual negative scores per chain. The face/context bundle contains 37,140
positive scores and 371,400 negative scores across 45 groups. Its recovered target
scores exactly match the historical CSV, and its AUCs match the retained summaries
to floating-point precision.

Validation checks the 84 table values at one decimal percentage point and 45
face/context AUCs at three decimals, matching the paper's reporting. Cascade
summaries are additionally compared with retained unrounded values within 2e-6.
Refitted coefficients may differ slightly because the original saved operators
are float32; observed maximum coefficient differences are below 3e-5. R² and
b_u are checked at three decimals. These numerical tolerances are not statistical
confidence intervals or a promise of identical fresh GPU runs.

## What the operator exports support

The design matrix is `[donor_raw, target_raw, 1]`. `train.npz` supplies XᵀX and
XᵀY, allowing the ridge solution to be recomputed without loading the old A/B/c.
The intercept is not regularized. As in the original fitting code, training Gram
products use float32; the released solver uses float64. `test.npz` contains held-out
moments for squared-error evaluation and b_u. R² averages per-coordinate
`1 - MSE / variance`, using the original float32 target variances.

`split_manifest.csv` lists source row, donor identity, target identity, train/test
membership and inner-validation assignment. Validation checks no donor or target
identity appears in both the final training and test sets.

FaceFusion's seed-1 five-fold validation uses identity folds and mean validation
MSE. Cross-fold donor/target triplets are excluded during CV but are included in
the final training solve. The auxiliary runs use one seed-11 80/20 row holdout
and highest validation cosine. Their inner holdout is **not identity-disjoint**.
This differs from the paper's broad five-fold description.

Held-out fit cosine and auxiliary candidate-selection cosine use per-sample
predicted-dot-true and predicted-norm components exported for the recovered
solutions. Those scalar components reproduce the original candidates' metrics;
they do not permit independently evaluating arbitrary new models. R² and b_u
are independently evaluated from moments after solving. Main operator refitting
does not establish reproducibility of the separate baseline or learning-curve fits.

## Fresh image experiments

[EXTERNAL_RESOURCES.md](EXTERNAL_RESOURCES.md) provides dataset, tool, checkpoint
and installation links, expected filenames, version guidance and download gaps.
[pipeline/README.md](pipeline/README.md) identifies the recognition protocol,
donor schedules, checkpoint fingerprints, source adapters and recovery commands.
This gives a researcher the methods and provenance needed to reconstruct the
pipeline with their own authorized dataset access. It is not a tested clean-machine
installation or a guarantee of historically identical images. Missing historical
hard-negative assignments and generation environment details remain material
limits, documented in [AUDIT.txt](AUDIT.txt).
