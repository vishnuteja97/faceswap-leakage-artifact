# Face-swapping privacy artifact

This artifact provides code and data for studying identity leakage in face swapping,
including single-swap evaluations and 25-pass cascades.

## Setup

Use Python 3.9+ in an isolated environment. From the repository root, install
NumPy and Matplotlib:

```sh
python -m pip install -r requirements.txt
```

The commands below use bundled data and run on CPU; no image or model downloads
are required. Tested with Python 3.10.19, NumPy 2.2.6 and Matplotlib 3.10.8.

Optional: set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` to limit CPU
thread use. These settings can avoid slowdowns from excessive parallelism;
they are not required for correct results.

## Bundled data

The `data/` folder contains measurements saved from the image experiments.
The scripts analyse these measurements without generating new face swaps.

- **Single swaps** (`similarity_scores.csv`): face-similarity scores comparing
  swapped faces with target, donor and unrelated identities.
- **Cascades** (`cascade_*_25pass.npz`): target and unrelated-identity similarity
  scores at each of the 25 successive swaps.
- **Operator fitting** (`operator_fit/`): aggregate training and test statistics
  for refitting the linear model, plus saved values for cosine evaluation.
  `operator_*.npz` contains fitted matrices for comparison and spectral analysis;
  `cross_tool_summary.json` contains recorded fit summaries.
- **Controls** (`transplant_scores.npz`, `hard_nonmember_scores.csv`): similarity
  scores for face/context separation and reconstructed harder-negative comparisons.
- **Recorded summaries** (`recorded/`): model predictions, learning-curve results
  and the paper's harder-negative results, used for plots and comparisons.

## Run the experiments

| Experiment | Command | Output |
|---|---|---|
| Single-swap identity leakage | `python reproduce_part1.py` | AUC, TPR, balanced accuracy and similarity summaries for seven tools and two recognizers |
| 25-pass cascades | `python reproduce_part2.py` | Per-pass leakage, AUC, TPR and fitted decay rates for FaceFusion, BlendFace and CanonSwap |
| Linear operator fitting | `python refit_operators.py --cv` | Fit quality, operator spectra and regularization selection for the three tools |
| Controls | `python reproduce_controls.py` | Face/context separation AUC and reconstructed harder-negative results |

The harder-negative reconstruction is shown alongside the paper's reported values;
it does not exactly reproduce the original assignments.

## Inspect the results

Numerical results appear in the terminal. The cascade analysis should produce:

| Tool | Pass-25 full-image AUC | Pass-8..25 geometric rate | Fitted rho(B) |
|---|---:|---:|---:|
| FaceFusion | 0.602 | 0.966 | 0.894 |
| BlendFace | 0.599 | 0.935 | 0.920 |
| CanonSwap | 0.553 | 0.946 | 0.854 |

Generate the plots with:

```sh
python make_figures.py --output ../artifact_figures
```

This writes ten PDF figures to `../artifact_figures/`. Model prediction and
learning-curve plots use recorded summaries, labelled in the figures.

For background on the image experiments, see the [external data and tool links](EXTERNAL_RESOURCES.md).

Code and bundled derived data are provided under the [MIT license](LICENSE).
