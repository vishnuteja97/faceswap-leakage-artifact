# Recovery and image-pipeline reference

The top-level reviewer commands run from bundled scores and aggregate statistics.
The scripts here explain how those exports were recovered from the retained
research inputs. They need inputs that are not distributed in this artifact.
No new swaps were generated for this update.
Dataset, model-download and environment links are collected in
[EXTERNAL_RESOURCES.md](../EXTERNAL_RESOURCES.md), including the exact variants
used and the unbundled author-trained FaceShifter checkpoint. The primary
FaceShifter source link follows the paper's citation to maum-ai/faceshifter.

## Source snapshots and generation provenance

`source/research/` preserves cascade generation, scoring, transplant and operator
fitting scripts. `source/source-data/data/D1/swaps/` contains the seven single-swap
adapters and alignment helpers. `source/source-data/data/D2/pairs/` preserves the
earlier harder-negative selection code. Only the original machine's home-prefix
defaults were replaced with `/opt/reproduction`; these are reference sources,
not turnkey commands for a fresh directory. `orchestration_sources.json` records
their original hashes and that modification.

For a fresh setup, obtain the underlying tools and their dependencies, the models
and the published VGGFace2-HQ image corpus under their respective access terms.
Configure the source defaults for your directories, retain the already aligned
HQ inputs with the per-tool adapter defaults, and use dataset-relative paths in the released
manifests to locate the exact source/gallery photographs. Run each swapper in its
own compatible environment. The score-only NumPy requirements do not install
these image-generation dependencies.

`data/provenance/model_versions.json` records current local commits, checkpoint
SHA-256 values and modified files for FaceFusion, BlendFace and CanonSwap. The
corresponding `*_current_checkout.patch` files preserve those local changes.
They identify the current inspected setup; they are not independently frozen
historical generation versions. Model weights and third-party libraries are not
bundled. The adapters show the actual model choices and parameters, including
FaceFusion's `hyperswap_1a_256` model.

`data/provenance/executed_donors_<tool>.json` gives each completed chain's source
photograph and ordered 25-donor schedule, incorporating recorded substitutions.
There are 46 recorded FaceFusion substitutions and none in the retained auxiliary
completion records. A completed/skipped status does not independently prove that
no earlier unlogged substitution happened. `execution_log_hashes.json` identifies
the logs used. For a new execution, record each actual donor, input/output hashes,
failures and retries as it runs. Feed the previous generated image into each next
pass; retain failures instead of replacing them with original photographs.

The original auxiliary deep script reuses passes 1..3 and the FaceFusion deep
script reuses passes 1..5. Their deep-stage entrypoints therefore require those
existing earlier outputs. The released 25-donor schedules are a clearer guide for
building an entirely fresh sequence; merely running a deep-stage script will not
create all of its prerequisite images.

## Face/context recognition protocol

`recover_transplants.py` reruns the original fixed ellipse compositor and
Buffalo-L recognizer on **existing** images. The ellipse center is `(0.50w, 0.54h)`
and its axes are `(0.30w, 0.34h)`; the Gaussian feather parameter is 21 pixels at
width 512 (kernel width 43). Inputs are directly resized to 112×112 for the
`w600k_r50.onnx` recognition model; this recovery does not redetect/re-align faces.
The benchmark inputs are already aligned/restored VGGFace2-HQ photographs.

For chain index i in a tool panel of N chains, the ten control indices are
`(i + 1 + j * (N // 11)) % N`, j = 0..9. The context image is the first readable
target photograph when scanning from i+7, excluding the target and ten control
identities. FaceFusion uses at most 30 target-gallery images; auxiliary tools use
all stored target images except the source photograph. Scoring takes each gallery's
median cosine, then pools all ten negative-gallery scores per chain for the ROC.

`data/provenance/transplant_recovery.json` contains context/control assignments,
selected image hashes and runtime/model fingerprints. Each checkpoint contains
15 conditions: three no-swap controls and full/face-only/context-only at passes
1, 5, 10 and 25. Exported NPZ fields are `tool`, `chain_id`, `condition`, `passes`,
`target` (one scalar per row) and `nonmember` (ten scalars per row).

The recovery was tested in Python 3.12 with NumPy 2.2.6, OpenCV 4.12,
InsightFace and ONNX Runtime GPU 1.23.2; exact package versions and model hash are
in the provenance JSON. The CUDA run completed all 2,476 chains successfully at
approximately 4.5 chains/second on the local machine. This is a recovery timing,
not an estimate for generating 25-pass swaps or for another GPU.

## Maintainer commands with retained inputs

Run from the artifact root. Replace the example absolute paths below with your
own; keep recovery work outside the release directory. `RESEARCH` is the original
research tree, `SOURCE_DATA` its companion data store, `IMAGES` the aligned
VGGFace2-HQ directory with identity subdirectories, and `RECOVERY` a new work directory.
Use a NumPy environment for statistics/exports and the recognition environment
described above for `recover_transplants.py`.

```sh
export RESEARCH=/path/to/Cascade_Privacy_Dynamics
export SOURCE_DATA=/path/to/source-store/data
export IMAGES=/path/to/aligned/vggface2
export RECOGNIZER=/path/to/w600k_r50.onnx
export RECOVERY=/path/to/new-recovery-work
export OPENBLAS_NUM_THREADS=1

python pipeline/export_operator_statistics.py \
  --research-root "$RESEARCH" --output data/operator_fit

python pipeline/recover_transplants.py \
  --research-root "$RESEARCH" --source-data-root "$SOURCE_DATA" \
  --image-root "$IMAGES" --model "$RECOGNIZER" \
  --output "$RECOVERY/transplant"

python pipeline/recover_hard_nonmembers.py \
  --source-data-root "$SOURCE_DATA" --artifact-root . \
  --selection-images all --output "$RECOVERY/hard"

python pipeline/export_recovered_controls.py \
  --checkpoint-dir "$RECOVERY/transplant" --artifact-root . \
  --hard-dir "$RECOVERY/hard"
```

For a recognition smoke test, add `--limit 2` and use a separate output directory.
The ten-control assignments still use the full panel. A smoke directory is not
sufficient input for the full-table exporter. For long runs, use a persistent
terminal session such as tmux, retain the log/exit status and monitor the printed
throughput/ETA. Recovery resumes completed per-chain checkpoints with the same
script/model/runtime metadata; changing these requires a fresh output directory.

Required private inputs are checked by the scripts as files are opened:

| Operation | Required retained inputs |
|---|---|
| Operator statistics | `RESEARCH/../outputs/embeddings/facefusion_vggface2_train_metadata_embeddings.npz`, auxiliary embedding NPZs, and original operator NPZs used to check recovery |
| Transplants | Original deep manifests, existing D1/D3/deep-cascade images, aligned source images, `SOURCE_DATA/D2/embeddings/vggface2_raw_embeddings.npz`, recognition model |
| Harder negatives | Same raw embedding store, D1 pairs, D2 saved MAAD rankings, seven D1 swap-embedding caches, released single-swap score CSV |
| Execution provenance | Original generation completion logs, D3 pairs, deep manifests, local third-party checkouts and their checkpoints |

`export_execution_provenance.py --help` describes its explicit root arguments.
The original base export remains `export_camera_ready.py`; run it before these
additional exports if rebuilding the entire release. After reviewing exports,
regenerate plots, run the numerical scripts/tests, then freeze and validate:

```sh
python freeze_release.py --archive ../artifact_camera_ready.zip
python validate.py --manuscript "$RESEARCH/paper/camera_ready.tex"
```

The harder-negative default is a **new reconstruction**, not the recovered
historical assignment. `--selection-images five` reproduces a second, older
candidate-sampling convention; its sensitivity results are also included. Neither
matches all historical values. See AUDIT.txt for the comparison and truncated
candidate-group limitation.
