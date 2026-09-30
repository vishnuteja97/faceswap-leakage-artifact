# External data, models and setup

The score-analysis and aggregate-refitting commands in [WALKTHROUGH.md](WALKTHROUGH.md)
need none of these downloads. These resources are for researchers rebuilding the
image experiments. Publisher/repository pages were reviewed on 2026-09-30;
large archives and cloud-hosted weights were not downloaded again or tested in a
fresh environment. Cloud links can require login or encounter quota limits.

## Dataset and attribute annotations

| Resource | Source / download | How it is used here |
|---|---|---|
| VGGFace2-HQ | [Project and preparation instructions](https://github.com/NNNNAI/VGGFace2-HQ), including its [Google Drive folder](https://drive.google.com/drive/folders/1ZHy7jrd6cGb2lUa4qYugXe41G_Ef9Ibw?usp=sharing) and alternative provider link | The paper uses the published, pre-aligned, GFPGAN-restored HQ images. Preserve `identity/image.jpg` names so the released manifests resolve. |
| MAAD-Face | [Project and annotation schema](https://github.com/pterhoer/MAAD-Face), [v1.0 annotation release](https://github.com/pterhoer/MAAD-Face/releases/tag/MAADFACE) | Attribute annotations for harder-negative candidate selection; this is not an image download. Join to the source image filenames/identities. |
| Original VGGFace2 | [Oxford dataset page](https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/) | Provenance reference only: Oxford no longer provides the original dataset download. Raw VGGFace2 is not a drop-in replacement for the HQ images used here. |

The [HQ publisher's preparation instructions](https://github.com/NNNNAI/VGGFace2-HQ#generate-the-hq-dataset-by-yourself-if-you-want-to-do-so)
explain restoration/alignment if rebuilding the derivative itself. For this
paper's evaluation, use the published HQ images and the bundled adapter defaults;
do not apply an additional universal alignment or restoration pass. The adapters
already distinguish pre-aligned inputs from in-the-wild inputs. Dataset access
and reuse remain subject to the providers' terms.

## Recognition models

Use the [InsightFace model zoo](https://github.com/deepinsight/insightface/blob/master/model_zoo/README.md)
and its [model-pack release](https://github.com/deepinsight/insightface/releases/tag/model-zoo)
for `buffalo_l` and `antelopev2`. Package setup is documented in the
[InsightFace Python package](https://github.com/deepinsight/insightface/tree/master/python-package).
The publisher lists pretrained models for non-commercial research use.

| Model | File and location | Artifact use |
|---|---|---|
| `buffalo_l` | `~/.insightface/models/buffalo_l/w600k_r50.onnx` | Main score/cascade/control recognizer. Pass this ONNX file to `recover_transplants.py --model`. |
| `antelopev2` | `~/.insightface/models/antelopev2/glintr100.onnx` | Second recognizer in the single-swap table. Obtain the full pack if running its detection/alignment components. |

The tested recovery uses InsightFace **0.7.3**, not the latest package by default.
Its other versions are in `data/provenance/transplant_recovery.json`. The model
SHA-256 for `w600k_r50.onnx` is
`4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43`.
Use `sha256sum /path/to/w600k_r50.onnx` to check a downloaded model before recovery.
Current download locations do not establish that new archives are byte-identical
to historical ones; compare the actual model file. CanonSwap's separate
`antelope` directory follows its own setup instructions below.

## Swap tools and their weights

Paths in this table are relative to each tool's checkout. The included adapters
are in `pipeline/source/source-data/data/D1/swaps/`.

| Tool | Code and setup | Weights / expected input |
|---|---|---|
| FaceFusion | [Repository](https://github.com/facefusion/facefusion), [installation guide](https://docs.facefusion.io/installation) | [`models-3.3.0` assets](https://github.com/facefusion/facefusion-assets/releases/tag/models-3.3.0): use `hyperswap_1a_256.onnx` at `.assets/models/hyperswap_1a_256.onnx`. The pinned checkout's downloader also resolves supporting models. |
| BlendFace / BlendSwap | [Repository and installation](https://github.com/mapooon/BlendFace#installation), [face-swapping instructions](https://github.com/mapooon/BlendFace#face-swapping) | [Publisher's BlendSwap checkpoint](https://drive.google.com/file/d/1ssTKnNVGomtrtPl57EOGzKNz62bh6mSs/view?usp=sharing), placed at `swapping/checkpoints/blendswap.pth`. Use the swapping checkpoint, not just the BlendFace identity encoder. |
| CanonSwap | [Environment setup](https://github.com/Pixel-Talk/CanonSwap#environment-setup), [all model downloads and directory layout](https://github.com/Pixel-Talk/CanonSwap#model-download) | [Main checkpoint](https://drive.google.com/file/d/1uDWiIam1jziU918iOZY2ATE2dw9aqYAr/view?usp=drive_link) at `pretrained_weights/combined_weights.pth`; also obtain the documented InsightFace, ArcFace and landmark files. |
| DiffFace | [Repository/setup](https://github.com/hxngiee/DiffFace), [pretrained-weight download section](https://github.com/hxngiee/DiffFace#download-pretrained-weights) | Publisher's linked weight collection: `checkpoints/{Arcface.tar,FaceParser.pth,GazeEstimator.pt,Model.pt}`. |
| DiffSwap | [Repository and installation/weight link](https://github.com/wl-zhao/DiffSwap#installation) | Publisher's checkpoint collection: `checkpoints/diffswap.pth` plus supporting files, including `shape_predictor_68_face_landmarks.dat` required by this adapter. |
| E4S | [Repository](https://github.com/e4s2022/e4s), [installation and complete checkpoint layout](https://github.com/e4s2022/e4s/blob/main/INSTALLATION.md) | [RGI checkpoint](https://drive.google.com/file/d/1cyJTYRO5G4kcugAcgSJ7cMsE96GzV_hq/view?usp=share_link), `pretrained_ckpts/e4s/iteration_300000.pt`; obtain the parser, FaceVid2Vid and GPEN dependencies listed in the installation document too. |
| FaceShifter reimplementation cited in the paper | [maum-ai/faceshifter](https://github.com/maum-ai/faceshifter), [training and Docker setup](https://github.com/maum-ai/faceshifter#training), [data preparation](https://github.com/maum-ai/faceshifter#datasets), [inference](https://github.com/maum-ai/faceshifter#inference) | The camera-ready paper states that the authors trained their own checkpoint on CelebA-HQ. The repository supplies implementation/training instructions; it is not a download for the authors' trained weights. Those weights are not bundled or linked in this artifact. |

The FaceShifter source above follows the footnote in `camera_ready.tex`, confirmed
by the paper authors. The upstream project is an unofficial AEI-Net
reimplementation. Its pretrained ArcFace link is for the identity encoder, not
the authors' trained swapping model. Retraining with its instructions does not
by itself recreate the exact checkpoint evaluated in the paper.

Local provenance qualification: the inspected checkout's Git remote points to
[graydove/FaceShifter](https://github.com/graydove/FaceShifter), and the bundled
adapter expects `saved_models/G_latest_280000.pth`,
`saved_models/D_latest_280000.pth` and `face_modules/model_ir_se50.pth`.
That checkout observation does not establish the training origin of those weights;
it should not replace the paper's cited implementation. The connection between
the cited training implementation, the local inference adapter and the exact
evaluated checkpoint remains to be documented. Exact fresh-inference reproduction
requires the authors' evaluated generator weights and the matching loading code.

For the three cascade tools, start from the inspected commits below and apply the
included local patch, if nonempty. These are current checkout fingerprints, not
independently attested historical generation versions. Full checkpoint SHA-256
values and patch paths are in [model_versions.json](data/provenance/model_versions.json).

| Tool | Inspected commit |
|---|---|
| FaceFusion | `420d738a6bd25041ca38628b68a9f94b5f9bd837` |
| BlendFace | `0203d91d083eedc5b5b554f9b2380f060532587a` |
| CanonSwap | `dd5100f6348edb4db32c1e5b3e99ce8b7d5f422a` |

## Setup order

1. Obtain VGGFace2-HQ and preserve its identity/filename layout. Use the released
   pair/gallery/donor manifests to select inputs. Obtain MAAD-Face only if rebuilding
   the attribute-based control.
2. Create a separate environment for each swapper using its linked setup guide.
   Use the relevant older dependency versions where the inspected checkout
   requires them; the [PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/)
   and [ONNX Runtime CUDA compatibility guide](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
   help match the runtime to the installed driver/CUDA libraries. FaceFusion's
   setup guide also covers FFmpeg and platform dependencies.
3. Download the exact named checkpoints and their supporting models. Check known
   hashes against the artifact, apply the recorded patches, and configure the
   `/opt/reproduction` defaults in the source snapshots for your machine.
4. Run a small adapter smoke test using the selected manifest inputs, then build
   sequential cascades using each previous generated image as the next target.
   Follow [pipeline/README.md](pipeline/README.md) for gallery scoring, controls,
   recovery commands and provenance requirements.

These links resolve where to obtain dependencies and which variants to select.
They do not provide the retained private intermediate embedding stores or replace
the missing historical assignments/environment records identified in AUDIT.txt.
A clean-machine end-to-end generation run has not been validated by this release.
