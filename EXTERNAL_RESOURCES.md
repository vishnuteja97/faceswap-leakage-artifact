# External data and tools

These links describe the data and models behind the experiments. They are not
needed for the [bundled-data reproduction commands](README.md). This artifact
does not include an image-generation pipeline.

## Data

| Resource | Source | Purpose |
|---|---|---|
| VGGFace2-HQ | [Project and downloads](https://github.com/NNNNAI/VGGFace2-HQ) | Published, pre-aligned and restored images used for evaluation. |
| MAAD-Face | [Project](https://github.com/pterhoer/MAAD-Face), [annotations](https://github.com/pterhoer/MAAD-Face/releases/tag/MAADFACE) | Attributes used for harder-negative selection. |

## Recognition models

The bundled single-swap scores use **Buffalo-L** through InsightFace and
**FaceNet512** through DeepFace. Cascade scores use Buffalo-L.

- [InsightFace setup](https://github.com/deepinsight/insightface/tree/master/python-package)
  and [model packs](https://github.com/deepinsight/insightface/releases/tag/model-zoo):
  Buffalo-L uses `w600k_r50.onnx`.
- [DeepFace setup and supported models](https://github.com/serengil/deepface):
  select `Facenet512`.

## Swap tools

Checkpoint paths below are relative to each tool's checkout. Follow the linked
project instructions for installation and supporting models.

| Tool | Code and setup | Weights / expected input |
|---|---|---|
| FaceFusion | [Repository](https://github.com/facefusion/facefusion), [installation guide](https://docs.facefusion.io/installation) | [`models-3.3.0` assets](https://github.com/facefusion/facefusion-assets/releases/tag/models-3.3.0): use `hyperswap_1a_256.onnx` at `.assets/models/hyperswap_1a_256.onnx`. FaceFusion also downloads supporting models. |
| BlendFace / BlendSwap | [Repository and installation](https://github.com/mapooon/BlendFace#installation), [face-swapping instructions](https://github.com/mapooon/BlendFace#face-swapping) | [Publisher's BlendSwap checkpoint](https://drive.google.com/file/d/1ssTKnNVGomtrtPl57EOGzKNz62bh6mSs/view?usp=sharing), placed at `swapping/checkpoints/blendswap.pth`. Use the swapping checkpoint, not just the BlendFace identity encoder. |
| CanonSwap | [Environment setup](https://github.com/Pixel-Talk/CanonSwap#environment-setup), [all model downloads and directory layout](https://github.com/Pixel-Talk/CanonSwap#model-download) | [Main checkpoint](https://drive.google.com/file/d/1uDWiIam1jziU918iOZY2ATE2dw9aqYAr/view?usp=drive_link) at `pretrained_weights/combined_weights.pth`; also obtain the documented InsightFace, ArcFace and landmark files. |
| DiffFace | [Repository/setup](https://github.com/hxngiee/DiffFace), [pretrained-weight download section](https://github.com/hxngiee/DiffFace#download-pretrained-weights) | Publisher's linked weight collection: `checkpoints/{Arcface.tar,FaceParser.pth,GazeEstimator.pt,Model.pt}`. |
| DiffSwap | [Repository and installation/weight link](https://github.com/wl-zhao/DiffSwap#installation) | Publisher's checkpoint collection: `checkpoints/diffswap.pth` plus supporting files, including `shape_predictor_68_face_landmarks.dat` required by this adapter. |
| E4S | [Repository](https://github.com/e4s2022/e4s), [installation and complete checkpoint layout](https://github.com/e4s2022/e4s/blob/main/INSTALLATION.md) | [RGI checkpoint](https://drive.google.com/file/d/1cyJTYRO5G4kcugAcgSJ7cMsE96GzV_hq/view?usp=share_link), `pretrained_ckpts/e4s/iteration_300000.pt`; obtain the parser, FaceVid2Vid and GPEN dependencies listed in the installation document too. |
| FaceShifter reimplementation cited in the paper | [maum-ai/faceshifter](https://github.com/maum-ai/faceshifter), [training and Docker setup](https://github.com/maum-ai/faceshifter#training), [data preparation](https://github.com/maum-ai/faceshifter#datasets), [inference](https://github.com/maum-ai/faceshifter#inference) | The authors trained their own checkpoint on CelebA-HQ. Their evaluated weights are not included in this artifact. |

Dataset and model access is governed by each provider's terms. The external
projects supply their own setup instructions; these links do not constitute a
tested procedure for recreating this paper's image experiments from scratch.
