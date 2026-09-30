# Reproducibility Artifact: Toward Interpretable Privacy Guarantees in Face-Swapping Anonymization

This artifact reproduces the headline quantitative results of the paper from
released derived data, using two short Python scripts and no GPU. It is fully
anonymous and contains no author, institution, or file-system information. Raw
images and the large swap corpora are intentionally omitted (for size and
anonymity); the released embedding-derived scores and fitted operators are
sufficient to recompute every number reported in the paper's tables and main
text.

## Contents

```
reproduce_part1.py        Part 1: single-swap leakage, membership inference, re-identification
reproduce_part2.py        Part 2: operator spectra, cascade two-rate decay, cross-tool ordering
requirements.txt          numpy, scipy
data/
  similarity_scores.csv   per-(pair, tool, recognizer) target/non-member/donor cosine
                          similarities (median and max gallery aggregation)
  reid_ranks.npz          closed-set rank of the true target among 948 candidates
  operator_facefusion.npz fitted affine operator (A, B, c) + spectra, raw ArcFace space
  operator_blendface.npz
  operator_canonswap.npz
  cross_tool_summary.json per-tool fit cos, R^2, rho(B), sigma_max(B), b_u, late ratio
  cascade_measurements.csv 478 five-pass FaceFusion dilution chains (per-pass similarities)
  provenance/
    d1_pairs_manifest.csv         pair_id -> donor/target/non-member identity + image
    cascade_chains_manifest.jsonl chain_id -> target + per-pass donor identities + images
```

Identity labels in the score/operator files are opaque integers and contain no
personal or path information. The `provenance/` manifests record exactly which
source images produced each data file, so a reviewer who obtains the underlying
public dataset can rebuild the pipeline from scratch (see "Full reproduction"
below).

## Setup

```
python -m pip install -r requirements.txt
```

## Running

```
python reproduce_part1.py
python reproduce_part2.py
```

Each script runs in a few seconds on a CPU.

## What this reproduces

**`reproduce_part1.py`** (VGGFace2-HQ, `buffalo_l`, median gallery aggregation):

- Per-tool mean swap-to-donor / target / non-member similarity and the fraction
  of pairs in which the swap is closer to the donor than the target (the
  donor-transfer / "which tools anonymize at all" result). Five of the seven
  tools are donor-dominated in >93% of pairs.
- Membership-inference AUC and **TPR at FPR <= 1% and <= 0.1%** (target vs.
  non-member), reproducing the paper's membership-inference table, e.g.
  BlendFace TPR@1% = 61.3, FaceFusion = 30.2, E4S = 12.4.
- Closed-set re-identification rank-1/5/10/50 rates against a 948-candidate
  gallery (e.g. BlendFace rank-1 = 34.2%, FaceFusion = 13.7%, chance = 0.1%).

**`reproduce_part2.py`**:

- Spectra of the fitted target-transfer operator B for FaceFusion, BlendFace,
  CanonSwap: spectral radius rho(B) vs. largest singular value sigma_max(B),
  showing rho(B) < 1 while sigma_max(B) >> rho(B) (highly non-normal). rho(B) is
  independently recomputed from B as a consistency check.
- The measured five-pass FaceFusion dilution series (0.616 -> 0.091 -> ... ->
  0.047), the non-member floor (~0.006), and the excess-over-floor ratios
  (0.139, 0.677, 0.873, 0.915, 0.893): the **late-pass ratio 0.893 matches the
  independently fitted rho(B) = 0.894** to three decimals (the paper's central
  two-rate-decay result).
- The cross-tool ordering: rho(B) orders the cascade tail decay (BlendFace
  slowest, CanonSwap fastest), matching the measured late ratios.

## Full reproduction from source images (optional)

The two scripts above reproduce all reported numbers from the released
embedding-derived data, with no images or GPU required. For reviewers who wish
to rebuild the pipeline end-to-end, the `data/provenance/` manifests give the
exact image-level inputs:

- `d1_pairs_manifest.csv` lists, for every `pair_id`, the donor, target, and
  non-member identities and the specific image used for each. The `pair_id`
  column joins directly to `similarity_scores.csv` and `reid_ranks.npz`.
- `cascade_chains_manifest.jsonl` lists, for every `chain_id`, the target image
  and the fresh donor image used at each of the five passes; `chain_id` joins to
  `cascade_measurements.csv`. The scoring gallery for a chain is all other
  images of its `target_id`.

Image paths are dataset-relative (`<identity>/<file>`); the identity folders are
the public VGGFace2-HQ identifiers. A reviewer can therefore: (i) obtain
VGGFace2-HQ and the public face-swapping tools, (ii) regenerate the swaps named in the
manifests, (iii) extract `buffalo_l` / Facenet512 embeddings, and (iv) recompute
the scores and operators. We do not redistribute the images or swaps themselves
(dataset license and size).

## Scope

This artifact reproduces the paper's analysis and results from released,
embedding-derived data. It does not itself regenerate swap images, run
face-swapping tools, or train models; those require the full image corpora and
GPUs and are out of scope for this lightweight artifact, though the provenance
manifests above make end-to-end reproduction possible. All recognizers and
face-swapping tools referenced are publicly available research systems.

## License

The code and derived data files in this artifact are released under the MIT
License (see `LICENSE`). The provenance manifests reference images from the
public VGGFace2-HQ dataset, which remains subject to its own dataset terms; we
redistribute no source images. The face recognition models and face-swapping
tools referenced are the property of their respective authors under their own
licenses.
