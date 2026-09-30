#!/usr/bin/env python3
"""Select FaceFusion deep-cascade chains (no GPU). Builds a manifest of N chains, each:
  - one original target identity t0 (with a held-out gallery of its other images),
  - K fresh donor identities (one image each).

STRICT NO-REUSE GUARANTEE (VGGFace2): every swap-input image used anywhere in the whole
experiment is globally unique. Donor identities are assigned distinctly across ALL chains and
ALL passes (N*K distinct identities), each contributing exactly one (never-reused) image, and
they are disjoint from the N target identities. Target source images are one-per-identity and
distinct. The held-out gallery of each target excludes its source image. An explicit assertion
verifies that no image row is used twice across all targets and donors.

Writes cascade/outputs/chains_manifest.jsonl and chains_summary.json. Validates that all
image paths exist so the GPU job can't silently skip.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
RAW_NPZ = Path("/opt/reproduction/S&P2027-Artifact/data/D2/embeddings/vggface2_raw_embeddings.npz")
VGG_ROOT = Path("/opt/reproduction/VGGface2_None_norm_512_true_bygfpgan")

N_CHAINS = 500
K_PASSES = 5
MIN_IMAGES = 6        # need 1 for t0 source + >=5 for gallery
GALLERY_CAP = 30      # cap gallery size per target
SEED = 7


def main():
    z = np.load(RAW_NPZ, allow_pickle=True)
    paths = np.asarray(z["paths"], dtype=object)
    uids = np.asarray(z["unique_identity_ids"], dtype=object)
    s_idx = np.asarray(z["identity_start_idx"], dtype=np.int64)
    e_idx = np.asarray(z["identity_end_idx"], dtype=np.int64)
    counts = e_idx - s_idx
    uid2pos = {str(u): i for i, u in enumerate(uids)}
    eligible = np.where(counts >= MIN_IMAGES)[0]
    rng = np.random.default_rng(SEED)
    rng.shuffle(eligible)

    need_donors = N_CHAINS * K_PASSES
    if len(eligible) < N_CHAINS + need_donors:
        raise RuntimeError(
            f"not enough eligible identities for no-reuse: need {N_CHAINS + need_donors}, "
            f"have {len(eligible)}")

    target_pos = eligible[:N_CHAINS]
    # globally distinct donor identities: one per (chain, pass) slot, disjoint from targets
    donor_pos_flat = eligible[N_CHAINS:N_CHAINS + need_donors]

    def first_image(uid_pos: int) -> str:
        return str(paths[s_idx[uid_pos]])

    used_rows = set()       # every swap-input image row used anywhere (uniqueness check)
    chains = []
    for ci, tp in enumerate(target_pos):
        tid = str(uids[tp])
        rng_local = np.random.default_rng(SEED * 1000 + ci)
        # gallery = all images of t0 EXCEPT the source image (row s_idx[tp])
        gal_idx = list(range(s_idx[tp] + 1, e_idx[tp]))
        if len(gal_idx) > GALLERY_CAP:
            gal_idx = rng_local.choice(gal_idx, size=GALLERY_CAP, replace=False).tolist()
        # this chain's K donor identities (a distinct slice of the global donor list)
        d_pos = donor_pos_flat[ci * K_PASSES:(ci + 1) * K_PASSES]
        donor_ids = [str(uids[int(p)]) for p in d_pos]
        donor_rows = [int(s_idx[int(p)]) for p in d_pos]   # first image of each donor
        donor_images = [str(paths[r]) for r in donor_rows]

        # uniqueness bookkeeping: source target image + all donor images
        for r in [int(s_idx[tp])] + donor_rows:
            assert r not in used_rows, f"image row {r} reused (chain {ci})"
            used_rows.add(r)

        chains.append({
            "chain_id": f"chain_{ci:04d}",
            "t0_id": tid,
            "t0_image": str(paths[s_idx[tp]]),
            "t0_image_idx": int(s_idx[tp]),                # row into RAW_NPZ embeddings
            "t0_gallery_idx": [int(j) for j in gal_idx],   # rows into RAW_NPZ embeddings
            "donor_ids": donor_ids,
            "donor_images": donor_images,
            "donor_emb_idx": donor_rows,                   # rows into RAW_NPZ embeddings
        })

    # validate existence (count missing) and re-verify global image uniqueness
    missing = 0
    all_input_images = []
    for c in chains:
        all_input_images.append(c["t0_image"])
        all_input_images.extend(c["donor_images"])
        if not Path(c["t0_image"]).is_file():
            missing += 1
        for di in c["donor_images"]:
            if not Path(di).is_file():
                missing += 1
    n_unique_images = len(set(all_input_images))
    assert n_unique_images == len(all_input_images), "duplicate swap-input image detected"

    with (OUT / "chains_manifest.jsonl").open("w") as f:
        for c in chains:
            f.write(json.dumps(c) + "\n")
    summary = {
        "n_chains": len(chains), "k_passes": K_PASSES, "min_images": MIN_IMAGES,
        "gallery_cap": GALLERY_CAP, "seed": SEED,
        "n_eligible_identities": int(len(eligible)),
        "n_distinct_donor_identities": int(need_donors),
        "n_distinct_target_identities": int(N_CHAINS),
        "n_total_swap_input_images": int(len(all_input_images)),
        "n_unique_swap_input_images": int(n_unique_images),
        "no_reuse_guarantee": n_unique_images == len(all_input_images),
        "n_missing_image_paths": int(missing),
        "raw_npz": str(RAW_NPZ), "vgg_root": str(VGG_ROOT),
        "total_swaps_to_generate": len(chains) * K_PASSES,
    }
    (OUT / "chains_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
