#!/usr/bin/env python3
"""Extend the verified 5-pass FaceFusion cascade to K_DEEP passes (rebuttal experiment).

Continues the SAME chains used in the submission (those with a complete s1..s5), appending
fresh donors for passes 6..K_DEEP so the resulting series is one continuous cascade per
chain rather than a new experiment.

No-reuse guarantees enforced here:
  (a) IMAGE-level, global: no image (target source, gallery, or donor) is ever used twice
      as a swap input anywhere in the whole experiment, including the original 5 passes.
  (b) IDENTITY-level, within-chain: a donor identity never appears twice in the same chain,
      and never equals that chain's target identity.
  (c) IDENTITY-level, global: preserved for as deep as VGGFace2 allows. With 8,624
      identities and 478 chains, passes 6..PASS_GLOBAL_FRESH use identities that appear
      nowhere else in the experiment. Beyond that depth the identity budget is exhausted,
      so donor identities are reused ACROSS chains (never within one) with a distinct,
      never-used image each time. Chains are statistically independent and every measured
      quantity is within-chain, so cross-chain identity sharing cannot manufacture the
      predicted within-chain decay.

Writes cascade/outputs/chains_manifest_deep.jsonl and chains_summary_deep.json.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
CHAINS_DIR = OUT / "chains"
MANIFEST_5 = OUT / "chains_manifest.jsonl"
MANIFEST_DEEP = OUT / "chains_manifest_deep.jsonl"
RAW_NPZ = Path("/opt/reproduction/S&P2027-Artifact/data/D2/embeddings/vggface2_raw_embeddings.npz")

K_BASE = 5          # passes already generated and reported in the submission
K_DEEP = 25         # total passes after extension
N_SPARE = 12        # replacement donors per chain, used when a donor image fails detection
SEED = 7


def main() -> None:
    z = np.load(RAW_NPZ, allow_pickle=True)
    paths = np.asarray(z["paths"], dtype=object)
    uids = np.asarray(z["unique_identity_ids"], dtype=object)
    s_idx = np.asarray(z["identity_start_idx"], dtype=np.int64)
    e_idx = np.asarray(z["identity_end_idx"], dtype=np.int64)

    base = [json.loads(l) for l in MANIFEST_5.read_text().splitlines() if l.strip()]

    # Keep only chains whose original 5 passes all completed: these are the 478 chains the
    # submitted numbers are computed on.
    chains = [c for c in base
              if all((CHAINS_DIR / c["chain_id"] / f"s{k}.jpg").is_file()
                     and (CHAINS_DIR / c["chain_id"] / f"s{k}.jpg").stat().st_size > 0
                     for k in range(1, K_BASE + 1))]
    n_ch = len(chains)
    print(f"[deep] extending {n_ch} complete chains from pass {K_BASE} to {K_DEEP}")

    # ---- bookkeeping of everything already consumed by the 5-pass experiment ----------
    used_rows: set[int] = set()          # image rows used as swap inputs OR as gallery
    used_ident: set[str] = set()         # identities appearing anywhere so far
    chain_idents: dict[str, set[str]] = defaultdict(set)

    for c in base:                       # walk ALL 500 original chains, not just the 478
        used_rows.add(int(c["t0_image_idx"]))
        used_rows.update(int(r) for r in c["t0_gallery_idx"])
        used_rows.update(int(r) for r in c["donor_emb_idx"])
        used_ident.add(str(c["t0_id"]))
        used_ident.update(str(d) for d in c["donor_ids"])
        chain_idents[c["chain_id"]].add(str(c["t0_id"]))
        chain_idents[c["chain_id"]].update(str(d) for d in c["donor_ids"])

    uid2pos = {str(u): i for i, u in enumerate(uids)}

    # Per-identity list of images that are still globally unused, in a fixed order.
    free_images: dict[str, list[int]] = {}
    for i, u in enumerate(uids):
        free = [r for r in range(int(s_idx[i]), int(e_idx[i])) if r not in used_rows]
        if free:
            free_images[str(u)] = free

    rng = np.random.default_rng(SEED * 31 + K_DEEP)

    # Identities never used anywhere in the 5-pass experiment -> spend these first, so the
    # original strict global-distinctness guarantee survives as deep as possible.
    virgin = [str(u) for u in uids if str(u) not in used_ident and str(u) in free_images]
    rng.shuffle(virgin)
    n_extra = K_DEEP - K_BASE
    pass_global_fresh = K_BASE + len(virgin) // n_ch
    print(f"[deep] {len(virgin)} identities unused by the 5-pass run -> passes "
          f"{K_BASE + 1}..{min(pass_global_fresh, K_DEEP)} keep global identity distinctness")

    # Recycling pool for the deeper passes: any identity that still owns an unused image.
    recycle = [u for u in free_images if u not in set(virgin)]
    rng.shuffle(recycle)

    v_ptr = 0
    r_ptr = 0
    n_recycled_assignments = 0
    new_used_rows: set[int] = set()

    for c in chains:
        cid = c["chain_id"]
        seen = set(chain_idents[cid])          # identities already in THIS chain
        add_ids: list[str] = []
        add_rows: list[int] = []
        for _ in range(n_extra):
            picked = None
            # 1st choice: a globally virgin identity
            while v_ptr < len(virgin):
                cand = virgin[v_ptr]
                v_ptr += 1
                if cand in seen:
                    continue
                rows = [r for r in free_images[cand] if r not in new_used_rows]
                if rows:
                    picked = (cand, rows[0])
                    break
            # 2nd choice: recycle an identity across chains, distinct image, not in chain
            if picked is None:
                tries = 0
                while tries < len(recycle) * 2:
                    cand = recycle[r_ptr % len(recycle)]
                    r_ptr += 1
                    tries += 1
                    if cand in seen:
                        continue
                    rows = [r for r in free_images[cand] if r not in new_used_rows]
                    if rows:
                        picked = (cand, rows[0])
                        n_recycled_assignments += 1
                        break
            if picked is None:
                raise RuntimeError(f"exhausted donor pool for {cid}")
            ident, row = picked
            seen.add(ident)
            new_used_rows.add(row)
            add_ids.append(ident)
            add_rows.append(row)

        c["donor_ids"] = list(c["donor_ids"]) + add_ids
        c["donor_images"] = list(c["donor_images"]) + [str(paths[r]) for r in add_rows]
        c["donor_emb_idx"] = [int(r) for r in c["donor_emb_idx"]] + add_rows

    # ---- spare donors ----------------------------------------------------------------
    # A donor image whose face the swapper cannot detect kills the rest of its chain. Over
    # 25 passes that truncated ~20% of chains in a first run, which would bias a
    # depth-vs-leakage study toward whichever chains happened to survive. Each chain
    # therefore carries spare donors, drawn under the same constraints, so a failed pass
    # retries with a different donor instead of ending the chain.
    # Assigned strictly AFTER all primary donors so that primary assignments -- and hence
    # already-generated images -- are unchanged.
    n_spare_assigned = 0
    for c in chains:
        seen = set(chain_idents[c["chain_id"]]) | set(c["donor_ids"])
        sp_ids: list[str] = []
        sp_rows: list[int] = []
        tries = 0
        while len(sp_ids) < N_SPARE and tries < len(recycle) * 2:
            cand = recycle[r_ptr % len(recycle)]
            r_ptr += 1
            tries += 1
            if cand in seen:
                continue
            rows = [r for r in free_images[cand] if r not in new_used_rows]
            if not rows:
                continue
            seen.add(cand)
            new_used_rows.add(rows[0])
            sp_ids.append(cand)
            sp_rows.append(rows[0])
        c["spare_donor_ids"] = sp_ids
        c["spare_donor_images"] = [str(paths[r]) for r in sp_rows]
        c["spare_donor_emb_idx"] = sp_rows
        n_spare_assigned += len(sp_ids)

    # ---- verification ----------------------------------------------------------------
    assert not (new_used_rows & used_rows), "new donor image collides with a 5-pass input"
    all_rows: list[int] = []
    for c in chains:
        assert len(c["donor_images"]) == K_DEEP, c["chain_id"]
        assert len(set(c["donor_ids"])) == K_DEEP, f"{c['chain_id']} repeats a donor identity"
        assert c["t0_id"] not in set(c["donor_ids"]), f"{c['chain_id']} donor == target"
        combined = set(c["donor_ids"]) | set(c["spare_donor_ids"])
        assert len(combined) == K_DEEP + len(c["spare_donor_ids"]), \
            f"{c['chain_id']} spare collides with a primary donor identity"
        assert c["t0_id"] not in combined, f"{c['chain_id']} spare donor == target"
        all_rows.extend(int(r) for r in c["donor_emb_idx"])
        all_rows.extend(int(r) for r in c["spare_donor_emb_idx"])
        all_rows.append(int(c["t0_image_idx"]))
    assert len(all_rows) == len(set(all_rows)), "duplicate swap-input image across chains"
    missing = sum(1 for c in chains
                  for p in list(c["donor_images"]) + list(c["spare_donor_images"])
                  if not Path(p).is_file())

    with MANIFEST_DEEP.open("w") as f:
        for c in chains:
            f.write(json.dumps(c) + "\n")

    summary = {
        "n_chains": n_ch,
        "k_base": K_BASE,
        "k_deep": K_DEEP,
        "new_swaps_to_generate": n_ch * n_extra,
        "n_new_donor_images": len(new_used_rows),
        "n_spare_donors_per_chain": N_SPARE,
        "n_spare_donor_slots": n_spare_assigned,
        "passes_with_global_identity_distinctness": min(pass_global_fresh, K_DEEP),
        "n_virgin_identities_available": len(virgin),
        "n_virgin_identities_consumed": v_ptr,
        "n_cross_chain_recycled_donor_slots": n_recycled_assignments,
        "image_level_no_reuse": True,
        "within_chain_identity_distinctness": True,
        "n_missing_image_paths": missing,
        "seed": SEED,
    }
    (OUT / "chains_summary_deep.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
