#!/usr/bin/env python3
"""Build the deep-cascade (K=25) donor manifest for BlendFace and CanonSwap.

Extends the D3 three-pass dilution chains (~1000 per tool, same pairs for both tools) with
fresh donors for passes 4..K_DEEP, mirroring the protocol of the FaceFusion deep cascade
(select_chains_deep.py). Both tools share ONE donor schedule per pair, exactly as D3 gave
every tool the same donor2/donor3.

No-reuse guarantees, per the FaceFusion deep protocol:
  (a) IMAGE-level, global: no image is used twice as a swap input anywhere in this
      experiment (including the D1/D3 passes 1..3 and both tools' extensions).
  (b) IDENTITY-level, within-chain: a donor identity never appears twice in the same
      chain and never equals the chain's target (or its D1 donor / donor2 / donor3 /
      non-member).
  (c) IDENTITY-level, global: passes 4..PASS_GLOBAL_FRESH use identities that appear
      nowhere in D1/D3; beyond that the identity budget is exhausted and identities are
      reused ACROSS chains (never within one) with a distinct, never-used image each time.

Writes cascade/outputs/chains_manifest_deep_aux.jsonl and chains_summary_deep_aux.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
ART = Path("/opt/reproduction/S&P2027-Artifact/data")
D3_PAIRS = ART / "D3/pairs/vggface2/pairs_seed_42.jsonl"
RAW_NPZ = ART / "D2/embeddings/vggface2_raw_embeddings.npz"
MANIFEST = OUT / "chains_manifest_deep_aux.jsonl"

TOOLS = ("blendface", "canonswap")
K_BASE = 3
K_DEEP = 25
N_SPARE = 10
SEED = 11


def swap_paths(pair: dict, tool: str) -> dict[int, Path]:
    return {
        1: ART / f"D1/swaps/{tool}/vggface2_seed_42" / pair["swap1_basename"],
        2: ART / f"D3/swaps/{tool}/swap2/vggface2_seed_42" / pair["swap2_basename"],
        3: ART / f"D3/swaps/{tool}/swap3/vggface2_seed_42" / pair["swap3_basename"],
    }


def main() -> None:
    z = np.load(RAW_NPZ, allow_pickle=True)
    paths = np.asarray(z["paths"], dtype=object)
    uids = np.asarray(z["unique_identity_ids"], dtype=object)
    s_idx = np.asarray(z["identity_start_idx"], np.int64)
    e_idx = np.asarray(z["identity_end_idx"], np.int64)
    path2row = {str(p): i for i, p in enumerate(paths)}

    pairs = [json.loads(l) for l in D3_PAIRS.read_text().splitlines() if l.strip()]

    # Chains eligible for extension: passes 1..3 complete for BOTH tools, so the two
    # cascades run on an identical chain panel.
    chains = []
    for p in pairs:
        ok = all(
            (sp := swap_paths(p, t)[k]).is_file() and sp.stat().st_size > 0
            for t in TOOLS for k in (1, 2, 3)
        )
        if ok:
            chains.append(p)
    print(f"[aux-deep] {len(chains)}/{len(pairs)} chains complete through pass 3 for both tools")

    # Identities & images already consumed anywhere in the D1/D3 experiment (all 1000
    # pairs, not just the complete chains).
    used_ident: set[str] = set()
    used_rows: set[int] = set()
    for p in pairs:
        for f_id, f_img in (("target_id", "target_image"), ("donor_id", "donor_image"),
                            ("donor2_id", "donor2_image"), ("donor3_id", "donor3_image"),
                            ("non_member_id", "non_member_anchor_image")):
            used_ident.add(str(p[f_id]))
            r = path2row.get(str(p[f_img]))
            if r is not None:
                used_rows.add(r)

    free_images: dict[str, list[int]] = {}
    for i, u in enumerate(uids):
        free = [r for r in range(int(s_idx[i]), int(e_idx[i])) if r not in used_rows]
        if free:
            free_images[str(u)] = free

    rng = np.random.default_rng(SEED * 31 + K_DEEP)
    virgin = [str(u) for u in uids if str(u) not in used_ident and str(u) in free_images]
    rng.shuffle(virgin)
    n_extra = K_DEEP - K_BASE
    pass_global_fresh = K_BASE + len(virgin) // max(len(chains), 1)
    print(f"[aux-deep] {len(virgin)} virgin identities -> passes "
          f"{K_BASE + 1}..{min(pass_global_fresh, K_DEEP)} keep global identity distinctness")

    recycle = [u for u in free_images if u not in set(virgin)]
    rng.shuffle(recycle)

    v_ptr = r_ptr = n_recycled = 0
    new_used_rows: set[int] = set()
    out_rows = []
    for p in chains:
        seen = {str(p["target_id"]), str(p["donor_id"]), str(p["donor2_id"]),
                str(p["donor3_id"]), str(p["non_member_id"])}
        add_ids: list[str] = []
        add_rows: list[int] = []
        for _ in range(n_extra):
            picked = None
            while v_ptr < len(virgin):
                cand = virgin[v_ptr]
                v_ptr += 1
                if cand in seen:
                    continue
                rows = [r for r in free_images[cand] if r not in new_used_rows]
                if rows:
                    picked = (cand, rows[0])
                    break
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
                        n_recycled += 1
                        break
            if picked is None:
                raise RuntimeError(f"exhausted donor pool for {p['pair_id']}")
            ident, row = picked
            seen.add(ident)
            new_used_rows.add(row)
            add_ids.append(ident)
            add_rows.append(row)

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

        t_row = path2row.get(str(p["target_image"]))
        out_rows.append({
            "pair_id": p["pair_id"],
            "target_id": p["target_id"],
            "target_image": p["target_image"],
            "target_image_idx": t_row,
            "swap_basenames": {"1": p["swap1_basename"], "2": p["swap2_basename"],
                               "3": p["swap3_basename"]},
            "donor_ids_deep": add_ids,                     # passes 4..K_DEEP
            "donor_images_deep": [str(paths[r]) for r in add_rows],
            "donor_emb_idx_deep": add_rows,
            "spare_donor_ids": sp_ids,
            "spare_donor_images": [str(paths[r]) for r in sp_rows],
        })

    # verification
    assert not (new_used_rows & used_rows)
    all_rows = list(new_used_rows)
    assert len(all_rows) == len(set(all_rows))
    for c in out_rows:
        assert len(c["donor_ids_deep"]) == n_extra
        assert len(set(c["donor_ids_deep"]) | set(c["spare_donor_ids"])) == \
            n_extra + len(c["spare_donor_ids"])

    with MANIFEST.open("w") as f:
        for c in out_rows:
            f.write(json.dumps(c) + "\n")

    summary = {
        "n_chains": len(out_rows), "k_base": K_BASE, "k_deep": K_DEEP,
        "tools": list(TOOLS),
        "new_swaps_per_tool": len(out_rows) * n_extra,
        "passes_with_global_identity_distinctness": min(pass_global_fresh, K_DEEP),
        "n_virgin_identities": len(virgin), "n_virgin_consumed": v_ptr,
        "n_cross_chain_recycled_slots": n_recycled,
        "n_spare_per_chain": N_SPARE, "seed": SEED,
    }
    (OUT / "chains_summary_deep_aux.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
