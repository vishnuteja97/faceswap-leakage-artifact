#!/usr/bin/env python3
"""
Build D2 donor-target-non_member pairs from D1, replacing only the non-member.

Selection per target:
  1. Rank all gallery identities by MAAD attribute score vs the target portrait.
  2. Write top-K MAAD matches (default 1000) to maad_similarity_seed_<seed>.jsonl
     for later processing (attribute-only rankings).
  3. Among identities tied at the maximum MAAD score, pick the best visual match
     using up to images_per_identity embeddings per candidate (max cosine).
  4. Enforce unique non_member_id across pairs (next-best if taken).

Requires:
  - data/D1/pairs/vggface2/pairs_seed_<seed>.jsonl
  - data/D2/MAAD_Face.csv
  - data/D2/embeddings/vggface2_raw_embeddings.npz (+ .index.json)
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

D2_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAAD = D2_ROOT / "MAAD_Face.csv"
DEFAULT_EMB_NPZ = D2_ROOT / "embeddings" / "vggface2_raw_embeddings.npz"
DEFAULT_EMB_INDEX = D2_ROOT / "embeddings" / "vggface2_raw_embeddings.index.json"
DEFAULT_D1_PAIRS = (
    Path(__file__).resolve().parents[2] / "D1" / "pairs" / "vggface2" / "pairs_seed_42.jsonl"
)

ATTRS = (
    "Male",
    "Young",
    "Middle_Aged",
    "Senior",
    "Asian",
    "White",
    "Black",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Gray_Hair",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def rel_path_from_abs(path_str: str) -> str:
    p = Path(path_str)
    return f"{p.parent.name}/{p.name}"


def maad_score(target: Dict[str, int], candidate: Dict[str, int]) -> float:
    score = 0.0
    for a in ATTRS:
        tv, cv = target[a], candidate[a]
        if tv == 0:
            continue
        if tv == cv:
            score += 1.0
        else:
            score -= 0.25
    return score


def sample_indices(start: int, end: int, k: int) -> np.ndarray:
    n = end - start
    if n <= 0:
        return np.array([], dtype=np.int64)
    if n <= k:
        return np.arange(start, end, dtype=np.int64)
    return np.linspace(start, end - 1, num=k, dtype=np.int64)


def build_maad_lookup(
    maad_csv: Path, needed_rel_paths: set[str]
) -> Dict[str, Dict[str, int]]:
    lookup: Dict[str, Dict[str, int]] = {}
    with maad_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["Filename"]
            if fn not in needed_rel_paths:
                continue
            lookup[fn] = {a: int(row[a]) for a in ATTRS}
    return lookup


def load_embedding_store(
    npz_path: Path, index_path: Path
) -> Tuple[np.ndarray, List[str], Dict[str, Tuple[int, int]], Dict[str, int]]:
    data = np.load(npz_path, allow_pickle=True)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    paths = [str(p) for p in data["paths"]]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    embeddings = embeddings / norms

    path_to_row = {rel_path_from_abs(p): i for i, p in enumerate(paths)}

    with index_path.open("r", encoding="utf-8") as f:
        index_obj = json.load(f)
    identity_to_range = {
        str(k): (int(v[0]), int(v[1])) for k, v in index_obj["identity_to_range"].items()
    }
    return embeddings, paths, identity_to_range, path_to_row


def identity_representative_labels(
    identity_id: str,
    start: int,
    end: int,
    paths: Sequence[str],
    maad_lookup: Dict[str, Dict[str, int]],
) -> Optional[Dict[str, int]]:
    for i in range(start, end):
        rel = rel_path_from_abs(paths[i])
        if rel in maad_lookup:
            return maad_lookup[rel]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build D2 pairs with MAAD + embedding non-members.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--d1-pairs-jsonl",
        type=Path,
        default=None,
        help="Defaults to data/D1/pairs/vggface2/pairs_seed_<seed>.jsonl",
    )
    parser.add_argument("--maad-csv", type=Path, default=DEFAULT_MAAD)
    parser.add_argument("--embeddings-npz", type=Path, default=DEFAULT_EMB_NPZ)
    parser.add_argument("--embeddings-index", type=Path, default=DEFAULT_EMB_INDEX)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--maad-top-k",
        type=int,
        default=1000,
        help="Per-target MAAD ranking list size (max 1000; capped by gallery size).",
    )
    parser.add_argument("--images-per-identity", type=int, default=5)
    parser.add_argument("--max-pairs", type=int, default=None, help="Debug: process first N pairs only.")
    args = parser.parse_args()

    d1_pairs_path = (
        args.d1_pairs_jsonl.resolve()
        if args.d1_pairs_jsonl is not None
        else (
            Path(__file__).resolve().parents[2]
            / "D1"
            / "pairs"
            / "vggface2"
            / f"pairs_seed_{args.seed}.jsonl"
        ).resolve()
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (Path(__file__).resolve().parent / "vggface2").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[start] d1_pairs={d1_pairs_path}")
    d1_pairs = load_jsonl(d1_pairs_path)
    if args.max_pairs is not None:
        d1_pairs = d1_pairs[: args.max_pairs]

    print("[load] embeddings...")
    embeddings, paths, identity_to_range, path_to_row = load_embedding_store(
        args.embeddings_npz.resolve(), args.embeddings_index.resolve()
    )
    needed_rel = set(path_to_row.keys())
    print(f"[load] embedding rows={len(paths)} identities={len(identity_to_range)}")

    print("[load] MAAD labels for embedded images (streaming CSV)...")
    maad_lookup = build_maad_lookup(args.maad_csv.resolve(), needed_rel)
    print(f"[load] MAAD rows matched to embeddings: {len(maad_lookup)}")

    identities = sorted(identity_to_range.keys())
    ident_rep_labels: Dict[str, Dict[str, int]] = {}
    for ident in identities:
        start, end = identity_to_range[ident]
        lab = identity_representative_labels(ident, start, end, paths, maad_lookup)
        if lab is not None:
            ident_rep_labels[ident] = lab
    print(f"[load] identities with MAAD labels: {len(ident_rep_labels)}")

    pairs_out: List[Dict[str, Any]] = []
    maad_sim_out: List[Dict[str, Any]] = []
    used_non_members: set[str] = set()
    skipped_no_target_emb = 0
    skipped_no_target_maad = 0

    for pair in d1_pairs:
        pair_id = pair["pair_id"]
        donor_id = pair["donor_id"]
        target_id = pair["target_id"]
        target_image = pair["target_image"]
        target_rel = rel_path_from_abs(target_image)

        if target_rel not in path_to_row:
            skipped_no_target_emb += 1
            continue
        if target_rel not in maad_lookup:
            skipped_no_target_maad += 1
            continue

        target_labels = maad_lookup[target_rel]
        target_row = path_to_row[target_rel]
        target_emb = embeddings[target_row]

        excluded = {donor_id, target_id}
        maad_ranked: List[Tuple[str, float]] = []
        for ident in identities:
            if ident in excluded or ident not in ident_rep_labels:
                continue
            maad_ranked.append((ident, maad_score(target_labels, ident_rep_labels[ident])))
        maad_ranked.sort(key=lambda x: x[1], reverse=True)

        store_k = min(args.maad_top_k, len(maad_ranked))
        maad_top_records: List[Dict[str, Any]] = []
        for rank, (ident, sc) in enumerate(maad_ranked[:store_k], start=1):
            start, end = identity_to_range[ident]
            anchor_path = paths[start]
            maad_top_records.append(
                {
                    "rank": rank,
                    "identity_id": ident,
                    "maad_score": sc,
                    "anchor_image": anchor_path,
                }
            )
        max_score = maad_ranked[0][1] if maad_ranked else None
        n_at_max = sum(1 for _, sc in maad_ranked if sc == max_score) if maad_ranked else 0
        maad_sim_out.append(
            {
                "pair_id": pair_id,
                "seed": args.seed,
                "target_id": target_id,
                "target_image": target_image,
                "target_maad_profile": {a: target_labels[a] for a in ATTRS},
                "max_maad_score": max_score,
                "n_identities_at_max_score": n_at_max,
                "n_candidates_ranked": len(maad_ranked),
                "n_stored": store_k,
                "top_matches": maad_top_records,
            }
        )

        tie_group = [(ident, sc) for ident, sc in maad_ranked if sc == max_score]
        embed_ranked: List[Tuple[str, float, str, float]] = []
        k = args.images_per_identity
        for ident, m_sc in tie_group:
            start, end = identity_to_range[ident]
            idxs = sample_indices(start, end, k)
            if idxs.size == 0:
                continue
            sims = embeddings[idxs] @ target_emb
            best_local = int(np.argmax(sims))
            best_idx = int(idxs[best_local])
            embed_ranked.append((ident, float(sims[best_local]), paths[best_idx], m_sc))

        embed_ranked.sort(key=lambda x: x[1], reverse=True)

        chosen_ident: Optional[str] = None
        chosen_path: Optional[str] = None
        chosen_cosine = float("nan")
        chosen_maad = float("nan")

        for ident, cos, img_path, m_sc in embed_ranked:
            if ident in used_non_members:
                continue
            chosen_ident = ident
            chosen_path = img_path
            chosen_cosine = cos
            chosen_maad = m_sc
            break

        if chosen_ident is None:
            # fallback: best MAAD not yet used
            for ident, m_sc in maad_ranked:
                if ident in used_non_members or ident in excluded:
                    continue
                start, end = identity_to_range[ident]
                chosen_ident = ident
                chosen_path = paths[start]
                chosen_maad = m_sc
                chosen_cosine = float("nan")
                break

        if chosen_ident is None:
            raise RuntimeError(f"Could not assign non-member for {pair_id}")

        used_non_members.add(chosen_ident)

        out_row = dict(pair)
        out_row["d1_non_member_id"] = pair.get("non_member_id")
        out_row["d1_non_member_anchor_image"] = pair.get("non_member_anchor_image")
        out_row["non_member_id"] = chosen_ident
        out_row["non_member_anchor_image"] = chosen_path
        out_row["non_member_maad_score"] = chosen_maad
        out_row["non_member_embed_cosine"] = chosen_cosine
        out_row["non_member_selection"] = "maad_max_tie_embed_max"
        out_row["non_member_tie_group_size"] = len(tie_group)
        out_row["d2_seed"] = args.seed
        pairs_out.append(out_row)

        if len(pairs_out) % 100 == 0:
            print(f"[progress] {len(pairs_out)}/{len(d1_pairs)} pairs")

    pairs_path = output_dir / f"pairs_seed_{args.seed}.jsonl"
    maad_sim_path = output_dir / f"maad_similarity_seed_{args.seed}.jsonl"
    manifest_path = output_dir / f"pairing_manifest_seed_{args.seed}.json"

    with pairs_path.open("w", encoding="utf-8") as f:
        for row in pairs_out:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    with maad_sim_path.open("w", encoding="utf-8") as f:
        for row in maad_sim_out:
            f.write(json.dumps(row, sort_keys=True) + "\n")

    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "source_d1_pairs": str(d1_pairs_path),
                "maad_csv": str(args.maad_csv.resolve()),
                "embeddings_npz": str(args.embeddings_npz.resolve()),
                "embeddings_index": str(args.embeddings_index.resolve()),
                "n_pairs": len(pairs_out),
                "maad_top_k": args.maad_top_k,
                "embed_pool": "maad_max_score_tie_group",
                "images_per_identity": args.images_per_identity,
                "skipped_no_target_emb": skipped_no_target_emb,
                "skipped_no_target_maad": skipped_no_target_maad,
                "unique_non_members": len(used_non_members),
                "pairs_jsonl": str(pairs_path),
                "maad_similarity_jsonl": str(maad_sim_path),
                "selection": "MAAD max-score tie group, then max cosine over K images/identity; unique non-members",
            },
            f,
            indent=2,
            sort_keys=True,
        )

    # Point index at D2 copy for downstream tools
    index_path = args.embeddings_index.resolve()
    with index_path.open("r", encoding="utf-8") as f:
        idx_obj = json.load(f)
    idx_obj["npz_path"] = str(args.embeddings_npz.resolve())
    with index_path.open("w", encoding="utf-8") as f:
        json.dump(idx_obj, f)

    elapsed = time.time() - t0
    print(f"[done] pairs={len(pairs_out)} -> {pairs_path}")
    print(f"[done] maad_similarity -> {maad_sim_path}")
    print(f"[done] manifest -> {manifest_path}")
    print(f"[done] elapsed_sec={elapsed:.1f}")


if __name__ == "__main__":
    main()
