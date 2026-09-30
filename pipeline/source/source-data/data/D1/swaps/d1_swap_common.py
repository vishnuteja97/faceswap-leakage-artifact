import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

D1_ROOT = Path(__file__).resolve().parent.parent
SWAPS_ROOT = Path(__file__).resolve().parent


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_output_name(pair: Dict[str, Any]) -> str:
    pair_id = str(pair.get("pair_id", "pair_unknown"))
    donor_id = str(pair.get("donor_id", "donor_unknown"))
    target_id = str(pair.get("target_id", "target_unknown"))
    return f"swap__{pair_id}__donor_{donor_id}__target_{target_id}.jpg"


def donor_image_path(pair: Dict[str, Any]) -> Path:
    raw = pair.get("donor_image") or pair.get("donor_path")
    if not raw:
        raise KeyError("pair missing donor_image or donor_path")
    return Path(raw)


def target_image_path(pair: Dict[str, Any]) -> Path:
    raw = pair.get("target_image") or pair.get("target_path")
    if not raw:
        raise KeyError("pair missing target_image or target_path")
    return Path(raw)


def resolve_run_paths(
    tool: str,
    dataset: str,
    seed: int,
    pairs_jsonl: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    metadata_path: Optional[Path] = None,
) -> Tuple[Path, Path, Path]:
    run_tag = f"{dataset}_seed_{seed}"
    pairs_dir = D1_ROOT / "pairs" / dataset
    tool_dir = SWAPS_ROOT / tool

    resolved_pairs = (
        pairs_jsonl.resolve()
        if pairs_jsonl is not None
        else (pairs_dir / f"pairs_seed_{seed}.jsonl").resolve()
    )
    resolved_output = (
        output_dir.resolve()
        if output_dir is not None
        else (tool_dir / run_tag).resolve()
    )
    resolved_metadata = (
        metadata_path.resolve()
        if metadata_path is not None
        else (tool_dir / f"{tool}_run_{run_tag}.jsonl").resolve()
    )
    return resolved_pairs, resolved_output, resolved_metadata


def add_dataset_seed_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        type=str,
        choices=("vggface2",),
        default="vggface2",
        help="Dataset whose pair file and swap output folder to use.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pairs-jsonl",
        type=Path,
        default=None,
        help="Defaults to pairs/<dataset>/pairs_seed_<seed>.jsonl under data/D1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to swaps/<tool>/<dataset>_seed_<seed> under data/D1.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Defaults to swaps/<tool>/<tool>_run_<dataset>_seed_<seed>.jsonl under data/D1.",
    )
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", default=True)
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")


def add_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split pairs across N parallel jobs (pair i -> shard i %% N).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This job's shard index in [0, num_shards).",
    )


def select_shard_pairs(
    pairs: List[Dict[str, Any]], shard_index: int, num_shards: int
) -> List[Dict[str, Any]]:
    if num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
    if num_shards == 1:
        return pairs
    return [p for i, p in enumerate(pairs) if i % num_shards == shard_index]


def sharded_metadata_path(metadata_path: Path, shard_index: int, num_shards: int) -> Path:
    if num_shards <= 1:
        return metadata_path
    return metadata_path.with_name(
        f"{metadata_path.stem}_shard{shard_index}of{num_shards}{metadata_path.suffix}"
    )


def write_metadata_line(
    meta_f,
    *,
    index: int,
    pair: Dict[str, Any],
    swap_image: Path,
    status: str,
    return_code: int,
    stderr_tail: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    record: Dict[str, Any] = {
        "index": index,
        "pair_id": pair.get("pair_id"),
        "donor_id": pair.get("donor_id"),
        "target_id": pair.get("target_id"),
        "non_member_id": pair.get("non_member_id"),
        "swap_image": str(swap_image),
        "status": status,
        "return_code": return_code,
    }
    if stderr_tail:
        record["stderr_tail"] = stderr_tail
    if extra:
        record.update(extra)
    meta_f.write(json.dumps(record, sort_keys=True) + "\n")
