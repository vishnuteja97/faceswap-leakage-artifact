"""
Run FaceFusion swaps for D1 donor-target pairs.

Loads FaceFusion once in a single process and reuses model weights across pairs
(faster than spawning a new ``conda run … headless-run`` subprocess per pair).

Requires the facefusion conda env:

  conda activate facefusion
  python data/D1/swaps/run_facefusion.py --dataset vggface2 --seed 42
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from d1_swap_common import (
    add_dataset_seed_args,
    build_output_name,
    donor_image_path,
    load_jsonl,
    resolve_run_paths,
    target_image_path,
    write_metadata_line,
)

FACEFUSION_ROOT_DEFAULT = Path("/opt/reproduction/facefusion")
TOOL = "facefusion"


def _ensure_facefusion_importable(facefusion_root: Path) -> None:
    root = facefusion_root.resolve()
    if not (root / "facefusion").is_dir():
        raise FileNotFoundError(f"FaceFusion root not found: {root}")
    os.chdir(root)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def _bootstrap_facefusion_argv(
    *,
    donor: Path,
    target: Path,
    output: Path,
    jobs_path: Path,
    face_swapper_model: str,
    execution_providers: List[str],
    execution_device_ids: List[int],
    log_level: str,
) -> List[str]:
    return [
        "facefusion.py",
        "headless-run",
        "-s",
        str(donor),
        "-t",
        str(target),
        "-o",
        str(output),
        "--jobs-path",
        str(jobs_path),
        "--processors",
        "face_swapper",
        "--face-swapper-model",
        face_swapper_model,
        "--execution-providers",
        *execution_providers,
        "--execution-device-ids",
        *[str(device_id) for device_id in execution_device_ids],
        "--log-level",
        log_level,
    ]


def init_facefusion_session(
    facefusion_root: Path,
    *,
    bootstrap_donor: Path,
    bootstrap_target: Path,
    bootstrap_output: Path,
    jobs_path: Path,
    face_swapper_model: str,
    execution_providers: List[str],
    execution_device_ids: List[int],
    log_level: str,
) -> Tuple[Any, Any, Any, Any]:
    """Parse CLI defaults once and warm up FaceFusion (models + detectors)."""
    _ensure_facefusion_importable(facefusion_root)

    from facefusion import logger, state_manager  # noqa: E402
    from facefusion.args import apply_args  # noqa: E402
    from facefusion.core import (  # noqa: E402
        common_pre_check,
        conditional_process,
        pre_check,
        processors_pre_check,
    )
    from facefusion.processors.core import get_processors_modules  # noqa: E402
    from facefusion.program import create_program  # noqa: E402
    from facefusion.program_helper import validate_args  # noqa: E402

    if not pre_check():
        raise RuntimeError("FaceFusion pre_check failed (deps / python version)")

    argv = _bootstrap_facefusion_argv(
        donor=bootstrap_donor,
        target=bootstrap_target,
        output=bootstrap_output,
        jobs_path=jobs_path,
        face_swapper_model=face_swapper_model,
        execution_providers=execution_providers,
        execution_device_ids=execution_device_ids,
        log_level=log_level,
    )
    old_argv = sys.argv
    try:
        sys.argv = argv
        program = create_program()
        if not validate_args(program):
            raise RuntimeError("invalid FaceFusion CLI arguments")
        parsed = vars(program.parse_args())
        apply_args(parsed, state_manager.init_item)
        logger.init(state_manager.get_item("log_level"))
    finally:
        sys.argv = old_argv

    if not common_pre_check() or not processors_pre_check():
        raise RuntimeError("FaceFusion module pre_check failed during bootstrap")

    for processor_module in get_processors_modules(state_manager.get_item("processors")):
        if not processor_module.pre_process("output"):
            raise RuntimeError("FaceFusion processor pre_process failed during bootstrap")

    error_code = conditional_process()
    if error_code != 0:
        raise RuntimeError(f"FaceFusion bootstrap swap failed with code {error_code}")

    return state_manager, apply_args, conditional_process, get_processors_modules


def run_one_swap(
    *,
    state_manager: Any,
    apply_args: Any,
    conditional_process: Any,
    get_processors_modules: Any,
    common_pre_check: Any,
    processors_pre_check: Any,
    donor: Path,
    target: Path,
    output: Path,
) -> bool:
    output.parent.mkdir(parents=True, exist_ok=True)
    state_manager.set_item("source_paths", [str(donor.resolve())])
    state_manager.set_item("target_path", str(target.resolve()))
    state_manager.set_item("output_path", str(output.resolve()))

    if not common_pre_check() or not processors_pre_check():
        return False

    for processor_module in get_processors_modules(state_manager.get_item("processors")):
        if not processor_module.pre_process("output"):
            return False

    return conditional_process() == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run FaceFusion swaps for D1 pairs (single-process batch)."
    )
    add_dataset_seed_args(parser)
    parser.add_argument("--facefusion-root", type=Path, default=FACEFUSION_ROOT_DEFAULT)
    parser.add_argument("--face-swapper-model", type=str, default="hyperswap_1a_256")
    parser.add_argument("--execution-providers", nargs="+", default=["cuda"])
    parser.add_argument("--execution-device-id", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="warn")
    parser.add_argument(
        "--jobs-path",
        type=Path,
        default=None,
        help="Temp jobs dir for FaceFusion bootstrap (default: output_dir/.jobs).",
    )
    args = parser.parse_args()

    pairs_jsonl, output_dir, metadata_path = resolve_run_paths(
        TOOL,
        args.dataset,
        args.seed,
        args.pairs_jsonl,
        args.output_dir,
        args.metadata_path,
    )

    pairs = load_jsonl(pairs_jsonl)
    if args.max_pairs is not None:
        pairs = pairs[: args.max_pairs]

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    jobs_path = (
        args.jobs_path.resolve()
        if args.jobs_path is not None
        else (output_dir / ".jobs").resolve()
    )
    jobs_path.mkdir(parents=True, exist_ok=True)

    execution_device_ids = [args.execution_device_id]
    device_label = f"cuda:{args.execution_device_id}"

    pending: List[Tuple[int, Dict[str, Any], Path, Path, Path]] = []
    skipped = 0
    for index, pair in enumerate(pairs, start=1):
        donor = donor_image_path(pair).resolve()
        target = target_image_path(pair).resolve()
        out_path = output_dir / build_output_name(pair)

        if args.skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        pending.append((index, pair, donor, target, out_path))

    print(
        f"[start] dataset={args.dataset} total={len(pairs)} "
        f"to_run={len(pending)} skipped_existing={skipped} device={device_label} "
        f"output_dir={output_dir}"
    )

    ok_count = 0
    fail_count = 0
    run_start = time.perf_counter()

    with metadata_path.open("w", encoding="utf-8") as meta_f:
        for index, pair in enumerate(pairs, start=1):
            out_path = output_dir / build_output_name(pair)
            if args.skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
                write_metadata_line(
                    meta_f,
                    index=index,
                    pair=pair,
                    swap_image=out_path,
                    status="skipped_existing",
                    return_code=0,
                )

        if not pending:
            print("[done] nothing to run")
            return

        warmup_pair = pairs[0]
        warmup_donor = donor_image_path(warmup_pair).resolve()
        warmup_target = target_image_path(warmup_pair).resolve()
        warmup_out = jobs_path / "_warmup.jpg"
        if not warmup_donor.is_file() or not warmup_target.is_file():
            raise RuntimeError("warmup pair_0000 images not found")

        t0 = time.perf_counter()
        state_manager, apply_args, conditional_process, get_processors_modules = (
            init_facefusion_session(
                args.facefusion_root.resolve(),
                bootstrap_donor=warmup_donor,
                bootstrap_target=warmup_target,
                bootstrap_output=warmup_out,
                jobs_path=jobs_path,
                face_swapper_model=args.face_swapper_model,
                execution_providers=args.execution_providers,
                execution_device_ids=execution_device_ids,
                log_level=args.log_level,
            )
        )
        from facefusion.core import common_pre_check, processors_pre_check  # noqa: E402

        bootstrap_s = time.perf_counter() - t0
        print(f"[warmup] models loaded in {bootstrap_s:.1f}s (pair_0000)")

        for index, pair, donor, target, out_path in pending:
            if not donor.is_file():
                fail_count += 1
                write_metadata_line(
                    meta_f,
                    index=index,
                    pair=pair,
                    swap_image=out_path,
                    status="failed",
                    return_code=1,
                    stderr_tail=f"donor not found: {donor}",
                )
                continue
            if not target.is_file():
                fail_count += 1
                write_metadata_line(
                    meta_f,
                    index=index,
                    pair=pair,
                    swap_image=out_path,
                    status="failed",
                    return_code=1,
                    stderr_tail=f"target not found: {target}",
                )
                continue

            ok = run_one_swap(
                state_manager=state_manager,
                apply_args=apply_args,
                conditional_process=conditional_process,
                get_processors_modules=get_processors_modules,
                common_pre_check=common_pre_check,
                processors_pre_check=processors_pre_check,
                donor=donor,
                target=target,
                output=out_path,
            )
            status = "ok" if ok else "failed"
            if ok:
                ok_count += 1
            else:
                fail_count += 1

            write_metadata_line(
                meta_f,
                index=index,
                pair=pair,
                swap_image=out_path,
                status=status,
                return_code=0 if ok else 1,
                extra={"face_swapper_model": args.face_swapper_model},
            )
            print(f"[{index}/{len(pairs)}] pair={pair.get('pair_id')} status={status}")

    elapsed = time.perf_counter() - run_start
    rate = ok_count / elapsed if elapsed > 0 else 0.0
    print(
        f"[done] ok={ok_count} failed={fail_count} skipped_existing={skipped} "
        f"elapsed={elapsed:.1f}s ({rate * 60:.1f} swaps/min) metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
