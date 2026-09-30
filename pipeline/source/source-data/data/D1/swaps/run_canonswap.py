"""
Run CanonSwap image-to-image swaps for D1 donor-target pairs.

Loads the pipeline once (from the CanonSwap repo root) and runs one swap per pair.
Outputs are written with standard D1 naming under swaps/canonswap/<dataset>_seed_<seed>/.

Run from the CanonSwap conda env, e.g.:
  conda activate CanonSwap
  export PYTHONNOUSERSITE=1
  python data/D1/swaps/run_canonswap.py --seed 42
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

os.environ.setdefault("PYTHONNOUSERSITE", "1")

from d1_swap_common import (
    add_dataset_seed_args,
    build_output_name,
    donor_image_path,
    load_jsonl,
    resolve_run_paths,
    target_image_path,
    write_metadata_line,
)

CANONSWAP_ROOT_DEFAULT = Path("/opt/reproduction/CanonSwap")
TOOL = "canonswap"


def fast_check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def init_canonswap_pipeline(
    canonswap_root: Path,
    *,
    dataset: str = "vggface2",
    device_id: int = 0,
    use_half_precision: bool = False,
):
    """Initialize CanonSwap once; must run with cwd=canonswap_root."""
    root = canonswap_root.resolve()
    if not (root / "pretrained_weights" / "combined_weights.pth").is_file():
        raise FileNotFoundError(
            f"CanonSwap weights not found under {root / 'pretrained_weights'}"
        )

    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.config.argument_config import ArgumentConfig  # noqa: E402
    from src.config.crop_config import CropConfig  # noqa: E402
    from src.config.inference_config import InferenceConfig  # noqa: E402
    from src.can_swap_pipeline_e2e import CanSwapPipeline  # noqa: E402

    # VGGFace2 inputs are 512x512 pre-aligned squares; without forcing driving
    # crop, pasteback skips target_M_c2o_lst and raises UnboundLocalError.
    flag_crop_driving = dataset == "vggface2"
    inference_cfg = InferenceConfig(
        flag_stitching=False,
        flag_pasteback=True,
        flag_use_half_precision=use_half_precision,
        flag_crop_driving_video=flag_crop_driving,
        device_id=device_id,
    )
    crop_cfg = CropConfig()
    pipeline = CanSwapPipeline(inference_cfg=inference_cfg, crop_cfg=crop_cfg)
    return pipeline, ArgumentConfig


def canonswap_output_name(donor: Path, target: Path) -> str:
    from src.utils.helper import basename  # noqa: WPS433

    return f"{basename(str(donor))}--{basename(str(target))}.jpg"


def run_canonswap_pair(
    donor: Path,
    target: Path,
    out_path: Path,
    pipeline,
    argument_config_cls,
    work_dir: Path,
    *,
    device_id: int = 0,
    use_half_precision: bool = False,
) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    pair_args = argument_config_cls(
        source=str(donor.resolve()),
        driving=str(target.resolve()),
        output_dir=str(work_dir.resolve()),
        device_id=device_id,
        flag_use_half_precision=use_half_precision,
        flag_pasteback=True,
        flag_do_crop=True,
    )
    pipeline.execute(pair_args)

    generated = work_dir / canonswap_output_name(donor, target)
    if not generated.is_file():
        raise FileNotFoundError(f"CanonSwap did not write expected output: {generated}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    shutil.move(str(generated), str(out_path))

    concat_name = canonswap_output_name(donor, target).replace(".jpg", "_concat.jpg")
    concat_path = work_dir / concat_name
    if concat_path.is_file():
        concat_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CanonSwap image-to-image swaps for D1 pairs."
    )
    add_dataset_seed_args(parser)
    parser.add_argument("--canonswap-root", type=Path, default=CANONSWAP_ROOT_DEFAULT)
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="CUDA device id passed to CanonSwap.",
    )
    parser.add_argument(
        "--use-half-precision",
        action="store_true",
        default=False,
        help="Enable FP16 in CanonSwap (may cause black boxes on some GPUs).",
    )
    args = parser.parse_args()

    if not fast_check_ffmpeg():
        raise RuntimeError(
            "ffmpeg not found. Install in the CanonSwap env: conda install -n CanonSwap -c conda-forge ffmpeg"
        )

    canonswap_root = args.canonswap_root.resolve()
    if not canonswap_root.is_dir():
        raise RuntimeError(f"CanonSwap root not found: {canonswap_root}")

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
    work_dir = output_dir / "_canonswap_work"

    print(
        f"[start] dataset={args.dataset} pairs={len(pairs)} "
        f"output_dir={output_dir} device={args.device_id} "
        f"half_precision={args.use_half_precision}"
    )

    pipeline, argument_config_cls = init_canonswap_pipeline(
        canonswap_root,
        dataset=args.dataset,
        device_id=args.device_id,
        use_half_precision=args.use_half_precision,
    )

    ok_count, fail_count, skip_count = 0, 0, 0
    with metadata_path.open("w", encoding="utf-8") as meta_f:
        for index, pair in enumerate(pairs, start=1):
            donor = donor_image_path(pair).resolve()
            target = target_image_path(pair).resolve()
            out_path = output_dir / build_output_name(pair)

            if args.skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
                skip_count += 1
                write_metadata_line(
                    meta_f,
                    index=index,
                    pair=pair,
                    swap_image=out_path,
                    status="skipped_existing",
                    return_code=0,
                )
                continue

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

            stderr_tail = ""
            try:
                run_canonswap_pair(
                    donor,
                    target,
                    out_path,
                    pipeline,
                    argument_config_cls,
                    work_dir,
                    device_id=args.device_id,
                    use_half_precision=args.use_half_precision,
                )
                status = "ok"
                return_code = 0
                ok_count += 1
            except Exception as exc:
                status = "failed"
                return_code = 1
                fail_count += 1
                stderr_tail = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
                stderr_tail = "\n".join(stderr_tail.strip().splitlines()[-20:])

            write_metadata_line(
                meta_f,
                index=index,
                pair=pair,
                swap_image=out_path,
                status=status,
                return_code=return_code,
                stderr_tail=stderr_tail,
                extra={
                    "mode": "image_to_image",
                    "device_id": args.device_id,
                    "use_half_precision": args.use_half_precision,
                },
            )
            print(f"[{index}/{len(pairs)}] pair={pair.get('pair_id')} status={status}")

    print(
        f"[done] total={len(pairs)} ok={ok_count} failed={fail_count} "
        f"skipped_existing={skip_count} metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
