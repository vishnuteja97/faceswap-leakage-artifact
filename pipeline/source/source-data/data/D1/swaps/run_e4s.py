"""
Run E4S face swaps for D1 donor-target pairs.

Run from the e4s conda env, e.g.:
  python data/D1/swaps/run_e4s.py --seed 42
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import cv2

from d1_swap_common import (
    add_dataset_seed_args,
    build_output_name,
    donor_image_path,
    load_jsonl,
    resolve_run_paths,
    target_image_path,
    write_metadata_line,
)

E4S_ROOT_DEFAULT = Path("/opt/reproduction/e4s")
TOOL = "e4s"


def run_e4s_swap(
    donor_path: Path,
    target_path: Path,
    output_path: Path,
    e4s_root: Path,
    *,
    need_crop: bool = False,
    only_target_crop: bool = False,
) -> Tuple[int, str]:
    face_swap_script = e4s_root / "scripts" / "face_swap.py"
    out_dir = output_path.parent / f"_e4s_tmp_{output_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    stderr_tail = ""
    try:
        cmd = [
            sys.executable,
            str(face_swap_script),
            "--source",
            str(donor_path),
            "--target",
            str(target_path),
            "--output_dir",
            str(out_dir),
        ]
        if need_crop:
            cmd.append("--need-crop")
        if only_target_crop:
            cmd.append("--only-target-crop")
        run_env = {**os.environ, "PYTHONPATH": str(e4s_root), "TMPDIR": "/tmp"}
        proc = subprocess.run(cmd, cwd=str(e4s_root), env=run_env, capture_output=True, text=True)
        if proc.returncode != 0:
            stderr_tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            return proc.returncode, stderr_tail

        expected_name = f"swap_{donor_path.stem}_to_{target_path.stem}.png"
        generated = out_dir / expected_name
        if not generated.exists():
            candidates = list(out_dir.glob("swap_*.png"))
            if not candidates:
                return 1, "e4s produced no swap_*.png output"
            generated = candidates[0]

        image_bgr = cv2.imread(str(generated))
        if image_bgr is None:
            return 1, f"failed to read e4s output: {generated}"
        if not cv2.imwrite(str(output_path), image_bgr):
            return 1, f"failed to write {output_path}"
        return 0, ""
    finally:
        if out_dir.exists():
            try:
                shutil.rmtree(out_dir)
            except OSError:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E4S swaps for D1 pairs.")
    add_dataset_seed_args(parser)
    parser.add_argument("--e4s-root", type=Path, default=E4S_ROOT_DEFAULT)
    parser.add_argument(
        "--align-faces",
        action="store_true",
        default=None,
        help="Use E4S dlib detect/align + paste-back. Default: off for pre-aligned VGGFace2.",
    )
    parser.add_argument(
        "--no-align-faces",
        dest="align_faces",
        action="store_false",
        help="Disable face alignment (resize whole image to 1024x1024).",
    )
    parser.add_argument(
        "--only-target-crop",
        action="store_true",
        default=False,
        help="Align target only; donor is resized (passed through to E4S).",
    )
    args = parser.parse_args()

    align_faces = args.align_faces if args.align_faces is not None else False

    e4s_root = args.e4s_root.resolve()
    if not (e4s_root / "scripts" / "face_swap.py").is_file():
        raise RuntimeError(f"E4S script not found under {e4s_root}")

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
    print(
        f"[start] dataset={args.dataset} align_faces={align_faces} "
        f"only_target_crop={args.only_target_crop}"
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

            return_code, stderr_tail = run_e4s_swap(
                donor,
                target,
                out_path,
                e4s_root,
                need_crop=align_faces and not args.only_target_crop,
                only_target_crop=align_faces and args.only_target_crop,
            )
            status = "ok" if return_code == 0 else "failed"
            if status == "ok":
                ok_count += 1
            else:
                fail_count += 1

            write_metadata_line(
                meta_f,
                index=index,
                pair=pair,
                swap_image=out_path,
                status=status,
                return_code=return_code,
                stderr_tail=stderr_tail,
                extra={"align_faces": align_faces, "only_target_crop": args.only_target_crop},
            )
            print(f"[{index}/{len(pairs)}] pair={pair.get('pair_id')} status={status}")

    print(
        f"[done] total={len(pairs)} ok={ok_count} failed={fail_count} "
        f"skipped_existing={skip_count} metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
