"""
Run DiffFace diffusion swaps for D1 donor-target pairs.

DiffFace expects one source (donor) and one target image under data/src and
data/targ. The model is loaded once; each pair is staged, swapped, and written
with standard D1 naming (512x512 JPEG).

Run from the DiffFace conda env, e.g.:
  cd "/opt/reproduction/S&P2027-Artifact/data/D1/swaps"
  conda run -n DiffFace python run_diffface.py --dataset vggface2 --seed 42

  # 4 GPUs in parallel:
  bash run_diffface_4gpu.sh
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2
from PIL import Image

from d1_swap_common import (
    add_dataset_seed_args,
    add_shard_args,
    build_output_name,
    donor_image_path,
    load_jsonl,
    resolve_run_paths,
    select_shard_pairs,
    sharded_metadata_path,
    target_image_path,
    write_metadata_line,
)

DIFFFACE_ROOT_DEFAULT = Path("/opt/reproduction/DiffFace")
TOOL = "diffface"
PAIR_IMAGE = "0000.png"
VGG_OUTPUT_SIZE = 512


def copy_as_png(src: Path, dst: Path) -> None:
    image = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst, format="PNG")


def reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def distance_from_rank_stem(stem: str) -> float:
    """Rank filenames encode ArcFace distance with the decimal point removed (.4f)."""
    if not stem.isdigit():
        return float("inf")
    return int(stem) / 10_000.0


def find_best_ranked_png(rank_dir: Path) -> Optional[Path]:
    if not rank_dir.is_dir():
        return None
    candidates = list(rank_dir.glob("*.png"))
    if not candidates:
        return None
    return min(candidates, key=lambda p: distance_from_rank_stem(p.stem))


def find_last_iteration_png(pair_work: Path, iterations_num: int) -> Optional[Path]:
    """DiffFace writes ``{pair_work}/0_{iteration}.png`` for step index 0."""
    last = pair_work / f"0_{iterations_num - 1}.png"
    if last.is_file():
        return last
    matches = sorted(pair_work.glob("0_*.png"))
    return matches[-1] if matches else None


def upscale_and_write_jpg(rgb_path: Path, output_path: Path, size: int = VGG_OUTPUT_SIZE) -> None:
    image_bgr = cv2.imread(str(rgb_path))
    if image_bgr is None:
        raise RuntimeError(f"failed to read swap output: {rgb_path}")
    h, w = image_bgr.shape[:2]
    if h != size or w != size:
        image_bgr = cv2.resize(image_bgr, (size, size), interpolation=cv2.INTER_LANCZOS4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise RuntimeError(f"failed to write {output_path}")


class DiffFaceRunner:
    """Loads DiffFace once and runs one pair at a time via ImageEditor."""

    def __init__(
        self,
        diffface_root: Path,
        *,
        gpu_id: int = 0,
        skip_timesteps: int = 25,
        timestep_respacing: str = "100",
        iterations_num: int = 4,
        aug_num: int = 8,
        seed: int = 404,
        enforce_background: bool = True,
        masking_threshold: int = 30,
        ddim: bool = False,
        shard_index: int = 0,
        num_shards: int = 1,
    ) -> None:
        self.diffface_root = diffface_root.resolve()
        shard_tag = f"_shard{shard_index}" if num_shards > 1 else ""
        self.src_dir = self.diffface_root / "data" / f"src{shard_tag}"
        self.targ_dir = self.diffface_root / "data" / f"targ{shard_tag}"
        self.work_base = self.diffface_root / "data" / f"d1_pair_outputs{shard_tag}"
        self.iterations_num = iterations_num

        os.environ.setdefault("PYTHONNOUSERSITE", "1")
        os.environ["DIFFACE_SRC_DIR"] = str(self.src_dir)
        os.environ["DIFFACE_TARG_DIR"] = str(self.targ_dir)
        os.chdir(self.diffface_root)
        if str(self.diffface_root) not in sys.path:
            sys.path.insert(0, str(self.diffface_root))

        from optimization.image_editor import ImageEditor  # noqa: WPS433

        args = argparse.Namespace(
            batch_size=1,
            skip_timesteps=skip_timesteps,
            ddim=ddim,
            timestep_respacing=timestep_respacing,
            enforce_background=enforce_background,
            aug_num=aug_num,
            seed=seed,
            gpu_id=gpu_id,
            output_path=str(self.work_base),
            output_file="output.png",
            iterations_num=iterations_num,
            masking_threshold=masking_threshold,
        )

        for ckpt in (
            "Arcface.tar",
            "FaceParser.pth",
            "GazeEstimator.pt",
            "Model.pt",
        ):
            path = self.diffface_root / "checkpoints" / ckpt
            if not path.is_file():
                raise FileNotFoundError(f"DiffFace checkpoint missing: {path}")

        print("[diffface] loading models ...", flush=True)
        t0 = time.perf_counter()
        self.editor = ImageEditor(args)
        print(f"[diffface] ready in {time.perf_counter() - t0:.1f}s", flush=True)

    def stage_pair(self, donor_path: Path, target_path: Path) -> None:
        reset_dir(self.src_dir)
        reset_dir(self.targ_dir)
        copy_as_png(donor_path, self.src_dir / PAIR_IMAGE)
        copy_as_png(target_path, self.targ_dir / PAIR_IMAGE)

    def run_pair(self, pair_id: str) -> Path:
        pair_work = self.work_base / pair_id
        if pair_work.exists():
            shutil.rmtree(pair_work)
        pair_work.mkdir(parents=True, exist_ok=True)
        self.editor.args.output_path = str(pair_work)
        self.editor.edit_image_by_prompt()

        rank_dir = pair_work / "Rank0"
        best = find_best_ranked_png(rank_dir)
        if best is None:
            best = find_last_iteration_png(pair_work, self.iterations_num)
        if best is None:
            raise FileNotFoundError(
                f"DiffFace produced no output under {pair_work} (checked Rank0 and 0_*.png)"
            )
        return best


def run_diffface_swap(
    runner: DiffFaceRunner,
    donor_path: Path,
    target_path: Path,
    output_path: Path,
    pair_id: str,
) -> Tuple[int, str]:
    try:
        runner.stage_pair(donor_path, target_path)
        swap_png = runner.run_pair(pair_id)
        upscale_and_write_jpg(swap_png, output_path)
        return 0, ""
    except Exception as exc:
        return 1, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiffFace swaps for D1 pairs.")
    add_dataset_seed_args(parser)
    add_shard_args(parser)
    parser.add_argument("--diffface-root", type=Path, default=DIFFFACE_ROOT_DEFAULT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--skip-timesteps", type=int, default=25)
    parser.add_argument("--timestep-respacing", type=str, default="100")
    parser.add_argument("--iterations-num", type=int, default=4)
    parser.add_argument("--aug-num", type=int, default=8)
    parser.add_argument("--diffface-seed", type=int, default=404, help="DiffFace RNG seed.")
    parser.add_argument(
        "--no-enforce-background",
        dest="enforce_background",
        action="store_false",
        help="Disable DiffFace final background enforcement.",
    )
    parser.set_defaults(enforce_background=True)
    parser.add_argument("--ddim", action="store_true", help="Use DDIM instead of DDPM.")
    args = parser.parse_args()

    diffface_root = args.diffface_root.resolve()
    if not diffface_root.is_dir():
        raise RuntimeError(f"DiffFace root not found: {diffface_root}")

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
    total_pairs = len(pairs)
    pairs = select_shard_pairs(pairs, args.shard_index, args.num_shards)
    metadata_path = sharded_metadata_path(metadata_path, args.shard_index, args.num_shards)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    resume = metadata_path.is_file() and metadata_path.stat().st_size > 0
    meta_mode = "a" if resume else "w"

    print(
        f"[start] dataset={args.dataset} shard={args.shard_index}/{args.num_shards} "
        f"pairs={len(pairs)}/{total_pairs} gpu_id={args.gpu_id} "
        f"output_dir={output_dir} metadata={metadata_path.name} resume_metadata={resume}",
        flush=True,
    )

    runner = DiffFaceRunner(
        diffface_root,
        gpu_id=args.gpu_id,
        skip_timesteps=args.skip_timesteps,
        timestep_respacing=args.timestep_respacing,
        iterations_num=args.iterations_num,
        aug_num=args.aug_num,
        seed=args.diffface_seed,
        enforce_background=args.enforce_background,
        ddim=args.ddim,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )

    ok_count, fail_count, skip_count = 0, 0, 0
    with metadata_path.open(meta_mode, encoding="utf-8") as meta_f:
        if resume:
            meta_f.write(
                json.dumps(
                    {
                        "event": "resume",
                        "time": datetime.now(timezone.utc).isoformat(),
                        "dataset": args.dataset,
                    }
                )
                + "\n"
            )
        for index, pair in enumerate(pairs, start=1):
            donor = donor_image_path(pair).resolve()
            target = target_image_path(pair).resolve()
            out_path = output_dir / build_output_name(pair)
            pair_id = str(pair.get("pair_id", f"pair_{index:04d}"))

            if args.skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
                skip_count += 1
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
                print(f"[{index}/{len(pairs)}] pair={pair_id} status=failed")
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
                print(f"[{index}/{len(pairs)}] pair={pair_id} status=failed")
                continue

            t_pair = time.perf_counter()
            return_code, stderr_tail = run_diffface_swap(
                runner, donor, target, out_path, pair_id
            )
            elapsed = time.perf_counter() - t_pair
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
                extra={
                    "shard_index": args.shard_index,
                    "num_shards": args.num_shards,
                    "iterations_num": args.iterations_num,
                    "skip_timesteps": args.skip_timesteps,
                    "timestep_respacing": args.timestep_respacing,
                    "elapsed_s": round(elapsed, 2),
                },
            )
            print(
                f"[{index}/{len(pairs)}] pair={pair_id} status={status} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    print(
        f"[done] total={len(pairs)} ok={ok_count} failed={fail_count} "
        f"skipped_existing={skip_count} metadata={metadata_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
