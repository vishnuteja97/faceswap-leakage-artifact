"""
Run DiffSwap face swaps for D1 donor-target pairs.

Loads the diffusion model once per job, then preprocesses and swaps each pair.

Run from the diffswap conda env, e.g.:
  cd "/opt/reproduction/S&P2027-Artifact/data/D1/swaps"
  conda run -n diffswap python run_diffswap.py --dataset vggface2 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader

from d1_swap_common import (
    add_dataset_seed_args,
    build_output_name,
    donor_image_path,
    load_jsonl,
    resolve_run_paths,
    target_image_path,
    write_metadata_line,
)

DIFFSWAP_ROOT_DEFAULT = Path("/opt/reproduction/DiffSwap")
TOOL = "diffswap"
DEFAULT_PAIR_IMAGE = "0000.png"


def default_portrait_workdir(dataset: str) -> str:
    return f"data/portrait_d1_{dataset}"


def reset_portrait(portrait_dir: Path) -> None:
    if portrait_dir.exists():
        shutil.rmtree(portrait_dir)
    for sub in (
        "source",
        "target",
        "align/source",
        "align/target",
        "landmark",
        "mtcnn",
        "swap_res",
        "swap_res_ori",
    ):
        (portrait_dir / sub).mkdir(parents=True, exist_ok=True)
    json.dump(
        {"source": [], "target": []},
        open(portrait_dir / "error_img.json", "w", encoding="utf-8"),
        indent=4,
    )


def copy_as_png(src: Path, dst: Path) -> None:
    image = Image.open(src).convert("RGB")
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst, format="PNG")


def find_swap_png(portrait_dir: Path, ckpt_stem: str, tgt_scale: float, pair_image: str) -> Optional[Path]:
    scale_dir = portrait_dir / "swap_res" / f"{ckpt_stem}_{tgt_scale}"
    if not scale_dir.is_dir():
        return None
    stem = pair_image.rsplit(".", 1)[0]
    direct = scale_dir / stem / pair_image
    if direct.is_file():
        return direct
    matches = list(scale_dir.rglob("*.png"))
    return matches[0] if matches else None


def find_pasted_png(
    portrait_dir: Path, ckpt_stem: str, tgt_scale: float, pair_image: str
) -> Optional[Path]:
    scale_dir = portrait_dir / "swap_res_ori" / f"{ckpt_stem}_{tgt_scale}"
    stem = pair_image.rsplit(".", 1)[0]
    direct = scale_dir / stem / pair_image
    if direct.is_file():
        return direct
    matches = list(scale_dir.rglob("*.png")) if scale_dir.is_dir() else []
    return matches[0] if matches else None


class DiffSwapRunner:
    """Single-GPU DiffSwap model kept loaded across many pairs."""

    def __init__(
        self,
        diffswap_root: Path,
        checkpoint: str,
        gpu: str,
        tgt_scale: float,
        ddim_steps: int,
    ) -> None:
        self.diffswap_root = diffswap_root.resolve()
        self.checkpoint = checkpoint
        self.tgt_scale = tgt_scale
        self.ddim_steps = ddim_steps
        self.ckpt_path = self.diffswap_root / checkpoint
        if not self.ckpt_path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {self.ckpt_path}")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["PYTHONNOUSERSITE"] = "1"
        os.chdir(self.diffswap_root)
        if str(self.diffswap_root) not in sys.path:
            sys.path.insert(0, str(self.diffswap_root))

        from omegaconf import OmegaConf

        from ldm.data.portrait import Portrait
        from ldm.models.diffusion.ddim import DDIMSampler
        from ldm.util import instantiate_from_config
        from tests.faceswap_portrait import perform_swap

        self._Portrait = Portrait
        self._perform_swap = perform_swap

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"[diffswap] loading model on {self.device} ...", flush=True)
        t0 = time.perf_counter()
        config = OmegaConf.load("configs/diffswap/default-project.yaml")
        model = instantiate_from_config(config.model)
        model.init_from_ckpt(str(self.checkpoint))
        model.eval()
        model = model.to(self.device)
        model.cond_stage_model.affine_crop = True
        model.cond_stage_model.swap = True
        self.model = model
        self.ddim_sampler = DDIMSampler(model, tgt_scale=tgt_scale)
        self.ckpt_stem = self.ckpt_path.stem
        print(
            f"[diffswap] model ready in {time.perf_counter() - t0:.1f}s "
            f"(ddim_steps={ddim_steps})",
            flush=True,
        )

    def run_inference(self, portrait_rel: str) -> None:
        import tests.faceswap_portrait as faceswap_portrait

        faceswap_portrait.save_dir = portrait_rel
        dataset = self._Portrait(portrait_rel)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
        for batch in loader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            self._perform_swap(
                self.model,
                batch,
                self.ckpt_stem,
                self.ddim_sampler,
                ddim_steps=self.ddim_steps,
            )


def preprocess_pair(
    donor_path: Path,
    target_path: Path,
    diffswap_root: Path,
    portrait_rel: str,
    pair_image: str,
    env: dict,
) -> Tuple[int, str]:
    portrait_dir = diffswap_root / portrait_rel
    reset_portrait(portrait_dir)
    copy_as_png(donor_path, portrait_dir / "source" / pair_image)
    copy_as_png(target_path, portrait_dir / "target" / pair_image)

    os.chdir(diffswap_root)
    if str(diffswap_root) not in sys.path:
        sys.path.insert(0, str(diffswap_root))

    from pipeline import crop_ffhq, get_lmk_256, get_lmk_ori

    get_lmk_ori(data_path=portrait_rel, save_path=f"{portrait_rel}/landmark")
    crop_ffhq(
        data_path=portrait_rel,
        save_path=f"{portrait_rel}/align",
        affine_path=f"{portrait_rel}/affines.json",
        landmark_path=f"{portrait_rel}/landmark/landmark_ori.pkl",
    )
    get_lmk_256(
        data_path=f"{portrait_rel}/align",
        save_path=f"{portrait_rel}/landmark",
        error_path=f"{portrait_rel}/error_img.json",
    )

    def run_cmd(cmd: list) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd, cwd=str(diffswap_root), env=env, capture_output=True, text=True
        )

    proc = run_cmd(
        [sys.executable, "data_preprocessing/detection/detcect_faces_portrait.py", "0"]
    )
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        return proc.returncode, tail or "mtcnn detect failed"

    proc = run_cmd([sys.executable, "data_preprocessing/detection/merge_mtcnn_portrait.py"])
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        return proc.returncode, tail or "mtcnn merge failed"

    proc = run_cmd([sys.executable, "-m", "data_preprocessing.align.face_align_portrait"])
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
        return proc.returncode, tail or "face_align_portrait failed"

    return 0, ""


def finalize_pair_output(
    portrait_dir: Path,
    ckpt_stem: str,
    tgt_scale: float,
    pair_image: str,
    output_path: Path,
    paste_back: bool,
    diffswap_root: Path,
    portrait_rel: str,
) -> Tuple[int, str]:
    swap_png = find_swap_png(portrait_dir, ckpt_stem, tgt_scale, pair_image)
    if swap_png is None:
        return 1, "no swap PNG under swap_res"

    if paste_back:
        os.chdir(diffswap_root)
        if str(diffswap_root) not in sys.path:
            sys.path.insert(0, str(diffswap_root))
        from pipeline import paste

        paste(
            data_root=str(portrait_dir / "swap_res"),
            dst_dir=str(portrait_dir / "swap_res_ori"),
            ori_tgt_path=str(portrait_dir / "target"),
            affine_path=str(portrait_dir / "affines.json"),
        )
        result_png = find_pasted_png(portrait_dir, ckpt_stem, tgt_scale, pair_image)
        if result_png is None:
            return 1, "paste_back produced no image under swap_res_ori"
    else:
        result_png = swap_png

    image_bgr = cv2.imread(str(result_png))
    if image_bgr is None:
        return 1, f"failed to read swap output: {result_png}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_bgr):
        return 1, f"failed to write {output_path}"
    return 0, ""


def run_diffswap_swap(
    donor_path: Path,
    target_path: Path,
    output_path: Path,
    diffswap_root: Path,
    runner: DiffSwapRunner,
    *,
    portrait_rel: str,
    pair_image: str = DEFAULT_PAIR_IMAGE,
    tgt_scale: float = 0.01,
    paste_back: bool = True,
    env: Optional[dict] = None,
) -> Tuple[int, str]:
    diffswap_root = diffswap_root.resolve()
    portrait_dir = diffswap_root / portrait_rel
    if env is None:
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{diffswap_root}:{env.get('PYTHONPATH', '')}"
        env["DIFFSWAP_PORTRAIT"] = portrait_rel
        env["PYTHONNOUSERSITE"] = "1"

    prev_cwd = Path.cwd()
    try:
        code, err = preprocess_pair(
            donor_path, target_path, diffswap_root, portrait_rel, pair_image, env
        )
        if code != 0:
            return code, err

        runner.run_inference(portrait_rel)

        return finalize_pair_output(
            portrait_dir,
            runner.ckpt_stem,
            tgt_scale,
            pair_image,
            output_path,
            paste_back,
            diffswap_root,
            portrait_rel,
        )
    finally:
        os.chdir(prev_cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DiffSwap swaps for D1 pairs.")
    add_dataset_seed_args(parser)
    parser.add_argument("--diffswap-root", type=Path, default=DIFFSWAP_ROOT_DEFAULT)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/diffswap.pth",
        help="Checkpoint path relative to --diffswap-root.",
    )
    parser.add_argument("--gpu", type=str, default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument(
        "--portrait-workdir",
        type=str,
        default=None,
        help="DiffSwap work dir under repo (default: data/portrait_d1_<dataset>).",
    )
    parser.add_argument(
        "--tgt-scale",
        type=float,
        default=0.01,
        help=(
            "Target-guided DDIM strength (lower -> more donor identity, "
            "higher -> more target). Try 0 for stronger donor."
        ),
    )
    parser.add_argument(
        "--ddim-steps",
        type=int,
        default=200,
        help="DDIM sampling steps (default 200, same as upstream).",
    )
    parser.add_argument(
        "--pair-image-name",
        type=str,
        default=DEFAULT_PAIR_IMAGE,
        help="Filename used inside DiffSwap data/portrait for the pair.",
    )
    parser.add_argument(
        "--no-paste-back",
        dest="paste_back",
        action="store_false",
        help="Keep 256x256 aligned swap only (no paste into target frame).",
    )
    parser.set_defaults(paste_back=True)
    args = parser.parse_args()

    diffswap_root = args.diffswap_root.resolve()
    if not diffswap_root.is_dir():
        raise RuntimeError(f"DiffSwap root not found: {diffswap_root}")

    predictor = diffswap_root / "checkpoints/shape_predictor_68_face_landmarks.dat"
    if not predictor.is_file():
        raise RuntimeError(f"dlib predictor not found: {predictor}")

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

    portrait_rel = args.portrait_workdir or default_portrait_workdir(args.dataset)

    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    resume = metadata_path.is_file() and metadata_path.stat().st_size > 0
    meta_mode = "a" if resume else "w"

    print(
        f"[start] dataset={args.dataset} pairs={len(pairs)} gpu={args.gpu} "
        f"paste_back={args.paste_back} portrait_workdir={portrait_rel} "
        f"output_dir={output_dir} resume_metadata={resume}"
    )

    runner = DiffSwapRunner(
        diffswap_root,
        args.checkpoint,
        args.gpu,
        args.tgt_scale,
        args.ddim_steps,
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
                print(f"[{index}/{len(pairs)}] pair={pair.get('pair_id')} status=failed")
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
                print(f"[{index}/{len(pairs)}] pair={pair.get('pair_id')} status=failed")
                continue

            t_pair = time.perf_counter()
            return_code, stderr_tail = run_diffswap_swap(
                donor,
                target,
                out_path,
                diffswap_root,
                runner,
                portrait_rel=portrait_rel,
                pair_image=args.pair_image_name,
                tgt_scale=args.tgt_scale,
                paste_back=args.paste_back,
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
                    "paste_back": args.paste_back,
                    "tgt_scale": args.tgt_scale,
                    "checkpoint": args.checkpoint,
                    "portrait_workdir": portrait_rel,
                    "elapsed_s": round(elapsed, 2),
                },
            )
            print(
                f"[{index}/{len(pairs)}] pair={pair.get('pair_id')} "
                f"status={status} elapsed={elapsed:.1f}s"
            )

    print(
        f"[done] total={len(pairs)} ok={ok_count} failed={fail_count} "
        f"skipped_existing={skip_count} metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
