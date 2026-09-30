"""
Run BlendFace (AEI-Net + BlendFace) swaps for D1 donor-target pairs.

BlendFace expects 112x112 donor crops and 256x256 target crops. VGGFace2 uses
pre-aligned 512px crops resized to the model input sizes; the 256px swap is
upscaled back to 512px.

Run from the blendface conda env, e.g.:
  conda run -n blendface python data/D1/swaps/run_blendface.py --seed 42 --max-pairs 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from d1_swap_common import (
    add_dataset_seed_args,
    build_output_name,
    donor_image_path,
    load_jsonl,
    resolve_run_paths,
    target_image_path,
    write_metadata_line,
)

BLENDFACE_ROOT_DEFAULT = Path("/opt/reproduction/BlendFace")
BLENDSWAP_CHECKPOINT = "swapping/checkpoints/blendswap.pth"
DONOR_SIZE = 112
TARGET_SIZE = 256
VGG_OUTPUT_SIZE = 512
TOOL = "blendface"


def load_blendswap_model(blendface_root: Path, device_str: str):
    swapping_dir = blendface_root / "swapping"
    if not swapping_dir.is_dir():
        raise FileNotFoundError(f"BlendFace swapping dir not found: {swapping_dir}")

    checkpoint = blendface_root / BLENDSWAP_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"BlendSwap checkpoint not found: {checkpoint}")

    if str(swapping_dir) not in sys.path:
        sys.path.insert(0, str(swapping_dir))

    from blendswap import BlendSwap  # noqa: E402

    device = torch.device(device_str)
    model = BlendSwap()
    model.load_state_dict(torch.load(str(checkpoint), map_location="cpu"))
    model.eval()
    model.to(device)
    return model, device


def run_blendswap(
    model,
    device: torch.device,
    donor_pil: Image.Image,
    target_pil: Image.Image,
) -> np.ndarray:
    donor = donor_pil.convert("RGB").resize((DONOR_SIZE, DONOR_SIZE), Image.LANCZOS)
    target = target_pil.convert("RGB").resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    to_tensor = transforms.ToTensor()
    source_img = to_tensor(donor).unsqueeze(0).to(device)
    target_img = to_tensor(target).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(target_img, source_img)

    swap_rgb = (
        output.permute(0, 2, 3, 1)[0].cpu().numpy() * 255.0
    ).clip(0, 255).astype(np.uint8)
    return swap_rgb


def swap_pair(
    model,
    device: torch.device,
    donor_path: Path,
    target_path: Path,
    output_path: Path,
) -> None:
    donor_pil = Image.open(donor_path).convert("RGB")
    target_pil = Image.open(target_path).convert("RGB")
    swap_rgb = run_blendswap(model, device, donor_pil, target_pil)

    if swap_rgb.shape[0] != VGG_OUTPUT_SIZE or swap_rgb.shape[1] != VGG_OUTPUT_SIZE:
        swap_rgb = cv2.resize(
            swap_rgb,
            (VGG_OUTPUT_SIZE, VGG_OUTPUT_SIZE),
            interpolation=cv2.INTER_LANCZOS4,
        )

    if not cv2.imwrite(str(output_path), cv2.cvtColor(swap_rgb, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"failed to write {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BlendFace swaps for D1 pairs.")
    add_dataset_seed_args(parser)
    parser.add_argument("--blendface-root", type=Path, default=BLENDFACE_ROOT_DEFAULT)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    blendface_root = args.blendface_root.resolve()
    if not blendface_root.is_dir():
        raise RuntimeError(f"BlendFace root not found: {blendface_root}")

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

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[start] dataset={args.dataset} pairs={len(pairs)} device={device_str} "
        f"output_dir={output_dir}"
    )

    model, device = load_blendswap_model(blendface_root, device_str)

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

            status = "ok"
            return_code = 0
            stderr_tail = ""
            try:
                swap_pair(model, device, donor, target, out_path)
                ok_count += 1
            except Exception as exc:
                status = "failed"
                return_code = 1
                stderr_tail = str(exc)
                fail_count += 1

            write_metadata_line(
                meta_f,
                index=index,
                pair=pair,
                swap_image=out_path,
                status=status,
                return_code=return_code,
                stderr_tail=stderr_tail,
                extra={"checkpoint": str(blendface_root / BLENDSWAP_CHECKPOINT)},
            )
            print(
                f"[{index}/{len(pairs)}] pair={pair.get('pair_id')} "
                f"status={status} -> {out_path.name}"
            )

    print(
        f"[done] total={len(pairs)} ok={ok_count} failed={fail_count} "
        f"skipped_existing={skip_count} metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
