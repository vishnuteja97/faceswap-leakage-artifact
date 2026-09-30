"""
Run FaceShifter swaps for D1 donor-target pairs.

Run from a conda env with FaceShifter deps (torch, cv2, PIL), e.g.:
  conda run -n faceshifter python data/D1/swaps/run_faceshifter.py --dataset vggface2 --seed 42
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from d1_swap_common import (
    add_dataset_seed_args,
    build_output_name,
    donor_image_path,
    load_jsonl,
    resolve_run_paths,
    target_image_path,
    write_metadata_line,
)

FACESHIFTER_ROOT_DEFAULT = Path("/opt/reproduction/FaceShifter")
DEFAULT_GENERATOR = "G_latest_280000.pth"
DEFAULT_DISCRIMINATOR = "D_latest_280000.pth"
ARCface_CHECKPOINT = "face_modules/model_ir_se50.pth"
TOOL = "faceshifter"


def resolve_saved_model_checkpoint(
    faceshifter_root: Path, checkpoint: Optional[str], default_name: str
) -> Path:
    name = checkpoint or default_name
    path = Path(name)
    if not path.is_absolute():
        path = faceshifter_root / "saved_models" / path
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_faceshifter_models(faceshifter_root: Path, generator_path: Path, device_str: str):
    import torch
    import torchvision.transforms as transforms
    from face_modules.model import Backbone
    from face_modules.mtcnn import MTCNN
    from network.AEI_Net import AEI_Net

    arcface_path = faceshifter_root / ARCface_CHECKPOINT
    if not arcface_path.is_file():
        raise FileNotFoundError(f"ArcFace checkpoint not found: {arcface_path}")

    device = torch.device(device_str)
    detector = MTCNN()

    generator = AEI_Net(c_id=512)
    generator.eval()
    generator.load_state_dict(
        torch.load(str(generator_path), map_location=torch.device("cpu"))
    )
    generator = generator.to(device)

    arcface = Backbone(50, 0.6, "ir_se").to(device)
    arcface.eval()
    arcface.load_state_dict(
        torch.load(str(arcface_path), map_location=device), strict=False
    )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return generator, arcface, detector, device, transform


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FaceShifter swaps for D1 pairs.")
    add_dataset_seed_args(parser)
    parser.add_argument("--faceshifter-root", type=Path, default=FACESHIFTER_ROOT_DEFAULT)
    parser.add_argument("--generator-checkpoint", type=str, default=DEFAULT_GENERATOR)
    parser.add_argument("--discriminator-checkpoint", type=str, default=DEFAULT_DISCRIMINATOR)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    faceshifter_root = args.faceshifter_root.resolve()
    if not faceshifter_root.is_dir():
        raise RuntimeError(f"FaceShifter root not found: {faceshifter_root}")

    generator_path = resolve_saved_model_checkpoint(
        faceshifter_root, args.generator_checkpoint, DEFAULT_GENERATOR
    )
    discriminator_path = resolve_saved_model_checkpoint(
        faceshifter_root, args.discriminator_checkpoint, DEFAULT_DISCRIMINATOR
    )

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

    os.chdir(faceshifter_root)
    sys.path.insert(0, str(faceshifter_root))
    sys.path.insert(0, str(faceshifter_root / "face_modules"))

    import torch
    from do_faceswaps import do_faceswap  # noqa: E402

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[start] generator={generator_path.name} device={device_str}")
    generator, arcface, detector, device, transform = load_faceshifter_models(
        faceshifter_root, generator_path, device_str
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

            ok = do_faceswap(
                str(donor),
                str(target),
                str(out_path),
                generator,
                arcface,
                detector,
                device,
                transform,
            )
            status = "ok" if ok else "failed"
            return_code = 0 if ok else 1
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
                extra={
                    "generator_checkpoint": str(generator_path),
                    "discriminator_checkpoint": str(discriminator_path),
                },
            )
            print(f"[{index}/{len(pairs)}] pair={pair.get('pair_id')} status={status}")

    print(
        f"[done] total={len(pairs)} ok={ok_count} failed={fail_count} "
        f"skipped_existing={skip_count} metadata={metadata_path}"
    )


if __name__ == "__main__":
    main()
