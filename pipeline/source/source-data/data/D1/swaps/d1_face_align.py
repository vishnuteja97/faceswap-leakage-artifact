"""Detect / align / paste-back helpers for in-the-wild D1 swaps."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

FACESHIFTER_ROOT_DEFAULT = Path("/opt/reproduction/FaceShifter")


class FaceAlignError(Exception):
    pass


def load_mtcnn(faceshifter_root: Path):
    root = faceshifter_root.resolve()
    face_modules = root / "face_modules"
    if not face_modules.is_dir():
        raise FileNotFoundError(f"FaceShifter face_modules not found: {face_modules}")
    for path in (root, face_modules):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    from face_modules.mtcnn import MTCNN  # noqa: E402

    return MTCNN()


def feather_mask(crop_size: int, dilate_iter: int = 20) -> np.ndarray:
    center = (crop_size - 1) / 2.0
    yy, xx = np.ogrid[:crop_size, :crop_size]
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / center
    mask = 1.0 - np.minimum(dist, 1.0)
    return cv2.dilate(mask.astype(np.float32), None, iterations=dilate_iter)


def align_pair(
    detector,
    donor_path: Path,
    target_path: Path,
    crop_size: int = 512,
) -> Tuple[Image.Image, Image.Image, np.ndarray, np.ndarray]:
    donor_pil = Image.open(donor_path).convert("RGB")
    target_pil = Image.open(target_path).convert("RGB")
    target_bgr = cv2.cvtColor(np.array(target_pil), cv2.COLOR_RGB2BGR)

    try:
        donor_aligned = detector.align(donor_pil, crop_size=(crop_size, crop_size))
    except Exception as exc:
        raise FaceAlignError(f"donor align failed: {exc}") from exc
    if donor_aligned is None:
        raise FaceAlignError("donor: no face detected")

    try:
        aligned = detector.align(
            target_pil,
            crop_size=(crop_size, crop_size),
            return_trans_inv=True,
        )
    except Exception as exc:
        raise FaceAlignError(f"target align failed: {exc}") from exc
    if aligned is None:
        raise FaceAlignError("target: no face detected")

    target_aligned, trans_inv = aligned
    return donor_aligned, target_aligned, trans_inv, target_bgr


def paste_back(
    target_bgr: np.ndarray,
    swap_rgb: np.ndarray,
    trans_inv: np.ndarray,
    crop_size: int,
) -> np.ndarray:
    if swap_rgb.shape[0] != crop_size or swap_rgb.shape[1] != crop_size:
        swap_rgb = cv2.resize(
            swap_rgb,
            (crop_size, crop_size),
            interpolation=cv2.INTER_LANCZOS4,
        )

    swap_bgr = cv2.cvtColor(swap_rgb, cv2.COLOR_RGB2BGR)
    height, width = target_bgr.shape[:2]
    target_f = target_bgr.astype(np.float64) / 255.0
    mask = feather_mask(crop_size)

    swapped_warped = cv2.warpAffine(
        swap_bgr.astype(np.float64) / 255.0,
        trans_inv,
        (width, height),
        borderValue=(0, 0, 0),
    )
    mask_warped = cv2.warpAffine(mask, trans_inv, (width, height), borderValue=0.0)
    mask_warped = np.expand_dims(mask_warped, axis=2)

    blended = mask_warped * swapped_warped + (1.0 - mask_warped) * target_f
    return np.clip(blended * 255.0, 0, 255).astype(np.uint8)
