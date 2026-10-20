"""Frame discovery and DeepfakeBench-compatible preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch


def _sorted_jpgs(path: Path) -> tuple[Path, ...]:
    return tuple(sorted(p for p in path.glob("*.jpg") if p.is_file()))


def resolve_frames_dir(frames_dir: str | Path) -> Path:
    """Accept either a stage-1 video directory or its ``frames`` directory."""
    path = Path(frames_dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Frames directory not found: {path}")
    crops = path / "cropped_faces"
    if crops.is_dir() and _sorted_jpgs(crops):
        return crops
    frames = path / "frames"
    if frames.is_dir() and _sorted_jpgs(frames):
        return frames
    if _sorted_jpgs(path):
        return path
    raise FileNotFoundError(f"No .jpg frames found under {path}")


def iter_frame_paths(frames_dir: str | Path) -> tuple[str, ...]:
    return tuple(str(p) for p in _sorted_jpgs(resolve_frames_dir(frames_dir)))


def select_frame_indices(
    n_frames: int,
    count: int,
    *,
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> list[int]:
    """Evenly spaced frames, optionally jittered within each temporal bin."""
    if n_frames < 1 or count < 1:
        raise ValueError("n_frames and count must be positive")
    if n_frames < count:
        raise ValueError(f"Need {count} frames, video has only {n_frames}")
    count = min(count, n_frames)
    if count == 1:
        if training and rng is not None:
            return [int(rng.integers(0, n_frames))]
        return [n_frames // 2]
    base = np.linspace(0, n_frames - 1, count).astype(int)
    if not training or rng is None:
        return [int(i) for i in base]
    bins = np.linspace(0, n_frames, count + 1)
    picks: list[int] = []
    for i in range(count):
        lo = int(np.floor(bins[i]))
        hi = max(lo + 1, int(np.floor(bins[i + 1])))
        picks.append(int(rng.integers(lo, min(hi, n_frames))))
    if len(set(picks)) < count:
        return [int(i) for i in base]
    return picks


def _jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    )
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def augment_kyc(
    image_rgb: np.ndarray,
    resolution: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Low-resolution KYC augmentation applied after the cubic resize."""
    image = cv2.resize(
        image_rgb, (resolution, resolution), interpolation=cv2.INTER_CUBIC
    )
    if rng.random() < 0.5:
        low = int(rng.integers(max(48, resolution // 4), max(49, resolution // 2)))
        image = cv2.resize(image, (low, low), interpolation=cv2.INTER_AREA)
        image = cv2.resize(
            image, (resolution, resolution), interpolation=cv2.INTER_CUBIC
        )
    if rng.random() < 0.5:
        image = _jpeg_compress(image, int(rng.integers(30, 91)))
    if rng.random() < 0.5:
        image = cv2.flip(image, 1)
    if rng.random() < 0.3:
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=float(rng.uniform(0.3, 1.2)))
    if rng.random() < 0.3:
        noise = rng.normal(0.0, float(rng.uniform(1.0, 4.0)), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        alpha = float(rng.uniform(0.90, 1.10))
        beta = float(rng.uniform(-8.0, 8.0))
        image = np.clip(
            image.astype(np.float32) * alpha + beta, 0, 255
        ).astype(np.uint8)
    return image


def preprocess_array(
    bgr: np.ndarray,
    *,
    resolution: int,
    mean: Sequence[float],
    std: Sequence[float],
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    """Mirror DeepfakeBench test preprocessing: BGR->RGB, cubic resize, /255, normalize."""
    if bgr is None or bgr.size == 0:
        raise ValueError("Empty image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if training:
        if rng is None:
            rng = np.random.default_rng(0)
        rgb = augment_kyc(rgb, resolution, rng)
    else:
        rgb = cv2.resize(
            rgb, (resolution, resolution), interpolation=cv2.INTER_CUBIC
        )
    array = rgb.astype(np.float32) / 255.0
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(
        std, dtype=np.float32
    )
    tensor = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))
    return tensor


def preprocess_frame(
    path: str | Path,
    *,
    resolution: int,
    mean: Sequence[float],
    std: Sequence[float],
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Could not read frame: {path}")
    return preprocess_array(
        bgr,
        resolution=resolution,
        mean=mean,
        std=std,
        training=training,
        rng=rng,
    )


def stack_frames(
    frame_paths: Sequence[str | Path],
    indices: Iterable[int],
    *,
    resolution: int,
    mean: Sequence[float],
    std: Sequence[float],
    training: bool = False,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    tensors = [
        preprocess_frame(
            frame_paths[i],
            resolution=resolution,
            mean=mean,
            std=std,
            training=training,
            rng=rng,
        )
        for i in indices
    ]
    if not tensors:
        raise ValueError("No frames selected")
    return torch.stack(tensors)
