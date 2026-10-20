"""Video-level inference for DeepfakeBench detectors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from dfb_backend.config import (
    DEFAULT_DETECTOR,
    DEFAULT_AGGREGATION,
    DEFAULT_FRAMES,
    DEFAULT_MIN_FRAMES,
    DEFAULT_TOPK_RATIO,
    FRAMES_ROOT,
)
from dfb_backend.loader import load_detector, reset_detector_buffers, resolve_device
from dfb_backend.preprocess import (
    iter_frame_paths,
    preprocess_frame,
    resolve_frames_dir,
    select_frame_indices,
    stack_frames,
)

_MODEL_CACHE: dict[tuple[str, str, str], tuple[torch.nn.Module, dict[str, Any]]] = {}


def fake_prob_to_real(fake_prob: float | np.ndarray) -> float | np.ndarray:
    """Convert DeepfakeBench P(fake) to the project's P(real) convention.

    This is the single probability-convention boundary between DeepfakeBench
    (fake = class 1) and the KYC verdict (REAL when P(real) >= threshold).
    """
    array = np.asarray(fake_prob, dtype=np.float64)
    if not np.isfinite(array).all() or ((array < 0.0) | (array > 1.0)).any():
        raise ValueError("fake_prob must be finite and in [0, 1]")
    real = 1.0 - array
    return float(real) if real.ndim == 0 else real


def aggregate_frame_probs(
    probs: Sequence[float],
    method: str = "mean",
    topk_ratio: float = 0.2,
) -> float:
    """Aggregate per-frame P(fake) into one video-level P(fake)."""
    values = np.asarray(list(probs), dtype=np.float64)
    if values.size == 0:
        raise ValueError("No frame probabilities to aggregate")
    if not np.isfinite(values).all():
        raise ValueError("Frame probabilities must be finite")
    if method == "mean":
        return float(values.mean())
    if method == "median":
        return float(np.median(values))
    if method == "topk":
        if not 0.0 < float(topk_ratio) <= 1.0:
            raise ValueError("topk_ratio must be in (0, 1]")
        k = max(1, int(math.ceil(values.size * float(topk_ratio))))
        return float(np.sort(values)[-k:].mean())
    raise ValueError(f"Unknown aggregation {method!r}")


def get_model(
    detector: str = DEFAULT_DETECTOR,
    *,
    device: str | torch.device | None = "auto",
    checkpoint: str | Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    device_obj = resolve_device(device)
    key = (detector, str(device_obj), str(checkpoint or ""))
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = load_detector(
            detector,
            device=device_obj,
            checkpoint=checkpoint,
            strict=True,
        )
    return _MODEL_CACHE[key]


def _logits_to_fake_probs(outputs: dict[str, Any]) -> np.ndarray:
    prob = outputs.get("prob")
    if prob is not None:
        values = torch.as_tensor(prob).detach().float().reshape(-1)
    else:
        cls = outputs.get("cls")
        if cls is None:
            raise RuntimeError("Detector output has neither 'prob' nor 'cls'")
        values = torch.softmax(torch.as_tensor(cls).detach().float(), dim=1)[:, 1]
    return values.cpu().numpy().astype(np.float64)


@torch.inference_mode()
def run_model_on_tensor(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Run a (N, 3, H, W) tensor and return per-frame P(fake)."""
    if images.ndim != 4:
        raise ValueError(f"Expected NCHW tensor, got shape {tuple(images.shape)}")
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, images.shape[0], max(1, int(batch_size))):
        batch = images[start : start + batch_size].to(device, non_blocking=True)
        labels = torch.zeros(batch.shape[0], dtype=torch.long, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            outputs = model({"image": batch, "label": labels}, inference=True)
        output.append(_logits_to_fake_probs(outputs))
        reset_detector_buffers(model)
    return np.concatenate(output, axis=0)


def _load_video_tensor(
    paths: Sequence[str],
    indices: Sequence[int],
    *,
    resolution: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    return stack_frames(
        paths,
        indices,
        resolution=resolution,
        mean=mean,
        std=std,
        training=False,
    )


def predict_video(
    video_path: str | Path | None = None,
    *,
    frames_dir: str | Path | None = None,
    detector: str = DEFAULT_DETECTOR,
    frames: int = DEFAULT_FRAMES,
    min_frames: int = DEFAULT_MIN_FRAMES,
    aggregation: str = DEFAULT_AGGREGATION,
    topk_ratio: float = DEFAULT_TOPK_RATIO,
    threshold: float = 0.5,
    device: str | torch.device | None = "auto",
    model: torch.nn.Module | None = None,
    meta: dict[str, Any] | None = None,
    checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Predict P(real) for one stage-1 video directory.

    Returns ``verdict='INCONCLUSIVE'`` with ``prob_real=None`` when fewer than
    ``min_frames`` accepted frames exist, matching the pipeline's exit-code-3
    contract.
    """
    if frames_dir is None:
        if video_path is None:
            raise ValueError("frames_dir or video_path is required")
        frames_dir = Path(FRAMES_ROOT) / Path(video_path).stem
    try:
        resolved = resolve_frames_dir(frames_dir)
        paths = iter_frame_paths(resolved)
    except FileNotFoundError as exc:
        return {
            "backend": "dfb",
            "detector": detector,
            "frames_dir": str(frames_dir),
            "prob_real": None,
            "verdict": "INCONCLUSIVE",
            "confidence": None,
            "reason": str(exc),
            "frames_used": 0,
            "frame_fake_probs": [],
        }

    if len(paths) < int(min_frames):
        return {
            "backend": "dfb",
            "detector": detector,
            "frames_dir": str(resolved),
            "prob_real": None,
            "verdict": "INCONCLUSIVE",
            "confidence": None,
            "reason": (
                f"Need at least {min_frames} accepted frames, found {len(paths)}"
            ),
            "frames_available": len(paths),
            "frames_used": 0,
            "frame_fake_probs": [],
        }

    if model is None or meta is None:
        model, meta = get_model(detector, device=device, checkpoint=checkpoint)
    device_obj = resolve_device(device)
    selected = select_frame_indices(len(paths), min(int(frames), len(paths)))
    tensor = _load_video_tensor(
        paths,
        selected,
        resolution=int(meta["resolution"]),
        mean=meta["mean"],
        std=meta["std"],
    )
    frame_fake = run_model_on_tensor(model, tensor, device=device_obj)
    video_fake = aggregate_frame_probs(frame_fake, aggregation, topk_ratio)
    prob_real = float(fake_prob_to_real(video_fake))
    verdict = "REAL" if prob_real >= float(threshold) else "FAKE"
    return {
        "backend": "dfb",
        "detector": detector,
        "frames_dir": str(resolved),
        "frames_available": len(paths),
        "frames_used": len(selected),
        "aggregation": aggregation,
        "topk_ratio": float(topk_ratio),
        "frame_fake_probs": [round(float(v), 6) for v in frame_fake],
        "video_fake_prob": round(float(video_fake), 6),
        "prob_real": round(prob_real, 6),
        "verdict": verdict,
        "confidence": round(abs(prob_real - 0.5) * 2.0, 6),
        "threshold": float(threshold),
        "checkpoint": meta.get("checkpoint"),
        "resolution": int(meta["resolution"]),
    }


def predict_records(
    records: Iterable[Any],
    model: torch.nn.Module,
    meta: dict[str, Any],
    *,
    frames: int = DEFAULT_FRAMES,
    min_frames: int = DEFAULT_MIN_FRAMES,
    aggregation: str = DEFAULT_AGGREGATION,
    topk_ratio: float = DEFAULT_TOPK_RATIO,
    batch_videos: int = 4,
    device: str | torch.device | None = "auto",
    return_frames: bool = False,
) -> list[dict[str, Any]]:
    """Batched video-level inference for aligned ``VideoRecord`` objects."""
    device_obj = resolve_device(device)
    model.eval()
    records = list(records)
    results: list[dict[str, Any]] = []
    pending: list[tuple[Any, list[int], torch.Tensor]] = []

    def flush() -> None:
        if not pending:
            return
        lengths = [tensor.shape[0] for _, _, tensor in pending]
        batch = torch.cat([tensor for _, _, tensor in pending], dim=0)
        fake_probs = run_model_on_tensor(model, batch, device=device_obj)
        offset = 0
        for (record, selected, _tensor), n in zip(pending, lengths):
            frame_fake = fake_probs[offset : offset + n]
            offset += n
            video_fake = aggregate_frame_probs(frame_fake, aggregation, topk_ratio)
            prob_real = float(fake_prob_to_real(video_fake))
            item = {
                "video_path": record.video_path,
                "video_key": record.video_key,
                "group": record.group,
                "split": record.split,
                "frames_used": len(selected),
                "video_fake_prob": round(float(video_fake), 6),
                "prob_real": round(prob_real, 6),
                "y_true_quantum": int(record.label),
                "y_true_fake": int(1 - int(record.label)),
                "rppg": list(record.rppg) if hasattr(record, "rppg") else None,
            }
            if return_frames:
                item["frame_fake_probs"] = [
                    round(float(v), 6) for v in frame_fake
                ]
            results.append(item)
        pending.clear()

    for record in records:
        paths = tuple(record.frame_paths)
        if len(paths) < int(min_frames):
            results.append(
                {
                    "video_path": record.video_path,
                    "video_key": record.video_key,
                    "group": record.group,
                    "split": record.split,
                    "frames_used": 0,
                    "prob_real": None,
                    "y_true_quantum": int(record.label),
                    "y_true_fake": int(1 - int(record.label)),
                    "rppg": list(record.rppg),
                    "reason": (
                        f"Need at least {min_frames} frames, found {len(paths)}"
                    ),
                }
            )
            continue
        selected = select_frame_indices(len(paths), min(int(frames), len(paths)))
        tensor = stack_frames(
            paths,
            selected,
            resolution=int(meta["resolution"]),
            mean=meta["mean"],
            std=meta["std"],
            training=False,
        )
        pending.append((record, selected, tensor))
        if len(pending) >= max(1, int(batch_videos)):
            flush()
    flush()
    return results
