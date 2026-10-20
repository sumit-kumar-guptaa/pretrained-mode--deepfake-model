"""Dataset construction for the ViT stage.

Videos are aligned to the existing fused split manifest, the labelled rPPG
CSV, and stage-1 accepted frames. Splits are always at video level; the
assertions below abort on any video/group overlap.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from quantum.config import LABEL_FAKE, LABEL_REAL
from quantum.data import csv_to_quantum_label

from vit.config import VitConfig

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class VideoRecord:
    video_path: str
    video_key: str
    group: str
    split: str
    frames_dir: str
    frame_paths: tuple[str, ...]
    rppg: tuple[float, ...]
    label: int

    @property
    def n_frames(self) -> int:
        return len(self.frame_paths)


def normalise_key(value: str | Path) -> str:
    return str(value).replace("\\", "/").strip().lower()


def video_stem(value: str | Path) -> str:
    return Path(str(value).replace("\\", "/")).stem.lower()


def frame_paths_for_video(video_dir: Path) -> tuple[Path, ...]:
    """Return accepted face crops, preferring saved crops when present."""
    crops_dir = video_dir / "cropped_faces"
    if crops_dir.is_dir():
        crops = sorted(p for p in crops_dir.glob("*.jpg") if p.is_file())
        if crops:
            return tuple(crops)
    frames_dir = video_dir / "frames"
    if not frames_dir.is_dir():
        frames_dir = video_dir
    frames = sorted(p for p in frames_dir.glob("*.jpg") if p.is_file())
    return tuple(frames)


def build_video_dir_index(frames_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    if not frames_root.is_dir():
        return index
    for child in sorted(frames_root.iterdir()):
        if not child.is_dir():
            continue
        for key in {normalise_key(child.name), video_stem(child.name)}:
            if key in index and index[key] != child:
                raise ValueError(
                    f"Ambiguous stage-1 frame directory for key {key!r}: "
                    f"{index[key]} and {child}"
                )
            index[key] = child
    return index


def find_frames_dir(frames_root: Path, video_path: str, index: dict[str, Path] | None = None) -> Path | None:
    index = index or build_video_dir_index(frames_root)
    for key in (
        normalise_key(Path(video_path).with_suffix("")),
        normalise_key(Path(video_path).name),
        video_stem(video_path),
    ):
        if key in index:
            return index[key]
    return None


def load_split_manifest(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {path}. Build the fused quantum dataset "
            "first (from WORKING/: python -m quantum.pipeline --build-data) so "
            "ViT and VQC use the identical video split."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows")
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"Split manifest has no rows: {path}")
    return payload


def load_rppg_rows(path: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """Index labelled rPPG rows by exact path and by unique video stem."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"rPPG feature CSV not found: {path}. Run RPPG/extract_dataset_features.py first."
        )
    by_path: dict[str, dict] = {}
    by_stem: dict[str, dict] = {}
    duplicate_stems: set[str] = set()
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            video_path = str(row.get("video_path", "")).strip()
            if not video_path:
                continue
            by_path[normalise_key(video_path)] = row
            stem = video_stem(video_path)
            if stem in by_stem:
                duplicate_stems.add(stem)
            else:
                by_stem[stem] = row
    for stem in duplicate_stems:
        by_stem.pop(stem, None)
    return by_path, by_stem


def _rppg_row_for(path: str, by_path: dict[str, dict], by_stem: dict[str, dict]) -> dict | None:
    return by_path.get(normalise_key(path)) or by_stem.get(video_stem(path))


def _rppg_vector(row: dict, feature_names: Sequence[str]) -> tuple[float, ...]:
    values: list[float] = []
    for name in feature_names:
        if name not in row:
            raise ValueError(f"rPPG CSV row is missing feature {name!r}")
        value = float(row[name])
        if not np.isfinite(value):
            raise ValueError(f"rPPG feature {name!r} is non-finite")
        values.append(value)
    return tuple(values)


def align_video_records(
    manifest: dict,
    frames_root: Path,
    rppg_csv: Path,
    feature_names: Sequence[str],
    min_frames: int = 1,
) -> tuple[list[VideoRecord], dict]:
    """Align manifest rows to frames/rPPG and return video-level records."""
    frames_root = Path(frames_root)
    by_path, by_stem = load_rppg_rows(Path(rppg_csv))
    dir_index = build_video_dir_index(frames_root)

    records: list[VideoRecord] = []
    stats = {
        "manifest_rows": 0,
        "missing_frames": 0,
        "missing_rppg": 0,
        "too_few_frames": 0,
        "invalid_rppg": 0,
        "aligned": 0,
    }
    for video_path, item in manifest["rows"].items():
        stats["manifest_rows"] += 1
        split = str(item.get("split", ""))
        group = str(item.get("group", ""))
        if split not in SPLITS or not group:
            raise ValueError(f"Invalid split manifest entry for {video_path}: {item}")
        video_dir = find_frames_dir(frames_root, str(video_path), dir_index)
        if video_dir is None:
            stats["missing_frames"] += 1
            continue
        paths = frame_paths_for_video(video_dir)
        if not paths:
            stats["missing_frames"] += 1
            continue
        if len(paths) < min_frames:
            stats["too_few_frames"] += 1
            continue
        row = _rppg_row_for(str(video_path), by_path, by_stem)
        if row is None:
            stats["missing_rppg"] += 1
            continue
        try:
            rppg = _rppg_vector(row, feature_names)
            label = csv_to_quantum_label(int(round(float(row["label"]))))
        except (KeyError, TypeError, ValueError):
            stats["invalid_rppg"] += 1
            continue
        if label not in (LABEL_REAL, LABEL_FAKE):
            stats["invalid_rppg"] += 1
            continue
        records.append(
            VideoRecord(
                video_path=str(video_path),
                video_key=video_stem(video_path),
                group=group,
                split=split,
                frames_dir=str(video_dir),
                frame_paths=tuple(str(p) for p in paths),
                rppg=rppg,
                label=label,
            )
        )
    stats["aligned"] = len(records)
    if not records:
        raise ValueError(
            "No videos could be aligned across split manifest, rPPG CSV, and "
            f"stage-1 frames under {frames_root}. Stats: {stats}"
        )
    return records, stats


def assert_zero_video_overlap(records: Iterable[VideoRecord]) -> None:
    records = list(records)
    by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    groups: dict[str, set[str]] = {split: set() for split in SPLITS}
    for record in records:
        by_split.setdefault(record.split, set()).add(record.video_key)
        groups.setdefault(record.split, set()).add(record.group)
    for left in SPLITS:
        for right in SPLITS:
            if left >= right:
                continue
            video_overlap = by_split[left] & by_split[right]
            if video_overlap:
                raise AssertionError(
                    f"Video leakage between {left} and {right}: {sorted(video_overlap)[:10]}"
                )
            group_overlap = groups[left] & groups[right]
            if group_overlap:
                raise AssertionError(
                    f"Group leakage between {left} and {right}: {sorted(group_overlap)[:10]}"
                )


def split_records(records: Iterable[VideoRecord]) -> dict[str, list[VideoRecord]]:
    records = list(records)
    assert_zero_video_overlap(records)
    splits = {split: [] for split in SPLITS}
    for record in records:
        splits[record.split].append(record)
    counts = {
        split: {
            "videos": len(items),
            "real": sum(r.label == LABEL_REAL for r in items),
            "fake": sum(r.label == LABEL_FAKE for r in items),
        }
        for split, items in splits.items()
    }
    if not splits["train"] or not splits["val"]:
        raise ValueError(f"Train/val split is empty after alignment: {counts}")
    return splits


def cap_records(records: list[VideoRecord], max_videos: int | None, seed: int = 42) -> list[VideoRecord]:
    """Deterministically cap total videos while preserving every non-empty split."""
    if not max_videos or len(records) <= max_videos:
        return records
    rng = random.Random(seed)
    by_split = {split: [r for r in records if r.split == split] for split in SPLITS}
    present = [split for split in SPLITS if by_split[split]]
    if max_videos < len(present):
        raise ValueError(
            f"max_videos={max_videos} cannot preserve all {len(present)} non-empty splits"
        )
    total = len(records)
    extras = max_videos - len(present)
    exact = {
        split: extras * len(by_split[split]) / total
        for split in present
    }
    quotas = {split: 1 + int(np.floor(exact[split])) for split in present}
    remaining = max_videos - sum(quotas.values())
    order = sorted(
        present,
        key=lambda split: (-(exact[split] - np.floor(exact[split])), SPLITS.index(split)),
    )
    while remaining > 0:
        added = False
        for split in order:
            if remaining <= 0:
                break
            if quotas[split] < len(by_split[split]):
                quotas[split] += 1
                remaining -= 1
                added = True
        if not added:
            break
    capped: list[VideoRecord] = []
    for split in SPLITS:
        if split not in quotas:
            continue
        items = list(by_split[split])
        rng.shuffle(items)
        capped.extend(items[: quotas[split]])
    rng.shuffle(capped)
    return capped


def sample_frame_indices(
    n_frames: int,
    count: int,
    rng: np.random.Generator | None = None,
    jitter: bool = False,
) -> list[int]:
    if n_frames < count:
        raise ValueError(f"Need {count} frames, video has only {n_frames}")
    if count == 1:
        return [int(rng.integers(0, n_frames)) if jitter and rng else n_frames // 2]
    if jitter and rng is not None:
        bins = np.linspace(0, n_frames, count + 1)
        picks = [
            int(rng.integers(int(np.floor(bins[i])), max(int(np.floor(bins[i])) + 1, int(np.floor(bins[i + 1])))))
            for i in range(count)
        ]
        picks = [min(max(p, 0), n_frames - 1) for p in picks]
        if len(set(picks)) < count:
            picks = list(np.linspace(0, n_frames - 1, count, dtype=int))
        return picks
    return list(np.linspace(0, n_frames - 1, count, dtype=int))


def _jpeg_compress(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return image
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def augment_frame(image_rgb: np.ndarray, cfg: VitConfig, rng: np.random.Generator) -> np.ndarray:
    """Low-resolution KYC augmentation (train only)."""
    image = image_rgb
    image = cv2.resize(image, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)

    if rng.random() < 0.5:
        low = int(rng.integers(cfg.augment_downscale_min, cfg.augment_downscale_max + 1))
        image = cv2.resize(image, (low, low), interpolation=cv2.INTER_AREA)
        image = cv2.resize(image, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)

    if rng.random() < 0.5:
        quality = int(rng.integers(cfg.augment_jpeg_quality_min, cfg.augment_jpeg_quality_max + 1))
        image = _jpeg_compress(image, quality)

    if rng.random() < 0.20:
        sigma = float(rng.uniform(0.4, 1.8))
        image = cv2.GaussianBlur(image, (3, 3), sigmaX=sigma)

    if rng.random() < 0.20:
        noise = rng.normal(0.0, float(rng.uniform(1.5, 5.0)), image.shape)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.5:
        image = cv2.flip(image, 1)

    if rng.random() < 0.5:
        alpha = float(rng.uniform(0.90, 1.10))
        beta = float(rng.uniform(-8.0, 8.0))
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.4:
        gains = rng.uniform(0.95, 1.05, size=3).astype(np.float32)
        image = np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)

    if rng.random() < 0.4:
        scale = float(rng.uniform(0.85, 1.0))
        side = max(16, int(round(cfg.image_size * scale)))
        max_off = cfg.image_size - side
        x = int(rng.integers(0, max_off + 1))
        y = int(rng.integers(0, max_off + 1))
        image = image[y:y + side, x:x + side]
        image = cv2.resize(image, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_LINEAR)

    return image


def preprocess_frame(
    path: str | Path,
    cfg: VitConfig,
    training: bool,
    rng: np.random.Generator | None = None,
) -> torch.Tensor:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise IOError(f"Could not read frame: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if training:
        if cfg.augment:
            assert rng is not None
            rgb = augment_frame(rgb, cfg, rng)
        else:
            rgb = cv2.resize(rgb, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
    else:
        rgb = cv2.resize(rgb, (cfg.image_size, cfg.image_size), interpolation=cv2.INTER_AREA)
    array = rgb.astype(np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


class VitVideoDataset(Dataset):
    def __init__(
        self,
        records: Sequence[VideoRecord],
        cfg: VitConfig,
        training: bool,
        seed: int | None = None,
    ) -> None:
        self.records = list(records)
        self.cfg = cfg
        self.training = training
        self.seed = cfg.seed if seed is None else seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int):
        record = self.records[index]
        rng = np.random.default_rng(self.seed + self.epoch * 100_003 + index)
        picks = sample_frame_indices(
            len(record.frame_paths),
            self.cfg.frames_per_video,
            rng=rng,
            jitter=self.training,
        )
        frames = torch.stack(
            [
                preprocess_frame(record.frame_paths[i], self.cfg, self.training, rng)
                for i in picks
            ]
        )
        rppg = torch.tensor(record.rppg, dtype=torch.float32)
        label = torch.tensor(float(record.label), dtype=torch.float32)
        return frames, rppg, label, index


def make_balanced_sampler(records: Sequence[VideoRecord], seed: int = 42) -> WeightedRandomSampler:
    labels = np.asarray([r.label for r in records], dtype=int)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[labels]
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )


def prepare_inference_frames(
    frames_dir: str | Path,
    count: int,
    image_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[str]]:
    paths = frame_paths_for_video(Path(frames_dir))
    if len(paths) < count:
        raise ValueError(f"Need at least {count} accepted frames, found {len(paths)}")
    indices = sample_frame_indices(len(paths), count, rng=None, jitter=False)
    cfg = VitConfig(image_size=image_size, frames_per_video=count, min_frames=count, pretrained=False)
    tensors = [preprocess_frame(paths[i], cfg, training=False) for i in indices]
    return torch.stack(tensors).to(device), [str(paths[i]) for i in indices]
