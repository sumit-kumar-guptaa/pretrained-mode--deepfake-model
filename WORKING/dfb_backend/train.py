"""Low-LR DFDC fine-tuning for DeepfakeBench detectors."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from dfb_backend.config import (
    DEFAULT_DETECTOR,
    DFB_OUTPUT_DIR,
    FRAMES_ROOT,
    RPPG_CSV,
    SPLIT_MANIFEST,
    finetuned_checkpoint,
    finetuned_metadata,
)
from dfb_backend.fusion import choose_conservative_threshold
from dfb_backend.loader import (
    load_detector,
    quantum_label_to_dfb_label,
    reset_detector_buffers,
    resolve_device,
    set_backbone_frozen,
)
from dfb_backend.predict import predict_records
from dfb_backend.preprocess import stack_frames, select_frame_indices
from quantum.config import LABEL_FAKE, LABEL_REAL, RPPG_FEATURE_NAMES
from quantum.evaluation import balanced_accuracy


@dataclass
class FineTuneConfig:
    detector: str = DEFAULT_DETECTOR
    epochs: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_videos: int = 4
    frames_per_video: int = 8
    val_frames: int = 16
    freeze_backbone_epochs: int = 1
    patience: int = 4
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    label_smoothing: float = 0.05
    amp: bool = True
    seed: int = 42
    device: str = "auto"
    max_train_videos: int | None = None

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_videos < 1 or self.frames_per_video < 1:
            raise ValueError("batch_videos and frames_per_video must be positive")
        if not 0.0 <= self.freeze_backbone_epochs <= self.epochs:
            raise ValueError("freeze_backbone_epochs must be in [0, epochs]")


class DfbVideoDataset(Dataset):
    def __init__(
        self,
        records: Sequence[Any],
        meta: dict[str, Any],
        cfg: FineTuneConfig,
        *,
        training: bool,
        frames: int,
    ) -> None:
        self.records = list(records)
        self.meta = meta
        self.cfg = cfg
        self.training = training
        self.frames = int(frames)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        record = self.records[index]
        rng = np.random.default_rng(
            self.cfg.seed + self.epoch * 100_003 + index
        )
        picks = select_frame_indices(
            len(record.frame_paths),
            min(self.frames, len(record.frame_paths)),
            training=self.training,
            rng=rng,
        )
        tensor = stack_frames(
            record.frame_paths,
            picks,
            resolution=int(self.meta["resolution"]),
            mean=self.meta["mean"],
            std=self.meta["std"],
            training=self.training,
            rng=rng,
        )
        label = quantum_label_to_dfb_label(int(record.label))
        return tensor, torch.tensor(label, dtype=torch.long), index


def _collate(batch):
    tensors, labels, indices = zip(*batch)
    return torch.stack(tensors), torch.stack(labels), list(indices)


def _balanced_sampler(records: Sequence[Any], seed: int):
    labels = np.asarray([int(r.label) for r in records], dtype=int)
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = 1.0 / counts[labels]
    generator = torch.Generator()
    generator.manual_seed(seed)
    from torch.utils.data import WeightedRandomSampler

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )


def _metrics(y_true_quantum: Sequence[int], prob_real: Sequence[float]) -> dict[str, Any]:
    from quantum.evaluation import balanced_accuracy, classification_metrics, choose_validation_threshold

    y = np.asarray(y_true_quantum, dtype=int)
    p = np.asarray(prob_real, dtype=float)
    payload = classification_metrics(y, p, decision_threshold=0.5)
    payload["balanced_accuracy_at_0.5"] = balanced_accuracy(y, p, 0.5)
    if len(np.unique(y)) > 1:
        threshold = choose_validation_threshold(y, p, method="youden")
        payload["youden_threshold"] = float(threshold)
        payload["balanced_accuracy_at_youden"] = balanced_accuracy(y, p, threshold)
    return payload


def _evaluate_split(
    records: Sequence[Any],
    model: torch.nn.Module,
    meta: dict[str, Any],
    cfg: FineTuneConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    device = resolve_device(cfg.device)
    results = predict_records(
        records,
        model,
        meta,
        frames=cfg.val_frames,
        min_frames=min(8, cfg.val_frames),
        aggregation="mean",
        batch_videos=cfg.batch_videos,
        device=device,
        return_frames=False,
    )
    usable = [r for r in results if r.get("prob_real") is not None]
    if not usable:
        raise ValueError("Validation produced no usable videos")
    metrics = _metrics(
        [r["y_true_quantum"] for r in usable],
        [r["prob_real"] for r in usable],
    )
    metrics["n_videos"] = len(usable)
    metrics["n_skipped"] = len(results) - len(usable)
    return metrics, results


def _make_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * float(warmup_ratio)))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return 0.1 + 0.9 * (step / max(1, warmup))
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def finetune(
    cfg: FineTuneConfig | None = None,
    *,
    records: Sequence[Any] | None = None,
) -> dict[str, Any]:
    cfg = cfg or FineTuneConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)

    if records is None:
        from vit.dataset import align_video_records, cap_records, load_split_manifest, split_records

        manifest = load_split_manifest(SPLIT_MANIFEST)
        records, stats = align_video_records(
            manifest,
            FRAMES_ROOT,
            RPPG_CSV,
            RPPG_FEATURE_NAMES,
            min_frames=8,
        )
        if cfg.max_train_videos:
            records = cap_records(list(records), cfg.max_train_videos, seed=cfg.seed)
        splits = split_records(records)
        print(f"  aligned records: {stats}")
    else:
        from vit.dataset import split_records

        splits = split_records(list(records))

    train_records = splits["train"]
    val_records = splits["val"]
    if not train_records or not val_records:
        raise ValueError("Fine-tuning requires non-empty train and val splits")

    model, meta = load_detector(cfg.detector, device=device, strict=True)
    meta = {**meta, "finetuned": True}

    train_ds = DfbVideoDataset(
        train_records, meta, cfg, training=True, frames=cfg.frames_per_video
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_videos,
        sampler=_balanced_sampler(train_records, cfg.seed),
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=_collate,
        drop_last=False,
    )

    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
    )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    total_steps = max(1, len(train_loader) * cfg.epochs)
    scheduler = _make_scheduler(optimizer, total_steps, cfg.warmup_ratio)

    DFB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_path = finetuned_checkpoint(cfg.detector)
    best_meta_path = finetuned_metadata(cfg.detector)
    log_path = DFB_OUTPUT_DIR / f"finetune_{cfg.detector}_log.jsonl"

    best_auc = -1.0
    best_epoch = -1
    best_threshold = 0.5
    best_val: dict[str, Any] = {}
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        freeze = epoch <= cfg.freeze_backbone_epochs
        frozen = set_backbone_frozen(model, freeze)
        if freeze:
            model.eval()
            for module in model.modules():
                if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                    module.eval()
        else:
            model.train()
        train_ds.set_epoch(epoch)
        running = 0.0
        seen = 0
        for step, (videos, labels, _indices) in enumerate(train_loader, start=1):
            b, t = videos.shape[:2]
            images = videos.reshape(b * t, *videos.shape[2:]).to(
                device, non_blocking=True
            )
            targets = labels.repeat_interleave(t).to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=cfg.amp and device.type == "cuda",
            ):
                outputs = model({"image": images, "label": targets}, inference=True)
                logits = outputs["cls"]
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            reset_detector_buffers(model)
            running += float(loss.detach()) * int(images.shape[0])
            seen += int(images.shape[0])

        train_loss = running / max(1, seen)
        val_metrics, val_results = _evaluate_split(val_records, model, meta, cfg)
        usable = [r for r in val_results if r.get("prob_real") is not None]
        val_auc = float(val_metrics.get("auc_roc") or 0.0)
        threshold = choose_conservative_threshold(
            [r["y_true_quantum"] for r in usable],
            [r["prob_real"] for r in usable],
        )
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_auc": val_auc,
            "val_balanced_accuracy_conservative": balanced_accuracy(
                [r["y_true_quantum"] for r in usable],
                [r["prob_real"] for r in usable],
                threshold,
            ),
            "val_threshold": threshold,
            "frozen_backbone": freeze,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
        print(
            f"  epoch {epoch:02d}/{cfg.epochs:02d} loss={train_loss:.4f} "
            f"val_auc={val_auc:.4f} thr={threshold:.3f} "
            f"bal_acc={row['val_balanced_accuracy_conservative']:.4f} freeze={freeze}"
        )

        if val_auc > best_auc + 1e-5:
            best_auc = val_auc
            best_epoch = epoch
            best_threshold = threshold
            best_val = val_metrics
            stale = 0
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, best_path)
            payload = {
                "detector": cfg.detector,
                "mode": "finetuned",
                "best_epoch": epoch,
                "val_metrics": val_metrics,
                "decision_threshold": threshold,
                "preprocessing": {
                    "resolution": int(meta["resolution"]),
                    "mean": meta["mean"],
                    "std": meta["std"],
                },
                "label_convention": {"dfb_fake": 1, "dfb_real": 0, "quantum_real": 1},
                "config": asdict(cfg),
                "checkpoint": str(best_path),
            }
            best_meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"  early stop after {stale} stale epochs")
                break

    elapsed = time.perf_counter() - started
    summary = {
        "detector": cfg.detector,
        "mode": "finetuned",
        "best_epoch": best_epoch,
        "best_val_auc": best_auc,
        "best_val_metrics": best_val,
        "decision_threshold": best_threshold,
        "checkpoint": str(best_path),
        "metadata": str(best_meta_path),
        "epochs_run": len(history),
        "elapsed_seconds": round(elapsed, 2),
        "history": history,
    }
    (DFB_OUTPUT_DIR / f"finetune_{cfg.detector}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
