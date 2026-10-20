"""Two-stage ViT training, threshold calibration, and grouped cross-validation."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from quantum.config import LABEL_REAL
from RPPG.features import RPPGFeatures

from vit.config import VitConfig
from vit.dataset import (
    VideoRecord,
    VitVideoDataset,
    make_balanced_sampler,
)
from vit.evaluation import (
    collect_logits,
    evaluate_video_level,
    logits_to_probabilities,
    tune_calibration_and_threshold,
)
from vit.model import (
    VitClassifier,
    build_model,
    build_param_groups,
    cosine_warmup_factor,
    count_trainable_parameters,
    save_checkpoint,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def fit_rppg_scaler(records: Sequence[VideoRecord]) -> StandardScaler:
    if not records:
        raise ValueError("Cannot fit rPPG scaler on an empty training split")
    matrix = np.asarray([record.rppg for record in records], dtype=np.float64)
    scaler = StandardScaler()
    scaler.fit(matrix)
    scaler.scale_[scaler.scale_ == 0] = 1.0
    return scaler


def _criterion(cfg: VitConfig, pos_weight: float | None, device: torch.device):
    weight = (
        torch.tensor([pos_weight], dtype=torch.float32, device=device)
        if pos_weight
        else None
    )
    bce = nn.BCEWithLogitsLoss(pos_weight=weight, reduction="none")
    if cfg.loss == "bce":
        return lambda logits, targets: bce(logits, targets).mean()

    def focal(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = bce(logits, targets)
        probability = torch.sigmoid(logits)
        p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
        return (((1.0 - p_t) ** cfg.focal_gamma) * loss).mean()

    return focal


def _make_loader(
    records: Sequence[VideoRecord],
    cfg: VitConfig,
    training: bool,
    device: torch.device,
    seed: int,
) -> DataLoader:
    dataset = VitVideoDataset(records, cfg, training=training, seed=seed)
    sampler = make_balanced_sampler(records, seed=seed) if training and cfg.balanced_sampler else None
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=bool(training and sampler is None),
        sampler=sampler,
        num_workers=max(0, cfg.num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        persistent_workers=False,
    )


def _validation_auc(
    model: VitClassifier,
    records: Sequence[VideoRecord],
    cfg: VitConfig,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    logits, labels, _ = collect_logits(
        model, records, cfg, device, num_workers=0
    )
    probabilities = logits_to_probabilities(logits, 1.0)
    if len(np.unique(labels)) < 2:
        return float("nan"), logits, labels
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, probabilities)), logits, labels


def _copy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_model(
    train_records: Sequence[VideoRecord],
    val_records: Sequence[VideoRecord],
    cfg: VitConfig,
    checkpoint_path: Path,
    device: torch.device | None = None,
    save: bool = True,
    config_name: str = "vit_config.json",
) -> dict[str, Any]:
    if not train_records or not val_records:
        raise ValueError("Train and validation records must both be non-empty")
    seed_everything(cfg.seed)
    device = device or resolve_device(cfg.device)
    scaler = fit_rppg_scaler(train_records)

    model = build_model(cfg, device)
    model.set_rppg_scaler(scaler.mean_, scaler.scale_)

    labels = np.asarray([record.label for record in train_records], dtype=int)
    positives = int((labels == LABEL_REAL).sum())
    negatives = int((labels != LABEL_REAL).sum())
    pos_weight = None
    if cfg.use_pos_weight and positives > 0:
        pos_weight = float(negatives) / float(positives)

    loss_fn = _criterion(cfg, pos_weight, device)
    use_amp = device.type == "cuda" and cfg.mixed_precision
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    output_dir = Path(checkpoint_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"{Path(checkpoint_path).stem}_training_log.jsonl"
    log_path.write_text("", encoding="utf-8")

    best_state = _copy_state(model)
    best_metric = float("-inf")
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    start = time.perf_counter()

    stages = [("stage_a", cfg.stage_a_epochs), ("stage_b", cfg.stage_b_epochs)]
    for stage_name, epochs in stages:
        if epochs <= 0:
            continue
        if stage_name == "stage_a":
            model.set_backbone_trainable(0 if cfg.freeze_backbone_stage_a else cfg.unfreeze_blocks)
        else:
            model.set_backbone_trainable(cfg.unfreeze_blocks)
        print(
            f"  [{stage_name}] epochs={epochs} "
            f"trainable={count_trainable_parameters(model):,}"
        )
        stale_epochs = 0
        param_groups = build_param_groups(model, cfg)
        if not param_groups:
            raise RuntimeError(f"No trainable parameters in {stage_name}")
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.999))
        loader = _make_loader(train_records, cfg, True, device, cfg.seed)
        steps_per_epoch = math.ceil(len(loader) / cfg.grad_accum_steps)
        total_steps = max(1, steps_per_epoch * epochs)
        warmup_steps = int(round(total_steps * cfg.warmup_ratio))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: cosine_warmup_factor(step, total_steps, warmup_steps),
        )

        for epoch in range(1, epochs + 1):
            model.train()
            loader.dataset.set_epoch(epoch - 1 + (0 if stage_name == "stage_a" else cfg.stage_a_epochs))
            running_loss = 0.0
            seen = 0
            optimizer.zero_grad(set_to_none=True)
            for step, (frames, rppg, targets, _indices) in enumerate(loader):
                frames = frames.to(device, non_blocking=True)
                rppg = rppg.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp,
                ):
                    logits = model(frames, rppg if cfg.use_rppg else None).squeeze(-1)
                    loss = loss_fn(logits, targets) / cfg.grad_accum_steps
                amp_scaler.scale(loss).backward()
                if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(loader):
                    amp_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group["params"]],
                        cfg.grad_clip,
                    )
                    amp_scaler.step(optimizer)
                    amp_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                running_loss += float(loss.item()) * cfg.grad_accum_steps * len(frames)
                seen += len(frames)

            val_auc, val_logits, val_labels = _validation_auc(model, val_records, cfg, device)
            if not np.isfinite(val_auc):
                from quantum.evaluation import balanced_accuracy

                val_auc = balanced_accuracy(
                    val_labels,
                    logits_to_probabilities(val_logits, 1.0),
                )
            epoch_payload = {
                "stage": stage_name,
                "epoch": epoch,
                "train_loss": running_loss / max(1, seen),
                "val_auc": float(val_auc),
                "lr_head": float(max(group["lr"] for group in optimizer.param_groups)),
                "elapsed_seconds": round(time.perf_counter() - start, 3),
            }
            history.append(epoch_payload)
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(epoch_payload) + "\n")
            print(
                f"    epoch {epoch:02d}/{epochs} loss={epoch_payload['train_loss']:.4f} "
                f"val_AUC={val_auc:.4f}"
            )
            if val_auc > best_metric + cfg.min_delta:
                best_metric = val_auc
                best_epoch = len(history)
                best_state = _copy_state(model)
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= cfg.patience:
                    print(f"    early stopping at epoch {epoch} (patience {cfg.patience})")
                    break
        model.load_state_dict(best_state)

    if not history:
        raise ValueError("No training epochs configured; set stage_a_epochs or stage_b_epochs")
    model.load_state_dict(best_state)
    model.eval()
    best_val_auc, val_logits, val_labels = _validation_auc(model, val_records, cfg, device)
    if np.isfinite(best_val_auc):
        best_metric = best_val_auc
    temperature, threshold = tune_calibration_and_threshold(val_logits, val_labels, cfg)
    cfg.temperature = temperature
    cfg.decision_threshold = threshold
    extra = {
        "use_rppg": cfg.use_rppg,
        "rppg_feature_order": list(RPPGFeatures.feature_names()),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "class_counts": {"real": positives, "fake": negatives},
        "best_epoch": best_epoch,
        "best_val_auc": best_metric,
        "history": history,
        "device": str(device),
        "train_videos": len(train_records),
        "val_videos": len(val_records),
    }
    if save:
        save_checkpoint(model, cfg, checkpoint_path, extra=extra)
        cfg.save_json(output_dir / config_name)
        print(f"  saved checkpoint: {checkpoint_path}")
    return {
        "model": model,
        "cfg": cfg,
        "scaler": scaler,
        "temperature": temperature,
        "threshold": threshold,
        "best_val_auc": best_metric,
        "best_epoch": best_epoch,
        "history": history,
        "checkpoint": str(checkpoint_path),
    }


def _grouped_val_split(
    train_records: Sequence[VideoRecord],
    cfg: VitConfig,
) -> tuple[list[VideoRecord], list[VideoRecord]]:
    groups = sorted({record.group for record in train_records})
    if len(groups) < 2:
        raise ValueError("Need at least two subject groups for cross-validation")
    rng = random.Random(cfg.seed)
    rng.shuffle(groups)
    n_val = max(1, int(round(len(groups) * 0.15)))
    val_groups = set(groups[:n_val])
    fit = [record for record in train_records if record.group not in val_groups]
    val = [record for record in train_records if record.group in val_groups]
    if not fit or not val:
        raise ValueError("Grouped validation split produced an empty partition")
    return fit, val


def cross_validate(
    records: Sequence[VideoRecord],
    cfg: VitConfig,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """5-fold grouped-by-video CV; train/val/test never share a video."""
    from sklearn.model_selection import StratifiedGroupKFold

    if len(records) < cfg.cv_folds:
        raise ValueError(f"Need at least {cfg.cv_folds} videos for cross-validation")
    device = device or resolve_device(cfg.device)
    labels = np.asarray([record.label for record in records], dtype=int)
    groups = np.asarray([record.group for record in records], dtype=object)
    splitter = StratifiedGroupKFold(
        n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.seed
    )
    folds: list[dict[str, Any]] = []
    metric_keys = ("accuracy", "balanced_accuracy", "f1", "auc_roc", "specificity", "ece")
    for fold, (train_idx, test_idx) in enumerate(splitter.split(np.zeros(len(records)), labels, groups)):
        train_all = [records[i] for i in train_idx]
        test_records = [records[i] for i in test_idx]
        fit_records, val_records = _grouped_val_split(train_all, cfg)
        fold_cfg = replace(cfg, seed=cfg.seed + fold)
        result = train_model(
            fit_records,
            val_records,
            fold_cfg,
            Path(cfg.output_dir) / f"cv_fold_{fold}.pt",
            device=device,
            save=False,
        )
        payload = evaluate_video_level(
            result["model"], test_records, val_records, fold_cfg, device, num_workers=0
        )
        metrics = payload["test"]
        folds.append({key: float(metrics[key]) for key in metric_keys if metrics.get(key) is not None})
        print(
            f"  fold {fold + 1}/{cfg.cv_folds}: "
            f"AUC={metrics.get('auc_roc')} balanced_acc={metrics.get('balanced_accuracy'):.4f}"
        )
    mean = {key: float(np.mean([fold[key] for fold in folds if key in fold])) for key in metric_keys}
    std = {key: float(np.std([fold[key] for fold in folds if key in fold])) for key in metric_keys}
    return {"mean": mean, "std": std, "folds": folds, "n_folds": cfg.cv_folds}
