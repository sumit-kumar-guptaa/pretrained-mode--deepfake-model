"""ViT training/evaluation CLI and reusable video inference entry point."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from quantum.config import LABEL_REAL, RPPG_FEATURE_NAMES
from RPPG.features import RPPGFeatures

from vit.config import (
    ABLATION_CHECKPOINT,
    DEFAULT_CHECKPOINT,
    FRAMES_ROOT,
    RPPG_CSV,
    SPLIT_MANIFEST,
    VitConfig,
)
from vit.dataset import (
    VideoRecord,
    _rppg_row_for,
    _rppg_vector,
    align_video_records,
    cap_records,
    find_frames_dir,
    load_rppg_rows,
    load_split_manifest,
    prepare_inference_frames,
    split_records,
)
from vit.evaluation import (
    build_comparison_table,
    evaluate_video_level,
    format_comparison_table,
    save_evaluation_artifacts,
)
from vit.model import load_checkpoint
from vit.train import cross_validate, resolve_device, seed_everything, train_model


def _feature_names() -> list[str]:
    names = list(RPPGFeatures.feature_names())
    if names != list(RPPG_FEATURE_NAMES):
        raise AssertionError(
            "rPPG feature order changed: RPPGFeatures.feature_names() no longer "
            "matches quantum.config.RPPG_FEATURE_NAMES"
        )
    return names


def prepare_records(
    cfg: VitConfig,
    manifest_path: Path | None = None,
    frames_root: Path | None = None,
    rppg_csv: Path | None = None,
) -> tuple[dict[str, list[VideoRecord]], dict[str, Any]]:
    feature_names = _feature_names()
    manifest = load_split_manifest(Path(manifest_path or SPLIT_MANIFEST))
    records, stats = align_video_records(
        manifest,
        frames_root=Path(frames_root or FRAMES_ROOT),
        rppg_csv=Path(rppg_csv or RPPG_CSV),
        feature_names=feature_names,
        min_frames=cfg.min_frames,
    )
    capped = cap_records(records, cfg.max_videos, seed=cfg.seed)
    if len(capped) != len(records):
        stats["capped_from"] = len(records)
        stats["aligned"] = len(capped)
    splits = split_records(capped)
    stats["splits"] = {
        split: {
            "videos": len(items),
            "real": sum(item.label == LABEL_REAL for item in items),
            "fake": sum(item.label != LABEL_REAL for item in items),
        }
        for split, items in splits.items()
    }
    return splits, stats


def _predict_tensor(
    model,
    cfg: VitConfig,
    frames: torch.Tensor,
    rppg: Sequence[float] | None,
    device: torch.device,
) -> float:
    frames = frames.unsqueeze(0).to(device)
    rppg_tensor = None
    if cfg.use_rppg:
        if rppg is None:
            raise ValueError("This ViT checkpoint requires rPPG features")
        vector = np.asarray(rppg, dtype=np.float32)
        if vector.shape != (20,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"Expected 20 finite rPPG features, got {vector.shape}")
        rppg_tensor = torch.from_numpy(vector).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=device.type == "cuda" and cfg.mixed_precision,
    ):
        logit = model(frames, rppg_tensor).reshape(-1)[0]
    probability = torch.sigmoid(logit.float() / max(1e-6, float(cfg.temperature))).item()
    return float(probability)


def _inconclusive(reason: str, checkpoint: str | None = None) -> dict[str, Any]:
    return {
        "backend": "vit",
        "prob_real": None,
        "verdict": "INCONCLUSIVE",
        "label": "INCONCLUSIVE",
        "confidence": None,
        "reason": reason,
        "checkpoint": checkpoint,
    }


def predict_video(
    video_path: str | Path,
    frames_dir: str | Path | None = None,
    rppg_features: Sequence[float] | None = None,
    checkpoint_path: str | Path | None = None,
    cfg: VitConfig | None = None,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Run ViT inference from accepted frames plus rPPG features.

    ``rppg_features`` may be omitted for offline dataset videos that are present
    in the training rPPG CSV; arbitrary uploads must supply the vector.
    """
    if checkpoint_path is not None:
        checkpoint = Path(checkpoint_path)
    elif cfg is not None:
        checkpoint = Path(cfg.output_dir) / DEFAULT_CHECKPOINT.name
    else:
        checkpoint = DEFAULT_CHECKPOINT
    torch_device = resolve_device(str(device))
    try:
        model, saved_cfg, _payload = load_checkpoint(checkpoint, torch_device)
    except (OSError, RuntimeError, ValueError) as exc:
        return _inconclusive(f"ViT checkpoint unavailable: {exc}", str(checkpoint))
    cfg = saved_cfg

    resolved_frames = Path(frames_dir) if frames_dir else find_frames_dir(FRAMES_ROOT, str(video_path))
    if resolved_frames is None:
        return _inconclusive(
            f"No accepted stage-1 frames found for {Path(video_path).name}", str(checkpoint)
        )

    if cfg.use_rppg and rppg_features is None:
        try:
            by_path, by_stem = load_rppg_rows(RPPG_CSV)
            row = _rppg_row_for(str(video_path), by_path, by_stem)
            if row is not None:
                rppg_features = _rppg_vector(row, _feature_names())
        except (OSError, ValueError):
            rppg_features = None
        if rppg_features is None:
            return _inconclusive(
                "rPPG features unavailable for this video; cannot run fused ViT",
                str(checkpoint),
            )

    try:
        frames, frame_paths = prepare_inference_frames(
            resolved_frames, cfg.frames_per_video, cfg.image_size, torch_device
        )
    except (OSError, ValueError) as exc:
        return _inconclusive(str(exc), str(checkpoint))

    try:
        probability = _predict_tensor(model, cfg, frames, rppg_features, torch_device)
    except (OSError, RuntimeError, ValueError) as exc:
        return _inconclusive(f"ViT inference failed: {exc}", str(checkpoint))

    verdict = "REAL" if probability >= cfg.decision_threshold else "FAKE"
    confidence = abs(probability - 0.5) * 2.0
    return {
        "backend": "vit",
        "prob_real": probability,
        "verdict": verdict,
        "label": verdict,
        "confidence": confidence,
        "reason": None,
        "decision_threshold": cfg.decision_threshold,
        "temperature": cfg.temperature,
        "use_rppg": cfg.use_rppg,
        "backbone": cfg.backbone,
        "temporal_head": cfg.temporal_head,
        "checkpoint": str(checkpoint),
        "frames_used": len(frame_paths),
        "frame_paths": frame_paths,
    }


def _train_one(
    cfg: VitConfig,
    splits: dict[str, list[VideoRecord]],
    checkpoint: Path,
    metrics_prefix: str,
    device: torch.device,
    config_name: str = "vit_config.json",
) -> dict[str, Any]:
    train_records = splits["train"]
    val_records = splits["val"]
    test_records = splits.get("test", [])
    if not test_records:
        print("  [warn] test split is empty; training metrics will be validation-only")
    print(
        f"  train={len(train_records)} val={len(val_records)} test={len(test_records)} "
        f"backbone={cfg.backbone} rppg={cfg.use_rppg}"
    )
    cfg.save_json(cfg.output_dir / config_name)
    result = train_model(
        train_records,
        val_records,
        cfg,
        checkpoint,
        device=device,
        save=True,
        config_name=config_name,
    )
    model = result["model"]
    payload: dict[str, Any] = {
        "training": {
            "best_epoch": result["best_epoch"],
            "best_val_auc": result["best_val_auc"],
            "temperature": result["temperature"],
            "threshold": result["threshold"],
            "history": result["history"],
        },
        "checkpoint": str(checkpoint),
    }
    if test_records:
        evaluation = evaluate_video_level(
            model, test_records, val_records, cfg, device, num_workers=0
        )
        evaluation["training"] = payload["training"]
        paths = save_evaluation_artifacts(evaluation, cfg.output_dir, prefix=metrics_prefix)
        _print_metrics(evaluation["test"], metrics_prefix)
        payload.update({"test": evaluation["test"], "artifacts": {k: str(v) for k, v in paths.items()}})
    return payload


def _print_metrics(metrics: dict, prefix: str) -> None:
    print(
        f"  [{prefix}] acc={metrics.get('accuracy')} balanced={metrics.get('balanced_accuracy')} "
        f"auc={metrics.get('auc_roc')} f1={metrics.get('f1')} ece={metrics.get('ece')}"
    )


def _evaluate_checkpoint(
    splits: dict[str, list[VideoRecord]],
    checkpoint: Path,
    prefix: str,
    device: torch.device,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    model, saved_cfg, _payload = load_checkpoint(checkpoint, device)
    if not splits.get("test"):
        raise ValueError("Test split is empty; cannot evaluate held-out metrics")
    saved_cfg.output_dir = Path(output_dir or checkpoint.parent)
    evaluation = evaluate_video_level(
        model, splits["test"], splits["val"], saved_cfg, device, num_workers=0
    )
    save_evaluation_artifacts(evaluation, saved_cfg.output_dir, prefix=prefix)
    _print_metrics(evaluation["test"], prefix)
    return evaluation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ViT/rPPG video-level deepfake classifier (DFDC only)"
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--train", action="store_true", help="Train the configured ViT")
    action.add_argument("--evaluate", action="store_true", help="Evaluate a saved checkpoint")
    action.add_argument("--all", action="store_true", help="Train, evaluate, and compare")
    action.add_argument(
        "--cross-validate",
        action="store_true",
        help="Run grouped 5-fold cross-validation (no held-out reuse)",
    )

    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--temporal-head", choices=("attention_pool", "transformer", "mean_logits"), default=None)
    parser.add_argument("--no-rppg", action="store_true", help="Disable the rPPG fusion branch")
    parser.add_argument("--max-videos", type=int, default=None, help="Deterministic smoke subset")
    parser.add_argument("--frames-per-video", type=int, default=None)
    parser.add_argument("--min-frames", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--stage-a-epochs", type=int, default=None)
    parser.add_argument("--stage-b-epochs", type=int, default=None)
    parser.add_argument("--cv-folds", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--frames-root", type=Path, default=None)
    parser.add_argument("--rppg-csv", type=Path, default=None)
    parser.add_argument("--no-pretrained", action="store_true", help="Do not load timm ImageNet weights")
    return parser


def _config_from_args(args: argparse.Namespace, base: VitConfig | None = None) -> VitConfig:
    cfg = base or VitConfig()
    updates: dict[str, Any] = {}
    for name in (
        "backbone",
        "temporal_head",
        "max_videos",
        "frames_per_video",
        "min_frames",
        "image_size",
        "batch_size",
        "stage_a_epochs",
        "stage_b_epochs",
        "cv_folds",
        "num_workers",
        "device",
        "output_dir",
    ):
        value = getattr(args, name, None)
        if value is not None:
            updates[name] = value
    if args.no_rppg:
        updates["use_rppg"] = False
    if args.no_pretrained:
        updates["pretrained"] = False
    if "frames_per_video" in updates and args.min_frames is None and base is None:
        updates["min_frames"] = updates["frames_per_video"]
    return replace(cfg, **updates)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = _config_from_args(args)
    checkpoint = args.checkpoint or (cfg.output_dir / DEFAULT_CHECKPOINT.name)
    device = resolve_device(cfg.device)
    seed_everything(cfg.seed)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate:
        model_cfg = load_checkpoint(checkpoint, device)[1]
        model_cfg.output_dir = args.output_dir or checkpoint.parent
        splits, stats = prepare_records(model_cfg, args.manifest, args.frames_root, args.rppg_csv)
        print(json.dumps(stats, indent=2))
        payload = _evaluate_checkpoint(
            splits, checkpoint, "vit", device, output_dir=model_cfg.output_dir
        )
        comparison = build_comparison_table(model_cfg.output_dir)
        print(format_comparison_table(comparison))
        return 0 if payload else 1

    if args.cross_validate:
        splits, stats = prepare_records(cfg, args.manifest, args.frames_root, args.rppg_csv)
        print(json.dumps(stats, indent=2))
        cfg.save_json(cfg.output_dir / "vit_config.json")
        all_records = splits["train"] + splits["val"] + splits["test"]
        print(f"Cross-validating {len(all_records)} videos with {cfg.cv_folds} grouped folds")
        payload = cross_validate(all_records, cfg, device)
        (cfg.output_dir / "cv_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload["mean"], indent=2))
        return 0

    splits, stats = prepare_records(cfg, args.manifest, args.frames_root, args.rppg_csv)
    print(json.dumps(stats, indent=2))
    started = time.perf_counter()

    if args.all:
        main_cfg = cfg
        main_result = _train_one(main_cfg, splits, checkpoint, "vit", device)
        if main_cfg.use_rppg:
            only_cfg = replace(
                cfg,
                use_rppg=False,
                temporal_head="mean_logits",
                output_dir=cfg.output_dir,
            )
            only_checkpoint = cfg.output_dir / ABLATION_CHECKPOINT.name
            only_result = _train_one(
                only_cfg,
                splits,
                only_checkpoint,
                "vit_only",
                device,
                config_name="vit_only_config.json",
            )
        else:
            only_result = main_result
        comparison = build_comparison_table(cfg.output_dir)
        (cfg.output_dir / "training_summary.json").write_text(
            json.dumps(
                {
                    "vit": main_result,
                    "vit_only": only_result,
                    "comparison": comparison,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(format_comparison_table(comparison))
        return 0

    result = _train_one(cfg, splits, checkpoint, "vit", device)
    comparison = build_comparison_table(cfg.output_dir)
    print(format_comparison_table(comparison))
    (cfg.output_dir / "training_summary.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(2)
