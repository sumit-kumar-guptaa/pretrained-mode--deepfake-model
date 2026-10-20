"""Evaluation harness for the DeepfakeBench image backend.

Usage (from WORKING/)::

    python -m dfb_backend.evaluation --zero-shot
    python -m dfb_backend.evaluation --finetune --detectors xception
    python -m dfb_backend.evaluation --fuse --detector xception

Video-level thresholds are selected on the validation split only; the test
split is scored once with the selected threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dfb_backend.config import (
    BASELINE_METRICS,
    DEFAULT_AGGREGATION,
    DEFAULT_DETECTOR,
    DEFAULT_FRAMES,
    DEFAULT_TOPK_RATIO,
    DFB_OUTPUT_DIR,
    FRAMES_ROOT,
    QUANTUM_METRICS,
    RPPG_CSV,
    SPLIT_MANIFEST,
    finetuned_checkpoint,
    finetuned_metadata,
    metrics_path,
    scores_path,
)
from dfb_backend.fusion import choose_conservative_threshold, fit_fusion
from dfb_backend.loader import load_detector, resolve_device
from dfb_backend.predict import aggregate_frame_probs, predict_records
from quantum.config import RPPG_FEATURE_NAMES
from quantum.evaluation import balanced_accuracy, classification_metrics


def _load_records(max_videos: int | None = None):
    from vit.dataset import align_video_records, cap_records, load_split_manifest, split_records

    manifest = load_split_manifest(SPLIT_MANIFEST)
    records, stats = align_video_records(
        manifest,
        FRAMES_ROOT,
        RPPG_CSV,
        RPPG_FEATURE_NAMES,
        min_frames=8,
    )
    if max_videos:
        records = cap_records(list(records), max_videos, seed=42)
    splits = split_records(records)
    counts = {
        split: {
            "videos": len(items),
            "real": sum(int(r.label) == 1 for r in items),
            "fake": sum(int(r.label) == 0 for r in items),
        }
        for split, items in splits.items()
    }
    return records, splits, counts, stats


def _eer(y_true: np.ndarray, prob_real: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, prob_real)
    fnr = 1.0 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def _metrics(y_true: Sequence[int], prob_real: Sequence[float], threshold: float) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_real, dtype=float)
    payload = classification_metrics(y, p, decision_threshold=threshold)
    payload["balanced_accuracy"] = balanced_accuracy(y, p, threshold)
    payload["eer"] = _eer(y, p)
    payload["threshold_used"] = float(threshold)
    payload["n_videos"] = int(len(y))
    return payload


def _frame_auc(records: Sequence[dict[str, Any]]) -> float | None:
    y: list[int] = []
    p: list[float] = []
    for record in records:
        frame_probs = record.get("frame_fake_probs")
        if not frame_probs:
            continue
        y.extend([int(record["y_true_fake"])] * len(frame_probs))
        p.extend(float(v) for v in frame_probs)
    if len(set(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, p))


def _plot_curves(y_true, prob_real, threshold: float, detector: str, mode: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, roc_curve

    out = DFB_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_real, dtype=float)

    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p)
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y, p)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"AUC-ROC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"DFB {detector} ({mode}) ROC")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"roc_{detector}_{mode}.png", dpi=150)
        plt.close(fig)

    pred = (p >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, labels=[0, 1], ax=ax, colorbar=False
    )
    ax.set_title(f"DFB {detector} ({mode}) confusion")
    fig.tight_layout()
    fig.savefig(out / f"confusion_{detector}_{mode}.png", dpi=150)
    plt.close(fig)

    bins = np.linspace(0.0, 1.0, 11)
    centers, observed = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p > lo) & (p <= hi)
        if mask.any():
            centers.append((lo + hi) / 2.0)
            observed.append(float(y[mask].mean()))
    fig, ax = plt.subplots(figsize=(5, 5))
    if centers:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        ax.plot(centers, observed, "o-", label=f"DFB {detector}")
    ax.set_xlabel("Mean predicted P(real)")
    ax.set_ylabel("Observed fraction REAL")
    ax.set_title(f"DFB {detector} ({mode}) calibration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / f"calibration_{detector}_{mode}.png", dpi=150)
    plt.close(fig)


def evaluate_detector(
    detector: str,
    mode: str,
    *,
    frames: int = DEFAULT_FRAMES,
    aggregation: str = DEFAULT_AGGREGATION,
    topk_ratio: float = DEFAULT_TOPK_RATIO,
    max_videos: int | None = None,
    batch_videos: int = 4,
    device: str = "auto",
) -> dict[str, Any]:
    all_records, splits, counts, stats = _load_records(max_videos)
    checkpoint = finetuned_checkpoint(detector) if mode == "finetuned" else None
    if mode == "finetuned" and (checkpoint is None or not checkpoint.is_file()):
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {checkpoint}. Run --finetune first."
        )
    model, meta = load_detector(
        detector, device=device, checkpoint=checkpoint, strict=True
    )
    meta = {**meta, "mode": mode}
    if mode == "finetuned":
        sidecar = finetuned_metadata(detector)
        if sidecar.is_file():
            meta["finetuned_metadata"] = json.loads(sidecar.read_text(encoding="utf-8"))

    print(
        f"  {detector}/{mode}: {len(all_records)} videos "
        f"{counts} | frames={frames} aggregation={aggregation}"
    )
    results = predict_records(
        all_records,
        model,
        meta,
        frames=frames,
        min_frames=min(8, frames),
        aggregation=aggregation,
        topk_ratio=topk_ratio,
        batch_videos=batch_videos,
        device=device,
        return_frames=True,
    )
    usable = [r for r in results if r.get("prob_real") is not None]
    skipped = len(results) - len(usable)
    if skipped:
        print(f"  skipped {skipped} videos with too few frames")

    by_split = {
        split: [r for r in usable if r["split"] == split]
        for split in ("train", "val", "test")
    }
    for split, items in by_split.items():
        if not items:
            raise ValueError(f"{detector}/{mode}: no usable {split} videos")

    y_val = [r["y_true_quantum"] for r in by_split["val"]]
    p_val = [r["prob_real"] for r in by_split["val"]]
    threshold = choose_conservative_threshold(y_val, p_val)

    metrics = {"splits": {}, "threshold": float(threshold)}
    for split, items in by_split.items():
        metrics["splits"][split] = _metrics(
            [r["y_true_quantum"] for r in items],
            [r["prob_real"] for r in items],
            threshold,
        )
        metrics["splits"][split]["at_0.5"] = _metrics(
            [r["y_true_quantum"] for r in items],
            [r["prob_real"] for r in items],
            0.5,
        )["balanced_accuracy"]
    metrics["primary_test"] = metrics["splits"]["test"]
    metrics["frame_auc_test"] = _frame_auc(by_split["test"])
    metrics.update(
        {
            "backend": "dfb",
            "detector": detector,
            "mode": mode,
            "frames_per_video": int(frames),
            "aggregation": aggregation,
            "topk_ratio": float(topk_ratio),
            "counts": counts,
            "alignment": stats,
            "skipped_videos": skipped,
            "meta": {k: v for k, v in meta.items() if k != "finetuned_metadata"},
        }
    )
    if "finetuned_metadata" in meta:
        metrics["finetuned_metadata"] = meta["finetuned_metadata"]

    DFB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_payload = {
        "detector": detector,
        "mode": mode,
        "frames_per_video": int(frames),
        "aggregation": aggregation,
        "topk_ratio": float(topk_ratio),
        "threshold": float(threshold),
        "records": [
            {
                "video_path": r["video_path"],
                "video_key": r["video_key"],
                "group": r["group"],
                "split": r["split"],
                "prob_real": r["prob_real"],
                "y_true_quantum": r["y_true_quantum"],
                "y_true_fake": r["y_true_fake"],
                "frames_used": r["frames_used"],
                "rppg": r["rppg"],
                "frame_fake_probs": r.get("frame_fake_probs"),
            }
            for r in usable
        ],
    }
    scores_path(detector, mode).write_text(
        json.dumps(scores_payload, indent=2), encoding="utf-8"
    )
    metrics_path(detector, mode).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    _plot_curves(
        [r["y_true_quantum"] for r in by_split["test"]],
        [r["prob_real"] for r in by_split["test"]],
        threshold,
        detector,
        mode,
    )
    print(
        f"  test: AUC={metrics['primary_test']['auc_roc']:.4f} "
        f"bal_acc={metrics['primary_test']['balanced_accuracy']:.4f} "
        f"thr={threshold:.3f} frame_AUC={metrics['frame_auc_test']}"
    )
    return metrics


def _metric_row(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = payload.get("primary_test") or payload.get("metrics") or payload
    return {
        "auc": metrics.get("auc_roc"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "eer": metrics.get("eer"),
        "ece": metrics.get("ece"),
        "brier": metrics.get("brier_score"),
        "accuracy": metrics.get("accuracy"),
        "n": metrics.get("n_videos") or metrics.get("n_total"),
    }


def _quantum_row() -> dict[str, Any] | None:
    if not QUANTUM_METRICS.is_file():
        return None
    payload = json.loads(QUANTUM_METRICS.read_text(encoding="utf-8"))
    metrics = payload.get("metrics") or payload
    return {
        "method": "Hybrid VQC (fused)",
        "auc": metrics.get("auc_roc"),
        "balanced_accuracy": payload.get("balanced_accuracy")
        or metrics.get("balanced_accuracy"),
        "eer": metrics.get("eer"),
        "ece": metrics.get("ece"),
        "brier": metrics.get("brier_score"),
        "accuracy": metrics.get("accuracy"),
        "n": metrics.get("n_total"),
    }


def _baseline_rows() -> list[dict[str, Any]]:
    if not BASELINE_METRICS.is_file():
        return []
    payload = json.loads(BASELINE_METRICS.read_text(encoding="utf-8"))
    rows = []
    for name, metrics in payload.items():
        if not isinstance(metrics, dict) or "skipped" in metrics:
            continue
        rows.append(
            {
                "method": f"classical:{name}",
                "auc": metrics.get("auc_roc"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "eer": metrics.get("eer"),
                "ece": metrics.get("ece"),
                "brier": metrics.get("brier_score"),
                "accuracy": metrics.get("accuracy"),
                "n": metrics.get("n_total"),
            }
        )
    return rows


def write_comparison() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quantum = _quantum_row()
    if quantum:
        rows.append(quantum)
    rows.extend(_baseline_rows())
    for path in sorted(DFB_OUTPUT_DIR.glob("metrics_*_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("backend") != "dfb":
            continue
        row = _metric_row(payload)
        row["method"] = f"dfb:{payload['detector']}:{payload['mode']}"
        rows.append(row)

    def fmt(value, digits=4):
        return "" if value is None else f"{float(value):.{digits}f}"

    lines = [
        "| Method | AUC-ROC | Balanced Acc | EER | ECE | Brier | n |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {fmt(row.get('auc'))} | "
            f"{fmt(row.get('balanced_accuracy'))} | {fmt(row.get('eer'))} | "
            f"{fmt(row.get('ece'))} | {fmt(row.get('brier'))} | "
            f"{row.get('n') or ''} |"
        )
    table = "\n".join(lines)
    (DFB_OUTPUT_DIR / "comparison.md").write_text(table + "\n", encoding="utf-8")
    (DFB_OUTPUT_DIR / "comparison.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DeepfakeBench image-backend evaluation")
    parser.add_argument("--zero-shot", action="store_true", help="Evaluate upstream checkpoints")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune on DFDC train split")
    parser.add_argument("--fuse", action="store_true", help="Fit image + rPPG late fusion")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR)
    parser.add_argument(
        "--detectors",
        nargs="+",
        default=None,
        help="Detectors for --zero-shot/--finetune (default: all zero-shot, --detector for finetune)",
    )
    parser.add_argument("--mode", default="finetuned", choices=["zero_shot", "finetuned"])
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument(
        "--aggregation", default="mean", choices=["mean", "median", "topk"]
    )
    parser.add_argument("--topk", type=float, default=0.2)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--batch-videos", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-frames", type=int, default=8)
    parser.add_argument("--freeze-epochs", type=int, default=1)
    parser.add_argument("--comparison", action="store_true", help="Write comparison table")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    DFB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.zero_shot:
        detectors = args.detectors or ["xception", "ucf", "efficientnetb4"]
        for detector in detectors:
            print(f"[zero-shot] {detector}")
            evaluate_detector(
                detector,
                "zero_shot",
                frames=args.frames,
                aggregation=args.aggregation,
                topk_ratio=args.topk,
                max_videos=args.max_videos,
                batch_videos=args.batch_videos,
                device=args.device,
            )

    if args.finetune:
        from dfb_backend.train import FineTuneConfig, finetune

        detectors = args.detectors or [args.detector]
        for detector in detectors:
            print(f"[finetune] {detector}")
            summary = finetune(
                FineTuneConfig(
                    detector=detector,
                    epochs=args.epochs,
                    lr=args.lr,
                    frames_per_video=args.train_frames,
                    freeze_backbone_epochs=args.freeze_epochs,
                    max_train_videos=None,
                    device=args.device,
                )
            )
            print(
                f"  best epoch {summary['best_epoch']} val AUC "
                f"{summary['best_val_auc']:.4f} -> {summary['checkpoint']}"
            )
            evaluate_detector(
                detector,
                "finetuned",
                frames=args.frames,
                aggregation=args.aggregation,
                topk_ratio=args.topk,
                max_videos=args.max_videos,
                batch_videos=args.batch_videos,
                device=args.device,
            )

    if args.fuse:
        detectors = args.detectors or [args.detector]
        for detector in detectors:
            print(f"[fusion] {detector}/{args.mode}")
            payload = fit_fusion(detector, args.mode)
            test = payload["metrics"]["fusion_test"]
            image = payload["metrics"]["image_only_test"]
            print(
                f"  image-only AUC={image['auc_roc']:.4f} -> "
                f"image+rPPG AUC={test['auc_roc']:.4f} "
                f"(thr={payload['threshold']:.3f}, T={payload['temperature']:.3f})"
            )

    if args.comparison or args.zero_shot or args.finetune:
        rows = write_comparison()
        print("\n" + (DFB_OUTPUT_DIR / "comparison.md").read_text(encoding="utf-8"))
        print(f"rows={len(rows)} -> {DFB_OUTPUT_DIR / 'comparison.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
