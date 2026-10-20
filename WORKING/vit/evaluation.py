"""Video-level metrics, calibration, plots, and VQC/classical comparison."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from quantum.config import DecisionConfig
from quantum.evaluation import (
    balanced_accuracy,
    choose_validation_threshold,
    classification_metrics,
    decision_bins,
    expected_calibration_error,
)

from vit.config import VIT_OUTPUT_DIR, VitConfig
from vit.dataset import VideoRecord, VitVideoDataset
from vit.model import VitClassifier


def fit_temperature(logits: Sequence[float], labels: Sequence[int]) -> float:
    """Fit one scalar temperature on validation logits (NLL minimisation)."""
    z = torch.tensor(np.asarray(logits, dtype=np.float64), dtype=torch.float64)
    y = torch.tensor(np.asarray(labels, dtype=np.float64), dtype=torch.float64)
    if z.numel() == 0 or len(torch.unique(y)) < 2:
        return 1.0
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature).clamp(0.05, 20.0)
        loss = loss_fn(z / temperature, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).clamp(0.05, 20.0).item())


def logits_to_probabilities(logits: Sequence[float], temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return 1.0 / (1.0 + np.exp(-z / float(temperature)))


def collect_logits(
    model: VitClassifier,
    records: Sequence[VideoRecord],
    cfg: VitConfig,
    device: torch.device,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    from torch.utils.data import DataLoader

    dataset = VitVideoDataset(records, cfg, training=False)
    loader = DataLoader(
        dataset,
        batch_size=batch_size or cfg.batch_size,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
    )
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_indices: list[int] = []
    model.eval()
    with torch.inference_mode():
        for frames, rppg, labels, indices in loader:
            frames = frames.to(device, non_blocking=True)
            rppg = rppg.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda" and cfg.mixed_precision,
            ):
                logits = model(frames, rppg if cfg.use_rppg else None)
            all_logits.append(logits.detach().float().cpu().numpy().reshape(-1))
            all_labels.append(labels.numpy())
            all_indices.extend(int(i) for i in indices)
    return (
        np.concatenate(all_logits) if all_logits else np.asarray([]),
        np.concatenate(all_labels) if all_labels else np.asarray([]),
        all_indices,
    )


def tune_calibration_and_threshold(
    val_logits: Sequence[float],
    val_labels: Sequence[int],
    cfg: VitConfig,
) -> tuple[float, float]:
    if len(np.unique(np.asarray(val_labels))) < 2:
        return 1.0, 0.5
    temperature = fit_temperature(val_logits, val_labels) if cfg.calibrate_temperature else 1.0
    probabilities = logits_to_probabilities(val_logits, temperature)
    threshold = choose_validation_threshold(
        np.asarray(val_labels, dtype=int),
        probabilities,
        method=cfg.threshold_method,
    )
    return temperature, float(threshold)


def evaluate_video_level(
    model: VitClassifier,
    test_records: Sequence[VideoRecord],
    val_records: Sequence[VideoRecord],
    cfg: VitConfig,
    device: torch.device,
    num_workers: int = 0,
) -> dict[str, Any]:
    val_logits, val_labels, _ = collect_logits(
        model, val_records, cfg, device, num_workers=num_workers
    )
    temperature, threshold = tune_calibration_and_threshold(val_logits, val_labels, cfg)
    val_prob = logits_to_probabilities(val_logits, temperature)

    test_logits, test_labels, _ = collect_logits(
        model, test_records, cfg, device, num_workers=num_workers
    )
    test_prob = logits_to_probabilities(test_logits, temperature)
    metrics = classification_metrics(test_labels, test_prob, decision_threshold=threshold)
    metrics["balanced_accuracy"] = balanced_accuracy(
        test_labels, test_prob, decision_threshold=threshold
    )

    payload: dict[str, Any] = {
        "backend": "vit",
        "use_rppg": cfg.use_rppg,
        "backbone": cfg.backbone,
        "temporal_head": cfg.temporal_head,
        "temperature": temperature,
        "decision_threshold": threshold,
        "n_test_videos": len(test_records),
        "n_val_videos": len(val_records),
        "test": metrics,
        "validation": {
            "n": len(val_records),
            "auc_roc": _safe_auc(val_labels, val_prob),
            "balanced_accuracy": balanced_accuracy(
                val_labels, val_prob, decision_threshold=threshold
            ),
            "ece": expected_calibration_error(val_labels, val_prob),
        },
        "decision_bins": decision_bins(
            test_labels,
            test_prob,
            cfg=DecisionConfig(decision_threshold=threshold),
        ),
        "predictions": [
            {
                "video_path": record.video_path,
                "label": int(record.label),
                "prob_real": float(prob),
                "prediction": int(prob >= threshold),
            }
            for record, prob in zip(test_records, test_prob)
        ],
    }
    return payload


def _safe_auc(labels: Sequence[int], probabilities: Sequence[float]) -> float | None:
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(labels, probabilities))


def _plot_roc(labels, probabilities, threshold: float, path: Path) -> None:
    if len(np.unique(labels)) < 2:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(labels, probabilities)
    auc = roc_auc_score(labels, probabilities)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC-ROC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ViT Video-Level ROC")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(labels, probabilities, threshold: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    predictions = (np.asarray(probabilities) >= threshold).astype(int)
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["FAKE", "REAL"]).plot(ax=ax, colorbar=False)
    ax.set_title("ViT Video-Level Confusion Matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_calibration(labels, probabilities, path: Path, n_bins: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, observed = [], []
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (probabilities > low) & (probabilities <= high)
        if mask.any():
            centers.append((low + high) / 2.0)
            observed.append(float(np.asarray(labels)[mask].mean()))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    if centers:
        ax.plot(centers, observed, "o-", label="ViT")
    ax.set_xlabel("Mean predicted P(real)")
    ax.set_ylabel("Observed fraction REAL")
    ax.set_title("ViT Calibration")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_evaluation_artifacts(
    payload: dict[str, Any],
    output_dir: Path,
    prefix: str = "vit",
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"metrics_{prefix}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    labels = np.asarray([p["label"] for p in payload["predictions"]], dtype=int)
    probabilities = np.asarray([p["prob_real"] for p in payload["predictions"]], dtype=float)
    threshold = float(payload["decision_threshold"])
    paths = {
        "metrics": json_path,
        "roc": output_dir / f"{prefix}_roc.png",
        "confusion": output_dir / f"{prefix}_confusion.png",
        "calibration": output_dir / f"{prefix}_calibration.png",
    }
    _plot_roc(labels, probabilities, threshold, paths["roc"])
    _plot_confusion(labels, probabilities, threshold, paths["confusion"])
    _plot_calibration(labels, probabilities, paths["calibration"])
    return paths


def _load_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _extract_metrics(payload: dict | None) -> dict | None:
    if not payload:
        return None
    if "test" in payload and isinstance(payload["test"], dict):
        return payload["test"]
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        return payload["metrics"]
    return payload


def _pick_best_classical(payload: dict | None) -> tuple[str | None, dict | None]:
    if not payload:
        return None, None
    candidates = []
    for name, metrics in payload.items():
        if not isinstance(metrics, dict) or "skipped" in metrics:
            continue
        auc = metrics.get("auc_roc")
        if auc is None:
            auc = metrics.get("balanced_accuracy", 0.0)
        candidates.append((float(auc), name, metrics))
    if not candidates:
        return None, None
    _, name, metrics = max(candidates, key=lambda item: item[0])
    return name, metrics


def build_comparison_table(
    output_dir: Path | None = None,
    quantum_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir or VIT_OUTPUT_DIR)
    quantum_dir = Path(
        quantum_dir
        or Path(output_dir).parent / "quantum"
    )
    vqc_payload = _load_json(quantum_dir / "metrics_quantum.json")
    baseline_payload = _load_json(quantum_dir / "metrics_baselines.json")
    vit_payload = _load_json(output_dir / "metrics_vit.json")
    vit_only_payload = _load_json(output_dir / "metrics_vit_only.json")
    if vit_payload and vit_payload.get("use_rppg") is False:
        vit_only_payload = vit_only_payload or vit_payload
        vit_payload = None

    best_name, best_metrics = _pick_best_classical(baseline_payload)
    rows: dict[str, dict | None] = {
        "VQC": _extract_metrics(vqc_payload),
        f"Best classical ({best_name or 'unavailable'})": best_metrics,
        "ViT-only": _extract_metrics(vit_only_payload),
        "ViT+rPPG": _extract_metrics(vit_payload),
    }
    columns = (
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "specificity",
        "auc_roc",
        "ece",
    )
    lines = [
        "| Model | " + " | ".join(columns) + " |",
        "| --- | " + " | ".join("---:" for _ in columns) + " |",
    ]
    for name, metrics in rows.items():
        cells = []
        for column in columns:
            value = metrics.get(column) if metrics else None
            cells.append("n/a" if value is None else f"{float(value):.4f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    comparison = {
        "table_markdown": table,
        "rows": {
            name: (None if metrics is None else {c: metrics.get(c) for c in columns})
            for name, metrics in rows.items()
        },
        "sources": {
            "vqc": str(quantum_dir / "metrics_quantum.json"),
            "classical": str(quantum_dir / "metrics_baselines.json"),
            "vit": str(output_dir / "metrics_vit.json"),
            "vit_only": str(output_dir / "metrics_vit_only.json"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (output_dir / "comparison.md").write_text(table + "\n", encoding="utf-8")
    return comparison


def format_comparison_table(comparison: dict[str, Any]) -> str:
    return str(comparison.get("table_markdown", "Comparison unavailable"))
