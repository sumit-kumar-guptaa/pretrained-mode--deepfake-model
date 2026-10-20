"""Late fusion of DFB image evidence with the 20 rPPG features."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dfb_backend.config import (
    DFB_OUTPUT_DIR,
    DEFAULT_DETECTOR,
    fusion_artifact_path,
    scores_path,
)
from quantum.config import RPPG_FEATURE_NAMES
from quantum.evaluation import balanced_accuracy, classification_metrics, choose_validation_threshold

_EPS = 1e-6
MAX_FAKE_FALSE_ACCEPTANCE = 0.01


def _logit(prob: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(prob, dtype=np.float64), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(value, dtype=np.float64)))


def fit_temperature(prob_real: Sequence[float], y_true: Sequence[int]) -> float:
    """Fit a scalar temperature on validation data by minimising NLL."""
    p = np.clip(np.asarray(prob_real, dtype=np.float64), _EPS, 1.0 - _EPS)
    y = np.asarray(y_true, dtype=np.float64)
    logits = _logit(p)

    def nll(temperature: float) -> float:
        if temperature <= 0:
            return 1e12
        q = np.clip(_sigmoid(logits / temperature), _EPS, 1.0 - _EPS)
        return float(-np.mean(y * np.log(q) + (1.0 - y) * np.log(1.0 - q)))

    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
        return float(result.x) if result.success else 1.0
    except ImportError:
        grid = np.geomspace(0.05, 10.0, 80)
        return float(grid[int(np.argmin([nll(t) for t in grid]))])


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


def choose_conservative_threshold(
    y_true: Sequence[int],
    prob_real: Sequence[float],
    *,
    max_fake_false_acceptance: float = MAX_FAKE_FALSE_ACCEPTANCE,
) -> float:
    """Choose a REAL cutoff that caps fake-video acceptance on validation data.

    Youden's threshold is useful for balanced offline scoring, but it can be
    unsafe for KYC when false REAL decisions are the costly error. Among
    thresholds meeting the fake-acceptance target, retain the best balanced
    accuracy and prefer the less conservative cutoff on ties.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(prob_real, dtype=float)
    target = float(max_fake_false_acceptance)
    if len(y) != len(p) or len(y) == 0 or len(np.unique(y)) < 2:
        raise ValueError("Threshold selection requires paired validation data with both classes")
    if not 0.0 <= target <= 1.0:
        raise ValueError("max_fake_false_acceptance must be within [0, 1]")
    if not np.isfinite(p).all() or ((p < 0.0) | (p > 1.0)).any():
        raise ValueError("Validation probabilities must be finite and in [0, 1]")

    candidates = np.unique(np.r_[0.0, p, 1.0])
    eligible: list[tuple[float, float, float]] = []
    for threshold in candidates:
        predicted_real = p >= threshold
        fake = y == 0
        real = y == 1
        fake_acceptance = float(predicted_real[fake].mean())
        real_recall = float(predicted_real[real].mean())
        balanced = (real_recall + (1.0 - fake_acceptance)) / 2.0
        if fake_acceptance <= target:
            eligible.append((balanced, -float(threshold), float(threshold)))
    if not eligible:
        return 1.0
    return max(eligible)[2]


def _load_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing DFB score artifact: {path}. Run evaluation --zero-shot or --finetune first."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records") or []
    if not records:
        raise ValueError(f"No video records in {path}")
    return records


def _split_records(records: Sequence[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("split") == split and record.get("prob_real") is not None
    ]


def _matrix(records: Sequence[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = []
    rppg = []
    labels = []
    for record in records:
        values = record.get("rppg")
        if values is None or len(values) != len(RPPG_FEATURE_NAMES):
            raise ValueError(
                f"Score record for {record.get('video_path')} has rPPG length "
                f"{None if values is None else len(values)}; expected {len(RPPG_FEATURE_NAMES)}"
            )
        vector = np.asarray(values, dtype=np.float64)
        if not np.isfinite(vector).all():
            raise ValueError(f"Non-finite rPPG features for {record.get('video_path')}")
        image.append(float(record["prob_real"]))
        rppg.append(vector)
        labels.append(int(record["y_true_quantum"]))
    return (
        np.asarray(image, dtype=np.float64),
        np.asarray(rppg, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
    )


def _standardiser(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-9] = 1.0
    return mean, scale


def _transform(matrix: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (matrix - mean) / scale


def _design(image_prob: np.ndarray, rppg: np.ndarray, image_mean, image_scale, rppg_mean, rppg_scale) -> np.ndarray:
    z_image = (_logit(image_prob) - image_mean) / image_scale
    z_rppg = _transform(rppg, rppg_mean, rppg_scale)
    return np.column_stack([z_image, z_rppg])


def _fit_logistic(X: np.ndarray, y: np.ndarray, C: float):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(
        C=float(C),
        class_weight="balanced",
        max_iter=4000,
        solver="lbfgs",
    )
    model.fit(X, y)
    return model


def fit_fusion(
    detector: str = DEFAULT_DETECTOR,
    mode: str = "finetuned",
    *,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Fit image + rPPG late fusion on train, tune on val, report on test."""
    paths = {split: scores_path(detector, mode) for split in ("train", "val", "test")}
    per_split = {split: _load_records(path) for split, path in paths.items()}
    data = {}
    for split, records in per_split.items():
        split_records = _split_records(records, split)
        if not split_records:
            raise ValueError(f"No usable {split} records in {paths[split]}")
        data[split] = _matrix(split_records)
    image_train, rppg_train, y_train = data["train"]
    image_val, rppg_val, y_val = data["val"]
    image_test, rppg_test, y_test = data["test"]
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        raise ValueError("Fusion requires both classes in train and val")

    image_logits_train = _logit(image_train)
    image_mean = float(image_logits_train.mean())
    image_scale = float(image_logits_train.std() or 1.0)
    rppg_mean, rppg_scale = _standardiser(rppg_train)

    X_train = _design(
        image_train, rppg_train, image_mean, image_scale, rppg_mean, rppg_scale
    )
    X_val = _design(
        image_val, rppg_val, image_mean, image_scale, rppg_mean, rppg_scale
    )
    X_test = _design(
        image_test, rppg_test, image_mean, image_scale, rppg_mean, rppg_scale
    )

    best = (-1.0, 1.0)
    for C in (0.01, 0.1, 1.0, 10.0):
        candidate = _fit_logistic(X_train, y_train, C)
        p_val = candidate.predict_proba(X_val)[:, 1]
        if len(np.unique(y_val)) > 1:
            from sklearn.metrics import roc_auc_score

            auc = float(roc_auc_score(y_val, p_val))
        else:
            auc = 0.0
        if auc > best[0]:
            best = (auc, C)

    C = best[1]
    model = _fit_logistic(X_train, y_train, C)
    p_val_raw = model.predict_proba(X_val)[:, 1]
    temperature = fit_temperature(p_val_raw, y_val)
    p_val = _sigmoid(_logit(p_val_raw) / temperature)
    threshold = choose_conservative_threshold(y_val, p_val)

    p_test_raw = model.predict_proba(X_test)[:, 1]
    p_test = _sigmoid(_logit(p_test_raw) / temperature)

    image_threshold = choose_conservative_threshold(y_val, image_val)
    image_test_metrics = _metrics(y_test, image_test, image_threshold)
    fusion_test_metrics = _metrics(y_test, p_test, threshold)
    fusion_val_metrics = _metrics(y_val, p_val, threshold)

    # Deployment model: refit on train+val after all tuning is complete.
    image_all = np.concatenate([image_train, image_val])
    rppg_all = np.concatenate([rppg_train, rppg_val])
    y_all = np.concatenate([y_train, y_val])
    image_mean_final = float(_logit(image_all).mean())
    image_scale_final = float(_logit(image_all).std() or 1.0)
    rppg_mean_final, rppg_scale_final = _standardiser(rppg_all)
    X_all = _design(
        image_all,
        rppg_all,
        image_mean_final,
        image_scale_final,
        rppg_mean_final,
        rppg_scale_final,
    )
    final_model = _fit_logistic(X_all, y_all, C)

    payload = {
        "backend": "dfb",
        "detector": detector,
        "mode": mode,
        "fusion": "logistic_regression_late_fusion",
        "rppg_feature_names": list(RPPG_FEATURE_NAMES),
        "image_feature": "logit(prob_real)",
        "image_mean": image_mean_final,
        "image_scale": image_scale_final,
        "rppg_mean": [float(v) for v in rppg_mean_final],
        "rppg_scale": [float(v) for v in rppg_scale_final],
        "coef": [float(v) for v in final_model.coef_.reshape(-1)],
        "intercept": float(final_model.intercept_[0]),
        "temperature": float(temperature),
        "threshold": float(threshold),
        "C": float(C),
        "label_convention": {"real": 1, "fake": 0},
        "metrics": {
            "image_only_test": image_test_metrics,
            "fusion_test": fusion_test_metrics,
            "fusion_val": fusion_val_metrics,
            "image_only_threshold": float(image_threshold),
        },
        "counts": {
            "train": int(len(y_train)),
            "val": int(len(y_val)),
            "test": int(len(y_test)),
        },
        "sources": {split: str(path) for split, path in paths.items()},
    }
    out = Path(output_path) if output_path else fusion_artifact_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_fusion_artifact(path: Path | None = None) -> dict[str, Any] | None:
    path = Path(path) if path else fusion_artifact_path()
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if list(payload.get("rppg_feature_names", [])) != list(RPPG_FEATURE_NAMES):
        raise ValueError(
            "Fusion artifact rPPG feature order does not match RPPG_FEATURE_NAMES"
        )
    return payload


def predict_fused(
    image_prob_real: float,
    rppg_values: Sequence[float],
    artifact: dict[str, Any],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Apply the saved fusion model to one video."""
    values = np.asarray(list(rppg_values), dtype=np.float64)
    names = list(artifact["rppg_feature_names"])
    if values.size != len(names):
        raise ValueError(f"Expected {len(names)} rPPG values, got {values.size}")
    if not np.isfinite(values).all():
        raise ValueError("rPPG values must be finite")
    coef = np.asarray(artifact["coef"], dtype=np.float64)
    if coef.size != values.size + 1:
        raise ValueError("Fusion coefficient length does not match feature contract")
    z_image = (float(_logit(np.asarray([image_prob_real]))[0]) - float(artifact["image_mean"])) / float(
        artifact["image_scale"]
    )
    z_rppg = (values - np.asarray(artifact["rppg_mean"], dtype=np.float64)) / np.asarray(
        artifact["rppg_scale"], dtype=np.float64
    )
    design = np.concatenate([[z_image], z_rppg])
    logit = float(artifact["intercept"] + np.dot(coef, design))
    temperature = float(artifact.get("temperature", 1.0))
    fused_prob_real = float(_sigmoid(np.asarray([logit / max(temperature, 1e-6)]))[0])
    # rPPG is supporting evidence only. It must never turn a visual DFB
    # signal into a more permissive REAL decision.
    prob_real = min(float(image_prob_real), fused_prob_real)
    cut = float(artifact["threshold"] if threshold is None else threshold)
    return {
        "prob_real": round(prob_real, 6),
        "verdict": "REAL" if prob_real >= cut else "FAKE",
        "confidence": round(abs(prob_real - 0.5) * 2.0, 6),
        "threshold": cut,
        "image_prob_real": float(image_prob_real),
        "fusion": "image+rppg",
        "fused_prob_real": round(fused_prob_real, 6),
    }
