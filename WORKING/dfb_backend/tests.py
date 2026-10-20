"""Self-checks for the DFB backend (no network required)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from dfb_backend.config import DETECTOR_SPECS, fusion_artifact_path
from dfb_backend.fusion import (
    choose_conservative_threshold,
    load_fusion_artifact,
    predict_fused,
)
from dfb_backend.loader import (
    load_detector,
    quantum_label_to_dfb_label,
    set_backbone_frozen,
)
from dfb_backend.predict import (
    aggregate_frame_probs,
    fake_prob_to_real,
    predict_video,
    run_model_on_tensor,
)
from dfb_backend.preprocess import (
    iter_frame_paths,
    preprocess_array,
    select_frame_indices,
)
from quantum.config import RPPG_FEATURE_NAMES


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_probability_conversion() -> None:
    _check(np.isclose(fake_prob_to_real(0.9), 0.1), "fake_prob_to_real(0.9) must be 0.1")
    array = fake_prob_to_real(np.asarray([0.0, 0.25, 1.0]))
    _check(np.allclose(array, [1.0, 0.75, 0.0]), "array conversion is wrong")
    for bad in (-0.1, 1.1, float("nan")):
        try:
            fake_prob_to_real(bad)
        except ValueError:
            continue
        raise AssertionError(f"fake_prob_to_real accepted {bad}")
    _check(quantum_label_to_dfb_label(1) == 0, "quantum REAL must map to DFB real=0")
    _check(quantum_label_to_dfb_label(0) == 1, "quantum FAKE must map to DFB fake=1")


def test_aggregation() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.9]
    _check(abs(aggregate_frame_probs(values, "mean") - 0.38) < 1e-9, "mean wrong")
    _check(abs(aggregate_frame_probs(values, "median") - 0.3) < 1e-9, "median wrong")
    topk = aggregate_frame_probs(values, "topk", 0.4)
    _check(abs(topk - 0.65) < 1e-9, f"topk wrong: {topk}")
    try:
        aggregate_frame_probs([], "mean")
    except ValueError:
        pass
    else:
        raise AssertionError("empty aggregation must fail")


def test_conservative_threshold() -> None:
    y_true = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    prob_real = np.asarray([0.2, 0.4, 0.6, 0.7, 0.65, 0.75, 0.85, 0.95])
    threshold = choose_conservative_threshold(
        y_true, prob_real, max_fake_false_acceptance=0.25
    )
    _check(threshold >= 0.6, "threshold must reject at least 75% of fake samples")
    predictions = prob_real >= threshold
    _check(
        float(predictions[y_true == 0].mean()) <= 0.25,
        "conservative threshold exceeded fake acceptance target",
    )


def test_preprocess() -> None:
    resolution = 32
    white = np.full((12, 16, 3), 255, dtype=np.uint8)
    tensor = preprocess_array(
        white, resolution=resolution, mean=[0.5] * 3, std=[0.5] * 3
    )
    _check(tuple(tensor.shape) == (3, resolution, resolution), "shape wrong")
    _check(tensor.dtype == torch.float32, "dtype must be float32")
    _check(float(tensor.max()) <= 1.0 + 1e-5, "normalised max out of range")
    _check(abs(float(tensor.mean()) - 1.0) < 1e-4, "white must normalise to +1")

    black = np.zeros((12, 16, 3), dtype=np.uint8)
    dark = preprocess_array(
        black, resolution=resolution, mean=[0.5] * 3, std=[0.5] * 3
    )
    _check(abs(float(dark.mean()) + 1.0) < 1e-4, "black must normalise to -1")


def test_frame_discovery() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames = root / "frames"
        frames.mkdir()
        for i in range(3):
            cv2.imwrite(str(frames / f"{i:03d}.jpg"), np.zeros((8, 8, 3), np.uint8))
        paths = iter_frame_paths(root)
        _check(len(paths) == 3, "frames/ discovery failed")

        crops = root / "cropped_faces"
        crops.mkdir()
        for i in range(2):
            cv2.imwrite(str(crops / f"{i:03d}.jpg"), np.zeros((8, 8, 3), np.uint8))
        preferred = iter_frame_paths(root)
        _check(len(preferred) == 2, "cropped_faces must be preferred")

    _check(select_frame_indices(100, 5) == [0, 24, 49, 74, 99], "even sampling wrong")
    _check(select_frame_indices(100, 5, training=True, rng=np.random.default_rng(0)) != [], "jitter failed")


def test_insufficient_frames_inconclusive() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        frames = root / "frames"
        frames.mkdir()
        for i in range(3):
            cv2.imwrite(str(frames / f"{i:03d}.jpg"), np.zeros((8, 8, 3), np.uint8))
        result = predict_video(frames_dir=root, frames=16, min_frames=8)
        _check(result["prob_real"] is None, "too few frames must be inconclusive")
        _check(result["verdict"] == "INCONCLUSIVE", "verdict must be INCONCLUSIVE")
        _check(result["frames_used"] == 0, "inconclusive result must use zero frames")


def test_feature_contract() -> None:
    from RPPG.features import RPPGFeatures

    _check(
        list(RPPGFeatures.feature_names()) == list(RPPG_FEATURE_NAMES),
        "RPPG feature order drifted from quantum.config.RPPG_FEATURE_NAMES",
    )
    _check(
        len(RPPG_FEATURE_NAMES) == 20,
        f"expected 20 rPPG features, got {len(RPPG_FEATURE_NAMES)}",
    )


def test_fusion_inference() -> None:
    artifact = {
        "rppg_feature_names": list(RPPG_FEATURE_NAMES),
        "image_mean": 0.0,
        "image_scale": 1.0,
        "rppg_mean": [0.0] * len(RPPG_FEATURE_NAMES),
        "rppg_scale": [1.0] * len(RPPG_FEATURE_NAMES),
        "coef": [1.0] + [0.0] * len(RPPG_FEATURE_NAMES),
        "intercept": 0.0,
        "temperature": 1.0,
        "threshold": 0.5,
    }
    result = predict_fused(0.8, [0.0] * len(RPPG_FEATURE_NAMES), artifact)
    _check(result["prob_real"] > 0.5, "positive image logit should predict REAL")
    _check(result["verdict"] == "REAL", "verdict mismatch")
    boosted = dict(artifact)
    boosted["coef"] = [0.0] + [10.0] * len(RPPG_FEATURE_NAMES)
    boosted_result = predict_fused(0.8, [1.0] * len(RPPG_FEATURE_NAMES), boosted)
    _check(
        boosted_result["prob_real"] <= 0.8,
        "rPPG fusion must not increase the visual REAL probability",
    )
    result = predict_fused(0.2, [0.0] * len(RPPG_FEATURE_NAMES), artifact)
    _check(result["prob_real"] < 0.5, "negative image logit should predict FAKE")
    try:
        predict_fused(0.5, [0.0] * 3, artifact)
    except ValueError:
        pass
    else:
        raise AssertionError("fusion must reject wrong rPPG length")

    existing = load_fusion_artifact(fusion_artifact_path())
    if existing is not None:
        _check(
            list(existing["rppg_feature_names"]) == list(RPPG_FEATURE_NAMES),
            "saved fusion artifact feature order drifted",
        )


def test_detector_loading() -> None:
    for detector in DETECTOR_SPECS:
        model, meta = load_detector(detector, device="cpu", strict=True)
        _check(not meta["missing_keys"], f"{detector} has missing keys")
        _check(not meta["unexpected_keys"], f"{detector} has unexpected keys")
        _check(int(meta["resolution"]) in (256, 380), f"{detector} resolution odd")
        frozen = set_backbone_frozen(model, True)
        _check(frozen > 0, f"{detector} has no backbone parameters")
        set_backbone_frozen(model, False)
        images = torch.randn(2, 3, int(meta["resolution"]), int(meta["resolution"]))
        probs = run_model_on_tensor(model, images, device=torch.device("cpu"))
        _check(probs.shape == (2,), f"{detector} output shape {probs.shape}")
        _check(np.isfinite(probs).all(), f"{detector} non-finite probability")
        _check(((probs >= 0) & (probs <= 1)).all(), f"{detector} probability range")


def main() -> int:
    tests = [
        test_probability_conversion,
        test_aggregation,
        test_conservative_threshold,
        test_preprocess,
        test_frame_discovery,
        test_insufficient_frames_inconclusive,
        test_feature_contract,
        test_fusion_inference,
    ]
    if os.environ.get("DFB_RUN_MODEL_TESTS", "").lower() in {"1", "true", "yes"}:
        tests.append(test_detector_loading)
    for test in tests:
        test()
        print(f"[ok] {test.__name__}")
    if test_detector_loading not in tests:
        print("[skip] test_detector_loading (set DFB_RUN_MODEL_TESTS=1 to run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
