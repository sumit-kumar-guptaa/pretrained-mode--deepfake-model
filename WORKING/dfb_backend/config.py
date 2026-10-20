"""Paths and detector registry for the DeepfakeBench image backend."""

from __future__ import annotations

import os
from pathlib import Path

WORKING_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = Path(
    os.environ.get("MAJ_OUTPUT_ROOT", WORKING_ROOT.parent / "Scrape" / "output")
)

DFB_OUTPUT_DIR = OUTPUT_BASE / "dfb"
FRAMES_ROOT = OUTPUT_BASE / "frames" / "frame_sequences"
RPPG_CSV = OUTPUT_BASE / "rppg" / "dataset_features.csv"
SPLIT_MANIFEST = OUTPUT_BASE / "quantum" / "split_manifest_fused.json"
QUANTUM_METRICS = OUTPUT_BASE / "quantum" / "metrics_quantum.json"
BASELINE_METRICS = OUTPUT_BASE / "quantum" / "metrics_baselines.json"

DFB_ROOT = WORKING_ROOT / "third_party" / "DeepfakeBench"
DFB_TRAINING = DFB_ROOT / "training"
DFB_CONFIG_DIR = DFB_TRAINING / "config" / "detector"
DFB_WEIGHTS_DIR = DFB_TRAINING / "weights"
DFB_PRETRAINED_DIR = DFB_TRAINING / "pretrained"

DETECTOR_SPECS = {
    "xception": {
        "yaml": "xception.yaml",
        "checkpoint": "xception_best.pth",
        "module": "xception_detector",
        "pretrained": "xception-b5690688.pth",
    },
    "ucf": {
        "yaml": "ucf.yaml",
        "checkpoint": "ucf_best.pth",
        "module": "ucf_detector",
        "pretrained": "xception-b5690688.pth",
    },
    "efficientnetb4": {
        "yaml": "efficientnetb4.yaml",
        "checkpoint": "effnb4_best.pth",
        "module": "efficientnetb4_detector",
        "pretrained": None,
    },
}

DEFAULT_DETECTOR = "xception"
DEFAULT_FRAMES = 32
DEFAULT_MIN_FRAMES = 8
DEFAULT_AGGREGATION = "topk"
DEFAULT_TOPK_RATIO = 0.25
SUPPORTED_AGGREGATIONS = ("mean", "median", "topk")


def checkpoint_path(detector: str) -> Path:
    return DFB_WEIGHTS_DIR / DETECTOR_SPECS[detector]["checkpoint"]


def finetuned_checkpoint(detector: str) -> Path:
    return DFB_OUTPUT_DIR / f"{detector}_finetuned.pt"


def finetuned_metadata(detector: str) -> Path:
    return DFB_OUTPUT_DIR / f"{detector}_finetuned.json"


def fusion_artifact_path() -> Path:
    return DFB_OUTPUT_DIR / "fusion.json"


def scores_path(detector: str, mode: str) -> Path:
    return DFB_OUTPUT_DIR / f"scores_{detector}_{mode}.json"


def metrics_path(detector: str, mode: str) -> Path:
    return DFB_OUTPUT_DIR / f"metrics_{detector}_{mode}.json"
