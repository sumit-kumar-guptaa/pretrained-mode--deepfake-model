"""Configuration and artifact locations for the ViT decision stage.

The ViT stage consumes stage-1 accepted face frames and the 20 rPPG
features. Labels always follow the quantum convention (1 = real,
0 = fake) via ``quantum.data.csv_to_quantum_label``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from quantum.config import LABEL_FAKE, LABEL_REAL


WORKING_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = Path(
    os.environ.get("MAJ_OUTPUT_ROOT", WORKING_ROOT.parent / "Scrape" / "output")
)
VIT_OUTPUT_DIR = OUTPUT_BASE / "vit"
FRAMES_ROOT = OUTPUT_BASE / "frames" / "frame_sequences"
RPPG_CSV = OUTPUT_BASE / "rppg" / "dataset_features.csv"
SPLIT_MANIFEST = OUTPUT_BASE / "quantum" / "split_manifest_fused.json"

DEFAULT_CHECKPOINT = VIT_OUTPUT_DIR / "vit_model.pt"
ABLATION_CHECKPOINT = VIT_OUTPUT_DIR / "vit_only.pt"
DEFAULT_CONFIG_JSON = VIT_OUTPUT_DIR / "vit_config.json"
METRICS_JSON = VIT_OUTPUT_DIR / "metrics_vit.json"
ABLATION_METRICS_JSON = VIT_OUTPUT_DIR / "metrics_vit_only.json"
COMPARISON_JSON = VIT_OUTPUT_DIR / "comparison.json"
COMPARISON_MD = VIT_OUTPUT_DIR / "comparison.md"

ALLOWED_BACKBONES = (
    "vit_small_patch16_224.augreg_in21k_ft_in1k",
    "vit_base_patch16_224",
    "deit_small_patch16_224",
    "swin_tiny_patch4_window7_224",
)
TEMPORAL_HEADS = ("attention_pool", "transformer", "mean_logits")
LOSSES = ("bce", "focal")


@dataclass
class VitConfig:
    """All model/training/evaluation hyperparameters (JSON serialisable)."""

    backbone: str = "vit_small_patch16_224.augreg_in21k_ft_in1k"
    pretrained: bool = True
    image_size: int = 224
    frames_per_video: int = 16
    min_frames: int = 16

    temporal_head: str = "attention_pool"
    temporal_layers: int = 1
    temporal_heads: int = 4
    dropout: float = 0.2

    use_rppg: bool = True
    rppg_hidden_1: int = 64
    rppg_hidden_2: int = 32
    fusion_hidden: int = 256

    stage_a_epochs: int = 4
    stage_b_epochs: int = 20
    batch_size: int = 8
    grad_accum_steps: int = 1
    num_workers: int = 4
    freeze_backbone_stage_a: bool = True
    unfreeze_blocks: int = 4

    backbone_lr: float = 1e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.05
    layer_decay: float = 0.75
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    mixed_precision: bool = True
    patience: int = 6
    min_delta: float = 1e-4

    loss: str = "bce"
    focal_gamma: float = 2.0
    use_pos_weight: bool = True
    balanced_sampler: bool = True

    augment: bool = True
    augment_downscale_min: int = 64
    augment_downscale_max: int = 160
    augment_jpeg_quality_min: int = 30
    augment_jpeg_quality_max: int = 90

    decision_threshold: float = 0.5
    threshold_method: str = "youden"
    calibrate_temperature: bool = True
    temperature: float = 1.0

    seed: int = 42
    max_videos: int | None = None
    cv_folds: int = 5

    device: str = "auto"
    output_dir: Path = field(default_factory=lambda: VIT_OUTPUT_DIR)

    def __post_init__(self) -> None:
        if self.backbone not in ALLOWED_BACKBONES:
            raise ValueError(
                f"Unsupported backbone {self.backbone!r}; choose from {ALLOWED_BACKBONES}"
            )
        if self.temporal_head not in TEMPORAL_HEADS:
            raise ValueError(
                f"Unsupported temporal_head {self.temporal_head!r}; choose from {TEMPORAL_HEADS}"
            )
        if self.loss not in LOSSES:
            raise ValueError(f"Unsupported loss {self.loss!r}; choose from {LOSSES}")
        if self.frames_per_video < 1 or self.min_frames < 1:
            raise ValueError("frames_per_video and min_frames must be positive")
        if self.min_frames < self.frames_per_video:
            raise ValueError("min_frames must be at least frames_per_video")
        if self.temporal_head == "mean_logits" and self.use_rppg:
            raise ValueError(
                "mean_logits is a ViT-only baseline; set use_rppg=False to use it"
            )
        if self.temporal_head == "transformer" and self.temporal_layers < 1:
            raise ValueError("temporal_layers must be >= 1 for the transformer head")
        if self.temporal_layers > 2:
            raise ValueError("temporal_layers must be 1 or 2")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_size < 1 or self.grad_accum_steps < 1:
            raise ValueError("batch_size and grad_accum_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["label_convention"] = {"real": LABEL_REAL, "fake": LABEL_FAKE}
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VitConfig":
        allowed = {item.name for item in fields(cls)}
        clean = {k: v for k, v in payload.items() if k in allowed}
        if "output_dir" in clean:
            clean["output_dir"] = Path(clean["output_dir"])
        return cls(**clean)

    def save_json(self, path: Path | None = None) -> Path:
        path = Path(path) if path else self.output_dir / "vit_config.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


def load_config_json(path: Path) -> VitConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return VitConfig.from_dict(payload)
