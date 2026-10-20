"""ViT frame encoder, temporal aggregation, rPPG fusion, and checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from RPPG.features import RPPGFeatures

from vit.config import VitConfig


def _create_backbone(cfg: VitConfig) -> nn.Module:
    import timm

    model = timm.create_model(
        cfg.backbone,
        pretrained=bool(cfg.pretrained),
        num_classes=0,
        img_size=cfg.image_size,
    )
    return model


def _backbone_blocks(backbone: nn.Module) -> list[nn.Module]:
    for attribute in ("blocks", "layers"):
        candidate = getattr(backbone, attribute, None)
        if isinstance(candidate, (nn.ModuleList, nn.Sequential)):
            return list(candidate)
    return []


class AttentionPool(nn.Module):
    """Masked-free attention pooling over frame embeddings."""

    def __init__(self, dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Linear(dim, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(self.norm(embeddings)).squeeze(-1), dim=1)
        pooled = torch.sum(embeddings * weights.unsqueeze(-1), dim=1)
        return self.dropout(pooled)


class TemporalTransformer(nn.Module):
    """Small 1-2 layer temporal Transformer encoder with a CLS token."""

    def __init__(self, dim: int, cfg: VitConfig) -> None:
        super().__init__()
        if cfg.temporal_heads < 1 or dim % cfg.temporal_heads != 0:
            raise ValueError(
                f"temporal_heads={cfg.temporal_heads} must divide embedding dim {dim}"
            )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, cfg.frames_per_video + 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=cfg.temporal_heads,
            dim_feedforward=dim * 2,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.temporal_layers)
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position, std=0.02)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        batch, frames, _ = embeddings.shape
        cls = self.cls_token.expand(batch, -1, -1)
        tokens = torch.cat([cls, embeddings], dim=1)
        tokens = tokens + self.position[:, : frames + 1]
        encoded = self.encoder(tokens)
        return self.norm(encoded[:, 0])


class RPPGBranch(nn.Module):
    def __init__(self, input_dim: int, hidden_1: int, hidden_2: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.GELU(),
        )
        self.output_dim = hidden_2

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class VitClassifier(nn.Module):
    """Per-frame ViT embeddings -> temporal video embedding -> optional fusion."""

    def __init__(self, cfg: VitConfig, rppg_dim: int = 20) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone = _create_backbone(cfg)
        self.embed_dim = int(getattr(self.backbone, "num_features", 0))
        if self.embed_dim <= 0:
            raise ValueError(f"Could not determine embedding dim for {cfg.backbone}")

        self.temporal_head = cfg.temporal_head
        if cfg.temporal_head == "attention_pool":
            self.temporal = AttentionPool(self.embed_dim, cfg.dropout)
        elif cfg.temporal_head == "transformer":
            self.temporal = TemporalTransformer(self.embed_dim, cfg)
        elif cfg.temporal_head == "mean_logits":
            self.temporal = None
        else:  # guarded by config
            raise ValueError(cfg.temporal_head)

        self.use_rppg = cfg.use_rppg
        if self.use_rppg:
            self.register_buffer("rppg_mean", torch.zeros(rppg_dim))
            self.register_buffer("rppg_scale", torch.ones(rppg_dim))
            self.rppg_branch = RPPGBranch(
                rppg_dim, cfg.rppg_hidden_1, cfg.rppg_hidden_2, cfg.dropout
            )
            fused_dim = self.embed_dim + self.rppg_branch.output_dim
            self.classifier = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, cfg.fusion_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.fusion_hidden, 1),
            )
        else:
            self.rppg_branch = None
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, cfg.fusion_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.fusion_hidden, 1),
            )
        if cfg.temporal_head == "mean_logits":
            self.frame_logit = nn.Linear(self.embed_dim, 1)

    def set_rppg_scaler(self, mean, scale) -> None:
        if not self.use_rppg:
            return
        mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
        scale_tensor = torch.as_tensor(scale, dtype=torch.float32)
        if mean_tensor.shape != self.rppg_mean.shape or scale_tensor.shape != self.rppg_scale.shape:
            raise ValueError(
                f"rPPG scaler shape mismatch: mean {tuple(mean_tensor.shape)}, "
                f"scale {tuple(scale_tensor.shape)}, expected {tuple(self.rppg_mean.shape)}"
            )
        if torch.any(scale_tensor <= 0):
            raise ValueError("rPPG scaler scale values must be positive")
        self.rppg_mean.copy_(mean_tensor)
        self.rppg_scale.copy_(scale_tensor)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5:
            raise ValueError(f"frames must be (B, N, C, H, W), got {tuple(frames.shape)}")
        batch, n_frames = frames.shape[:2]
        flat = frames.reshape(batch * n_frames, *frames.shape[2:])
        embeddings = self.backbone(flat)
        return embeddings.reshape(batch, n_frames, -1)

    def forward(self, frames: torch.Tensor, rppg: torch.Tensor | None = None) -> torch.Tensor:
        embeddings = self.encode_frames(frames)
        if self.temporal_head == "mean_logits":
            return self.frame_logit(embeddings).mean(dim=1)
        video_embedding = self.temporal(embeddings)
        if self.use_rppg:
            if rppg is None:
                raise ValueError("rPPG features are required when use_rppg=True")
            scaled = (rppg - self.rppg_mean) / self.rppg_scale
            rppg_embedding = self.rppg_branch(scaled)
            video_embedding = torch.cat([video_embedding, rppg_embedding], dim=1)
        return self.classifier(video_embedding)

    def set_backbone_trainable(self, unfreeze_blocks: int) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        if unfreeze_blocks <= 0:
            return
        blocks = _backbone_blocks(self.backbone)
        for block in blocks[-unfreeze_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for attribute in ("norm", "layers_norm"):
            module = getattr(self.backbone, attribute, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad = True


def build_model(cfg: VitConfig, device: torch.device, rppg_dim: int = 20) -> VitClassifier:
    model = VitClassifier(cfg, rppg_dim=rppg_dim)
    model.to(device)
    return model


def _block_index(parameter_name: str) -> int | None:
    parts = parameter_name.split(".")
    for marker in ("blocks", "layers"):
        if marker in parts:
            position = parts.index(marker)
            if position + 1 < len(parts) and parts[position + 1].isdigit():
                return int(parts[position + 1])
    return None


def build_param_groups(model: VitClassifier, cfg: VitConfig) -> list[dict[str, Any]]:
    blocks = _backbone_blocks(model.backbone)
    n_blocks = len(blocks)
    groups: dict[tuple[float, float], dict[str, Any]] = {}

    def add(name: str, parameter: nn.Parameter, lr: float) -> None:
        decay = cfg.weight_decay if parameter.ndim >= 2 else 0.0
        key = (float(lr), float(decay))
        group = groups.setdefault(key, {"lr": float(lr), "weight_decay": float(decay), "params": [], "names": []})
        group["params"].append(parameter)
        group["names"].append(name)

    backbone_ids = {id(p) for p in model.backbone.parameters()}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in backbone_ids:
            block = _block_index(name)
            if block is None or n_blocks == 0:
                lr = cfg.backbone_lr
            else:
                depth = max(0, n_blocks - 1 - block)
                lr = cfg.backbone_lr * (cfg.layer_decay ** depth)
            add(name, parameter, lr)
        else:
            add(name, parameter, cfg.head_lr)
    return [group for group in groups.values() if group["params"]]


def save_checkpoint(
    model: VitClassifier,
    cfg: VitConfig,
    path: Path,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Save a weights-only state dict plus a sidecar JSON config."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, path)
    payload = cfg.to_dict()
    payload["checkpoint"] = str(path)
    if extra:
        payload.update(extra)
    config_path = path.with_suffix(".json")
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, config_path


def load_checkpoint(path: Path, device: torch.device) -> tuple[VitClassifier, VitConfig, dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"ViT checkpoint not found: {path}. Train one first: python -m vit.pipeline --train"
        )
    config_path = path.with_suffix(".json")
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint config JSON not found: {config_path}. Refusing to load an "
            "artifact without its training configuration."
        )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    saved_order = payload.get("rppg_feature_order")
    if saved_order is not None and list(saved_order) != list(RPPGFeatures.feature_names()):
        raise ValueError(
            "Checkpoint rPPG feature order does not match RPPGFeatures.feature_names(); "
            "retrain or realign the feature contract"
        )
    cfg = VitConfig.from_dict(payload)
    cfg.pretrained = False
    model = VitClassifier(cfg, rppg_dim=20)
    state = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(state, dict) or "model" not in state:
        raise RuntimeError(f"Invalid ViT checkpoint structure in {path}")
    model.load_state_dict(state["model"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg, payload


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_warmup_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))
