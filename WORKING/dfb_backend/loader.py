"""DeepfakeBench detector loading without importing its optional stack.

DeepfakeBench's ``detectors`` and ``loss`` package initialisers import
SlowFast/decord/fvcore, which are unnecessary for KYC inference and are not
installed on this host.  We install lightweight package stubs whose
``__path__`` points at the real directories, import only the detector/loss
modules that are actually needed, and let the upstream registries resolve
classes by name.
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import torch

from dfb_backend.config import (
    DETECTOR_SPECS,
    DFB_CONFIG_DIR,
    DFB_PRETRAINED_DIR,
    DFB_ROOT,
    DFB_TRAINING,
    DFB_WEIGHTS_DIR,
    checkpoint_path,
)

LABEL_FAKE = 1
LABEL_REAL = 0

_HEAD_MARKERS = (
    "last_linear",
    "_fc",
    "classifier",
    "head_sha",
    "head_spe",
    "head_con",
    "head_rec",
)


def quantum_label_to_dfb_label(label: int) -> int:
    """Map quantum labels (1 = real) to DeepfakeBench labels (1 = fake)."""
    if int(label) not in (0, 1):
        raise ValueError(f"label must be 0 or 1, got {label!r}")
    return 1 - int(label)


def _ensure_paths() -> None:
    for path in (str(DFB_ROOT), str(DFB_TRAINING)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _bootstrap() -> None:
    _ensure_paths()
    from metrics.registry import DETECTOR, LOSSFUNC

    detectors = sys.modules.get("detectors")
    if detectors is None or not getattr(detectors, "_dfb_stub", False):
        detectors = types.ModuleType("detectors")
        detectors.__path__ = [str(DFB_TRAINING / "detectors")]  # type: ignore[attr-defined]
        detectors.DETECTOR = DETECTOR  # type: ignore[attr-defined]
        detectors._dfb_stub = True  # type: ignore[attr-defined]
        sys.modules["detectors"] = detectors

    loss = sys.modules.get("loss")
    if loss is None or not getattr(loss, "_dfb_stub", False):
        loss = types.ModuleType("loss")
        loss.__path__ = [str(DFB_TRAINING / "loss")]  # type: ignore[attr-defined]
        loss.LOSSFUNC = LOSSFUNC  # type: ignore[attr-defined]
        loss._dfb_stub = True  # type: ignore[attr-defined]
        sys.modules["loss"] = loss

    for name in ("cross_entropy_loss", "contrastive_regularization", "l1_loss"):
        importlib.import_module(f"loss.{name}")


def resolve_device(device: str | torch.device | None = "auto") -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return resolved


def load_detector_config(detector: str) -> dict[str, Any]:
    if detector not in DETECTOR_SPECS:
        raise ValueError(
            f"Unknown detector {detector!r}; choose from {sorted(DETECTOR_SPECS)}"
        )
    _ensure_paths()
    import yaml

    spec = DETECTOR_SPECS[detector]
    config_path = DFB_CONFIG_DIR / spec["yaml"]
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing DeepfakeBench config: {config_path}")
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    pretrained_name = spec.get("pretrained")
    pretrained_path = DFB_PRETRAINED_DIR / pretrained_name if pretrained_name else None
    if pretrained_path is not None and pretrained_path.is_file():
        cfg["pretrained"] = str(pretrained_path)
    else:
        cfg["pretrained"] = False
    cfg.setdefault("backbone_config", {})["num_classes"] = 2
    return cfg


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            nested = payload.get(key)
            if isinstance(nested, dict) and nested and all(
                isinstance(k, str) for k in nested
            ):
                return nested
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint is not a state_dict-like mapping")
    return payload


def _strip_data_parallel(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[len("module.") :]: value for key, value in state.items()}
    return state


def load_detector(
    detector: str = "xception",
    *,
    device: str | torch.device | None = "auto",
    checkpoint: Path | str | None = None,
    strict: bool = True,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Instantiate a detector and load a DeepfakeBench checkpoint.

    Returns ``(model, metadata)``.  ``metadata`` records the resolved device,
    yaml path, checkpoint path, key accounting, and preprocessing contract so
    evaluation artifacts are self-describing.
    """
    if detector not in DETECTOR_SPECS:
        raise ValueError(
            f"Unknown detector {detector!r}; choose from {sorted(DETECTOR_SPECS)}"
        )
    _bootstrap()
    from metrics.registry import DETECTOR

    spec = DETECTOR_SPECS[detector]
    importlib.import_module(f"detectors.{spec['module']}")
    cfg = load_detector_config(detector)
    device_obj = resolve_device(device)

    model = DETECTOR[detector](cfg)
    ckpt_path = Path(checkpoint) if checkpoint else checkpoint_path(detector)
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"Missing {detector} checkpoint: {ckpt_path}. "
            "Download the DeepfakeBench v1.0.1 release weights into training/weights/."
        )
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        payload = torch.load(ckpt_path, map_location="cpu")
    state = _strip_data_parallel(_unwrap_state_dict(payload))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"{detector} checkpoint key mismatch: missing={missing[:5]} "
            f"({len(missing)}), unexpected={unexpected[:5]} ({len(unexpected)})"
        )
    model.to(device_obj)
    model.eval()

    meta = {
        "backend": "dfb",
        "detector": detector,
        "device": str(device_obj),
        "yaml": str(DFB_CONFIG_DIR / spec["yaml"]),
        "checkpoint": str(ckpt_path),
        "checkpoint_keys": len(state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "resolution": int(cfg["resolution"]),
        "mean": [float(v) for v in cfg.get("mean", [0.5, 0.5, 0.5])],
        "std": [float(v) for v in cfg.get("std", [0.5, 0.5, 0.5])],
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "pretrained_backbone": bool(cfg.get("pretrained")),
        "label_convention": {"dfb_fake": LABEL_FAKE, "dfb_real": LABEL_REAL},
        "weights_dir": str(DFB_WEIGHTS_DIR),
    }
    return model, meta


def reset_detector_buffers(model: torch.nn.Module) -> None:
    """UCF appends every batch to ``prob``/``label``; keep inference bounded."""
    for name in ("prob", "label"):
        value = getattr(model, name, None)
        if isinstance(value, list):
            value.clear()


def _is_head_parameter(name: str) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in _HEAD_MARKERS)


def set_backbone_frozen(model: torch.nn.Module, frozen: bool) -> int:
    """Freeze/unfreeze non-classifier parameters; returns frozen param count."""
    count = 0
    for name, param in model.named_parameters():
        if _is_head_parameter(name):
            param.requires_grad = True
            continue
        param.requires_grad = not frozen
        if frozen:
            count += 1
    return count
