"""Self-checks for the ViT stage (no pytest dependency).

Run from WORKING/:  python -m vit.tests
Exit code 0 = all checks passed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

WORKING_ROOT = Path(__file__).resolve().parent.parent
if str(WORKING_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKING_ROOT))
REPO_ROOT = WORKING_ROOT.parent
SCRAPE_ROOT = REPO_ROOT / "Scrape"
SCRAPE_ROOT.mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import torch

from quantum.config import LABEL_FAKE, LABEL_REAL, RPPG_FEATURE_NAMES
from quantum.data import csv_to_quantum_label
from RPPG.features import RPPGFeatures

from vit.config import VitConfig
from vit.dataset import (
    VideoRecord,
    assert_zero_video_overlap,
    sample_frame_indices,
)
from vit.evaluation import (
    build_comparison_table,
    evaluate_video_level,
    save_evaluation_artifacts,
)
from vit.model import VitClassifier, load_checkpoint, save_checkpoint
from vit.pipeline import predict_video

BACKBONE = "deit_small_patch16_224"
IMAGE_SIZE = 64


def _cfg(**overrides) -> VitConfig:
    values = {
        "backbone": BACKBONE,
        "pretrained": False,
        "image_size": IMAGE_SIZE,
        "frames_per_video": 2,
        "min_frames": 2,
        "batch_size": 2,
        "num_workers": 0,
        "dropout": 0.0,
        "fusion_hidden": 32,
        "output_dir": SCRAPE_ROOT,
    }
    values.update(overrides)
    return VitConfig(**values)


def _frames(batch: int = 2, count: int = 2) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(batch, count, 3, IMAGE_SIZE, IMAGE_SIZE, generator=generator)


def _record(video: str, group: str, split: str, label: int = LABEL_REAL) -> VideoRecord:
    return VideoRecord(
        video_path=f"{video}.mp4",
        video_key=video,
        group=group,
        split=split,
        frames_dir="",
        frame_paths=(),
        rppg=(0.0,) * len(RPPG_FEATURE_NAMES),
        label=label,
    )


def test_labels_and_feature_order():
    assert csv_to_quantum_label(0) == LABEL_REAL
    assert csv_to_quantum_label(1) == LABEL_FAKE
    assert RPPGFeatures.feature_names() == list(RPPG_FEATURE_NAMES)
    assert len(RPPG_FEATURE_NAMES) == 20


def test_frame_sampling_deterministic():
    first = sample_frame_indices(100, 16)
    second = sample_frame_indices(100, 16)
    assert first == second
    assert len(first) == 16
    assert first[0] == 0 and first[-1] == 99
    jittered = sample_frame_indices(100, 16, rng=np.random.default_rng(0), jitter=True)
    assert len(jittered) == 16
    assert jittered == sorted(jittered)


def test_cap_records_preserves_splits():
    from vit.dataset import cap_records

    records = [
        _record(f"train_{index}", f"g_train_{index}", "train")
        for index in range(4)
    ]
    records.append(_record("val", "g_val", "val"))
    records.append(_record("test", "g_test", "test"))
    capped = cap_records(records, 3, seed=42)
    assert len(capped) == 3
    assert {record.split for record in capped} == {"train", "val", "test"}
    try:
        cap_records(records, 2, seed=42)
    except ValueError:
        pass
    else:
        raise AssertionError("max_videos smaller than the split count must be rejected")

    skewed = [
        _record(f"train_{index}", f"g_train_{index}", "train")
        for index in range(8)
    ]
    skewed.append(_record("val", "g_val", "val"))
    skewed.append(_record("test", "g_test", "test"))
    capped = cap_records(skewed, 9, seed=42)
    assert len(capped) == 9
    assert {record.split for record in capped} == {"train", "val", "test"}


def test_zero_video_group_overlap():
    clean = [
        _record("a", "actor_a", "train"),
        _record("b", "actor_b", "val"),
        _record("c", "actor_c", "test"),
    ]
    assert_zero_video_overlap(clean)
    leaky = clean + [_record("a-other-file", "actor_a", "test")]
    try:
        assert_zero_video_overlap(leaky)
    except AssertionError:
        pass
    else:
        raise AssertionError("group leakage was not detected")


def test_head_shapes():
    cases = [
        ("attention_pool", True),
        ("transformer", True),
        ("mean_logits", False),
    ]
    for head, use_rppg in cases:
        cfg = _cfg(temporal_head=head, use_rppg=use_rppg)
        model = VitClassifier(cfg)
        model.eval()
        with torch.inference_mode():
            output = model(_frames(), torch.zeros(2, 20) if use_rppg else None)
        assert output.shape == (2, 1), (head, output.shape)


def test_one_batch_overfit():
    torch.manual_seed(3)
    cfg = _cfg(temporal_head="attention_pool", use_rppg=True)
    model = VitClassifier(cfg)
    model.set_backbone_trainable(0)
    model.train()
    frames = _frames(batch=2, count=2)
    rppg = torch.randn(2, 20, generator=torch.Generator().manual_seed(11))
    targets = torch.tensor([0.0, 1.0])
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-3, weight_decay=0.0)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    initial = None
    final = None
    for _step in range(80):
        optimizer.zero_grad()
        logits = model(frames, rppg).squeeze(-1)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        if initial is None:
            initial = float(loss.item())
        final = float(loss.item())

    predictions = (torch.sigmoid(model(frames, rppg).squeeze(-1)) >= 0.5).int()
    assert final < initial * 0.5, (initial, final)
    assert torch.equal(predictions, targets.int()), predictions


def test_stage_b_runs_after_stage_a_early_stop():
    import vit.train as train_module

    with tempfile.TemporaryDirectory(dir=SCRAPE_ROOT) as temporary:
        root = Path(temporary)
        records = []
        for index, split in enumerate(("train", "train", "val")):
            frame_dir = root / f"v{index}" / "frames"
            frame_dir.mkdir(parents=True)
            for frame in range(2):
                image = np.full((80, 80, 3), (index * 50 + frame * 20) % 255, dtype=np.uint8)
                cv2.imwrite(str(frame_dir / f"{frame:03d}.jpg"), image)
            records.append(
                VideoRecord(
                    video_path=f"v{index}.mp4",
                    video_key=f"v{index}",
                    group=f"g{index}",
                    split=split,
                    frames_dir=str(frame_dir.parent),
                    frame_paths=tuple(str(p) for p in sorted(frame_dir.glob("*.jpg"))),
                    rppg=(0.0,) * 20,
                    label=index % 2,
                )
            )

        cfg = _cfg(
            temporal_head="mean_logits",
            use_rppg=False,
            frames_per_video=1,
            min_frames=1,
            stage_a_epochs=2,
            stage_b_epochs=1,
            patience=1,
            min_delta=0.0,
            batch_size=2,
            output_dir=root / "vit",
        )
        original = train_module._validation_auc

        def constant_auc(*_args, **_kwargs):
            return 0.5, np.asarray([0.0, 1.0]), np.asarray([0, 1])

        train_module._validation_auc = constant_auc
        try:
            result = train_module.train_model(
                records[:2],
                records[2:],
                cfg,
                root / "vit" / "vit_model.pt",
                device=torch.device("cpu"),
                save=True,
                config_name="vit_only_config.json",
            )
        finally:
            train_module._validation_auc = original

        stages = [entry["stage"] for entry in result["history"]]
        assert "stage_a" in stages and "stage_b" in stages, stages
        assert (root / "vit" / "vit_only_config.json").is_file()


def test_checkpoint_round_trip_and_inconclusive():
    cfg = _cfg(temporal_head="attention_pool", use_rppg=True)
    model = VitClassifier(cfg)
    model.eval()
    frames = _frames()
    rppg = torch.zeros(2, 20)
    with torch.inference_mode():
        expected = model(frames, rppg)

    with tempfile.TemporaryDirectory(dir=SCRAPE_ROOT) as temporary:
        checkpoint = Path(temporary) / "vit_model.pt"
        save_checkpoint(model, cfg, checkpoint, extra={"temperature": 1.2})
        assert checkpoint.is_file()
        assert checkpoint.with_suffix(".json").is_file()
        loaded, loaded_cfg, payload = load_checkpoint(checkpoint, torch.device("cpu"))
        with torch.inference_mode():
            actual = loaded(frames, rppg)
        assert torch.allclose(expected, actual, atol=1e-6)
        assert loaded_cfg.frames_per_video == cfg.frames_per_video
        assert payload["temperature"] == 1.2

        bad_checkpoint = Path(temporary) / "bad_model.pt"
        save_checkpoint(
            model,
            cfg,
            bad_checkpoint,
            extra={"rppg_feature_order": ["wrong_order"]},
        )
        try:
            load_checkpoint(bad_checkpoint, torch.device("cpu"))
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched rPPG feature order must be rejected")

        frame_dir = Path(temporary) / "video" / "frames"
        frame_dir.mkdir(parents=True)
        for index in range(2):
            image = np.full((80, 80, 3), index * 30, dtype=np.uint8)
            cv2.imwrite(str(frame_dir / f"{index:03d}.jpg"), image)
        success = predict_video(
            "video.mp4",
            frames_dir=frame_dir,
            rppg_features=[0.0] * 20,
            checkpoint_path=checkpoint,
            device="cpu",
        )
        assert success["verdict"] in {"REAL", "FAKE"}
        assert 0.0 <= success["prob_real"] <= 1.0
        assert success["frames_used"] == 2

        result = predict_video(
            "missing.mp4",
            frames_dir=Path(temporary) / "does_not_exist",
            rppg_features=[0.0] * 20,
            checkpoint_path=checkpoint,
            device="cpu",
        )
        assert result["verdict"] == "INCONCLUSIVE"
        assert result["prob_real"] is None
        assert result["backend"] == "vit"

        missing = predict_video(
            "missing.mp4",
            frames_dir=frame_dir,
            rppg_features=[0.0] * 20,
            checkpoint_path=Path(temporary) / "missing.pt",
            device="cpu",
        )
        assert missing["verdict"] == "INCONCLUSIVE"
        assert missing["prob_real"] is None
        assert "checkpoint unavailable" in missing["reason"]


def test_evaluation_and_comparison_artifacts():
    with tempfile.TemporaryDirectory(dir=SCRAPE_ROOT) as temporary:
        root = Path(temporary)
        records: list[VideoRecord] = []
        for index in range(4):
            frame_dir = root / f"v{index}" / "frames"
            frame_dir.mkdir(parents=True)
            for frame in range(2):
                image = np.full((80, 80, 3), (index * 40 + frame * 10) % 255, dtype=np.uint8)
                cv2.imwrite(str(frame_dir / f"{frame:03d}.jpg"), image)
            markers = {
                "video_path": f"v{index}.mp4",
                "video_key": f"v{index}",
                "group": f"g{index}",
                "split": "val" if index < 2 else "test",
                "frames_dir": str(frame_dir.parent),
                "frame_paths": tuple(str(p) for p in sorted(frame_dir.glob("*.jpg"))),
                "rppg": tuple(float(index + j) for j in range(20)),
                "label": index % 2,
            }
            records.append(VideoRecord(**markers))

        output_dir = root / "vit"
        cfg = _cfg(
            temporal_head="attention_pool",
            use_rppg=True,
            output_dir=output_dir,
            batch_size=2,
        )
        model = VitClassifier(cfg).eval()
        payload = evaluate_video_level(
            model, records[2:], records[:2], cfg, torch.device("cpu")
        )
        artifacts = save_evaluation_artifacts(payload, output_dir, "vit")
        assert all(path.is_file() for path in artifacts.values())
        assert "auc_roc" in payload["test"]

        comparison = build_comparison_table(output_dir, quantum_dir=root / "quantum")
        assert "ViT+rPPG" in comparison["table_markdown"]
        assert (output_dir / "comparison.md").is_file()

        payload["use_rppg"] = False
        save_evaluation_artifacts(payload, output_dir, "vit")
        comparison = build_comparison_table(output_dir, quantum_dir=root / "quantum")
        assert comparison["rows"]["ViT+rPPG"] is None
        assert comparison["rows"]["ViT-only"] is not None


def test_config_guards():
    try:
        _cfg(temporal_head="mean_logits", use_rppg=True)
    except ValueError:
        pass
    else:
        raise AssertionError("mean_logits + rPPG must be rejected")
    assert _cfg(temporal_head="mean_logits", use_rppg=False).use_rppg is False
    try:
        _cfg(frames_per_video=2, min_frames=1)
    except ValueError:
        pass
    else:
        raise AssertionError("min_frames below frames_per_video must be rejected")
    assert _cfg(frames_per_video=2, min_frames=4).min_frames == 4


def main() -> int:
    tests = [
        test_labels_and_feature_order,
        test_frame_sampling_deterministic,
        test_cap_records_preserves_splits,
        test_zero_video_group_overlap,
        test_head_shapes,
        test_one_batch_overfit,
        test_stage_b_runs_after_stage_a_early_stop,
        test_checkpoint_round_trip_and_inconclusive,
        test_evaluation_and_comparison_artifacts,
        test_config_guards,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - self-check runner
            failures += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} checks passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
