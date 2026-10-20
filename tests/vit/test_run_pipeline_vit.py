"""Integration test for the run_pipeline.py --backend vit branch."""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRAPE_ROOT = REPO_ROOT / "Scrape"
SCRAPE_ROOT.mkdir(parents=True, exist_ok=True)


def test_run_pipeline_vit_branch(monkeypatch):
    import run_pipeline

    with tempfile.TemporaryDirectory(dir=SCRAPE_ROOT) as temporary:
        root = Path(temporary)
        video = root / "clip.mp4"
        video.write_bytes(b"not-a-real-video")
        out_json = root / "result.json"
        frames_dir = root / "frames"
        frames_dir.mkdir()

        def fake_frames_stage(_video_path):
            stats = {
                "sampled_frames": 30,
                "accepted_frames": 20,
                "rejections": {},
                "mean_quality_score": 0.8,
                "mean_face_confidence": 0.9,
                "temporal_coverage_ratio": 0.66,
            }
            return {"status": "success"}, stats, {"frames_dir": str(frames_dir)}

        def fake_rppg_stage(_video_path, _method="POS", _handoff=None):
            payload = {
                "method": "POS",
                "input_mode": "stage1_frames",
                "fps_used": 10.0,
                "n_frames_total": 20,
                "n_frames_usable": 20,
                "warnings": [],
                "features": {"heart_rate_bpm": 72.0, "signal_quality_index": 0.8},
            }
            return payload, [0.0] * 20, None

        def fake_predict_video(*_args, **_kwargs):
            return {
                "backend": "vit",
                "prob_real": 0.91,
                "verdict": "REAL",
                "confidence": 0.82,
                "reason": None,
            }

        monkeypatch.setattr(run_pipeline, "OUTPUT_ROOT", root)
        monkeypatch.setattr(run_pipeline, "run_frames_stage", fake_frames_stage)
        monkeypatch.setattr(run_pipeline, "run_rppg_stage", fake_rppg_stage)
        monkeypatch.setattr("vit.pipeline.predict_video", fake_predict_video)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "run_pipeline.py",
                "--backend",
                "vit",
                "--source",
                str(video),
                "--out",
                str(out_json),
            ],
        )

        exit_code = run_pipeline.main()
        assert exit_code == 0
        payload = json.loads(out_json.read_text(encoding="utf-8"))
        assert payload["backend"] == "vit"
        assert payload["stages"]["vit"]["verdict"] == "REAL"
        assert payload["stages"]["visual"]["status"] == "skipped"
        assert payload["verdict"]["label"] == "REAL"
        assert payload["diagnostics"]["backend"] == "vit"
