# Deepfake Video Detection for KYC (rPPG + Visual + Quantum ML)

[![GitHubTree](https://img.shields.io/badge/GitHubTree-Major__project-blue?style=flat-square)](https://githubtree.mgks.dev/repo/NYN-05/Major_project/main/?ref=badge)
![Fused mode only](https://img.shields.io/badge/mode-fused%20rPPG%2Bvisual-2ea043?style=flat-square)
![DFDC only](https://img.shields.io/badge/dataset-DFDC-blue?style=flat-square)

Detection of deepfake videos for video KYC in financial systems using
**rPPG (remote photoplethysmography)** as the physiological evidence layer,
**visual features** (ResNet50 + handcrafted) as the complementary appearance layer,
and **hybrid quantum-classical ML**, a **pretrained ViT**, or a **DeepfakeBench
image detector** as the final decision layer, designed for low-resolution videos.

## Contents

- [Architecture (Four-Stage Pipeline)](#architecture-four-stage-pipeline)
- [Repository Layout](#repository-layout)
- [Current Performance (Fused Mode, DFDC Only)](#current-performance-fused-mode-dfdc-only)
- [Components](#components)
- [Install](#install)
- [Run End-to-End (Inference)](#run-end-to-end-inference)
- [Frontend (Web UI)](#frontend-web-ui)
- [Verified Constraints (do not break)](#verified-constraints-do-not-break)
- [Docs](#docs)
- [Team](#team)

## Architecture (Four-Stage Pipeline)

[![Architecture diagram](https://gitdiagram.com/diagram-badge.svg)](https://gitdiagram.com/nyn-05/major_project?utm_source=readme&utm_medium=badge)

[![Architecture diagram of nyn-05/major_project](https://gitdiagram.com/nyn-05/major_project/diagram.png)](https://gitdiagram.com/nyn-05/major_project?utm_source=readme&utm_medium=picture)

```text
Input KYC video
  → Stage 1: Frame sampling + quality filtering    (WORKING/frame/)
  → Stage 2: rPPG signal extraction + features     (WORKING/RPPG/)
  → Stage 3: Visual feature extraction             (WORKING/visual/)
  → Stage 4A: QAOA selection → hybrid VQC verdict  (WORKING/quantum/)   [default]
  → Stage 4B: ViT frame encoder → optional rPPG fusion (WORKING/vit/) [alternative]
  → Stage 4C: DeepfakeBench detector → optional rPPG fusion (WORKING/dfb_backend/) [alternative]
  → Decision: REAL / FAKE
```

End-to-end orchestrator (`WORKING/run_pipeline.py`):

```text
frames → rPPG → visual → quantum (fused) → verdict (REAL / FAKE)   [--backend quantum]
frames → rPPG → ViT (frames + optional rPPG fusion) → verdict      [--backend vit]
frames → rPPG → DeepfakeBench detector (+ optional fusion) → verdict [--backend dfb]
```

> Quantum inference supports **fused mode** (rPPG + visual) only. The ViT backend
> consumes accepted face frames and optionally the 20 rPPG features; its ViT-only
> ablation disables rPPG. The DFB backend consumes accepted face frames, uses a
> fine-tuned Xception checkpoint by default when present, and optionally fuses the
> 20 rPPG features.
> Only **DFDC dataset** is supported for training. No FaceForensics++ (FF++).

## Repository Layout

| Path | Description |
| ---- | ----------- |
| `WORKING/` | Active project root (all four pipeline stages + end-to-end runner) |
| `WORKING/frame/` | Stage 1 — frame sampling, YOLO face detection, quality filtering |
| `WORKING/RPPG/` | Stage 2 — MediaPipe face ROIs → POS/CHROM pulse → 20 physiological features |
| `WORKING/visual/` | Stage 3 — ResNet50 + handcrafted visual features from face crops (39 features) |
| `WORKING/quantum/` | Stage 4A — QAOA feature selection → hybrid VQC → P(real) → verdict (run via `python -m quantum.pipeline`) |
| `WORKING/vit/` | Stage 4B — pretrained ViT frame encoder + temporal head + optional rPPG fusion (run via `python -m vit.pipeline`) |
| `WORKING/dfb_backend/` | Stage 4C — DeepfakeBench Xception/UCF/EfficientNet-B4 detector, optional DFDC fine-tuning and rPPG fusion |
| `WORKING/run_pipeline.py` | End-to-end orchestrator with `--backend quantum` (default), `vit`, or `dfb` |
| `WORKING/MAJ_OUTPUT_ROOT` | Default global regenerated-artifacts root (all `MAJ_OUTPUT_ROOT` dirs untracked); configurable via `MAJ_OUTPUT_ROOT` env var (defaults to `Scrape/output/`) |
| `MAJ_OUTPUT_ROOT/frames/` | Stage 1: `frame_sequences/`, `frame_extraction_*.json(l)`, `docs/` |
| `MAJ_OUTPUT_ROOT/rppg/` | Stage 2: `dataset_features.csv`, `rppg_classifier.pkl` + metadata, plots |
| `MAJ_OUTPUT_ROOT/visual/` | Stage 3: `visual_features.csv`, `fused_features.csv` |
| `MAJ_OUTPUT_ROOT/quantum/` | Stage 4A: `data_fused.npz`, `qaoa_selection_fused.json`, `feature_scaler_fused.json`, `hybrid_vqc_fused.pt`, metrics, plots |
| `MAJ_OUTPUT_ROOT/vit/` | Stage 4B: `vit_model.pt` + JSON config, `vit_only.pt` ablation, metrics, ROC/confusion/calibration plots, comparison table |
| `MAJ_OUTPUT_ROOT/dfb/` | Stage 4C: scores/metrics per detector, fine-tuned checkpoint + JSON, `fusion.json`, ROC/confusion/calibration plots |
| `MAJ_OUTPUT_ROOT/pipeline/` | `run_pipeline.py` result JSON |
| `frontend/` | Web UI: React + Vite frontend + stdlib-only API server (port 8000) |
| `Docs/` | Project docs: RMTT semester report, key findings, remediation plan, problem analysis, bottleneck diagnosis |
| `Scrape/` | Dev scratch: tests, temp scripts, debug artifacts (gitignored) |

## Current Performance (Fused Mode, DFDC Only)

| Metric                    | VQC Test    | VQC 5-Fold CV   | Best Classical (LR) |
| ------------------------- | ----------- | --------------- | ------------------- |
| Accuracy                  | 0.552       | 0.556 ± 0.019   | TBD                 |
| **Balanced Accuracy**     | **0.499**   | **0.545 ± TBD** | **TBD**             |
| Specificity (FAKE recall) | **0.702**   | **TBD ± TBD**   | 0.525               |
| AUC-ROC                   | 0.535       | 0.556 ± 0.019   | 0.582               |
| Decision Bins             | REAL / FAKE | —               | —                   |

*Run `python -m quantum.pipeline --all` from `WORKING/` to regenerate metrics from trained artifacts.*

### DeepfakeBench Backend (DFDC Fine-Tuned)

| Metric | Xception Zero-Shot | Xception + DFDC FT | Image + rPPG Fusion |
| --- | ---: | ---: | ---: |
| AUC-ROC | 0.6072 | 0.9715 | 0.9707 |
| Balanced Accuracy | 0.6081 | 0.9057 | **0.9083** |
| EER | 0.4026 | 0.0937 | **0.0937** |
| ECE | 0.0726 | 0.1071 | **0.0302** |
| Brier | 0.2434 | 0.0806 | **0.0686** |

For reference, the strongest classical baseline on the exact same split and the
59 fused features is RandomForest: AUC 0.8693, balanced accuracy 0.7758, EER
0.2134. Test split: 544 videos (298 real / 246 fake). Thresholds are selected on
the 546-video validation split only. UCF and EfficientNet-B4 were evaluated
zero-shot (test AUC 0.5383 and 0.5491 respectively); see
[`Docs/dfb_report.md`](Docs/dfb_report.md) for the full methodology and ablations.

ViT metrics are intentionally not listed until a DFDC run is completed locally.
`python -m vit.pipeline --all` writes `MAJ_OUTPUT_ROOT/vit/metrics_vit.json`,
`metrics_vit_only.json`, and `comparison.md`; the comparison table also includes
the existing VQC and best classical rows when their metrics files are present.

## Components

### 1. Frame Sampling & Quality (`WORKING/frame/`)

Modular face pipeline for video sources: frame sampling at a controlled
FPS, face detection (YOLO), quality assessment (blur / dark / overexposed / no-face / face-too-small / extreme-pose rules).
**Quality thresholds relaxed** for short/low-quality videos: blur 3.0, dark 30, bright 235, min face area 0.002, min seq len 32, conf 0.25.
Outputs accepted JPEGs + per-frame metadata JSONL for the rPPG stage.

```bash
# Frame extraction CLI (from WORKING/)
python -m frame.pipeline --source path/to/video.mp4 --input-dir path/to/videos/
```

(`python -m frame.pipeline` runs the extraction CLI; `run_pipeline.py` calls
`frame.pipeline.run_frame_sampling_quality_layer()` directly for stage 1.)

### 2. rPPG Pipeline (`WORKING/RPPG/`)

POS/CHROM pulse reconstruction from facial ROIs (left cheek, right cheek, forehead)
and a **20-feature** physiological vector per video:
heart rate, SNR, PRV, spectral entropy, MAD, signal quality index,
inter-region correlations, pulse morphology (peak width, dicrotic notch),
inter-ROI phase lag / pulse-transit-time proxy, motion contamination,
spectral/statistical probes (spectral flatness, centroid, kurtosis, phase coherence).
**Quality thresholds relaxed** for short/low-quality videos: min usable frames 24, min SQI 0.05.
Features are persisted with labels (1 = fake, 0 = real) in
`WORKING/MAJ_OUTPUT_ROOT/rppg/dataset_features.csv`, the direct data source for the quantum layer.

```bash
# From WORKING/ - rebuild MAJ_OUTPUT_ROOT/rppg/dataset_features.csv
#   --workers 0          = all CPU cores
#   --gpu                = GPU-accelerated face detection (YuNet ONNX CUDA)
python RPPG/extract_dataset_features.py --workers 0 --gpu

# Train rPPG classifier (side path, not used for final verdict)
python RPPG/train_classifier.py
```

Feature source: **DFDC only** (`archive/DFDC_Dataset` via `DFDC_DATASET_PATH` env var).
No FaceForensics++ (FF++) support.

### 3. Visual Pipeline (`WORKING/visual/`)

ResNet50 (ImageNet pre-trained) + handcrafted features from face crops saved by stage 1.
Extracts **39 features**: 16 deep (PCA-reduced ResNet50 GAP) + 10 LBP + 4 GLCM texture + 6 color + 3 DCT frequency.
Fuses with rPPG features to create the **fused feature set** (59 features) used for training.

```bash
# From WORKING/ - extract visual features from stage-1 frames
python visual/pipeline.py --frames-root MAJ_OUTPUT_ROOT/frames/frame_sequences --rppg-csv MAJ_OUTPUT_ROOT/rppg/dataset_features.csv --fuse --create-splits
```

### 4. Quantum Model (`WORKING/quantum/`)

Hybrid classical-quantum decision stage built with PennyLane + PyTorch. It consumes
the **fused (rPPG + visual) 59-feature vector directly** (same names/order as `RPPGFeatures.feature_names()` + `VISUAL_FEATURE_NAMES`);
no synthetic data is generated anywhere in the pipeline.

- **Data** — `data.py` builds `MAJ_OUTPUT_ROOT/quantum/data_fused.npz` from the real labeled fused feature table `MAJ_OUTPUT_ROOT/visual/fused_features.csv`.
  rPPG label `1 = fake` is flipped via the explicit, tested conversion
  `csv_to_quantum_label()` to the quantum convention `LABEL_REAL = 1, LABEL_FAKE = 0`
  (Phase 1A of the remediation plan). HR-plausibility filter (30–220 BPM, non-finite rejection)
  drops implausible rows at build time. Subject-grouped random split (seeded, per-clip for DFDC),
  persisted to `split_manifest_fused.json`.
- **QAOA feature selection** — `qaoa.py` selects the most informative fused features using
  a cost Hamiltonian with **supervised discrimination weights** (`qaoa._discrimination_weights`:
  sign-agnostic exact Mann-Whitney AUC strength `2*|AUC-0.5|`, deterministic, no sklearn)
  plus correlation redundancy penalty and cardinality constraint, optimized with COBYLA.
  8 parallel restarts via `ProcessPoolExecutor`. The circuit runs on the project's own
  **exact torch-native statevector simulator** (`qaoa_sim.QAOASimulator`, complex128;
  ~0.3–0.5 ms vs ~5.6 s per PennyLane call) and is cross-verified
  against PennyLane by the test suite.
- **Hybrid VQC** — `vqc.py`: angle-encoded features into a parameterized quantum
  circuit (`StronglyEntanglingLayers`) feeding a small classical head. The default
  quantum layer is the exact torch-native `QuantumLayerTorch` simulator (complex128,
  batched, same `weights` shape as the PennyLane layer so checkpoints are
  interchangeable); the legacy PennyLane QNode path remains behind `qnode_impl="pennylane"`
  for cross-verification. Training uses class-balanced focal loss (Phase 1B), cosine-annealed
  LR, gradient clipping, and early stopping on validation loss with restore of the
  best-validation checkpoint. CUDA head when available.
- **Evaluation** — `evaluation.py`: accuracy / precision / recall / F1 / AUC-ROC / ECE,
  KYC-friendly decision bins (real / fake), ROC, confusion-matrix and
  calibration-curve plots. **StratifiedKFold (5-fold) cross-validation** (mean ± std,
  balanced accuracy) for the VQC and every baseline. Classical baselines:
  RandomForest, MLP, LogisticRegression, calibrated LinearSVC, GaussianNB, XGBoost.
- **Orchestration** — `pipeline.py` drives the full training flow and exposes
  `predict_features()` as the single inference entry point used by `run_pipeline.py`.
  Hard-asserts Hamiltonian ≡ classical cost (`error < 1e-6`).

```bash
# From WORKING/: build data, QAOA selection, train VQC, evaluate, baselines (fused mode only)
python -m quantum.pipeline --all
# Self-checks (10/10): beta-alive ansatz, Hamiltonian≡classical cost,
# feature-contract sync, split determinism, grouping, sim cross-verification,
# checkpoint compat, label conversion
python -m quantum.tests
```

### 5. ViT Alternative Backend (`WORKING/vit/`)

A timm pretrained ViT is an alternative decision stage. It consumes
`frames_per_video` accepted face frames per video (default 16; cropped faces are
preferred when present, otherwise the stage-1 accepted frames are used), encodes
each frame, aggregates the frame embeddings with an attention-pooled or small
Transformer temporal head, and optionally fuses the 20 rPPG features through a
small MLP.

- **Backbones** — `vit_small_patch16_224.augreg_in21k_ft_in1k` (default),
  `vit_base_patch16_224`, `deit_small_patch16_224`, `swin_tiny_patch4_window7_224`.
- **Temporal heads** — `attention_pool` (default), `transformer`, `mean_logits`
  (ViT-only; requires rPPG fusion to be disabled).
- **Two-stage training** — Stage A freezes the backbone and trains the temporal +
  fusion heads; Stage B unfreezes the last 4 blocks with layer-wise LR decay.
  Uses AdamW, mixed precision, gradient clipping, balanced sampling/positive
  weighting, and early stopping on validation AUC.
- **Calibration** — one scalar temperature and the decision threshold are fitted
  on validation videos only, then applied unchanged to the test split.
- **Evaluation** — video-level accuracy, balanced accuracy, precision, recall,
  F1, specificity, AUC-ROC, ECE, ROC/confusion/calibration plots, and a
  VQC/classical/ViT comparison table.
- **Splits** — reuses `MAJ_OUTPUT_ROOT/quantum/split_manifest_fused.json`, so the
  ViT and VQC use identical video/group splits. Optional 5-fold grouped CV keeps
  every video/group in exactly one fold.

```bash
# From WORKING/ - requires frames, rPPG CSV, and the fused split manifest
python -m vit.pipeline --train
python -m vit.pipeline --evaluate
python -m vit.pipeline --all            # fused ViT + ViT-only (mean logits) + comparison
python -m vit.pipeline --cross-validate # grouped 5-fold CV
python -m vit.tests                     # self-checks (no dataset required)
python -m pytest -q ..\tests\vit        # tracked pytest entry points

# Deterministic smoke subset (still requires real aligned data)
python -m vit.pipeline --all --max-videos 50
```

Checkpoints are `state_dict` files with a sidecar JSON config and are loaded with
`weights_only=True`; no pickle is introduced. A missing checkpoint or less than
`frames_per_video` accepted frames yields `INCONCLUSIVE` (exit code 3), never a
fabricated probability.

### 6. DeepfakeBench Alternative Backend (`WORKING/dfb_backend/`)

A real image-based detector from DeepfakeBench replaces the VQC decision stage.
The wrapper imports the upstream Xception, UCF, or EfficientNet-B4 detector,
its YAML preprocessing config, and the FF++ checkpoint, then reproduces the DFB
test transform on the Stage-1 accepted frames (RGB, `INTER_CUBIC`, /255,
0.5/0.5/0.5 normalization). Sixteen frames per video are scored and aggregated
to a video-level `P(real)`.

- **Checkpoints** — `xception_best.pth`, `ucf_best.pth`, `effnb4_best.pth`
  from `WORKING/third_party/DeepfakeBench/training/weights/`; all load strictly.
- **DFDC fine-tuning** — optional Xception fine-tune (6 epochs, AdamW, AMP,
  frozen backbone for epoch 1, balanced sampling); best checkpoint saved as
  `MAJ_OUTPUT_ROOT/dfb/xception_finetuned.pt` with a JSON sidecar.
- **Fusion** — image logit + 20 rPPG features through balanced logistic
  regression; temperature and `C` are fitted on validation only. The deployed
  REAL cutoff is selected conservatively on validation data to cap fake-video
  false acceptance at 1%, rather than using a balanced Youden cutoff that can
  be unsafe for KYC.
- **Inference** — uses the fine-tuned checkpoint when present, falling back to
  the upstream zero-shot checkpoint otherwise. Deployment scores 32 evenly
  distributed accepted frames and uses the highest-scoring 25% (`topk`)
  instead of averaging away short deepfake evidence. rPPG fusion is supporting
  evidence only and cannot increase the visual DFB REAL probability. Too few
  frames yields `INCONCLUSIVE` (exit code 3), never a fabricated probability.
- **Security** — project-owned checkpoints and fusion artifacts are plain
  state dictionaries/JSON loaded with `weights_only=True`; upstream detector
  constructors load pretrained backbones with `weights_only=False` (upstream
  behavior, trusted checkpoints only).

```powershell
# From WORKING/
python -m dfb_backend.tests
python -m dfb_backend.evaluation --zero-shot --detectors xception ucf efficientnetb4
python -m dfb_backend.evaluation --finetune --detectors xception --epochs 6
python -m dfb_backend.evaluation --fuse --detector xception --mode finetuned
python run_pipeline.py --backend dfb --source path\to\video.mp4 --detector xception
```

DeepfakeBench is CC BY-NC 4.0 (non-commercial). See
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and
[`Docs/dfb_report.md`](Docs/dfb_report.md). The upstream clone is pinned at
`f188b1c105465e2e5377eb536a95022ae0e4522d`; see
`WORKING/third_party/DeepfakeBench.commit`.

## Install

```bash
# All Python stages (merged file: frame + RPPG + quantum + shared deps)
pip install -r requirements.txt

# Stage 3 (visual) - torchvision, scikit-image for handcrafted features
pip install torchvision scikit-image

# Stage 4 (quantum) - PennyLane, PyTorch, scikit-learn, scipy, matplotlib
pip install pennylane torch numpy scikit-learn scipy matplotlib

# Stage 4B (ViT) - timm (uses the installed PyTorch)
pip install timm

# Stage 4C (DeepfakeBench) - clone the pinned upstream repo, then install extras
git clone https://github.com/SCLBD/DeepfakeBench.git WORKING/third_party/DeepfakeBench
git -C WORKING/third_party/DeepfakeBench checkout f188b1c105465e2e5377eb536a95022ae0e4522d
pip install -r requirements-dfb.txt

# Frontend (optional)
cd frontend && npm install
```

## Run End-to-End (Inference)

```bash
# From WORKING/
python run_pipeline.py --source path/to/video.mp4 --method POS --out result.json
python run_pipeline.py --backend vit --source path/to/video.mp4 --out vit_result.json
python run_pipeline.py --backend dfb --source path/to/video.mp4 --detector xception --out dfb_result.json
```

The ViT backend requires `MAJ_OUTPUT_ROOT/vit/vit_model.pt` and its sidecar
`vit_model.json` (created by `python -m vit.pipeline --train` or `--all`).

Requires pre-trained quantum artifacts for **fused mode** (`MAJ_OUTPUT_ROOT/quantum/qaoa_selection_fused.json`, `hybrid_vqc_fused.pt`, `feature_scaler_fused.json`). If missing, run once from `WORKING/`:

```bash
# Build fused training data first (requires DFDC_DATASET_PATH env var)
export DFDC_DATASET_PATH=/path/to/DFDC_Dataset
python RPPG/extract_dataset_features.py --workers 0 --gpu
python visual/pipeline.py --frames-root MAJ_OUTPUT_ROOT/frames/frame_sequences --rppg-csv MAJ_OUTPUT_ROOT/rppg/dataset_features.csv --fuse --create-splits
# Then train quantum model
python -m quantum.pipeline --all
```

## Frontend (Web UI)

```bash
# From frontend/
python server.py          # Backend API on http://127.0.0.1:8000
npm run dev               # React dev server on http://localhost:5173 (proxies /api)
```

Upload a KYC video, watch the pipeline run live via SSE,
and read the verdict with its evidence dossier (accepted frames, physiological
features, backend probabilities and plots, the pulse waveform).

### API Contract (`frontend/server.py`)

```text
POST /api/detect              upload video (raw body + X-Filename + X-Backend: quantum|vit|dfb) → {job: <id>}
GET  /api/jobs/<id>           {done, error, video, signal, lines[-400:], result}
GET  /api/jobs/<id>/events    SSE: line / stage / result / signal / error
GET  /api/previous            last canonical pipeline result (with _signal)
GET  /api/health              {ok, platform, running, has_previous, sequences, artifacts}
GET  /api/artifacts?dir=      list files under MAJ_OUTPUT_ROOT/<rel>
GET  /api/files?rel=          serve artifact file (path traversal protected)
```

Upload limits: 200 MB max; magic-byte validation (MP4/MOV ftyp, AVI RIFF, WebM EBML).
Concurrency: max 2 jobs (429 on overflow). Jobs TTL: 1 h; frame sequences: 24 h.
CORS restricted to localhost origins. 30-min hard timeout per pipeline run.

### Frontend State Machine

```text
idle → selected (preview + metadata + Start button)
     → running  (7-stage pipeline + live panel + creeping progress)
     → done     (Verdict gauge, Insights, Signal canvas, Quantum flow, FrameSamples)
     → error    (inline banner)
```

Theme toggle persists `rppgqc.theme` in localStorage; respects `prefers-reduced-motion`. Responsive: 1920 → 375 px, no overflow.

## Verified Constraints (do not break)

> **No synthetic data.** The quantum layer consumes only the real fused feature table `MAJ_OUTPUT_ROOT/visual/fused_features.csv`. Never reintroduce a generator or a transform/bridge layer.

> **Feature contract:** `FUSED_FEATURE_NAMES` in `quantum/config.py` (59 features = 20 rPPG + 39 visual) must stay identical in name AND order to `RPPGFeatures.feature_names()` + `VISUAL_FEATURE_NAMES`. Keep the two lists in sync; `test_feature_contract_sync` guards it.

- **Only fused mode supported.** No `rppg_only`, `visual_only`, `rppg_base`, `rppg_cross_roi`, or ablation feature sets.
- **Only DFDC is used for training/evaluation data.** No FF++ videos or annotations are added. The DeepfakeBench backend may initialize from the published FF++ checkpoints; they are pretrained weights, not project training data. `DFDC_DATASET_PATH` must point to `archive/DFDC_Dataset`.
- **Label conventions differ per stage — do not unify:**
  - rPPG CSV: `1 = fake, 0 = real`
  - quantum: `LABEL_REAL = 1`, `LABEL_FAKE = 0`; `quantum/data.py` flips via the tested `csv_to_quantum_label()` (`y = 1 - csv_label` lives only inside that function)
  - rPPG RandomForest cross-check in `run_pipeline.py`: `1 = DEEPFAKE`
- **Gitignored artifacts:** `*.csv`, `*.json`, `*.pkl`, `*.mp4`, and all `MAJ_OUTPUT_ROOT` dirs are untracked — `dataset_features.csv`, `MAJ_OUTPUT_ROOT/visual/fused_features.csv`, `MAJ_OUTPUT_ROOT/quantum/*`, and the trained models will not appear in `git status`. Regenerating them is normal.
- **rPPG returns `features=None`** when usable frames < `min_usable_frames` (24); `run_pipeline.py` then emits an **INCONCLUSIVE** verdict (confidence `null`) and exits **3** per the AGENTS.md contract. New code must handle `None` and exit code 3.
- **Stage 1 feeds stage 2 & 3.** `run_pipeline.py` hands the frame stage's accepted JPEGs (`MAJ_OUTPUT_ROOT/frames/frame_sequences/<video>/frames/`) plus `frame_metadata.jsonl` to `RPPGPipeline.process_frames()` at the stage-1 sample rate (30 fps) and to `visual.extractor.compute_visual_features()` for face crops. rPPG no longer re-gates on blur/brightness (stage 1 did) but still runs MediaPipe per frame; `features=None` handling emits INCONCLUSIVE and exits 3 (same contract as above). If stage 1 fails or yields no frames, `run_pipeline.py` falls back to `RPPGPipeline.process_video()` (direct video read; `input_mode` in the result JSON records which path ran). Standalone RPPG scripts keep using `process_video`. `run_pipeline.py --signal-out <path>` writes the decimated stage-2 waveform JSON (same schema as the retired standalone `frontend/dump_signal.py`).
- **RandomForest cross-check** is an optional side path; the final verdict comes from the selected backend (`quantum` by default, `vit` with `--backend vit`, or `dfb` with `--backend dfb`).
- **rPPG classifier trust:** `MAJ_OUTPUT_ROOT/rppg/` must remain write-protected; `rppg_classifier.pkl` is `pickle.load`-ed by `run_pipeline.py` (arbitrary-code risk if replaced). Never move to shared hosting as-is.
- **QAOA selection weights:** the selection objective uses supervised discrimination weights (`qaoa._discrimination_weights`: sign-agnostic exact Mann-Whitney AUC strength `2*|AUC-0.5|`, deterministic, no sklearn). `QAOASelectionConfig.target_features` is 3. Keep `_mutual_info_weights` only as a documented alternative — do not reintroduce it into `select()`.
- **QAOA ansatz:** `_apply_qaoa` applies precomputed cost gates with `gamma` and X-mixer gates with `beta` separately. Regression guard: `test_beta_alive` in `quantum/tests.py`. Any refactor must keep beta alive and re-run `python -m quantum.tests` + `--all`.
- **Hamiltonian ≡ classical cost:** `_cost_terms` reproduces `_classical_cost` exactly (verified ~1e-14 on real data) and `pipeline.py` hard-asserts `error < 1e-6`.
- **Torch-native simulators (default):** the default QAOA and VQC circuit backends are the project's exact complex128 statevector simulators (`qaoa_sim.QAOASimulator`, `vqc.QuantumLayerTorch`) — ~20× faster than PennyLane and CPU-process-safe. PennyLane paths remain behind `device="pennylane"` / `qnode_impl="pennylane"` for cross-verification; `test_qaoa_sim_matches_pennylane` and `test_torch_layer_matches_pennylane` pin them to ≤1e-6/1e-5.
- **rPPG needs MediaPipe Face Landmarker**; the model auto-downloads on first run (internet required). Extraction can alternatively use GPU-accelerated YuNet face detection (`--gpu`). In `frame/`, only `yolov8n-face-lindevs.pt` auto-downloads; missing other presets raise `FileNotFoundError`.

## Docs

- `Docs/maj_proj_report_sem_6/SEM6_report/Final_report_draft_V3.pdf` — RMTT semester project report (historical snapshot)
- `Docs/maj_proj_report_sem_6/research_paper/` — 23+ research papers on deepfake detection, QML, and rPPG
- `Docs/charts/` — 5 performance comparison charts
- `Docs/QML/M1-M5.pdf` — 5 Quantum Machine Learning module PDFs
- `Docs/vit_report.md` — ViT backend architecture, training, artifacts, and evidence status
- `Docs/dfb_report.md` — DeepfakeBench backend, DFDC fine-tuning, fusion, and ablations
- `THIRD_PARTY_NOTICES.md` — DeepfakeBench attribution and CC BY-NC 4.0 notice
- `changes.md` — historical development log (Phases 1–10, superseded)

## Team

| Member    | Responsibility                                                        |
| --------- | --------------------------------------------------------------------- |
| **Jhashank** | rPPG-to-classifier pipeline, quantum model (Stage 3)               |
| **Sumit**    | Input-to-ROI pipeline (data, frames, face detection) (Stage 1)     |
| **Aswin**    | Preprocessing policy, evaluation, integration (Stage 2)            |
# pretrained-mode--deepfake-model
