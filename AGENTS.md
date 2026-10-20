# Repository Guide

## Project

This repository implements deepfake-video detection for KYC using:

- rPPG physiological evidence
- visual features from ResNet50 and handcrafted descriptors
- hybrid quantum-classical machine learning with PennyLane and PyTorch, an
  alternative pretrained-ViT decision backend, or a DeepfakeBench image detector,
  each with optional rPPG fusion
- two decision outcomes: `REAL` and `FAKE`

Active application code is located in [`WORKING/`](WORKING/) and
[`frontend/`](frontend/). The root contains documentation, PDFs, and
[`Scrape/`](Scrape/), the required location for development-only files.

## Mandatory Change Log

After every code change:

1. Add an entry to [`changes.md`](changes.md).
2. Include the date, time, affected area, and a short description.
3. Record the entry in the same work session; never defer it.

Use this format:

```text
| YYYY-MM-DD HH:mm IST | Area | Short description of the change. | Complete / Experimental / Retired |
```

## Repository Structure

```text
WORKING/
  frame/              Frame sampling, face detection, and quality filtering
  RPPG/               MediaPipe ROIs, POS/CHROM signals, and rPPG features
  visual/             ResNet50 and handcrafted visual features
  quantum/            QAOA selection, VQC training, and inference
  vit/                Pretrained ViT backend with optional rPPG fusion
  dfb_backend/        DeepfakeBench detector backend, DFDC fine-tuning, and fusion
  third_party/        Pinned DeepfakeBench clone, commit pin, and upstream license
  run_pipeline.py     End-to-end frame → rPPG → (visual → quantum | ViT | DFB) pipeline
  output/ (MAJ_OUTPUT_ROOT)  Regenerated, gitignored artifacts

frontend/
  server.py           Standard-library API, uploads, jobs, SSE, and artifacts
  src/                React + Vite frontend

Scrape/               Required location for tests, probes, logs, screenshots,
                      temporary scripts, and other development artifacts
```

## Development Rules

- Work from `WORKING/` for pipeline, quantum, and ViT commands.
- Work from `frontend/` for frontend commands.
- Do not run quantum commands from the repository root.
- Do not create temporary or test files outside [`Scrape/`](Scrape/).
- Do not commit generated artifacts, datasets, videos, models, credentials, or
  other secrets.
- Preserve existing behavior unless the requested change explicitly changes it.
- Prefer small, typed, testable changes that follow existing project patterns.
- Do not add synthetic data, bridge transforms, or unsupported datasets.

## Supported Architecture

- The default quantum backend supports **fused mode only**: rPPG plus visual features.
- The optional ViT backend consumes accepted face frames plus the existing 20
  rPPG features; `mean_logits` with rPPG disabled is the ViT-only ablation.
- The optional DFB backend consumes accepted face frames through the upstream
  DeepfakeBench Xception/UCF/EfficientNet-B4 detector. It uses the DFDC
  fine-tuned checkpoint when present, otherwise the upstream zero-shot
  checkpoint, and optionally fuses the 20 rPPG features.
- The supported dataset is **DFDC only**.
- FF++ videos/annotations must not be added. The DFB backend may initialize from
  the published FF++ checkpoints because they are pretrained weights, not
  project training data.
- The fused feature contract is defined by `FUSED_FEATURE_NAMES` in
  `WORKING/quantum/config.py`.
- The feature names and order must match the combined rPPG and visual feature
  definitions. Run `python -m quantum.tests` after changing this contract.
- The final verdict comes from the selected backend (`quantum` by default, `vit`
  or `dfb` when requested). The RandomForest result is an optional cross-check only.
- ViT and VQC must reuse `split_manifest_fused.json`; video/group overlap between
  train, validation, and test is a hard failure.
- ViT and DFB fine-tuned checkpoints are state dictionaries with JSON sidecars
  loaded using `weights_only=True`; do not introduce pickle-based checkpoints.
- DFB fusion artifacts (`dfb/fusion.json`) store scaler parameters and logistic
  regression coefficients as JSON; do not switch to pickle.
- Upstream DeepfakeBench detector constructors load pretrained backbones with
  `torch.load(..., weights_only=False)`. Only load trusted checkpoints and keep
  that upstream behavior documented in `THIRD_PARTY_NOTICES.md`.
- DFB thresholds and fusion parameters must be fitted on validation data only.
- The DFB backend is exposed through `run_pipeline.py --backend dfb` and through
  the frontend API/UI (`X-Backend: dfb`). The UI defaults to DFB because the
  local VQC artifacts are not trained in the active output root.
- If rPPG returns `features=None` because too few frames are usable, the
  pipeline must return an inconclusive result and exit with status 3.
- Stage 1 accepted frames feed both the rPPG and visual stages. If frame
  extraction fails or produces no frames, the pipeline may fall back to direct
  video processing.

## Data and Label Contracts

- `DFDC_DATASET_PATH` must point to `archive/DFDC_Dataset`.
- rPPG CSV labels use `1 = fake` and `0 = real`.
- Quantum labels use `LABEL_REAL = 1` and `LABEL_FAKE = 0`.
- The label conversion belongs only in
  `WORKING/quantum/data.py:csv_to_quantum_label()`.
- The rPPG RandomForest cross-check uses `1 = DEEPFAKE`.
- Do not unify these conventions without updating all dependent tests and
  documentation.

## Security Constraints

- Treat `MAJ_OUTPUT_ROOT/rppg/rppg_classifier.pkl` as untrusted serialized code.
- Keep `MAJ_OUTPUT_ROOT/rppg/` write-protected during inference.
- Never move the pickle model to shared hosting without a replacement strategy.
- Keep upload validation, size limits, path-traversal protection, CORS
  restrictions, concurrency limits, and job timeouts intact.
- Never commit API keys, passwords, tokens, private datasets, or credentials.

## Commands

Run these from `WORKING/`:

```powershell
# Full fused quantum pipeline
python -m quantum.pipeline --all

# Regression and contract tests
python -m quantum.tests

# ViT self-checks (no dataset required) and training/evaluation
python -m vit.tests
python -m pytest -q ..\tests\vit
python -m vit.pipeline --train
python -m vit.pipeline --all

# End-to-end inference (quantum default, ViT/DFB alternatives)
python run_pipeline.py --source path\to\video.mp4 --method POS --out result.json
python run_pipeline.py --backend vit --source path\to\video.mp4 --out vit_result.json
python run_pipeline.py --backend dfb --source path\to\video.mp4 --detector xception --out dfb_result.json

# DFB self-checks and evaluation
python -m dfb_backend.tests
python -m dfb_backend.evaluation --zero-shot --detectors xception ucf efficientnetb4
python -m dfb_backend.evaluation --finetune --detectors xception --epochs 6
python -m dfb_backend.evaluation --fuse --detector xception --mode finetuned

# Rebuild rPPG training features
python RPPG\extract_dataset_features.py

# Extract visual features and create the fused table
python visual\pipeline.py --frames-root output\frames\frame_sequences `
  --rppg-csv output\rppg\dataset_features.csv --fuse --create-splits
```

Run these from `frontend/`:

```powershell
# Start the API
python server.py

# Start the development frontend
npm run dev

# Build the frontend
npm run build
```

## Validation

Use the smallest relevant validation command for each change:

- Quantum configuration, QAOA, VQC, labels, or feature changes:
  `python -m quantum.tests`
- Full quantum or artifact-generation changes:
  `python -m quantum.pipeline --all`
- ViT model, dataset, calibration, or checkpoint changes:
  `python -m vit.tests` and `python -m pytest -q ..\tests\vit`
- DFB loader, preprocessing, prediction, fine-tuning, or fusion changes:
  `python -m dfb_backend.tests`; use `DFB_RUN_MODEL_TESTS=1` for strict checkpoint
  loading and an end-to-end `run_pipeline.py --backend dfb` inference test
- Frontend changes: `npm run build`
- API or pipeline changes: run the relevant Python self-checks and an
  end-to-end inference test when representative input is available.

Keep tests and generated validation files in [`Scrape/`](Scrape/) unless the
project already defines a tracked test location.

## API Contract

The API in `frontend/server.py` provides:

```text
POST /api/detect                  (optional X-Backend: quantum | vit | dfb, default quantum)
GET  /api/jobs/<id>
GET  /api/jobs/<id>/events
GET  /api/previous
GET  /api/health
GET  /api/artifacts?dir=<relative-output-directory>
GET  /api/files?rel=<relative-artifact-path>
```

Important behavior:

- Uploads are limited to 200 MB and validated by file signature.
- At most two pipeline jobs run concurrently.
- Job records expire after one hour; frame sequences expire after 24 hours.
- Artifact paths must remain traversal-protected.
- Pipeline progress and the generated signal are delivered through the job
  result/SSE flow.

## Documentation

- Record all changes in [`changes.md`](changes.md).
- Keep detailed technical findings in the relevant documentation under
  [`Docs/`](Docs/). The DFB backend is documented in `Docs/dfb_report.md`.
- Treat this file as the active repository rulebook; remove stale guidance
  when the architecture changes.
