# ViT Decision Backend

## Purpose

The ViT backend is an alternative to the QAOA + hybrid-VQC decision stage. It
answers the same video-level question using a pretrained image transformer over
accepted face frames, with an optional rPPG fusion branch. The quantum backend
remains the default; no quantum or rPPG feature contract was changed.

## Architecture

```text
accepted frames (N per video)
  -> timm ViT / DeiT / Swin backbone (per-frame embedding)
  -> temporal head: attention pooling | 1-2 layer Transformer | mean logits
  -> optional fusion with 20 rPPG features (standardised, 64 -> 32 MLP)
  -> MLP classifier -> logit -> P(real)
```

- Backbones: `vit_small_patch16_224.augreg_in21k_ft_in1k` (default),
  `vit_base_patch16_224`, `deit_small_patch16_224`,
  `swin_tiny_patch4_window7_224`.
- `mean_logits` averages per-frame logits and is only valid when rPPG fusion is
  disabled. It is the ViT-only ablation.
- rPPG inputs use `RPPGFeatures.feature_names()` order and are asserted equal to
  `quantum.config.RPPG_FEATURE_NAMES` (20 features).
- Labels use `quantum.data.csv_to_quantum_label()` only (`1 = REAL`, `0 = FAKE`).

## Data and splits

- Dataset: DFDC only.
- Data sources: stage-1 accepted frames, `dataset_features.csv`, and
  `split_manifest_fused.json`. The ViT stage loads the manifest built by the
  quantum/visual pipeline, so both backends use the same video and subject
  groups.
- Frame preference: `<video>/cropped_faces/*.jpg`, then `<video>/frames/*.jpg`.
- Alignment: exact path first, then unique video filename stem. Duplicate stems
  are ambiguous and are not matched by stem.
- Split assertions reject any video or group overlap between train, validation,
  and test.
- Optional `--max-videos` capping is deterministic and preserves split
  proportions. It is intended for smoke runs, not reporting.

## Training

| Setting | Default |
| --- | --- |
| Frames per video | 16 |
| Image size | 224 |
| Batch size | 8 |
| Stage A | 4 epochs, backbone frozen |
| Stage B | 20 epochs, last 4 blocks unfrozen |
| Optimizer | AdamW |
| Backbone LR | `1e-5` |
| Head LR | `1e-3` |
| Weight decay | `0.05` |
| Layer decay | `0.75` |
| Warmup | 10% of stage steps |
| Gradient clipping | `1.0` |
| Mixed precision | enabled on CUDA |
| Loss | BCE with positive weighting; focal available |
| Sampling | balanced sampler by default |
| Early stopping | validation AUC, patience 6 |
| Seed | 42 |

Training-only augmentation targets low-resolution KYC video: downscale/upscale,
JPEG compression, blur, noise, horizontal flip, colour jitter, and light crop
jitter. Validation and inference use deterministic resize and ImageNet
normalisation only.

## Calibration and metrics

1. Fit a scalar temperature on validation logits by minimising NLL.
2. Choose the decision threshold on validation probabilities (Youden J by
   default).
3. Freeze both values and evaluate the held-out test split.

Reported test metrics: accuracy, balanced accuracy, precision, recall, F1,
specificity, AUC-ROC, ECE, and KYC decision bins. ROC, confusion-matrix, and
calibration plots are written beside the metrics JSON. `--cross-validate` runs
subject-grouped folds and reserves an inner subject-grouped validation split for
calibration, keeping each test fold untouched.

## Commands

Run from `WORKING/`:

```powershell
python -m vit.tests
python -m vit.pipeline --train
python -m vit.pipeline --evaluate
python -m vit.pipeline --all
python -m vit.pipeline --cross-validate
python -m vit.pipeline --all --max-videos 50
```

End-to-end inference:

```powershell
python run_pipeline.py --backend vit --source path\to\video.mp4 --out result.json
```

## Artifacts

Under `MAJ_OUTPUT_ROOT/vit/`:

- `vit_model.pt` and `vit_model.json` - fused ViT checkpoint and sidecar config.
- `vit_only.pt` and `vit_only.json` - ViT-only `mean_logits` ablation.
- `vit_config.json` / `vit_only_config.json` - final main/ablation configurations.
- `metrics_vit.json`, `metrics_vit_only.json` - held-out metrics/predictions.
- `vit_roc.png`, `vit_confusion.png`, `vit_calibration.png` (and `vit_only_*`).
- `comparison.md`, `comparison.json` - VQC, best classical, ViT-only, ViT+rPPG.
- `training_summary.json`, `vit_model_training_log.jsonl`,
  `vit_only_training_log.jsonl`, `cv_results.json` when run.

Checkpoints contain only a `state_dict`; loading uses `weights_only=True` and
requires the sidecar JSON. Missing artifacts or fewer than 16 accepted frames
produce `INCONCLUSIVE` with exit code 3 rather than a probability.

## Result schema

`run_pipeline.py --backend vit` writes `stages.vit` with `backend`, `prob_real`,
`verdict`, `confidence`, threshold, temperature, backbone, temporal head, and
frame provenance. The top-level `verdict` and exit codes match the quantum path.
Confidence follows the existing KYC convention `abs(P(real) - 0.5) * 2`.

## Current evidence status

No DFDC dataset, extracted frames, rPPG CSV, fused split manifest, or trained
quantum/ViT artifacts are present in this workspace. Consequently no ViT
accuracy, AUC, confusion matrix, or VQC-vs-ViT comparison is claimed here.
`python -m vit.tests` validates contracts and mechanics without network access
or dataset-specific numbers.

To produce real results, align the DFDC data first, then run:

```powershell
python RPPG/extract_dataset_features.py --workers 0 --gpu
python visual/pipeline.py --frames-root MAJ_OUTPUT_ROOT/frames/frame_sequences `
  --rppg-csv MAJ_OUTPUT_ROOT/rppg/dataset_features.csv --fuse --create-splits
python -m vit.pipeline --all
```

The comparison table is populated only from measured metrics files; unavailable
rows remain `n/a`.
