# DeepfakeBench (DFB) Decision Backend

## Summary

The VQC stage can be replaced by a real image-based deepfake detector from
DeepfakeBench, with the existing rPPG features retained as an optional fusion
branch. The backend is implemented in `WORKING/dfb_backend/` and selected with
`run_pipeline.py --backend dfb`.

On the DFDC subset used by this project, a fine-tuned Xception detector reaches
**AUC 0.9715** and **balanced accuracy 0.9057** on the held-out test split.
Late fusion with rPPG raises balanced accuracy to **0.9083** and reduces ECE
from 0.1071 to 0.0302, at essentially unchanged AUC.

## Data and Splits

The rPPG feature table contains 2,775 labeled DFDC videos (1,248 fake, 1,527
real). After Stage 1/3 processing, 2,743 videos had fused features. The quantum
split manifest contains 2,742 rows (one additional video was dropped by the
heart-rate plausibility filter):

| Split | Manifest | Usable for DFB | Real | Fake |
| --- | ---: | ---: | ---: | ---: |
| Train | 1,646 | 1,630 | 900 | 730 |
| Validation | 548 | 546 | 301 | 245 |
| Test | 548 | 544 | 298 | 246 |
| Total | 2,742 | 2,720 | 1,499 | 1,221 |

Twenty-two manifest rows have fewer than the minimum eight Stage-1 frames and
are skipped. No frames, rPPG values, or validation/test labels are missing for
the remaining 2,720 videos.

## Preprocessing and Domain Gap

The DFB test transform is reproduced exactly:

1. Read the Stage-1 accepted JPEG with OpenCV (BGR to RGB).
2. Resize to the detector resolution with `cv2.INTER_CUBIC`.
3. Convert to float, divide by 255, normalize with the detector YAML
   mean/std (`0.5/0.5/0.5` for all three detectors).

The important mismatch is the source resolution: Stage-1 crops are
**112x112**, while DFB expects **256x256** (Xception, UCF) or **380x380**
(EfficientNet-B4). The crops are therefore upscaled by 2.29x or 3.39x. The
pretrained FF++ checkpoints are also applied zero-shot to DFDC, a different
dataset and compression pipeline.

A dlib 81-landmark aligned re-crop from the original videos was considered but
not run: dlib and the shape predictor are not installed, and the raw 3,293
videos would need to be decoded again. This remains the main preprocessing
limitation. Fine-tuning on DFDC closes most of the observed domain gap, which
suggests the remaining high-frequency information is sufficient for this
subset; the aligned re-crop ablation is left as future work.

## Models

| Detector | Checkpoint | Parameters | Input | Zero-shot source |
| --- | --- | ---: | ---: | --- |
| Xception | `xception_best.pth` | 21.9 M | 256 | FF++ |
| UCF | `ucf_best.pth` | 46.8 M | 256 | FF++ |
| EfficientNet-B4 | `effnb4_best.pth` | 17.6 M | 380 | FF++ |

All checkpoints load strictly with zero missing and zero unexpected keys.
EfficientNet-B4 is constructed with `pretrained=False` because the ImageNet
backbone file is not present; this does not affect checkpoint loading.

Videos are represented by 16 evenly spaced accepted frames. Frame probabilities
are converted to `P(real) = 1 - P(fake)` and aggregated to a video score.
`mean` aggregation was selected; `topk` (best 20% of frames) was tested on the
fine-tuned Xception and was slightly worse (test AUC 0.9683 vs 0.9715).

## Fine-Tuning

Xception was fine-tuned on DFDC using the train split only:

| Setting | Value |
| --- | --- |
| Initialization | FF++ `xception_best.pth` |
| Epochs | 6 |
| Optimizer | AdamW, lr 1e-4, weight decay 1e-4 |
| Schedule | 10% warmup + cosine decay |
| Batch | 4 videos x 8 frames (32 frames/step) |
| Validation | 16 frames/video |
| Regularization | AMP, gradient clip 1.0, label smoothing 0.05 |
| Freeze | Backbone frozen for epoch 1, BatchNorm kept in eval mode |
| Sampling | Class-balanced video sampler |
| Best epoch | 6 (validation AUC 0.9559) |
| Training time | 652 s on an RTX 4060 Laptop (8 GB) |

Only Xception was fine-tuned. UCF and EfficientNet-B4 were evaluated zero-shot;
their DFDC fine-tuning is left as future work.

## Results

All values are video-level on the 544-video test split. Thresholds were chosen
on validation only.

| Backend | AUC | Balanced acc. | EER | ECE | Brier |
| --- | ---: | ---: | ---: | ---: | ---: |
| Xception, zero-shot, mean | 0.6072 | 0.6081 | 0.4026 | 0.0726 | 0.2434 |
| UCF, zero-shot, mean | 0.5383 | 0.5350 | 0.4929 | 0.1933 | 0.2966 |
| EfficientNet-B4, zero-shot, mean | 0.5491 | 0.5539 | 0.4575 | 0.1784 | 0.2888 |
| **Xception, DFDC fine-tuned, mean** | **0.9715** | **0.9057** | **0.0937** | 0.1071 | 0.0806 |
| **Xception fine-tuned + rPPG fusion** | **0.9707** | **0.9083** | **0.0937** | **0.0302** | **0.0686** |
| RandomForest, 59 fused features | 0.8693 | 0.7758 | 0.2134 | 0.1250 | 0.1671 |
| LogisticRegression, 59 fused features | 0.7921 | 0.7179 | 0.2886 | 0.0479 | 0.1866 |
| MLP, 59 fused features | 0.7992 | 0.7169 | 0.2795 | 0.2242 | 0.2388 |

The classical rows were trained on the exact same aligned split
(1,630 / 546 / 544 videos) and the same 59 fused features, so they are a
controlled comparison for the decision layer. RandomForest is the strongest
classical baseline (AUC 0.8693); the fine-tuned Xception is 10.2 AUC points
higher, and fusion improves calibration further.

Fine-tuned Xception validation metrics: AUC 0.9559, balanced accuracy 0.8841
(at the Youden threshold), EER 0.1268, ECE 0.0329, Brier 0.0858.

A controlled VQC comparison could not be produced: the only VQC artifact on
disk is an older run over 20 rPPG-only features with a different 694-video test
split (AUC 0.4646, balanced accuracy 0.5000, all decisions "uncertain"). Those
numbers are not directly comparable and are therefore not mixed into this table.

Fusion uses the image logit plus the 20 rPPG features, a standard scaler, and
balanced logistic regression. The regularization strength (`C=0.01`),
temperature (0.479), and decision threshold (0.459) are selected on validation.
The reported test scores come from the train-fit model; the deployed artifact
is refit on train+validation after tuning.

## Interpretation

- Zero-shot FF++ detectors do not transfer well to 112x112 DFDC crops; Xception
  is the strongest of the three but still near chance.
- DFDC fine-tuning produces a large improvement (AUC 0.61 to 0.97), confirming
  that the domain gap, not the image content, is the main zero-shot failure.
- rPPG fusion does not improve AUC, but it improves balanced accuracy and
  substantially improves calibration. This is useful for KYC decisions, where
  confidence quality matters.
- A controlled VQC row is absent because the only VQC artifact on disk is an
  older 20-feature run with a different split (test AUC 0.4646, all-uncertain
  decisions). Run `python -m quantum.pipeline --all` on this output root to
  produce a same-split VQC comparison.

## Reproduction

From `WORKING/`:

```powershell
# Contract and fast tests
python -m dfb_backend.tests

# Strict checkpoint loading for all three detectors
$env:DFB_RUN_MODEL_TESTS="1"; python -m dfb_backend.tests; Remove-Item Env:DFB_RUN_MODEL_TESTS

# Zero-shot evaluation (writes scores/metrics/ROC/confusion/calibration)
python -m dfb_backend.evaluation --zero-shot --detectors xception ucf efficientnetb4 --frames 16 --aggregation mean

# Fine-tune Xception on DFDC
python -m dfb_backend.evaluation --finetune --detectors xception --epochs 6 --lr 1e-4 --train-frames 8 --freeze-epochs 1 --frames 16 --batch-videos 4

# Evaluate the fine-tuned checkpoint
python -m dfb_backend.evaluation --zero-shot --detectors xception --mode finetuned --frames 16 --aggregation mean

# Fit image + rPPG fusion
python -m dfb_backend.evaluation --fuse --detector xception --mode finetuned

# End-to-end inference with the DFB backend
python run_pipeline.py --backend dfb --source path\to\video.mp4 --detector xception
```

Artifacts are written to `MAJ_OUTPUT_ROOT/dfb/` (`scores_*.json`,
`metrics_*.json`, plots, `xception_finetuned.pt`, `fusion.json`).

## Limitations

- Stage-1 crops are 112x112; no aligned 256x256 re-crop was evaluated.
- Only Xception was fine-tuned; UCF and EfficientNet-B4 results are zero-shot.
- The DFDC subset is small (2,720 usable videos), so test differences below
  roughly one percentage point should not be over-interpreted.
- The fusion branch depends on valid rPPG features; when rPPG is unavailable,
  the image-only decision is used.
- Upstream detector constructors load pretrained backbones with
  `torch.load(..., weights_only=False)`; only trusted checkpoints should be used.
- DeepfakeBench is licensed CC BY-NC 4.0, so this integration is for
  non-commercial research. See `THIRD_PARTY_NOTICES.md`.

## License and Attribution

DeepfakeBench: Copyright (c) 2023, CUHK(SZ), CC BY-NC 4.0.
Repository: https://github.com/SCLBD/DeepfakeBench, pinned at
`f188b1c105465e2e5377eb536a95022ae0e4522d`. The upstream license text is kept at
`WORKING/third_party/DeepfakeBench.LICENSE`.
