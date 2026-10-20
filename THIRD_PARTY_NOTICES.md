# Third-Party Notices

This project integrates the following third-party component. The upstream
license and attribution are reproduced or preserved in this repository.

## DeepfakeBench

- **Project:** DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection
- **Authors:** Zhiyuan Yan, Yong Zhang, Xinhang Yuan, Siwei Lyu, Baoyuan Wu
- **Repository:** https://github.com/SCLBD/DeepfakeBench
- **Pinned commit:** `f188b1c105465e2e5377eb536a95022ae0e4522d`
  (also recorded in `WORKING/third_party/DeepfakeBench.commit`)
- **Copyright:** Copyright (c) 2023, CUHK(SZ). All rights reserved.
- **License:** Creative Commons Attribution-NonCommercial 4.0 International
  (CC BY-NC 4.0), https://creativecommons.org/licenses/by-nc/4.0/legalcode
- **Local license copy:** `WORKING/third_party/DeepfakeBench.LICENSE`

### How this project uses DeepfakeBench

The project imports the upstream detector definitions, YAML detector
configurations, and pretrained FF++ detector checkpoints to produce image-based
deepfake evidence. The wrapper code lives in `WORKING/dfb_backend/`; upstream
source files are not modified. The upstream package is cloned locally and is not
committed to this repository (see `.gitignore`).

The pretrained detector checkpoints are used for local, non-commercial academic
research only. They are not redistributed by this repository. Users must obtain
them from the upstream project and comply with the upstream license.

### Security note

Upstream DeepfakeBench detector constructors load pretrained backbone weights
with `torch.load(..., weights_only=False)`. This is an upstream implementation
detail and is required by the published checkpoints. Only load checkpoints from
trusted sources. Project-owned fine-tuned checkpoints and fusion artifacts use
plain state dictionaries and JSON (no pickle) and are loaded with
`weights_only=True` where applicable.

### Citation

If you use this backend, cite the DeepfakeBench paper:

```bibtex
@inproceedings{yan2023deepfakebench,
  title={DeepfakeBench: A Comprehensive Benchmark of Deepfake Detection},
  author={Yan, Zhiyuan and Zhang, Yong and Yuan, Xinhang and Lyu, Siwei and Wu, Baoyuan},
  booktitle={Advances in Neural Information Processing Systems Datasets and Benchmarks Track},
  year={2023}
}
```

## Datasets and other dependencies

- **DFDC (Deepfake Detection Challenge)** videos are used under the dataset's
  own terms and are not redistributed by this repository.
- Python packages (PyTorch, torchvision, scikit-learn, OpenCV, MediaPipe,
  PennyLane, and others) remain under their respective upstream licenses.
