"""DeepfakeBench image-based decision backend for the KYC pipeline."""

from dfb_backend.config import DEFAULT_DETECTOR, DETECTOR_SPECS
from dfb_backend.predict import fake_prob_to_real, predict_video

__all__ = [
    "DEFAULT_DETECTOR",
    "DETECTOR_SPECS",
    "fake_prob_to_real",
    "predict_video",
]
