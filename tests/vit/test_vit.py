"""Pytest entry points for the ViT stage self-checks."""

from vit.tests import (
    test_cap_records_preserves_splits as _cap_records,
    test_checkpoint_round_trip_and_inconclusive as _checkpoint,
    test_config_guards as _config_guards,
    test_evaluation_and_comparison_artifacts as _evaluation,
    test_frame_sampling_deterministic as _frame_sampling,
    test_head_shapes as _head_shapes,
    test_labels_and_feature_order as _labels,
    test_one_batch_overfit as _overfit,
    test_stage_b_runs_after_stage_a_early_stop as _stage_b,
    test_zero_video_group_overlap as _overlap,
)


def test_labels_and_feature_order():
    _labels()


def test_frame_sampling_deterministic():
    _frame_sampling()


def test_cap_records_preserves_splits():
    _cap_records()


def test_zero_video_group_overlap():
    _overlap()


def test_head_shapes():
    _head_shapes()


def test_one_batch_overfit():
    _overfit()


def test_stage_b_runs_after_stage_a_early_stop():
    _stage_b()


def test_checkpoint_round_trip_and_inconclusive():
    _checkpoint()


def test_evaluation_and_comparison_artifacts():
    _evaluation()


def test_config_guards():
    _config_guards()
