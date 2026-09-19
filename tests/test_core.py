from itertools import pairwise

import numpy as np

from asl_realtime.core import (
    DiagnosticCandidateGate,
    parse_class_list,
    prepare_i3d_clip,
    resample_indices,
)


def test_parse_class_list_requires_contiguous_indices():
    text = "0\tbook\n1\tdrink\n2\tcomputer\n"
    assert parse_class_list(text) == ["book", "drink", "computer"]


def test_resample_indices_preserves_endpoints_and_count():
    indices = resample_indices(source_count=5, output_count=8)
    assert len(indices) == 8
    assert indices[0] == 0
    assert indices[-1] == 4
    assert all(a <= b for a, b in pairwise(indices))


def test_prepare_i3d_clip_replicates_grayscale_and_normalizes():
    frames = [np.full((12, 20), value, dtype=np.uint8) for value in (0, 64, 255)]
    clip = prepare_i3d_clip(frames, output_frames=5, crop_size=8, equalize=False)
    assert clip.shape == (1, 3, 5, 8, 8)
    assert float(clip.min()) >= -1.0
    assert float(clip.max()) <= 1.0
    np.testing.assert_allclose(clip[:, 0], clip[:, 1])
    np.testing.assert_allclose(clip[:, 1], clip[:, 2])


def test_prepare_i3d_clip_matches_pinned_wlasl_crop_for_large_frame():
    frame = np.tile(np.arange(324, dtype=np.uint8), (244, 1))
    clip = prepare_i3d_clip([frame], output_frames=1, crop_size=224, equalize=False)
    recovered = np.rint((clip[0, 0, 0] + 1.0) * 127.5).astype(np.uint8)
    # Official loader does not resize when both dimensions are at least 226;
    # CenterCrop(224) therefore starts at x=50, y=10 for a 324x244 frame.
    assert recovered[0, 0] == frame[10, 50]
    assert recovered[-1, -1] == frame[233, 273]


def test_diagnostic_candidate_gate_abstains_below_threshold():
    gate = DiagnosticCandidateGate(min_probability=0.5, required_agreement=2)
    assert gate.update("book", 0.49) is None


def test_diagnostic_candidate_gate_requires_agreement_and_deduplicates():
    gate = DiagnosticCandidateGate(min_probability=0.4, required_agreement=2)
    assert gate.update("book", 0.8) is None
    assert gate.update("drink", 0.8) is None
    assert gate.update("drink", 0.8) == "drink"
    assert gate.update("drink", 0.9) is None
    assert gate.update("book", 0.8) is None
    assert gate.update("book", 0.8) == "book"
    assert gate.update("book", 0.1) is None
    assert gate.update("book", 0.9) is None
    assert gate.update("book", 0.9) == "book"


def test_diagnostic_candidate_gate_clears_history_when_window_is_unobservable():
    gate = DiagnosticCandidateGate(min_probability=0.4, required_agreement=2)
    assert gate.update("book", 0.8) is None
    assert gate.update("book", 0.9, observable=False) is None
    assert gate.update("book", 0.8) is None
