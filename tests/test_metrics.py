from asl_realtime.metrics import DetectionAccumulator, prediction_stability


def test_detection_accumulator_preserves_failures_and_streaks():
    metrics = DetectionAccumulator()
    for count in [0, 0, 1, 2, 0, 1]:
        metrics.add(count, inference_ms=2.0)
    summary = metrics.summary()
    assert summary["frames"] == 6
    assert summary["frames_with_any_hand"] == 3
    assert summary["any_hand_rate"] == 0.5
    assert summary["longest_missed_streak"] == 2
    assert summary["longest_detected_streak"] == 2
    assert summary["mean_inference_ms"] == 2.0


def test_prediction_stability_reports_adjacent_agreement():
    assert prediction_stability([]) is None
    assert prediction_stability(["a"]) is None
    assert prediction_stability(["a", "a", "b", "b"]) == 2 / 3
