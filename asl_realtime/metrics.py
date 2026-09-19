"""Metrics that retain detector misses and unstable recognition windows."""

from __future__ import annotations

from collections import Counter
from itertools import pairwise


class DetectionAccumulator:
    def __init__(self) -> None:
        self.counts: list[int] = []
        self.inference_ms: list[float] = []

    def add(self, hand_count: int, inference_ms: float) -> None:
        if hand_count < 0:
            raise ValueError("hand_count cannot be negative")
        self.counts.append(hand_count)
        self.inference_ms.append(float(inference_ms))

    @staticmethod
    def _longest(values: list[bool], expected: bool) -> int:
        best = current = 0
        for value in values:
            if value is expected:
                current += 1
                best = max(best, current)
            else:
                current = 0
        return best

    def summary(self) -> dict[str, float | int | dict[int, int]]:
        frames = len(self.counts)
        detected = [count > 0 for count in self.counts]
        histogram = Counter(self.counts)
        return {
            "frames": frames,
            "frames_with_any_hand": sum(detected),
            "any_hand_rate": sum(detected) / frames if frames else 0.0,
            "hand_count_histogram": dict(sorted(histogram.items())),
            "longest_missed_streak": self._longest(detected, False),
            "longest_detected_streak": self._longest(detected, True),
            "mean_inference_ms": (
                sum(self.inference_ms) / len(self.inference_ms)
                if self.inference_ms
                else 0.0
            ),
        }


def prediction_stability(labels: list[str]) -> float | None:
    """Fraction of adjacent windows retaining top-1; undefined before one comparison."""
    if len(labels) < 2:
        return None
    return sum(a == b for a, b in pairwise(labels)) / (len(labels) - 1)
