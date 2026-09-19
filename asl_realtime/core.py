"""Shared preprocessing and decision logic for the ASL feasibility pipeline."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

import cv2
import numpy as np


def parse_class_list(text: str) -> list[str]:
    """Parse WLASL's ``index<TAB>gloss`` list and verify its ordering."""
    labels: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        index_text, label = line.split(maxsplit=1)
        index = int(index_text)
        if index != len(labels):
            raise ValueError(f"expected class index {len(labels)}, got {index}")
        labels.append(label.strip())
    if not labels:
        raise ValueError("class list is empty")
    return labels


def resample_indices(source_count: int, output_count: int) -> list[int]:
    """Return monotonic endpoint-preserving indices for temporal resampling."""
    if source_count <= 0 or output_count <= 0:
        raise ValueError("source_count and output_count must be positive")
    if source_count == 1:
        return [0] * output_count
    return np.rint(np.linspace(0, source_count - 1, output_count)).astype(int).tolist()


def _wlasl_center_crop(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """Match the pinned WLASL loader: only upscale below crop+2, then crop."""
    height, width = frame.shape
    resize_floor = crop_size + 2
    if height < resize_floor or width < resize_floor:
        scale = resize_floor / min(height, width)
        frame = cv2.resize(frame, dsize=(0, 0), fx=scale, fy=scale)
    top = round((frame.shape[0] - crop_size) / 2.0)
    left = round((frame.shape[1] - crop_size) / 2.0)
    return frame[top : top + crop_size, left : left + crop_size]


def prepare_i3d_clip(
    frames: Iterable[np.ndarray],
    *,
    output_frames: int = 64,
    crop_size: int = 224,
    equalize: bool = True,
) -> np.ndarray:
    """Convert grayscale frames to I3D ``(1, 3, T, H, W)`` float input."""
    source = [np.asarray(frame) for frame in frames]
    if not source:
        raise ValueError("at least one frame is required")
    for frame in source:
        if frame.ndim != 2 or frame.dtype != np.uint8:
            raise ValueError("frames must be uint8 grayscale images")

    selected = [source[index] for index in resample_indices(len(source), output_frames)]
    processed = []
    for frame in selected:
        if equalize:
            frame = cv2.equalizeHist(frame)
        processed.append(_wlasl_center_crop(frame, crop_size))

    array = np.stack(processed).astype(np.float32) / 127.5 - 1.0
    array = np.repeat(array[:, None, :, :], 3, axis=1)
    array = np.transpose(array, (1, 0, 2, 3))
    return array[None, ...]


class DiagnosticCandidateGate:
    """Flag repeated diagnostic candidates; never treat them as translations."""

    def __init__(self, min_probability: float = 0.35, required_agreement: int = 2):
        if not 0 <= min_probability <= 1:
            raise ValueError("min_probability must be in [0, 1]")
        if required_agreement <= 0:
            raise ValueError("required_agreement must be positive")
        self.min_probability = min_probability
        self.required_agreement = required_agreement
        self._labels: deque[str] = deque(maxlen=required_agreement)
        self._last_emitted: str | None = None

    def update(self, label: str, probability: float, *, observable: bool = True) -> str | None:
        if not observable or probability < self.min_probability:
            self._labels.clear()
            self._last_emitted = None
            return None
        self._labels.append(label)
        agreed = len(self._labels) == self.required_agreement and len(set(self._labels)) == 1
        if agreed and label != self._last_emitted:
            self._last_emitted = label
            return label
        return None
