"""MediaPipe telemetry for real grayscale camera frames."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Self

import cv2
import mediapipe as mp
import numpy as np

PREPROCESS_MODES = ("none", "equalize", "clahe", "clahe-sharpen")


def preprocess_hand_frame(image: np.ndarray, mode: str) -> np.ndarray:
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"unknown preprocessing mode: {mode}")
    if mode == "none":
        return image
    if mode == "equalize":
        return cv2.equalizeHist(image)
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    if mode == "clahe-sharpen":
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return enhanced


class HandTracker:
    def __init__(self, model_path: str | Path, min_confidence: float = 0.3):
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def detect(self, image: np.ndarray, timestamp_ms: int, mode: str = "equalize") -> dict:
        timestamp_ms = max(timestamp_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        processed = preprocess_hand_frame(image, mode)
        rgb = np.ascontiguousarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB))
        started = time.perf_counter()
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
        )
        latency_ms = (time.perf_counter() - started) * 1000
        confidences = [
            float(handedness[0].score)
            for handedness in result.handedness
            if handedness
        ]
        return {
            "hand_count": len(result.hand_landmarks),
            "handedness_confidences": confidences,
            "latency_ms": latency_ms,
            "processed": processed,
            "landmarks": result.hand_landmarks,
        }
