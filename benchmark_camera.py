#!/usr/bin/env python3
"""Benchmark the real HM01B0 recording with MediaPipe and WLASL-2000 I3D.

This is an isolated-word feasibility diagnostic. It has no ground-truth ASL label
for the camera recording and therefore does not report recognition accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from asl_realtime.core import prepare_i3d_clip
from asl_realtime.hand_tracker import PREPROCESS_MODES, HandTracker
from asl_realtime.i3d_backend import WLASLI3DBackend
from asl_realtime.metrics import DetectionAccumulator, prediction_stability
from asl_realtime.sources import JsonReplaySource


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def window_starts(frame_count: int, window: int, stride: int) -> list[int]:
    if frame_count < window:
        return []
    starts = list(range(0, frame_count - window + 1, stride))
    final = frame_count - window
    if starts[-1] != final:
        starts.append(final)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, default=Path("HM01B0_images/HM01B0_images"))
    parser.add_argument("--fps", type=float, default=8.6)
    parser.add_argument("--window-source-frames", type=int, default=32)
    parser.add_argument("--model-frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--hand-model", type=Path, default=Path("hand_landmarker.task"))
    parser.add_argument("--weights", type=Path, default=Path("spikes/001-asl-i3d-camera/assets/wlasl2000_i3d_state_dict.bin"))
    parser.add_argument("--labels", type=Path, default=Path("spikes/001-asl-i3d-camera/assets/wlasl_class_list.txt"))
    parser.add_argument("--i3d-module", type=Path, default=Path("spikes/001-asl-i3d-camera/vendor/pytorch_i3d.py"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    records = list(JsonReplaySource(args.folder, fps=args.fps))
    frames = [record.image for record in records]
    starts = window_starts(len(frames), args.window_source_frames, args.stride)
    if not starts:
        raise SystemExit("recording is shorter than one recognition window")

    detector_results = {}
    for mode in PREPROCESS_MODES:
        accumulator = DetectionAccumulator()
        per_frame = []
        with HandTracker(args.hand_model, min_confidence=0.3) as tracker:
            for record in records:
                result = tracker.detect(
                    record.image,
                    round(record.timestamp_seconds * 1000),
                    mode=mode,
                )
                accumulator.add(result["hand_count"], result["latency_ms"])
                per_frame.append({
                    "sequence": record.sequence,
                    "timestamp_seconds": record.timestamp_seconds,
                    "hand_count": result["hand_count"],
                    "latency_ms": result["latency_ms"],
                })
        detector_results[mode] = {"summary": accumulator.summary(), "frames": per_frame}

    backend = WLASLI3DBackend(
        weights_path=args.weights,
        class_list_path=args.labels,
        model_module_path=args.i3d_module,
        device="cuda",
    )
    # One explicit warm-up is excluded from reported window latency.
    warmup_clip = prepare_i3d_clip(
        frames[starts[0] : starts[0] + args.window_source_frames],
        output_frames=args.model_frames,
        equalize=True,
    )
    backend.predict(warmup_clip, top_k=5)

    recognition_results = {}
    for mode, equalize in (("raw", False), ("equalized", True)):
        windows = []
        for start in starts:
            source_window = frames[start : start + args.window_source_frames]
            clip = prepare_i3d_clip(
                source_window,
                output_frames=args.model_frames,
                equalize=equalize,
            )
            top5, runtime = backend.predict(clip, top_k=5)
            windows.append({
                "start_frame": start + 1,
                "end_frame": start + args.window_source_frames,
                "start_seconds": start / args.fps,
                "end_seconds": (start + args.window_source_frames - 1) / args.fps,
                "top5": top5,
                **runtime,
            })
        top1_labels = [window["top5"][0]["label"] for window in windows]
        latencies = [window["latency_ms"] for window in windows]
        recognition_results[mode] = {
            "windows": windows,
            "summary": {
                "window_count": len(windows),
                "mean_latency_ms": statistics.mean(latencies),
                "max_latency_ms": max(latencies),
                "adjacent_top1_stability": prediction_stability(top1_labels),
                "unique_top1_labels": sorted(set(top1_labels)),
                "maximum_top1_probability": max(
                    window["top5"][0]["probability"] for window in windows
                ),
                "window_observation_seconds": args.window_source_frames / args.fps,
                "prediction_hop_seconds": args.stride / args.fps,
            },
        }

    source_files = sorted(
        path for path in args.folder.glob("frame_*.json") if not path.name.startswith("._")
    )
    report = {
        "status": "completed",
        "scope": "unlabelled isolated-word ASL feasibility diagnostic; not translation accuracy",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "folder": str(args.folder),
            "frames": len(frames),
            "resolution": list(frames[0].shape[::-1]),
            "fps": args.fps,
            "duration_seconds": len(frames) / args.fps,
            "frame_sha256": {path.name: sha256(path) for path in source_files},
        },
        "configuration": {
            "window_source_frames": args.window_source_frames,
            "model_frames": args.model_frames,
            "stride": args.stride,
            "crop_size": 224,
            "model": "WLASL-2000 I3D",
            "weights_sha256": sha256(args.weights),
            "class_list_sha256": sha256(args.labels),
            "hand_model_sha256": sha256(args.hand_model),
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "opencv": cv2.__version__,
            "mediapipe": mp.__version__,
            "numpy": np.__version__,
        },
        "hand_detection": detector_results,
        "recognition": recognition_results,
        "limitations": [
            "The camera sequence has no ground-truth ASL gloss, so predictions cannot be scored for correctness.",
            "WLASL-2000 is an isolated-word RGB benchmark model, not continuous translation.",
            "The model was not trained for 324x244 grayscale HM01B0 footage.",
            "Softmax scores are not calibrated confidence or evidence that a sign occurred.",
        ],
    }

    output = args.output or Path("experiments") / f"asl_camera_feasibility_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}" / "report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(output)
    print(json.dumps({
        "hand_detection": {mode: value["summary"] for mode, value in detector_results.items()},
        "recognition": {mode: value["summary"] for mode, value in recognition_results.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
