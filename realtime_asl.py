#!/usr/bin/env python3
"""Run MediaPipe telemetry and WLASL isolated-gloss inference on replay or serial."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

from asl_realtime.core import DiagnosticCandidateGate, prepare_i3d_clip
from asl_realtime.hand_tracker import HandTracker
from asl_realtime.i3d_backend import WLASLI3DBackend
from asl_realtime.sources import JsonReplaySource, SerialJPEGSource


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=("json", "serial"), default="json")
    parser.add_argument("--folder", type=Path, default=Path("HM01B0_images/HM01B0_images"))
    parser.add_argument("--port")
    parser.add_argument("--fps", type=float, default=8.6)
    parser.add_argument("--window-source-frames", type=int, default=32)
    parser.add_argument("--model-frames", type=int, default=64)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="explicitly select JSON-lines output without a GUI (currently the only mode)",
    )
    parser.add_argument("--preprocess", choices=("none", "equalize", "clahe", "clahe-sharpen"), default="equalize")
    parser.add_argument("--min-output-probability", type=float, default=0.35)
    parser.add_argument("--required-agreement", type=int, default=2)
    parser.add_argument("--min-window-hand-coverage", type=float, default=0.5)
    parser.add_argument("--hand-model", type=Path, default=Path("hand_landmarker.task"))
    parser.add_argument("--weights", type=Path, default=Path("spikes/001-asl-i3d-camera/assets/wlasl2000_i3d_state_dict.bin"))
    parser.add_argument("--labels", type=Path, default=Path("spikes/001-asl-i3d-camera/assets/wlasl_class_list.txt"))
    parser.add_argument("--i3d-module", type=Path, default=Path("spikes/001-asl-i3d-camera/vendor/pytorch_i3d.py"))
    args = parser.parse_args()

    if args.source == "serial":
        if not args.port:
            parser.error("--port is required for serial capture")
        source = SerialJPEGSource(args.port)
    else:
        source = JsonReplaySource(args.folder, fps=args.fps)

    backend = WLASLI3DBackend(
        weights_path=args.weights,
        class_list_path=args.labels,
        model_module_path=args.i3d_module,
        device="cuda",
    )
    gate = DiagnosticCandidateGate(args.min_output_probability, args.required_agreement)
    buffer = deque(maxlen=args.window_source_frames)
    prediction_number = 0

    with HandTracker(args.hand_model, min_confidence=0.3) as tracker:
        for record in source:
            hand = tracker.detect(
                record.image,
                round(record.timestamp_seconds * 1000),
                mode=args.preprocess,
            )
            buffer.append((record.image, hand["hand_count"] > 0))
            event = {
                "frame": record.sequence,
                "capture_seconds": record.timestamp_seconds,
                "hands": hand["hand_count"],
                "hand_latency_ms": round(hand["latency_ms"], 2),
            }
            if len(buffer) == args.window_source_frames and (
                record.sequence - args.window_source_frames
            ) % args.stride == 0:
                images = [image for image, _detected in buffer]
                hand_coverage = sum(detected for _image, detected in buffer) / len(buffer)
                observable = hand_coverage >= args.min_window_hand_coverage
                clip = prepare_i3d_clip(
                    images, output_frames=args.model_frames, equalize=True
                )
                top5, runtime = backend.predict(clip, top_k=5)
                prediction_number += 1
                candidate = gate.update(
                    top5[0]["label"],
                    top5[0]["probability"],
                    observable=observable,
                )
                event["isolated_gloss_prediction"] = {
                    "number": prediction_number,
                    "top5": top5,
                    "window_hand_coverage": hand_coverage,
                    "observable": observable,
                    "diagnostic_candidate_only": candidate,
                    "not_translation": True,
                    **runtime,
                }
            print(json.dumps(event), flush=True)
            if args.source == "json":
                time.sleep(max(0.0, 1.0 / args.fps))
            if args.max_frames and record.sequence >= args.max_frames:
                break


if __name__ == "__main__":
    main()
