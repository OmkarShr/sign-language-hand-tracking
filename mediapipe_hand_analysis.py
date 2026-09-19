import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def load_json_image(filename):
    with filename.open("r") as file:
        obj = json.load(file)

    wrapper = obj["image"]
    if wrapper.get("__ndarray__") is not True:
        raise ValueError("invalid NumPy wrapper")

    image = np.array(wrapper["data"], dtype=wrapper["dtype"])
    return image.reshape(wrapper["shape"])


def preprocess_image(image, mode):
    if mode == "none":
        return image
    if mode == "equalize":
        return cv2.equalizeHist(image)

    # CLAHE corrects local lighting without letting bright windows dominate.
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    if mode == "clahe-sharpen":
        blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, blurred, -0.5, 0)
    return enhanced


def draw_result(image, result):
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    height, width = image.shape

    for hand_index, landmarks in enumerate(result.hand_landmarks):
        points = [
            (int(landmark.x * width), int(landmark.y * height))
            for landmark in landmarks
        ]
        for start, end in HAND_CONNECTIONS:
            cv2.line(output, points[start], points[end], (0, 220, 0), 2)
        for point in points:
            cv2.circle(output, point, 3, (0, 0, 255), -1)

        category = result.handedness[hand_index][0]
        x = max(0, min(point[0] for point in points))
        y = max(18, min(point[1] for point in points) - 6)
        label = f"{category.category_name} {category.score:.2f}"
        cv2.putText(
            output, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.45, (255, 100, 0), 1, cv2.LINE_AA,
        )

    return output


def longest_run(values, expected):
    longest = current = 0
    for value in values:
        if value == expected:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure MediaPipe hand detection on HM01B0 JSON frames."
    )
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=Path("HM01B0_images/HM01B0_images"),
        help="folder containing frame_*.json files",
    )
    parser.add_argument("--fps", type=float, default=8.6)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--min-confidence", type=float, default=0.3)
    parser.add_argument(
        "--preprocess",
        choices=("none", "equalize", "clahe", "clahe-sharpen"),
        default="equalize",
        help="grayscale lighting and contrast correction",
    )
    parser.add_argument("--model", type=Path, default=Path("hand_landmarker.task"))
    parser.add_argument("--output-video", type=Path, default=Path("hand_detection.mp4"))
    parser.add_argument("--output-report", type=Path, default=Path("hand_detection_report.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    files = sorted(args.folder.glob("frame_*.json"))
    if not files:
        raise SystemExit(f"No frame_*.json files found in {args.folder}")
    if not args.model.is_file():
        raise SystemExit(f"MediaPipe model not found: {args.model}")

    first_image = load_json_image(files[0])
    height, width = first_image.shape
    video_size = (width * args.scale, height * args.scale)
    writer = cv2.VideoWriter(
        str(args.output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        video_size,
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not create video: {args.output_video}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=args.min_confidence,
        min_hand_presence_confidence=args.min_confidence,
        min_tracking_confidence=args.min_confidence,
    )

    frame_results = []
    invalid_frames = []
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            for index, filename in enumerate(files):
                try:
                    image = load_json_image(filename)
                    if image.ndim != 2:
                        raise ValueError(f"expected grayscale image, got shape {image.shape}")

                    processed = preprocess_image(image, args.preprocess)
                    rgb = np.ascontiguousarray(cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB))
                    timestamp_ms = round(index * 1000 / args.fps)
                    result = landmarker.detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                        timestamp_ms,
                    )

                    hands = []
                    for handedness in result.handedness:
                        category = handedness[0]
                        hands.append({
                            "label": category.category_name,
                            "confidence": round(float(category.score), 4),
                        })
                    frame_results.append({"file": filename.name, "hands": hands})

                    annotated = draw_result(processed, result)
                    annotated = cv2.resize(
                        annotated, video_size, interpolation=cv2.INTER_NEAREST
                    )
                    status = f"{filename.name} | hands: {len(hands)}"
                    cv2.putText(
                        annotated, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1, cv2.LINE_AA,
                    )
                    writer.write(annotated)
                except Exception as error:
                    invalid_frames.append({"file": filename.name, "error": str(error)})

                print(f"\rProcessed {index + 1}/{len(files)}", end="", flush=True)
    finally:
        writer.release()
    print()

    hand_counts = [len(frame["hands"]) for frame in frame_results]
    detected = sum(count > 0 for count in hand_counts)
    total = len(frame_results)
    all_confidences = [
        hand["confidence"]
        for frame in frame_results
        for hand in frame["hands"]
    ]
    summary = {
        "input_folder": str(args.folder),
        "total_json_files": len(files),
        "valid_frames": total,
        "invalid_frames": len(invalid_frames),
        "frames_with_hands": detected,
        "detection_rate_percent": round(100 * detected / total, 2) if total else 0,
        "frames_with_zero_hands": hand_counts.count(0),
        "frames_with_one_hand": hand_counts.count(1),
        "frames_with_two_hands": hand_counts.count(2),
        "longest_detection_streak_frames": longest_run(
            [count > 0 for count in hand_counts], True
        ),
        "longest_missed_streak_frames": longest_run(
            [count == 0 for count in hand_counts], True
        ),
        "mean_handedness_confidence": (
            round(float(np.mean(all_confidences)), 4) if all_confidences else None
        ),
        "settings": {
            "fps": args.fps,
            "minimum_confidence": args.min_confidence,
            "preprocessing": args.preprocess,
            "model": str(args.model),
        },
    }
    report = {"summary": summary, "frames": frame_results, "errors": invalid_frames}
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"Annotated video: {args.output_video}")
    print(f"Detailed report: {args.output_report}")


if __name__ == "__main__":
    main()
