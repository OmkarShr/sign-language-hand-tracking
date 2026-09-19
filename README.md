# Sign Language Hand Tracking

Tools for capturing grayscale HM01B0 camera frames over USB serial, viewing the saved JSON sequence, and measuring MediaPipe hand-landmark detection.

## Included tools

- `outputwjpeg.py` requests JPEG frames from a microcontroller over serial and stores them as NumPy-compatible JSON.
- `jsonvisualiser.py` plays a folder of `frame_*.json` images.
- `mediapipe_hand_analysis.py` applies grayscale preprocessing, detects up to two hands, creates an annotated MP4, and writes a per-frame JSON report.
- `samples/` contains an example source image, annotated result, demo video, and analysis summary.

MediaPipe detects 21 landmarks and handedness for each hand. It does not classify sign-language words; a separate classifier must be trained from landmarks or images for that task.

## Setup

Python 3.12 was used for the current tests.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
curl -L --fail --output hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.

## Capture frames

Set `PORT` in `outputwjpeg.py` for the connected board. Common examples are `/dev/cu.usbmodem...` on macOS, `/dev/ttyACM0` on Linux, and `COM3` on Windows.

```bash
python outputwjpeg.py
```

The current serial protocol sends `c`, waits for `*RDY*`, reads a four-byte little-endian JPEG size, and then reads that many JPEG bytes. Capturing stops after 10 seconds by default.

## View JSON frames

```bash
python jsonvisualiser.py
```

Select the folder containing `frame_*.json` files. Controls are Space for play/pause, A/D for previous/next, R to restart, and Q to quit.

## Analyze hand detection

```bash
python mediapipe_hand_analysis.py path/to/json/folder
```

Without a folder argument, the analyzer uses `HM01B0_images/HM01B0_images`. It produces:

- `hand_detection.mp4`, an annotated video
- `hand_detection_report.json`, detection statistics and per-frame results

Global histogram equalization is the default preprocessing mode because it performed best on the sample recording. Other modes can be compared with `--preprocess none`, `equalize`, `clahe`, or `clahe-sharpen`.

## Sample result

The included 87-frame, 324x244 grayscale recording produced:

| Processing | Frames detected | Detection rate |
| --- | ---: | ---: |
| None | 18/87 | 20.69% |
| Histogram equalization | 32/87 | 36.78% |

The equalized run had a mean handedness confidence of 96.98%. Motion blur, hand overlap, grayscale input, and low camera resolution still limit detection.
