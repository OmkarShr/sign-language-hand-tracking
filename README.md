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

The equalized run had a mean handedness confidence of 96.98%. This is confidence conditioned on a detection, not landmark or recognition accuracy. Motion blur, hand overlap, grayscale input, and low camera resolution still limit detection.

## ASL realtime feasibility branch

Branch `experiment/asl-realtime-feasibility` adds an explicitly experimental pipeline for:

- replaying the real 324x244 HM01B0 sequence or reading the existing serial JPEG protocol;
- measuring MediaPipe hand coverage under four preprocessing modes;
- running a pinned WLASL-2000 I3D isolated-word model on the RTX 4060;
- suppressing user-facing output when too little of the window contains detected hands;
- producing auditable JSON reports with source and model hashes.

This is **isolated ASL gloss recognition, not continuous translation**. The bundled camera recording has no ASL ground-truth label, so its model outputs cannot be reported as accuracy.

Create the separate environment without changing the original `.venv`:

```bash
/home/omkar/miniconda3/envs/federated-ft/bin/python -m venv --system-site-packages .venv-asl
.venv-asl/bin/python -m pip install -r requirements-asl.txt
.venv-asl/bin/python setup_asl_assets.py
```

`setup_asl_assets.py` downloads the pinned model/code assets and rejects any SHA-256 mismatch. The checkpoint is intentionally not committed; its source and digest are recorded in `spikes/001-asl-i3d-camera/assets/provenance.json`.

Run the recorded-camera benchmark:

```bash
.venv-asl/bin/python benchmark_camera.py \
  --output experiments/manual-run/report.json
```

Run realtime replay:

```bash
.venv-asl/bin/python realtime_asl.py --source json
```

Run a connected HM01B0 board on Linux:

```bash
.venv-asl/bin/python realtime_asl.py --source serial --port /dev/ttyACM0
```

The default safety gate requires at least 50% of frames in a recognition window to contain a MediaPipe hand detection. `top5` and `diagnostic_candidate_only` are experiment telemetry, **not translations**. A candidate is emitted only after repeated observable windows and is debounced, but it must still be validated against labeled HM01B0 recordings before display to a user.

### Live test checklist

1. Connect the HM01B0 board by USB and identify its port:
   ```bash
   python -c 'from serial.tools import list_ports; [print(p.device, p.description) for p in list_ports.comports()]'
   ```
2. Start the stream, replacing the port if necessary:
   ```bash
   .venv-asl/bin/python realtime_asl.py \
     --source serial --port /dev/ttyACM0 --headless
   ```
3. Perform one supported isolated ASL sign at a time, return to a neutral pose between signs, and retain the JSON-lines output for analysis.

The program prints one JSON object per captured frame. `hands` reports MediaPipe detections. An `isolated_gloss_prediction` object appears when a full temporal window is available. Use Ctrl+C to stop.

The current live command does not draw a GUI overlay or translate sentences. It measures whether serial capture, hand visibility, GPU inference and cautious isolated-gloss candidates can operate together.
