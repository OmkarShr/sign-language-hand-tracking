# Spike 001: pretrained ASL model on HM01B0 footage

## Question

Given the existing 324x244 grayscale HM01B0 stream at approximately 8.6 FPS, can the laptop execute hand tracking and a real pretrained isolated-word ASL model fast enough for interactive use, and do the present images contain enough observable hand evidence to justify displaying predictions?

## Scope

- Camera sequence: 87 real HM01B0 frames, approximately 10.12 seconds.
- Detector: MediaPipe Hand Landmarker, video mode, confidence thresholds 0.3.
- Recognizer: WLASL-2000 I3D checkpoint, 2,000 isolated ASL glosses.
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB.
- This is not continuous sign-language translation.
- The HM01B0 sequence has no ground-truth ASL gloss, so no camera recognition accuracy is reported.

## Provenance

`assets/provenance.json` pins the source URLs, repository/model revisions, sizes, and SHA-256 hashes. Run `python setup_asl_assets.py` from the repository root to fetch and verify the ignored model/code assets. The 57.5 MB checkpoint is excluded from Git, loaded with `torch.load(..., weights_only=True)`, and must match the 2,000-class architecture strictly.

Two labelled WLASL test clips were used only to verify that the checkpoint, architecture, preprocessing and class-index mapping form a runnable pipeline:

- `00335`, ground truth `abdomen`: replicated-grayscale top-1 `abdomen` (0.9456).
- `00376`, ground truth `able`: replicated-grayscale top-1 `can` (0.9532), an incorrect result under the camera-oriented grayscale path.

Two samples do not estimate model accuracy. They establish one correct and one incorrect executable test case after matching the pinned WLASL center-crop and temporal max-pooling path; the model card reports WLASL-2000 top-1 32.48%, top-5 57.31% on its RGB source benchmark.

## Real-camera hand detection

All 87 frames were retained in each denominator.

| Preprocessing | Any hand | Longest missed streak | Mean detector time |
| --- | ---: | ---: | ---: |
| None | 18/87 (20.69%) | 29 frames | 16.22 ms |
| Global histogram equalization | 32/87 (36.78%) | 10 frames | 15.09 ms |
| CLAHE | 17/87 (19.54%) | 37 frames | 11.86 ms |
| CLAHE + sharpening | 16/87 (18.39%) | 37 frames | 11.90 ms |

Global equalization is best on this one sequence, but 55/87 frames still have no detected hand. Detection presence is not landmark correctness.

## GPU timing and output behavior

The first CUDA call is a warm-up and is excluded. Subsequent I3D timing:

| Source window | Model frames | Observation time | Mean I3D time raw/equalized | Equalized adjacent top-1 stability |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 32 | 1.86 s | 25.11 / 24.12 ms | 88.89% |
| 24 | 48 | 2.79 s | 34.41 / 34.70 ms | 66.67% |
| 32 | 64 | 3.72 s | 47.32 / 45.43 ms | 25.00% |

Peak CUDA allocation for the 64-frame path was approximately 372.16 MiB. Compute is well inside the 8.6 FPS acquisition budget. Observation-window duration, not GPU inference, dominates latency.

The apparent stability and softmax values are not evidence of correctness. On the unlabelled camera sequence, equalized windows mainly produced `compare`/`talk`, with a maximum top-1 probability of 0.4723. MediaPipe coverage remained poor. This is out-of-domain diagnostic behavior, not recognition accuracy.

The realtime pipeline reports top-5 values and a debounced `diagnostic_candidate_only` field, never a translation. It rejects windows with less than 50% hand-detection coverage. In the verified 40-frame replay, all four windows were rejected as unobservable and no candidate was emitted.

## Verdict: PARTIAL

### What worked

- The existing serial protocol, replay source, MediaPipe tracking, CUDA recognizer and output gate are integrated.
- The checkpoint and class map execute correctly on CUDA.
- Inference throughput and GPU memory are not current bottlenecks.
- Global histogram equalization measurably improves detector coverage on the supplied recording.

### What did not work

- The present camera recording provides detected hands in only 36.78% of frames under the best tested preprocessing.
- The pretrained RGB WLASL model can be confidently wrong on HM01B0 footage.
- There is no labelled HM01B0 sign set, so camera recognition accuracy remains unknown.
- The source model recognizes isolated ASL glosses; it cannot translate continuous signing.

### Recommendation

Do not abandon the camera solely for compute reasons. First collect a small labelled, signer-disjoint HM01B0 feasibility set and improve framing/lighting until hand observability is substantially higher. Evaluate top-1/top-5 and abstention on that set. Treat continuous translation as a later goal after isolated-word recognition and no-sign segmentation work.
