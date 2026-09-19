import cv2
import numpy as np
import serial
import struct
import time
import json
from pathlib import Path


# ==================================================
# SETTINGS
# ==================================================

PORT = "/dev/cu.usbmodem111101"
BAUD = 4000000

WIDTH = 324
HEIGHT = 244

SCALE = 2

DURATION = 10.0

SAVE_DIR = Path.home() / "Documents" / "HM01B0_images"

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# SERIAL
# ==================================================

ser = serial.Serial(
    PORT,
    BAUD,
    timeout=3,
    write_timeout=3
)

print("Opening:", PORT)

time.sleep(2)

ser.reset_input_buffer()


# ==================================================
# READ EXACTLY N BYTES
# ==================================================

def read_exact(n):

    data = bytearray()

    while len(data) < n:

        chunk = ser.read(n - len(data))

        if not chunk:
            return None

        data.extend(chunk)

    return bytes(data)


# ==================================================
# WAIT FOR *RDY*
# ==================================================

def wait_for_ready():

    target = b"*RDY*"

    buffer = bytearray()

    while True:

        b = ser.read(1)

        if not b:
            return False

        buffer.append(b[0])

        if len(buffer) > len(target):
            buffer.pop(0)

        if buffer == target:
            return True


# ==================================================
# SAVE NUMPY ARRAY AS JSON
# ==================================================

def save_image_json(img, frame_number):

    filename = (
        SAVE_DIR /
        f"frame_{frame_number:06d}.json"
    )

    data = {
        "image": {
            "__ndarray__": True,
            "dtype": str(img.dtype),
            "shape": list(img.shape),
            "data": img.tolist()
        }
    }

    with open(filename, "w") as f:
        json.dump(data, f)


# ==================================================
# OPEN WINDOW
# ==================================================

cv2.namedWindow(
    "HM01B0 Recording",
    cv2.WINDOW_NORMAL
)


# ==================================================
# RECORDING VARIABLES
# ==================================================

frame_number = 0

fps = 0.0

fps_count = 0

fps_timer = time.perf_counter()

recording_start = time.perf_counter()


print()
print("======================================")
print("HM01B0 RECORDING")
print("======================================")
print(f"Duration : {DURATION} seconds")
print(f"Save dir : {SAVE_DIR}")
print()
print("Recording...")
print()


# ==================================================
# MAIN RECORDING LOOP
# ==================================================

try:

    while True:

        # ==========================================
        # CHECK 10 SECOND LIMIT
        # ==========================================

        elapsed_total = (
            time.perf_counter()
            - recording_start
        )

        if elapsed_total >= DURATION:
            break


        # ==========================================
        # REQUEST FRAME
        # ==========================================

        ser.write(b"c")
        ser.flush()


        # ==========================================
        # WAIT FOR JPEG
        # ==========================================

        if not wait_for_ready():

            print(
                "\nTIMEOUT waiting for *RDY*"
            )

            continue


        # ==========================================
        # JPEG SIZE
        # ==========================================

        size_bytes = read_exact(4)

        if size_bytes is None:

            print(
                "\nTIMEOUT reading JPEG size"
            )

            continue


        jpeg_size = struct.unpack(
            "<I",
            size_bytes
        )[0]


        # ==========================================
        # VALIDATE SIZE
        # ==========================================

        if (
            jpeg_size <= 0
            or jpeg_size > 128 * 1024
        ):

            print(
                "\nINVALID JPEG SIZE:",
                jpeg_size
            )

            ser.reset_input_buffer()

            continue


        # ==========================================
        # RECEIVE JPEG
        # ==========================================

        jpeg_data = read_exact(
            jpeg_size
        )

        if jpeg_data is None:

            print(
                "\nINCOMPLETE JPEG"
            )

            continue


        # ==========================================
        # JPEG → NUMPY IMAGE
        # ==========================================

        img = cv2.imdecode(
            np.frombuffer(
                jpeg_data,
                dtype=np.uint8
            ),
            cv2.IMREAD_GRAYSCALE
        )


        if img is None:

            print(
                "\nJPEG DECODE FAILED"
            )

            continue


        # ==========================================
        # SAVE ORIGINAL IMAGE
        # ==========================================

        frame_number += 1

        save_image_json(
            img,
            frame_number
        )


        # ==========================================
        # FPS
        # ==========================================

        fps_count += 1

        fps_elapsed = (
            time.perf_counter()
            - fps_timer
        )

        if fps_elapsed >= 1.0:

            fps = (
                fps_count
                / fps_elapsed
            )

            fps_count = 0

            fps_timer = time.perf_counter()


        # ==========================================
        # DISPLAY
        # ==========================================

        img_big = cv2.resize(
            img,
            (
                WIDTH * SCALE,
                HEIGHT * SCALE
            ),
            interpolation=cv2.INTER_NEAREST
        )


        cv2.setWindowTitle(
            "HM01B0 Recording",
            f"Recording | "
            f"Time: {elapsed_total:.1f}/{DURATION:.0f}s | "
            f"FPS: {fps:.2f} | "
            f"Frame: {frame_number} | "
            f"JPEG: {jpeg_size / 1024:.1f} KB"
        )


        cv2.imshow(
            "HM01B0 Recording",
            img_big
        )


        # ==========================================
        # ALLOW EARLY QUIT
        # ==========================================

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            print(
                "\nRecording stopped manually."
            )

            break


# ==================================================
# CLEANUP
# ==================================================

finally:

    ser.close()

    cv2.destroyAllWindows()


# ==================================================
# SUMMARY
# ==================================================

actual_duration = (
    time.perf_counter()
    - recording_start
)

actual_fps = (
    frame_number / actual_duration
    if actual_duration > 0
    else 0
)


print()
print("======================================")
print("RECORDING COMPLETE")
print("======================================")
print(f"Duration : {actual_duration:.2f} seconds")
print(f"Frames   : {frame_number}")
print(f"FPS      : {actual_fps:.2f}")
print(f"Saved to : {SAVE_DIR}")
print("======================================")