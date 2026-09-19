import cv2
import numpy as np
import json
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog


# ==================================================
# SETTINGS
# ==================================================

SCALE = 2
FPS = 8.6

FRAME_DELAY = 1.0 / FPS


# ==================================================
# FOLDER SELECTOR
# ==================================================

def select_folder():

    root = tk.Tk()
    root.withdraw()

    root.attributes("-topmost", True)

    folder = filedialog.askdirectory(
        title="Select HM01B0 JSON Frames Folder"
    )

    root.destroy()

    if not folder:
        print("No folder selected.")
        raise SystemExit

    return Path(folder)


# ==================================================
# SELECT FOLDER
# ==================================================

IMAGE_DIR = select_folder()

print()
print("Selected folder:")
print(IMAGE_DIR)
print()


# ==================================================
# LOAD JSON IMAGE
# ==================================================

def load_json_image(filename):

    with open(filename, "r") as f:
        obj = json.load(f)

    wrapper = obj["image"]

    if wrapper.get("__ndarray__") is not True:
        raise ValueError(
            f"Invalid NumPy wrapper: {filename}"
        )

    img = np.array(
        wrapper["data"],
        dtype=wrapper["dtype"]
    )

    img = img.reshape(
        wrapper["shape"]
    )

    return img


# ==================================================
# FIND JSON FILES
# ==================================================

files = sorted(
    IMAGE_DIR.glob("frame_*.json")
)


if len(files) == 0:

    print("No frame_*.json files found in:")
    print(IMAGE_DIR)

    raise SystemExit


print(
    f"Found {len(files)} JSON frames."
)


# ==================================================
# LOAD ALL FRAMES
# ==================================================

print("Loading frames...")

frames = []

for i, filename in enumerate(files):

    try:

        img = load_json_image(
            filename
        )

        frames.append(img)

        print(
            f"\rLoaded {i + 1}/{len(files)}",
            end=""
        )

    except Exception as e:

        print(
            f"\nFailed to load:"
        )

        print(filename)

        print(e)


print()


if len(frames) == 0:

    print("No valid frames found.")
    raise SystemExit


print(
    f"Loaded {len(frames)} frames."
)


# ==================================================
# WINDOW
# ==================================================

cv2.namedWindow(
    "HM01B0 JSON Viewer",
    cv2.WINDOW_NORMAL
)


# ==================================================
# PLAYBACK VARIABLES
# ==================================================

index = 0

playing = True

last_frame_time = time.perf_counter()


# ==================================================
# MAIN LOOP
# ==================================================

while True:

    current_time = time.perf_counter()


    # ==============================================
    # DISPLAY FRAME
    # ==============================================

    img = frames[index]


    display = cv2.resize(
        img,
        (
            img.shape[1] * SCALE,
            img.shape[0] * SCALE
        ),
        interpolation=cv2.INTER_NEAREST
    )


    # ==============================================
    # WINDOW TITLE
    # ==============================================

    cv2.setWindowTitle(
        "HM01B0 JSON Viewer",

        f"Frame {index + 1}/{len(frames)} | "
        f"{img.shape[1]}x{img.shape[0]} | "
        f"{FPS} FPS | "
        f"{'PLAYING' if playing else 'PAUSED'}"
    )


    cv2.imshow(
        "HM01B0 JSON Viewer",
        display
    )


    # ==============================================
    # KEYBOARD
    # ==============================================

    key = cv2.waitKey(1) & 0xFF


    # ==============================================
    # QUIT
    # ==============================================

    if key == ord("q"):

        break


    # ==============================================
    # PLAY / PAUSE
    # ==============================================

    elif key == ord(" "):

        playing = not playing

        last_frame_time = (
            time.perf_counter()
        )


    # ==============================================
    # NEXT FRAME
    # ==============================================

    elif key == ord("d"):

        index += 1

        if index >= len(frames):
            index = 0

        last_frame_time = (
            time.perf_counter()
        )


    # ==============================================
    # PREVIOUS FRAME
    # ==============================================

    elif key == ord("a"):

        index -= 1

        if index < 0:
            index = len(frames) - 1

        last_frame_time = (
            time.perf_counter()
        )


    # ==============================================
    # RESET
    # ==============================================

    elif key == ord("r"):

        index = 0

        last_frame_time = (
            time.perf_counter()
        )


    # ==============================================
    # AUTOMATIC PLAYBACK
    # ==============================================

    if playing:

        now = time.perf_counter()

        elapsed = (
            now - last_frame_time
        )


        if elapsed >= FRAME_DELAY:

            frames_to_advance = max(
                1,
                int(
                    elapsed / FRAME_DELAY
                )
            )


            index = (
                index + frames_to_advance
            ) % len(frames)


            last_frame_time = now


# ==================================================
# CLEANUP
# ==================================================

cv2.destroyAllWindows()

print()
print("Viewer closed.")