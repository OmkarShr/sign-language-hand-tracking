"""Frame sources for recorded JSON sequences and the HM01B0 serial protocol."""

from __future__ import annotations

import json
import struct
import time
import warnings
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import serial


@dataclass(frozen=True)
class FrameRecord:
    image: np.ndarray
    sequence: int
    timestamp_seconds: float


def load_json_image(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    wrapper = payload["image"]
    if wrapper.get("__ndarray__") is not True:
        raise ValueError(f"invalid NumPy wrapper in {path}")
    image = np.asarray(wrapper["data"], dtype=wrapper["dtype"]).reshape(wrapper["shape"])
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError(f"expected uint8 grayscale image in {path}, got {image.dtype} {image.shape}")
    return image


class JsonReplaySource:
    def __init__(self, folder: str | Path, fps: float = 8.6):
        self.folder = Path(folder)
        self.fps = fps
        if fps <= 0:
            raise ValueError("fps must be positive")

    def __iter__(self) -> Iterator[FrameRecord]:
        files = sorted(
            path
            for path in self.folder.glob("frame_*.json")
            if not path.name.startswith("._")
        )
        if not files:
            raise FileNotFoundError(f"no frame_*.json files in {self.folder}")
        for sequence, path in enumerate(files, start=1):
            yield FrameRecord(
                image=load_json_image(path),
                sequence=sequence,
                timestamp_seconds=(sequence - 1) / self.fps,
            )


class SerialJPEGSource:
    """Read grayscale JPEGs from the existing ``c``/``*RDY*`` protocol."""

    READY = b"*RDY*"

    def __init__(
        self,
        port: str,
        *,
        baud: int = 4_000_000,
        timeout: float = 3.0,
        max_jpeg_bytes: int = 128 * 1024,
        max_consecutive_errors: int = 5,
    ):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.max_jpeg_bytes = max_jpeg_bytes
        self.max_consecutive_errors = max_consecutive_errors
        self.transport_failures = 0
        if max_consecutive_errors <= 0:
            raise ValueError("max_consecutive_errors must be positive")

    @staticmethod
    def _read_exact(connection: serial.Serial, size: int) -> bytes:
        result = bytearray()
        while len(result) < size:
            chunk = connection.read(size - len(result))
            if not chunk:
                raise TimeoutError(f"serial read timed out after {len(result)}/{size} bytes")
            result.extend(chunk)
        return bytes(result)

    def _wait_ready(self, connection: serial.Serial) -> None:
        window = bytearray()
        while True:
            value = connection.read(1)
            if not value:
                raise TimeoutError("serial read timed out waiting for *RDY*")
            window.extend(value)
            if len(window) > len(self.READY):
                del window[0]
            if bytes(window) == self.READY:
                return

    def _capture_one(
        self,
        connection: serial.Serial,
        *,
        sequence: int,
        start: float,
    ) -> FrameRecord:
        connection.write(b"c")
        connection.flush()
        self._wait_ready(connection)
        jpeg_size = struct.unpack("<I", self._read_exact(connection, 4))[0]
        if not 0 < jpeg_size <= self.max_jpeg_bytes:
            connection.reset_input_buffer()
            raise ValueError(f"invalid JPEG size: {jpeg_size}")
        jpeg = self._read_exact(connection, jpeg_size)
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError("JPEG decode failed")
        return FrameRecord(image, sequence, time.perf_counter() - start)

    def __iter__(self) -> Iterator[FrameRecord]:
        with serial.Serial(
            self.port,
            self.baud,
            timeout=self.timeout,
            write_timeout=self.timeout,
        ) as connection:
            time.sleep(2)
            connection.reset_input_buffer()
            start = time.perf_counter()
            sequence = 0
            consecutive_errors = 0
            while True:
                try:
                    record = self._capture_one(
                        connection, sequence=sequence + 1, start=start
                    )
                except (TimeoutError, ValueError) as error:
                    consecutive_errors += 1
                    self.transport_failures += 1
                    connection.reset_input_buffer()
                    warnings.warn(
                        f"serial frame failure {consecutive_errors}/"
                        f"{self.max_consecutive_errors}: {error}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    if consecutive_errors >= self.max_consecutive_errors:
                        raise RuntimeError(
                            "serial capture stopped after consecutive frame failures"
                        ) from error
                    continue
                consecutive_errors = 0
                sequence += 1
                yield record
