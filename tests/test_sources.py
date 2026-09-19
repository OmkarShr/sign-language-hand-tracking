import io
import json
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from asl_realtime.sources import JsonReplaySource, SerialJPEGSource


def _write_frame(path: Path, value: int) -> None:
    array = np.full((2, 3), value, dtype=np.uint8)
    payload = {
        "image": {
            "__ndarray__": True,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": array.tolist(),
        }
    }
    path.write_text(json.dumps(payload))


def test_json_replay_orders_frames_and_ignores_appledouble(tmp_path):
    _write_frame(tmp_path / "frame_000002.json", 2)
    _write_frame(tmp_path / "frame_000001.json", 1)
    _write_frame(tmp_path / "._frame_000003.json", 3)

    source = JsonReplaySource(tmp_path)
    frames = list(source)

    assert [int(frame.image[0, 0]) for frame in frames] == [1, 2]
    assert [frame.sequence for frame in frames] == [1, 2]
    assert frames[1].timestamp_seconds > frames[0].timestamp_seconds


class FakeSerial:
    def __init__(self, payload: bytes):
        self.stream = io.BytesIO(payload)
        self.writes = []

    def read(self, size: int) -> bytes:
        return self.stream.read(size)

    def write(self, data: bytes) -> None:
        self.writes.append(data)

    def flush(self) -> None:
        pass

    def reset_input_buffer(self) -> None:
        pass

    def close(self) -> None:
        pass


def _serial_source(payload: bytes) -> SerialJPEGSource:
    source = SerialJPEGSource.__new__(SerialJPEGSource)
    source.serial = FakeSerial(payload)
    source.max_jpeg_bytes = 128 * 1024
    return source


def test_serial_source_reads_existing_hm01b0_packet_format():
    ok, encoded = cv2.imencode(".jpg", np.full((4, 5), 120, dtype=np.uint8))
    assert ok
    jpeg = encoded.tobytes()
    source = _serial_source(b"boot-noise*RDY*" + struct.pack("<I", len(jpeg)) + jpeg)
    record = source._capture_one(source.serial, sequence=1, start=0.0)
    assert source.serial.writes == [b"c"]
    assert record.image.shape == (4, 5)


def test_serial_source_rejects_oversized_payload():
    source = _serial_source(b"*RDY*" + struct.pack("<I", 200 * 1024))
    with pytest.raises(ValueError, match="invalid JPEG size"):
        source._capture_one(source.serial, sequence=1, start=0.0)
