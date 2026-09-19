#!/usr/bin/env python3
"""Download pinned ASL experiment assets and verify every byte before use."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(
    url: str,
    target: Path,
    expected_sha256: str,
    *,
    expected_bytes: int,
    allow_file: bool = False,
) -> None:
    scheme = urlparse(url).scheme.lower()
    if scheme != "https" and not (allow_file and scheme == "file"):
        raise ValueError(f"only HTTPS downloads are allowed, got {scheme!r}")
    if expected_bytes <= 0 or expected_bytes > 1_000_000_000:
        raise ValueError(f"invalid expected byte count: {expected_bytes}")
    target = Path(target)
    if target.is_file() and target.stat().st_size == expected_bytes and file_sha256(target) == expected_sha256:
        print(f"verified existing {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    downloaded_bytes = 0
    with urlopen(url, timeout=120) as response, temporary.open("wb") as output:  # nosec B310 -- HTTPS enforced above
        while chunk := response.read(1024 * 1024):
            downloaded_bytes += len(chunk)
            if downloaded_bytes > expected_bytes:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"download exceeded expected size for {target}")
            output.write(chunk)
            digest.update(chunk)
    if downloaded_bytes != expected_bytes:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"byte-count mismatch for {target}: expected {expected_bytes}, got {downloaded_bytes}"
        )
    actual = digest.hexdigest()
    if actual != expected_sha256:
        temporary.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {target}: expected {expected_sha256}, got {actual}"
        )
    temporary.replace(target)
    print(f"downloaded and verified {target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("spikes/001-asl-i3d-camera/assets/provenance.json"),
    )
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    root = args.manifest.resolve().parents[1]
    for relative_path, record in manifest.items():
        target = (root / relative_path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"manifest path escapes asset root: {relative_path}")
        download_verified(
            record["url"],
            target,
            record["sha256"],
            expected_bytes=int(record["bytes"]),
        )


if __name__ == "__main__":
    main()
