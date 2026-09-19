import hashlib
import json

import pytest

from setup_asl_assets import download_verified


def test_download_verified_writes_only_matching_payload(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"model")
    target = tmp_path / "target.bin"
    digest = hashlib.sha256(b"model").hexdigest()
    download_verified(source.as_uri(), target, digest, expected_bytes=5, allow_file=True)
    assert target.read_bytes() == b"model"


def test_download_verified_rejects_hash_mismatch(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"wrong")
    target = tmp_path / "target.bin"
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        download_verified(source.as_uri(), target, "0" * 64, expected_bytes=5, allow_file=True)
    assert not target.exists()


def test_download_verified_rejects_non_https_by_default(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"model")
    with pytest.raises(ValueError, match="HTTPS"):
        download_verified(source.as_uri(), tmp_path / "target.bin", "0" * 64, expected_bytes=5)


def test_main_rejects_manifest_path_escape(tmp_path, monkeypatch):
    manifest = tmp_path / "spike" / "assets" / "provenance.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"../../escape": {"url": "https://example.com/x", "bytes": 1, "sha256": "0" * 64}}))
    monkeypatch.setattr("sys.argv", ["setup_asl_assets.py", "--manifest", str(manifest)])
    from setup_asl_assets import main

    with pytest.raises(ValueError, match="escapes"):
        main()
