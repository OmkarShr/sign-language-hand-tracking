import hashlib
import json

import pytest
import torch

from asl_realtime.i3d_backend import summarize_logits, verify_runtime_assets


def test_summarize_logits_uses_pinned_wlasl_temporal_max_pooling():
    logits = torch.tensor(
        [[[9.0, -9.0], [4.0, 4.0], [3.0, 3.0]]], dtype=torch.float32
    )
    result = summarize_logits(logits, ["a", "b", "c"], top_k=3)
    assert [row["label"] for row in result] == ["a", "b", "c"]
    assert result[0]["class_index"] == 0
    assert result[0]["probability"] > result[1]["probability"]
    assert abs(sum(row["probability"] for row in result) - 1.0) < 1e-6


def test_verify_runtime_assets_rejects_modified_executable(tmp_path):
    root = tmp_path / "spike"
    assets = root / "assets"
    vendor = root / "vendor"
    assets.mkdir(parents=True)
    vendor.mkdir()
    module = vendor / "model.py"
    module.write_bytes(b"safe")
    digest = hashlib.sha256(b"safe").hexdigest()
    manifest = assets / "provenance.json"
    manifest.write_text(json.dumps({"vendor/model.py": {"bytes": 4, "sha256": digest}}))
    assert verify_runtime_assets(manifest, [module]) == {"vendor/model.py": digest}
    module.write_bytes(b"evil")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_runtime_assets(manifest, [module])
