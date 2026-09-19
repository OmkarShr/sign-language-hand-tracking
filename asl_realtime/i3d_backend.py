"""Pinned WLASL-2000 I3D inference backend used by the feasibility experiment."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .core import parse_class_list


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_runtime_assets(manifest_path: Path, paths: list[Path]) -> dict[str, str]:
    """Verify ignored executable/model assets immediately before loading them."""
    manifest_path = manifest_path.resolve()
    root = manifest_path.parents[1]
    manifest = json.loads(manifest_path.read_text())
    verified: dict[str, str] = {}
    for path in paths:
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"asset is outside manifest root: {path}") from error
        if relative not in manifest:
            raise ValueError(f"asset is not pinned in manifest: {relative}")
        expected = manifest[relative]
        if resolved.stat().st_size != int(expected["bytes"]):
            raise ValueError(f"asset byte-count mismatch: {relative}")
        actual = _sha256(resolved)
        if actual != expected["sha256"]:
            raise ValueError(f"asset SHA-256 mismatch: {relative}")
        verified[relative] = actual
    return verified


def summarize_logits(
    logits: torch.Tensor, labels: list[str], *, top_k: int = 5
) -> list[dict[str, float | int | str]]:
    """Max-pool temporal logits as in the pinned WLASL evaluation path."""
    if logits.ndim == 3:
        logits = logits.amax(dim=2)
    if logits.ndim != 2 or logits.shape[0] != 1:
        raise ValueError(f"expected logits shape (1,C[,T]), got {tuple(logits.shape)}")
    if logits.shape[1] != len(labels):
        raise ValueError("logit count does not match class list")
    probabilities = torch.softmax(logits[0].float(), dim=0)
    values, indices = probabilities.topk(min(top_k, len(labels)))
    return [
        {
            "class_index": int(index),
            "label": labels[int(index)],
            "probability": float(value),
        }
        for value, index in zip(values.cpu(), indices.cpu())
    ]


class WLASLI3DBackend:
    """Load a pinned state dictionary without executing checkpoint pickle code."""

    def __init__(
        self,
        *,
        weights_path: str | Path,
        class_list_path: str | Path,
        model_module_path: str | Path,
        manifest_path: str | Path = "spikes/001-asl-i3d-camera/assets/provenance.json",
        device: str = "cuda",
    ):
        self.weights_path = Path(weights_path)
        class_list_path = Path(class_list_path)
        module_path = Path(model_module_path)
        self.verified_assets = verify_runtime_assets(
            Path(manifest_path), [self.weights_path, class_list_path, module_path]
        )
        self.labels = parse_class_list(class_list_path.read_text())
        if len(self.labels) != 2000:
            raise ValueError(f"expected 2000 WLASL labels, found {len(self.labels)}")

        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = requested

        spec = importlib.util.spec_from_file_location("pinned_wlasl_i3d", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load I3D module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = module.InceptionI3d(400, in_channels=3)
        model.replace_logits(2000)
        state = torch.load(self.weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        self.model = model.eval().to(self.device)

    def predict(
        self, clip: np.ndarray, *, top_k: int = 5
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        tensor = torch.from_numpy(np.ascontiguousarray(clip)).to(self.device)
        if tensor.dtype != torch.float32:
            tensor = tensor.float()

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            logits = self.model(tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        peak_mb = (
            torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
            if self.device.type == "cuda"
            else 0.0
        )
        return summarize_logits(logits, self.labels, top_k=top_k), {
            "latency_ms": elapsed * 1000,
            "peak_allocated_mb": peak_mb,
        }
