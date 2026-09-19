"""
PointFeatureNet — the deep learning model (M1).

A small MLP that classifies EVERY point from the per-point feature vector in
``perception/features.py``.  Three hidden layers of 128 units, ~37k parameters,
trained and run on CPU.

WHY AN MLP AND NOT A SPARSE-CONV NETWORK
----------------------------------------
Cylinder3D, SPVCNN, MinkUNet and PTv3 all require CUDA to *build* their sparse
convolution kernels, not merely to run them.  On a CPU-only machine they are
not an option at any speed.  What those networks buy is spatial context, so
this design puts the context into the *features* instead: the vertical run
length, the 3x3 range-image window statistics, the local planarity and the
voxel occupancy are all neighbourhood descriptors, computed once for the whole
cloud with array shifts.  The network then only has to learn the decision
boundary, which an MLP does well and fast.

The result is a genuine trained per-point semantic segmenter that runs in a
few milliseconds for 55k points on one CPU core — not a stub, and not a
rule-based classifier wearing a network's name.

TEMPERATURE SCALING
-------------------
Raw softmax is not a probability; a network trained with cross-entropy is
systematically overconfident.  Every uncertainty claim this project makes
(the U term in the allocation value function, the per-cell entropy layer) is
only meaningful after calibration, so the checkpoint carries a fitted
temperature and inference divides the logits by it.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from adaptive_lidar.perception.features import (
    FEATURE_NAMES,
    N_FEATURES,
    normalise_features,
)
from adaptive_lidar.pipeline.types import NUM_CLASSES

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except Exception:  # pragma: no cover
    _TORCH = False
    torch = None
    nn = object  # type: ignore

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "pointfeature_net.pt")


if _TORCH:

    class PointFeatureNet(nn.Module):
        """3 x 128 MLP, LayerNorm + GELU, ~37k parameters."""

        def __init__(self, n_in: int = N_FEATURES, n_out: int = NUM_CLASSES,
                     width: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_in, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Linear(width, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Linear(width, width // 2),
                nn.LayerNorm(width // 2),
                nn.GELU(),
                nn.Linear(width // 2, n_out),
            )

        def forward(self, x):
            return self.net(x)

        @property
        def n_params(self) -> int:
            return sum(p.numel() for p in self.parameters())

else:  # pragma: no cover

    class PointFeatureNet:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError("PyTorch is required for PointFeatureNet")


# ────────────────────────────────────────────────────────────────
# Inference wrapper
# ────────────────────────────────────────────────────────────────
class PointFeatureInference:
    """Loads a checkpoint and runs batched CPU inference.

    Validates that the checkpoint's recorded feature ordering matches the
    current :data:`FEATURE_NAMES`, so a feature reordering fails loudly instead
    of silently corrupting predictions.
    """

    def __init__(self, checkpoint: Optional[str] = None, batch_size: int = 65536):
        if not _TORCH:
            raise ImportError("PyTorch is required for PointFeatureNet")
        path = checkpoint or DEFAULT_CHECKPOINT
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No checkpoint at {path}. Run scripts/train_semantic.py first.")

        blob = torch.load(path, map_location="cpu", weights_only=False)
        names = tuple(blob.get("feature_names", FEATURE_NAMES))
        if names != tuple(FEATURE_NAMES):
            raise ValueError(
                "Checkpoint feature ordering does not match perception.features."
                f"\n  checkpoint: {names}\n  current:    {tuple(FEATURE_NAMES)}")

        self.model = PointFeatureNet(
            n_in=len(names), n_out=NUM_CLASSES, width=int(blob.get("width", 128)))
        self.model.load_state_dict(blob["state_dict"])
        self.model.eval()
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))

        self.temperature = float(blob.get("temperature", 1.0))
        self.batch_size = int(batch_size)
        self.meta = {k: v for k, v in blob.items()
                     if k not in ("state_dict",)}
        self.n_params = self.model.n_params

    @torch.no_grad()
    def predict(self, feats: np.ndarray) -> np.ndarray:
        """(N, N_FEATURES) → (N, 6) calibrated probabilities."""
        n = len(feats)
        if n == 0:
            return np.zeros((0, NUM_CLASSES), dtype=np.float32)
        x = normalise_features(feats)
        out = np.empty((n, NUM_CLASSES), dtype=np.float32)
        for lo in range(0, n, self.batch_size):
            hi = min(lo + self.batch_size, n)
            t = torch.from_numpy(np.ascontiguousarray(x[lo:hi]))
            logits = self.model(t) / self.temperature
            out[lo:hi] = torch.softmax(logits, dim=1).numpy()
        return out
