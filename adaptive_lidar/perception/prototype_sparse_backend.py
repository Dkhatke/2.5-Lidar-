"""
Prototype Sparse Backend.

A lightweight PyTorch network operating on sparse voxel coordinates.
Only processes SELECTED tile coordinates — NOT a dense 3D tensor.

PROTOTYPE NOTE:
  This implements the sparse-network INTERFACE without a full MinkUNet.
  Architectural concept is preserved:
    SPARSE COORDINATES → SPARSE FEATURES → LOCAL PROCESSING → SEMANTIC LOGITS

Future replacement: MinkUNet / SPVCNN pretrained checkpoint.
"""
from __future__ import annotations
import numpy as np
from typing import Tuple, Optional

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

NUM_CLASSES = 6


class _SparseVoxelNet(object):
    """
    Tiny MLP that operates on per-voxel feature vectors.
    Input features per voxel: [mean_x, mean_y, mean_z, point_count,
                                height_var, intensity_mean, ground_ratio]
    """
    _instance: Optional["_SparseVoxelNet"] = None

    def __init__(self, device: str = "cpu"):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        import torch.nn as nn
        self.device = torch.device(device)
        self.net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, NUM_CLASSES),
        ).to(self.device)
        # Note: weights are randomly initialised — this is a prototype
        # Future: load pretrained MinkUNet checkpoint here
        self.net.eval()

    def forward(self, features: np.ndarray) -> np.ndarray:
        """features: (M, 7) → probs: (M, 6)"""
        import torch
        x = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.net(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return probs


class PrototypeSparseBackend:
    """
    Lightweight sparse PyTorch backend.
    Operates only on selected sparse voxel coordinates.
    """
    name = "Prototype Sparse (PyTorch MLP — randomly initialised)"

    def __init__(self, device: str = "cpu"):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for PrototypeSparseBackend")
        self._net = _SparseVoxelNet(device)

    def predict_tile(
        self,
        points: np.ndarray,        # (N, 3)
        ground_mask: np.ndarray,   # (N,) bool
        height_mean: float,
        height_variance: float,
        intensity_mean: float,
        voxel_size: float = 0.20,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Voxelise tile at the assigned resolution,
        build one feature vector per occupied voxel,
        run the sparse MLP, average over voxels.

        Returns (probs, confidence, uncertainty).
        """
        if len(points) == 0:
            probs = np.ones(NUM_CLASSES) / NUM_CLASSES
            return probs.astype(np.float32), 1.0 / NUM_CLASSES, 1.0

        # Voxelise
        inv = 1.0 / voxel_size
        keys = np.floor(points * inv).astype(np.int32)
        unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

        n_voxels = len(unique_keys)
        feats = np.zeros((n_voxels, 7), dtype=np.float32)

        for vi in range(n_voxels):
            mask = inverse == vi
            vpts = points[mask]
            vgnd = ground_mask[mask] if len(ground_mask) == len(points) else np.zeros(mask.sum(), bool)
            feats[vi, 0] = vpts[:, 0].mean() / 50.0   # normalise to ~50m range
            feats[vi, 1] = vpts[:, 1].mean() / 50.0
            feats[vi, 2] = vpts[:, 2].mean() / 3.0    # normalise to ~3m height
            feats[vi, 3] = np.log1p(mask.sum()) / 6.0  # log-count
            feats[vi, 4] = np.var(vpts[:, 2]) / 4.0
            feats[vi, 5] = intensity_mean
            feats[vi, 6] = vgnd.mean()

        voxel_probs = self._net.forward(feats)          # (n_voxels, 6)
        probs = voxel_probs.mean(axis=0).astype(np.float32)
        probs = probs / probs.sum()

        confidence = float(probs.max())
        eps = 1e-9
        entropy = -np.sum(probs * np.log(probs + eps))
        uncertainty = float(entropy / np.log(NUM_CLASSES))

        return probs, confidence, uncertainty
