"""
Geometry-based semantic fallback.

PROTOTYPE NOTE: This is a rule-based classification using geometric features.
It is NOT a neural network. It is clearly labelled in the UI as
"Prototype: geometry fallback".

Future replacement: MinkUNet / SPVCNN pretrained sparse network.

Classes:
  0 ground_drivable
  1 ground_rough
  2 static_obstacle
  3 vehicle
  4 vru
  5 vegetation
"""
from __future__ import annotations
import numpy as np
from typing import Tuple


CLASS_NAMES = [
    "ground_drivable",
    "ground_rough",
    "static_obstacle",
    "vehicle",
    "vru",
    "vegetation",
]
NUM_CLASSES = 6


def classify_tile_geometry(
    tile_points: np.ndarray,
    ground_mask: np.ndarray,
    height_mean: float,
    height_variance: float,
    verticality: float,
    roughness: float,
    point_count: int,
) -> Tuple[np.ndarray, float, float]:
    """
    Classify a tile using geometry rules.

    Returns
    -------
    probs       : (6,) float32 — class probabilities (sum to 1)
    confidence  : float
    uncertainty : float  (1 - confidence, or entropy-based)
    """
    probs = np.ones(NUM_CLASSES, dtype=np.float32) * 0.05  # small prior

    if point_count == 0:
        probs[0] = 0.70  # assume drivable if empty
        probs = probs / probs.sum()
        return probs, float(probs.max()), 1.0 - float(probs.max())

    ground_ratio = ground_mask.mean() if len(ground_mask) > 0 else 0.0

    # Ground-like: mostly ground points, low variance
    if ground_ratio > 0.7:
        if roughness < 0.08:
            probs[0] = 0.75  # ground_drivable
            probs[1] = 0.15  # ground_rough
        else:
            probs[0] = 0.25
            probs[1] = 0.60  # ground_rough

    # Vehicle-like: medium height, wider footprint, box-ish
    elif (0.3 < height_mean < 2.0 and
          verticality > 0.3 and
          point_count > 30):
        probs[3] = 0.65  # vehicle
        probs[2] = 0.20  # static_obstacle

    # VRU-like: narrow footprint, human height range, vertical
    elif (0.5 < height_mean < 2.0 and
          verticality > 0.4 and
          point_count < 80):
        probs[4] = 0.60  # vru
        probs[3] = 0.20

    # Vegetation-like: irregular vertical spread + roughness
    elif (verticality > 0.2 and roughness > 0.3):
        probs[5] = 0.65  # vegetation
        probs[2] = 0.15

    # Static obstacle: large vertical structure
    elif (height_mean > 1.5 and verticality > 0.5):
        probs[2] = 0.70  # static_obstacle
        probs[3] = 0.15

    else:
        # Default: unknown ground
        probs[0] = 0.40
        probs[1] = 0.30
        probs[2] = 0.15

    # Normalise
    probs = np.clip(probs, 1e-6, None)
    probs /= probs.sum()

    confidence = float(probs.max())
    # Normalised entropy
    eps = 1e-9
    entropy = -np.sum(probs * np.log(probs + eps))
    max_entropy = np.log(NUM_CLASSES)
    uncertainty = float(entropy / max_entropy)

    return probs, confidence, uncertainty


class GeometryFallbackBackend:
    """
    Always-available geometry-rule semantic backend.
    Implements the same interface as SparseSemanticBackend.
    """
    name = "Geometry Fallback (prototype rules)"

    def predict_tile(
        self,
        points: np.ndarray,        # (N, 3)
        ground_mask: np.ndarray,   # (N,) bool
        height_mean: float,
        height_variance: float,
        verticality: float,
        roughness: float,
    ) -> Tuple[np.ndarray, float, float]:
        """Return (probs, confidence, uncertainty) for one tile."""
        return classify_tile_geometry(
            points, ground_mask,
            height_mean, height_variance,
            verticality, roughness,
            len(points),
        )
