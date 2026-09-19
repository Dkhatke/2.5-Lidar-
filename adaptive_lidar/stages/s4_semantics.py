"""
S4 — Foveated Sparse Semantic Perception.

ONLY processes tiles that were SELECTED in S3/S6.
This is the core of the adaptive computation idea:
  ~10–30% of tiles enter S4, not 100%.

The active backend is chosen by sparse_backend.build_backend().
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any

from adaptive_lidar.pipeline.types import Frame, PipelineContext, ResolutionLevel
from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.perception.sparse_backend import build_backend


class S4Semantics:
    def __init__(self, config: Dict[str, Any], backend: str = "auto"):
        self.cfg = config
        self._backend = build_backend(mode=backend, config=config)
        self.backend_name = self._backend.name

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S4", frame.timing):
            if not frame.tiles:
                return

            points = frame.points
            intensity = frame.intensity
            ground_mask = frame.ground_mask
            if ground_mask is None:
                ground_mask = np.zeros(len(points), dtype=bool)

            selected_count = 0
            for tile in frame.tiles:
                if not tile.selected:
                    # Default to geometry-prior for non-selected tiles
                    tile.semantic_probs = np.array(
                        [0.5, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)
                    tile.semantic_class = 0
                    tile.semantic_confidence = 0.5
                    tile.semantic_uncertainty = 0.5
                    continue

                selected_count += 1
                idx = tile.point_indices

                if len(idx) == 0:
                    tile.semantic_probs = np.ones(6, dtype=np.float32) / 6
                    tile.semantic_class = 0
                    tile.semantic_confidence = 1.0 / 6
                    tile.semantic_uncertainty = 1.0
                    continue

                tile_pts = points[idx]
                tile_gnd = ground_mask[idx]
                tile_int = intensity[idx]
                vsize = ResolutionLevel.size(tile.resolution_level)

                # Dispatch to the active backend
                backend = self._backend
                if hasattr(backend, "predict_tile"):
                    try:
                        if hasattr(backend, '_net'):  # PrototypeSparseBackend
                            probs, conf, unc = backend.predict_tile(
                                points=tile_pts,
                                ground_mask=tile_gnd,
                                height_mean=tile.height_mean,
                                height_variance=tile.height_variance,
                                intensity_mean=tile.intensity_mean,
                                voxel_size=vsize,
                            )
                        else:  # GeometryFallbackBackend or other
                            probs, conf, unc = backend.predict_tile(
                                tile_pts, tile_gnd,
                                tile.height_mean, tile.height_variance,
                                tile.verticality, tile.roughness,
                            )
                    except Exception:
                        # Fallback to geometry rules
                        from adaptive_lidar.perception.geometry_fallback import (
                            GeometryFallbackBackend,
                        )
                        fb = GeometryFallbackBackend()
                        probs, conf, unc = fb.predict_tile(
                            tile_pts, tile_gnd,
                            tile.height_mean, tile.height_variance,
                            tile.verticality, tile.roughness,
                        )
                else:
                    raise RuntimeError(f"Backend {backend} missing predict_tile()")

                tile.semantic_probs = probs.astype(np.float32)
                tile.semantic_class = int(np.argmax(probs))
                tile.semantic_confidence = float(conf)
                tile.semantic_uncertainty = float(unc)

                # Geometry-semantic disagreement — raises uncertainty
                _check_disagreement(tile)

                # Uncertainty feedback: update score_uncertainty for S6
                tile.score_uncertainty = max(tile.score_uncertainty, tile.semantic_uncertainty)

        # Update telemetry on context
        ctx.semantic_backend_name = self.backend_name


def _check_disagreement(tile):
    """
    Geometry says obstacle (high verticality) but semantics says ground?
    Raise uncertainty to flag for higher allocation in S6.
    """
    if tile.verticality > 0.5 and tile.semantic_class in (0, 1):
        # geometry says obstacle, semantics says ground → disagreement
        tile.semantic_uncertainty = min(tile.semantic_uncertainty + 0.2, 1.0)
