"""
SparseSemanticBackend — auto-detect + fallback chain.

Priority:
  1. MinkowskiBackend   (real sparse convolution)
  2. PrototypeSparse    (lightweight PyTorch)
  3. GeometryFallback   (pure numpy rules)

The active backend is always logged and shown in the dashboard.
"""
from __future__ import annotations
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def build_backend(mode: str = "auto", config: Dict[str, Any] = None):
    """
    Auto-detect and return the best available semantic backend.

    mode: "auto" | "minkowski" | "prototype" | "geometry"
    """
    config = config or {}

    if mode in ("auto", "minkowski"):
        try:
            from adaptive_lidar.perception.minkowski_backend import MinkowskiBackend
            backend = MinkowskiBackend()
            logger.info("Semantic backend: MinkowskiEngine ✓")
            return backend
        except Exception as e:
            if mode == "minkowski":
                logger.warning(f"MinkowskiEngine requested but unavailable: {e}")
            else:
                logger.info(f"MinkowskiEngine not available ({type(e).__name__}), falling back.")

    if mode in ("auto", "prototype"):
        try:
            from adaptive_lidar.perception.prototype_sparse_backend import PrototypeSparseBackend
            backend = PrototypeSparseBackend()
            logger.info("Semantic backend: Prototype Sparse (PyTorch) ✓")
            return backend
        except Exception as e:
            if mode == "prototype":
                logger.warning(f"PrototypeSparse requested but unavailable: {e}")
            else:
                logger.info(f"PrototypeSparse unavailable ({type(e).__name__}), falling back.")

    # Always works
    from adaptive_lidar.perception.geometry_fallback import GeometryFallbackBackend
    backend = GeometryFallbackBackend()
    logger.info("Semantic backend: Geometry Fallback (prototype rules) ✓")
    return backend
