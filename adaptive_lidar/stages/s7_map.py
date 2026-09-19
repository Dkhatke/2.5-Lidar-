"""
S7 — Adaptive 2.5D Map Update.
S8 — Temporal Fusion.
S9 — Output / Telemetry collation.
"""
# ── S7 ────────────────────────────────────────────────────────────────────
from __future__ import annotations
import numpy as np
from typing import Dict, Any

from adaptive_lidar.pipeline.types import Frame, PipelineContext
from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.mapping.adaptive_map import AdaptiveMap


class S7Map:
    def __init__(self, config: Dict[str, Any], ctx: PipelineContext):
        self.cfg = config
        self._amap = AdaptiveMap(config)
        # Share map reference with context
        ctx.map_cells = self._amap.cells
        self._amap_ref = self._amap

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S7", frame.timing):
            pts = frame.points
            gnd = frame.ground_mask
            if gnd is None:
                gnd = np.zeros(len(pts), dtype=bool)

            # Build point-level semantic label array from tile labels
            sem_labels = np.zeros(len(pts), dtype=np.int32)
            if frame.tiles:
                for tile in frame.tiles:
                    idx = tile.point_indices
                    if len(idx) > 0 and tile.semantic_class >= 0:
                        sem_labels[idx] = tile.semantic_class

            self._amap.update_from_points(
                points=pts,
                ground_mask=gnd,
                semantic_labels=sem_labels,
                semantic_probs_map={},
                tile_list=frame.tiles,
                timestamp=frame.timestamp,
            )
            # Expose the map helper for visualisation
            ctx._amap = self._amap


# ── S8 ────────────────────────────────────────────────────────────────────
class S8Fusion:
    """
    Temporal fusion — currently handled inside AdaptiveMap.update_from_points()
    via weighted running average. This stage is a hook for future upgrades.
    PROTOTYPE NOTE: Future replacement → probabilistic elevation mapping.
    """
    def __init__(self, config: Dict[str, Any], ctx: PipelineContext):
        self.cfg = config

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S8", frame.timing):
            pass  # Fusion logic runs inside S7/AdaptiveMap (temporal_decay)


# ── S9 ────────────────────────────────────────────────────────────────────
class S9Output:
    """Collate per-frame telemetry into frame.timing dict."""

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S9", frame.timing):
            tiles = frame.tiles or []
            n_high = sum(1 for t in tiles if t.resolution_level == 0)
            n_med = sum(1 for t in tiles if t.resolution_level in (1, 2))
            n_low = sum(1 for t in tiles if t.resolution_level in (3, 4))
            n_selected = sum(1 for t in tiles if t.selected)
            n_safety = sum(1 for t in tiles if t.safety_pinned)

            frame.timing["telemetry"] = {
                "point_count": len(frame.points),
                "total_tiles": len(tiles),
                "high_res_tiles": n_high,
                "medium_tiles": n_med,
                "low_res_tiles": n_low,
                "selected_tiles": n_selected,
                "safety_pinned": n_safety,
                "map_cells": len(ctx.map_cells),
                "frame_id": frame.frame_id,
                "backend": ctx.semantic_backend_name,
                "total_latency_ms": sum(
                    v for k, v in frame.timing.items()
                    if isinstance(v, float) and k.startswith("S")
                ),
            }
