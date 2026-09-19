"""
S9 — Output / telemetry collation.

Gathers every number the dashboard and the report generator display, so that
nothing downstream has to recompute or invent one.  Every field here was
measured this frame.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from adaptive_lidar.mapping.adaptive_map import uniform_grid_memory
from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.pipeline.types import (
    NUM_CLASSES,
    Frame,
    PipelineContext,
    ResolutionLevel,
    TrackState,
)

STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9")


class S9Output:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S9", frame.timing):
            tiles = frame.tiles or []
            amap = ctx.amap

            level_hist = (np.bincount(np.clip(frame.point_level, 0, 4), minlength=5)
                          if frame.point_level is not None else np.zeros(5, int))
            counts = amap.cell_counts() if amap is not None else {}
            n_cells = amap.n_cells if amap is not None else 0
            map_bytes = amap.nbytes() if amap is not None else 0

            # Uniform 5 cm reference: accumulated over the same frames and the
            # same sliding window as the adaptive map, counting only cells the
            # sensor actually saw.
            ref = getattr(ctx, "uniform_ref", None)
            if ref is not None:
                uni_cells, uni_bytes = ref.n_cells, ref.nbytes()
            else:
                pose = frame.pose if frame.pose is not None else np.eye(4)
                pw = frame.points @ pose[:3, :3].T + pose[:3, 3]
                uni_cells, uni_bytes = uniform_grid_memory(pw[:, :2], 0.05)

            instances = frame.instances or []
            cls_hist = (np.bincount(np.clip(frame.sem_class, 0, NUM_CLASSES - 1),
                                    minlength=NUM_CLASSES).tolist()
                        if frame.sem_class is not None else [0] * NUM_CLASSES)

            stage_ms = {s: float(frame.timing.get(s, 0.0)) for s in STAGES
                        if isinstance(frame.timing.get(s), float)}
            total = sum(stage_ms.values())

            frame.timing["telemetry"] = {
                "frame_id": frame.frame_id,
                "point_count": len(frame.points),
                "backend": ctx.semantic_backend_name,
                "policy": ctx.allocation_policy,
                "data_source": ctx.data_source,
                "budget": ctx.budget,

                "total_tiles": len(tiles),
                "safety_pinned": sum(1 for t in tiles if t.safety_pinned),
                "level_hist_points": level_hist.tolist(),
                "level_names": [ResolutionLevel.name(i) for i in range(5)],

                "map_cells": n_cells,
                "cells_per_level": counts,
                "map_bytes": map_bytes,
                "map_mb": map_bytes / 1e6,
                "uniform5_cells": uni_cells,
                "uniform5_bytes": uni_bytes,
                "uniform5_mb": uni_bytes / 1e6,
                "memory_reduction_pct": (
                    100.0 * (1.0 - map_bytes / uni_bytes) if uni_bytes else 0.0),

                "n_instances": len(instances),
                "n_moving": sum(1 for i in instances if i.state == TrackState.MOVING),
                "n_movable_stationary": sum(
                    1 for i in instances if i.state == TrackState.MOVABLE_BUT_STATIONARY),
                "class_hist": cls_hist,

                "stage_ms": stage_ms,
                "total_latency_ms": total,
                "fps": 1000.0 / total if total > 0 else 0.0,
            }
            ctx.telemetry = frame.timing["telemetry"]
