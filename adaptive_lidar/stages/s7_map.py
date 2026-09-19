"""
S7 — Adaptive 2.5D Map Update.

Takes the per-point resolution level from S6 and the STATIC points from the S8
gate's point of view, transforms them into WORLD coordinates using the pose,
and inserts them into the variable-resolution map.

WORLD COORDINATES ARE NOT OPTIONAL.  Inserting sensor-frame points into a
persistent map smears every static object across the road as the ego vehicle
moves — a failure that looks like sensor noise and is actually a missing
matrix multiply.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from adaptive_lidar.mapping.adaptive_map import AdaptiveMap, UniformReference
from adaptive_lidar.utils.grouping import morton_2d as _morton
from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.pipeline.types import Frame, PipelineContext


def to_world(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    if pose is None:
        return points
    return (points @ pose[:3, :3].T + pose[:3, 3]).astype(np.float32)


class S7Map:
    def __init__(self, config: Dict[str, Any], ctx: PipelineContext):
        self.cfg = config
        self.amap = AdaptiveMap(config)
        self.carve = bool(config.get("map", {}).get("carve_free_space", True))
        # Accumulated uniform 5 cm map over the SAME observed area and the same
        # sliding window, so the memory comparison is like-for-like rather than
        # one accumulated map against one frame's worth of uniform cells.
        self.uniform_ref = UniformReference(
            cell=0.05, window_m=float(config.get("map", {}).get("window_size", 200.0)))
        ctx.amap = self.amap
        ctx.uniform_ref = self.uniform_ref

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S7", frame.timing):
            n = len(frame.points)
            if n == 0:
                return

            # Points the MOS gate (S8, which runs first on the previous frame's
            # tracks) marked as belonging to a moving object never enter here.
            static = getattr(frame, "static_mask", None)
            if static is None:
                static = np.ones(n, bool)
            noise = getattr(frame, "noise_mask", None)
            if noise is not None and len(noise) == n:
                static = static & ~noise
            sel = np.flatnonzero(static)
            if sel.size == 0:
                return

            pose = frame.pose if frame.pose is not None else np.eye(4)
            pw = to_world(frame.points[sel], pose)
            gz_world = (frame.ground_z[sel] + pose[2, 3]).astype(np.float32) \
                if frame.ground_z is not None else np.zeros(len(sel), np.float32)

            levels = frame.point_level
            if levels is None:
                levels = np.full(n, 4, np.int8)

            # World-frame Morton codes. The frame's sensor-frame ordering is
            # only valid in world coordinates when the pose is a translation
            # plus a yaw that happens to preserve cell order, so the codes are
            # recomputed here and re-sorted once — still one sort, not five.
            inv0 = 1.0 / 0.05
            c0 = _morton(np.floor(pw[:, 0] * inv0).astype(np.int64),
                         np.floor(pw[:, 1] * inv0).astype(np.int64))
            world_order = np.argsort(c0, kind="quicksort")

            self.amap.frame_index = frame.frame_id
            self.amap.update_from_points(
                points=pw,
                levels=levels[sel],
                ground_z=gz_world,
                evidence=frame.sem_evidence[sel],
                intensity=frame.intensity_norm[sel],
                penetration=(frame.return_number[sel].astype(np.float32)
                             / np.maximum(frame.return_count[sel].astype(np.float32), 1.0)
                             if frame.return_count is not None else None),
                entropy=frame.sem_entropy[sel],
                disagreement=getattr(frame, "_geom_sem_disagree", None),
                safety_pinned=(frame.point_pinned[sel]
                               if frame.point_pinned is not None else None),
                timestamp=frame.timestamp,
                frame_index=frame.frame_id,
                code0=c0,
                sorted_order=world_order,
            )

            if self.carve:
                self.amap.carve_free_space(frame, pose)

            # The uniform 5 cm reference is exactly the distinct level-0
            # codes we just computed, so it costs a run scan rather than its
            # own Morton pass and sort.
            self.uniform_ref.update_codes(c0[world_order])

            # Slide the window so accumulation - and pose-drift smearing -
            # stays bounded.
            self.amap.slide_window(pose[:2, 3])
            self.uniform_ref.slide_window(pose[:2, 3])

            ctx.amap = self.amap
            frame.timing["S7_diag"] = {
                "cells_per_level": self.amap.cell_counts(),
                "n_cells": self.amap.n_cells,
                "map_bytes": self.amap.nbytes(),
                "n_static": int(sel.size),
                "n_gated_out": int(n - sel.size),
            }
