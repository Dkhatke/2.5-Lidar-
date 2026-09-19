"""
S6 — Final Allocation (the closed feedback loop).

Re-runs the allocation controller with everything S4 and S5 discovered:

    S4 entropy      -> U term  -> rho up -> finer
    S4 disagreement -> U term
    S5 moving_prob  -> D term
    geometric run   -> SAFETY PIN (not a term — a constraint)

and writes the per-point resolution level onto the frame.  That last step is
the fix for BROKEN 1: previously the level was computed here, shown in the
dashboard, and then thrown away, and the map was built at one uniform cell
size regardless.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from adaptive_lidar.mapping.allocation import (
    AllocationController,
    levels_to_points,
    tile_cells_at_level,
)
from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.pipeline.types import Frame, PipelineContext, VehicleProfile


class S6Allocation:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        self.ctrl = AllocationController(config)
        self.profile = VehicleProfile.wheeled()

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S6", frame.timing):
            tiles = frame.tiles
            n = len(frame.points)
            if not tiles:
                frame.point_level = np.full(n, 4, np.int8)
                return

            # Carry the disagreement signal into the U term.
            dis = getattr(frame, "_geom_sem_disagree", None)
            if dis is not None and frame.tile_of_point is not None:
                self._rollup_disagreement(frame)

            # The budget is a fraction of what a uniform 5 cm map of THIS frame
            # would cost, so the number on the slider means "this much of a
            # uniform 5 cm map's memory".
            cost_table = tile_cells_at_level(frame.morton, len(tiles))
            ref_cells = int(cost_table[:, 0].sum())
            levels = self.ctrl.allocate(
                tiles,
                budget=ctx.budget,
                policy=ctx.allocation_policy,
                vehicle_max_step=self.profile.max_step_m,
                reference_cells=ref_cells,
                cost_table=cost_table,
            )
            frame.tile_cost_table = cost_table

            # THE CONNECTION: per-tile level -> per-point level -> map cell size.
            frame.point_level = levels_to_points(
                tiles, levels, frame.tile_of_point, n)
            frame.tile_levels = levels

            pinned = np.array([t.safety_pinned for t in tiles], bool)
            frame.point_pinned = levels_to_points(
                tiles, pinned.astype(np.int8), frame.tile_of_point, n).astype(bool) \
                if len(tiles) else np.zeros(n, bool)

            diag = dict(self.ctrl.last_diag)
            diag["point_level_hist"] = np.bincount(
                np.clip(frame.point_level, 0, 4), minlength=5).tolist()
            frame.timing["S6_diag"] = diag

    @staticmethod
    def _uniform5_cells(frame: Frame) -> int:
        """Level-0 cells this frame's points occupy — the budget's unit."""
        from adaptive_lidar.utils.grouping import morton_2d
        p = frame.points
        if len(p) == 0:
            return 1
        ix = np.floor(p[:, 0] / 0.05).astype(np.int64)
        iy = np.floor(p[:, 1] / 0.05).astype(np.int64)
        return max(int(len(np.unique(morton_2d(ix, iy)))), 1)

    @staticmethod
    def _rollup_disagreement(frame: Frame):
        from adaptive_lidar.utils.grouping import segment_reduce
        mi = getattr(frame, "morton", None)
        if mi is None or mi.n_tiles == 0:
            return
        d = frame._geom_sem_disagree[mi.order].astype(np.float32)
        frac = segment_reduce(d, mi.tile_starts, "mean")
        for j in range(mi.n_tiles):
            t = frame.tiles[j]
            # A tile where a third of the points disagree is genuinely
            # ambiguous; one stray point is not.
            if frac[j] > 0.3:
                t.max_entropy = min(1.0, t.max_entropy + 0.25)
