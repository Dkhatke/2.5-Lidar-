"""
S3 — Pre-Allocation: Information Value Scoring.

Computes V(tile) = wg·G + ws·S + wu·U + wd·D
and assigns resolution levels before semantic perception runs.

After S4+S5 complete, S6 refines this using uncertainty + motion feedback.

Weights come from config.yaml — never hardcoded here.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List

from adaptive_lidar.pipeline.types import Frame, PipelineContext, Tile, ResolutionLevel
from adaptive_lidar.pipeline.timing import stage_timer


class S3Preallocation:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def _score_tiles(self, tiles: List[Tile], frame_count: int):
        """Compute normalised information value for each tile."""
        weights = self.cfg.get("weights", {})
        wg = weights.get("geometry", 0.35)
        ws = weights.get("semantic", 0.20)
        wu = weights.get("uncertainty", 0.25)
        wd = weights.get("dynamic", 0.20)

        th_cfg = self.cfg.get("thresholds", {})
        safe_range = th_cfg.get("safety", {}).get("max_range_pin", 20.0)

        # Gather raw geometry values for normalisation
        densities = np.array([t.density for t in tiles], dtype=np.float32)
        h_vars = np.array([t.height_variance for t in tiles], dtype=np.float32)
        verticals = np.array([t.verticality for t in tiles], dtype=np.float32)
        roughness = np.array([t.roughness for t in tiles], dtype=np.float32)
        boundaries = np.array([t.boundary_score for t in tiles], dtype=np.float32)
        ranges = np.array([t.range_mean for t in tiles], dtype=np.float32)
        counts = np.array([t.point_count for t in tiles], dtype=np.float32)

        def norm(arr):
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-6:
                return np.zeros_like(arr)
            return (arr - mn) / (mx - mn)

        # G — geometric complexity
        G = (
            0.30 * norm(h_vars) +
            0.35 * norm(verticals) +
            0.20 * norm(roughness) +
            0.15 * norm(boundaries)
        )

        # S — semantic relevance proxy (before actual semantics, use range + density)
        # Close tiles with decent density are more semantically relevant
        inv_range = norm(1.0 / (ranges + 1.0))
        S = 0.6 * inv_range + 0.4 * norm(densities)

        # U — uncertainty proxy (low density → high uncertainty; unknown area)
        low_density_thresh = self.cfg.get("thresholds", {}).get("uncertainty", {}).get(
            "low_density_points", 10)
        U = np.where(counts < low_density_thresh, 0.8, 0.2)
        # High boundary → uncertain region boundary
        U = np.clip(U + 0.3 * norm(boundaries), 0.0, 1.0)

        # D — dynamic relevance (zero on frame 0, updated in S6 from S5 motion)
        D = np.array([t.motion_probability for t in tiles], dtype=np.float32)

        # Combined information value
        V = wg * G + ws * S + wu * U + wd * D
        V = np.clip(V, 0.0, 1.0)

        # Estimated cost ≈ point count (normalised) — future: sparse network cost
        cost = np.clip(norm(counts + 1.0), 0.05, 1.0)

        for i, tile in enumerate(tiles):
            tile.score_geometry = float(G[i])
            tile.score_semantic = float(S[i])
            tile.score_uncertainty = float(U[i])
            tile.score_dynamic = float(D[i])
            tile.info_value = float(V[i])
            tile.estimated_cost = float(cost[i])

    def _assign_resolution(self, tiles: List[Tile], budget: float):
        """
        Rank tiles by priority = info_value / estimated_cost.
        Apply budget and assign resolution levels.
        Safety pins are always allocated regardless of budget.
        """
        res_cfg = self.cfg.get("resolution", {})
        thresh = res_cfg.get("thresholds", {})
        high_t = thresh.get("high", 0.65)
        med_t = thresh.get("medium", 0.35)

        th_cfg = self.cfg.get("thresholds", {})
        safe_range = th_cfg.get("safety", {}).get("max_range_pin", 20.0)
        ped_h_max = th_cfg.get("safety", {}).get("pedestrian_height_max", 2.2)
        ped_h_min = th_cfg.get("safety", {}).get("pedestrian_height_min", 0.8)
        ped_fp = th_cfg.get("safety", {}).get("pedestrian_footprint_max", 1.5)

        # Mark safety-pinned tiles
        for tile in tiles:
            reasons = []
            # Obstacle close by: high verticality + close range
            if tile.verticality > 0.5 and tile.range_mean < safe_range and tile.point_count > 5:
                reasons.append("nearby obstacle")
            # Pedestrian-sized cluster: narrow footprint + human-height vertical
            tsize = self.cfg.get("tiles", {}).get("size", 2.0)
            if (tile.verticality > 0.4 and
                    ped_h_min < tile.height_mean < ped_h_max and
                    tile.point_count > 3):
                reasons.append("pedestrian-sized")
            # High uncertainty + drivable boundary
            if tile.score_uncertainty > 0.7 and tile.range_mean < safe_range:
                reasons.append("uncertain drivable boundary")
            # Strong temporal change (motion)
            if tile.motion_probability > 0.5:
                reasons.append("dynamic object")

            tile.safety_pinned = len(reasons) > 0
            tile.safety_reason = ", ".join(reasons)

        # Sort by priority for budget allocation
        n_total = len(tiles)
        n_budget = max(1, int(n_total * budget))

        priorities = [(t.priority(), i) for i, t in enumerate(tiles)]
        priorities.sort(reverse=True)
        top_indices = set(i for _, i in priorities[:n_budget])

        for i, tile in enumerate(tiles):
            selected = (i in top_indices) or tile.safety_pinned
            tile.selected = selected

            if not selected:
                tile.resolution_level = ResolutionLevel.LEVEL_4  # 80 cm
                continue

            v = tile.info_value
            r = tile.range_mean

            if v >= high_t:
                level = ResolutionLevel.LEVEL_0  # 5 cm
            elif v >= med_t:
                # Medium tiles: finer if closer
                level = ResolutionLevel.LEVEL_1 if r < 20 else ResolutionLevel.LEVEL_2
            else:
                level = ResolutionLevel.LEVEL_3  # 40 cm

            tile.resolution_level = level

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S3", frame.timing):
            if not frame.tiles:
                return
            self._score_tiles(frame.tiles, ctx.frame_count)
            self._assign_resolution(frame.tiles, ctx.budget)
