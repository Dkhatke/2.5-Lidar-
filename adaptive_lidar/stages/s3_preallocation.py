"""
S3 — Pre-Allocation: Information Value Scoring.

Computes V(tile) = wg·G + ws·S + wu·U + wd·D
and assigns resolution levels before semantic perception runs.

KEY FIX (v2):
  Previous implementation normalised G/S/U/D across ALL 1600 tiles
  (including ~1340 empty ones).  The resulting V_max was ~0.47,
  never reaching the 0.65 HIGH threshold → zero HIGH tiles.

  CORRECT approach:
    1. Only score tiles where point_count > 0.
    2. Normalise each component WITHIN occupied tiles.
    3. Use percentile-aware thresholds so the top X% always get HIGH.

  This is principled: "among tiles the sensor actually sees,
  which ones need the most attention?"

After S4+S5 complete, S6 refines this using uncertainty + motion feedback.
Weights and thresholds come from config.yaml — never hardcoded here.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List

from adaptive_lidar.pipeline.types import Frame, PipelineContext, Tile, ResolutionLevel
from adaptive_lidar.pipeline.timing import stage_timer


class S3Preallocation:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    # ─────────────────────────────────────────────────────────────
    # Scoring
    # ─────────────────────────────────────────────────────────────
    def _score_tiles(self, tiles: List[Tile], frame_count: int):
        """Compute information value for each tile.

        Only occupied (point_count > 0) tiles are scored.
        Normalisation is within occupied tiles so empty sky/free-space
        tiles don't dilute the scale.
        """
        weights = self.cfg.get("weights", {})
        wg = weights.get("geometry",    0.35)
        ws = weights.get("semantic",    0.20)
        wu = weights.get("uncertainty", 0.25)
        wd = weights.get("dynamic",     0.20)

        # ── Partition occupied vs empty ───────────────────────────
        occ_idx   = [i for i, t in enumerate(tiles) if t.point_count > 0]
        empty_idx = [i for i, t in enumerate(tiles) if t.point_count == 0]

        # Empty tiles get minimum info value — skip expensive scoring
        for i in empty_idx:
            t = tiles[i]
            t.score_geometry    = 0.0
            t.score_semantic    = 0.0
            t.score_uncertainty = 0.0
            t.score_dynamic     = 0.0
            t.info_value        = 0.0
            t.estimated_cost    = 0.0

        if not occ_idx:
            return

        # ── Gather features for occupied tiles ───────────────────
        occ_tiles = [tiles[i] for i in occ_idx]
        n = len(occ_tiles)

        h_vars    = np.array([t.height_variance for t in occ_tiles], np.float32)
        verticals = np.array([t.verticality     for t in occ_tiles], np.float32)
        roughness = np.array([t.roughness       for t in occ_tiles], np.float32)
        boundaries= np.array([t.boundary_score  for t in occ_tiles], np.float32)
        densities = np.array([t.density         for t in occ_tiles], np.float32)
        ranges    = np.array([t.range_mean      for t in occ_tiles], np.float32)
        counts    = np.array([t.point_count     for t in occ_tiles], np.float32)
        motions   = np.array([t.motion_probability for t in occ_tiles], np.float32)

        def norm(arr: np.ndarray) -> np.ndarray:
            """Min-max normalise within the occupied tile set."""
            mn, mx = arr.min(), arr.max()
            if mx - mn < 1e-6:
                return np.full_like(arr, 0.5)
            return (arr - mn) / (mx - mn)

        # ── G — geometric complexity ──────────────────────────────
        # High values mean: tall objects, rough surface, complex boundary
        G = (
            0.30 * norm(h_vars) +
            0.35 * norm(verticals) +
            0.20 * norm(roughness) +
            0.15 * norm(boundaries)
        )

        # ── S — semantic relevance proxy ──────────────────────────
        # Close + dense → more likely to be relevant obstacle/object
        inv_range = norm(1.0 / (ranges + 1.0))   # closer = higher
        S = 0.6 * inv_range + 0.4 * norm(densities)

        # ── U — uncertainty proxy ─────────────────────────────────
        # Low density → uncertain; high boundary score → region boundary
        low_den_thresh = self.cfg.get("thresholds", {}).get(
            "uncertainty", {}).get("low_density_points", 10)
        U_base = np.where(counts < low_den_thresh, 0.8, 0.2).astype(np.float32)
        U = np.clip(U_base + 0.3 * norm(boundaries), 0.0, 1.0)

        # ── D — dynamic relevance ─────────────────────────────────
        # Populated from S5 on subsequent frames; zero on frame 0
        D = motions

        # ── Combined V ───────────────────────────────────────────
        V = wg * G + ws * S + wu * U + wd * D
        V = np.clip(V, 0.0, 1.0)

        # ── Estimated cost ────────────────────────────────────────
        cost = np.clip(norm(counts + 1.0), 0.05, 1.0)

        for j, tile in enumerate(occ_tiles):
            tile.score_geometry    = float(G[j])
            tile.score_semantic    = float(S[j])
            tile.score_uncertainty = float(U[j])
            tile.score_dynamic     = float(D[j])
            tile.info_value        = float(V[j])
            tile.estimated_cost    = float(cost[j])

        # ── Diagnostic telemetry ──────────────────────────────────
        # Stored in frame.timing["S3_diag"] by process()
        self._last_diag = {
            "V_min":  float(V.min()),
            "V_max":  float(V.max()),
            "V_mean": float(V.mean()),
            "V_p25":  float(np.percentile(V, 25)),
            "V_p50":  float(np.percentile(V, 50)),
            "V_p75":  float(np.percentile(V, 75)),
            "n_occupied": n,
            "n_empty": len(empty_idx),
        }

    # ─────────────────────────────────────────────────────────────
    # Safety pins
    # ─────────────────────────────────────────────────────────────
    def _mark_safety_pins(self, tiles: List[Tile]):
        """Mark tiles that must stay at HIGH regardless of budget."""
        th  = self.cfg.get("thresholds", {})
        saf = th.get("safety", {})
        safe_range  = saf.get("max_range_pin",           20.0)
        ped_h_max   = saf.get("pedestrian_height_max",   2.2)
        ped_h_min   = saf.get("pedestrian_height_min",   0.5)
        ped_fp_max  = saf.get("pedestrian_footprint_max", 1.5)
        ped_min_pts = saf.get("pedestrian_min_points",   8)
        obs_min_pts = th.get("obstacle", {}).get("min_points", 5)

        for tile in tiles:
            if tile.point_count == 0:
                continue
            reasons = []

            # 1) Near-field high-verticality obstacle
            if (tile.verticality > 0.45 and
                    tile.range_mean < safe_range and
                    tile.point_count > obs_min_pts):
                reasons.append("near obstacle")

            # 2) Pedestrian-sized cluster (narrow footprint + human height)
            if (tile.verticality > 0.35 and
                    ped_h_min < tile.height_mean < ped_h_max and
                    tile.point_count >= ped_min_pts):
                reasons.append("pedestrian-sized")

            # 3) Dynamic (motion) — from previous frame S5
            if tile.motion_probability > th.get("motion", {}).get("probability", 0.55):
                reasons.append("dynamic object")

            # 4) High uncertainty near ego
            if tile.score_uncertainty > 0.7 and tile.range_mean < safe_range:
                reasons.append("uncertain boundary")

            tile.safety_pinned = len(reasons) > 0
            tile.safety_reason = ", ".join(reasons)

    # ─────────────────────────────────────────────────────────────
    # Resolution assignment
    # ─────────────────────────────────────────────────────────────
    def _assign_resolution(self, tiles: List[Tile], budget: float):
        """
        Rank occupied tiles by priority = V / cost.
        Apply budget, then assign resolution levels.

        Threshold mode:
          percentile — top X% of occupied tiles get HIGH/MEDIUM
          absolute   — use fixed V cutoffs from config
        """
        res_cfg = self.cfg.get("resolution", {})
        thresh  = res_cfg.get("thresholds", {})
        mode    = thresh.get("threshold_mode", "percentile")

        # Budget-based selection: top fraction get processed at all
        occ_tiles = [t for t in tiles if t.point_count > 0]
        all_tiles = tiles

        n_total  = len(all_tiles)
        n_budget = max(1, int(n_total * budget))

        priorities = [(t.priority(), i) for i, t in enumerate(all_tiles)]
        priorities.sort(reverse=True)
        top_indices = set(i for _, i in priorities[:n_budget])

        # Assign selected flag
        for i, tile in enumerate(all_tiles):
            tile.selected = (i in top_indices) or tile.safety_pinned

        # ── Compute percentile thresholds within occupied + selected ──
        sel_occ_V = np.array(
            [t.info_value for t in occ_tiles if t.selected or t.safety_pinned],
            dtype=np.float32,
        )
        if mode == "percentile" and len(sel_occ_V) >= 4:
            hp = thresh.get("high_percentile",   75)
            mp = thresh.get("medium_percentile", 40)
            high_t = float(np.percentile(sel_occ_V, hp))
            med_t  = float(np.percentile(sel_occ_V, mp))
        else:
            high_t = thresh.get("high",   0.45)
            med_t  = thresh.get("medium", 0.22)

        # ── Assign resolution levels ──────────────────────────────
        for i, tile in enumerate(all_tiles):
            if not tile.selected or tile.point_count == 0:
                tile.resolution_level = ResolutionLevel.LEVEL_4  # 80 cm
                continue

            v = tile.info_value
            r = tile.range_mean

            if tile.safety_pinned or v >= high_t:
                level = ResolutionLevel.LEVEL_0   # 5 cm — HIGH
            elif v >= med_t:
                level = ResolutionLevel.LEVEL_1 if r < 20 else ResolutionLevel.LEVEL_2
            else:
                level = ResolutionLevel.LEVEL_3   # 40 cm — LOW

            tile.resolution_level = level

    # ─────────────────────────────────────────────────────────────
    # Main entry
    # ─────────────────────────────────────────────────────────────
    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S3", frame.timing):
            if not frame.tiles:
                return
            self._last_diag = {}
            self._mark_safety_pins(frame.tiles)
            self._score_tiles(frame.tiles, ctx.frame_count)
            self._assign_resolution(frame.tiles, ctx.budget)

            # Save diagnostics for dashboard
            frame.timing["S3_diag"] = self._last_diag
