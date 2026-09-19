"""
S6 — Final Allocation (Refined After S4 + S5).

Re-runs resolution assignment using:
  - S4 semantic uncertainty (if confidence < fallback threshold, U is raised)
  - S5 motion probability (D is updated from centroid tracking)
  - Updated score_uncertainty / score_dynamic per tile

S3: preliminary — geometry only
S6: refined — geometry + semantics + motion (CLOSED FEEDBACK LOOP)

This is one of the key innovations:

S4 uncertainty ↑  →  score_uncertainty ↑  →  priority ↑  →  resolution ↑
S5 motion ↑       →  score_dynamic ↑     →  priority ↑  →  resolution ↑

Telemetry records before/after priority changes for explainability.
"""
from __future__ import annotations
import numpy as np
from typing import Dict, Any

from adaptive_lidar.pipeline.types import Frame, PipelineContext, ResolutionLevel
from adaptive_lidar.pipeline.timing import stage_timer


class S6Allocation:
    def __init__(self, config: Dict[str, Any]):
        self.cfg = config

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S6", frame.timing):
            if not frame.tiles:
                return

            weights = self.cfg.get("weights", {})
            wg = weights.get("geometry",    0.35)
            ws = weights.get("semantic",    0.20)
            wu = weights.get("uncertainty", 0.25)
            wd = weights.get("dynamic",     0.20)

            res_cfg  = self.cfg.get("resolution", {})
            thresh   = res_cfg.get("thresholds", {})
            mode     = thresh.get("threshold_mode", "percentile")

            conf_fb = self.cfg.get("semantics", {}).get(
                "confidence_fallback_threshold", 0.45)
            mot_prob_thresh = self.cfg.get("thresholds", {}).get(
                "motion", {}).get("probability", 0.55)

            motion_changes = 0
            unc_changes    = 0

            for tile in frame.tiles:
                prev_v = tile.info_value

                # ── Refresh D from S5 motion ──────────────────────
                if tile.motion_probability > tile.score_dynamic:
                    tile.score_dynamic = tile.motion_probability
                    motion_changes += 1

                # ── Refresh U from S4 semantic uncertainty ─────────
                # Low confidence → uncertainty is meaningfully high
                if tile.semantic_confidence > 0 and tile.semantic_confidence < conf_fb:
                    effective_unc = max(tile.score_uncertainty, tile.semantic_uncertainty)
                    if effective_unc > tile.score_uncertainty:
                        tile.score_uncertainty = effective_unc
                        unc_changes += 1

                # ── Safety re-check: motion → pin ─────────────────
                if tile.motion_probability > mot_prob_thresh:
                    if not tile.safety_pinned:
                        tile.safety_pinned = True
                        tile.safety_reason = (
                            tile.safety_reason + ", dynamic"
                            if tile.safety_reason else "dynamic object"
                        )

                # ── Recompute combined V ──────────────────────────
                v = (wg * tile.score_geometry   +
                     ws * tile.score_semantic    +
                     wu * tile.score_uncertainty +
                     wd * tile.score_dynamic)
                tile.info_value = float(np.clip(v, 0.0, 1.0))

            # ── Re-apply budget with refined scores ───────────────
            budget  = ctx.budget
            n_total = len(frame.tiles)
            n_budget= max(1, int(n_total * budget))

            priorities = [(t.priority(), i) for i, t in enumerate(frame.tiles)]
            priorities.sort(reverse=True)
            top_indices = set(i for _, i in priorities[:n_budget])

            for i, tile in enumerate(frame.tiles):
                tile.selected = (i in top_indices) or tile.safety_pinned

            # ── Percentile thresholds within selected occupied ─────
            sel_occ_V = np.array(
                [t.info_value for t in frame.tiles
                 if t.point_count > 0 and (t.selected or t.safety_pinned)],
                dtype=np.float32,
            )
            if mode == "percentile" and len(sel_occ_V) >= 4:
                hp     = thresh.get("high_percentile",   75)
                mp     = thresh.get("medium_percentile", 40)
                high_t = float(np.percentile(sel_occ_V, hp))
                med_t  = float(np.percentile(sel_occ_V, mp))
            else:
                high_t = thresh.get("high",   0.45)
                med_t  = thresh.get("medium", 0.22)

            for i, tile in enumerate(frame.tiles):
                if not tile.selected or tile.point_count == 0:
                    tile.resolution_level = ResolutionLevel.LEVEL_4
                    continue

                v = tile.info_value
                r = tile.range_mean

                if tile.safety_pinned or v >= high_t:
                    level = ResolutionLevel.LEVEL_0
                elif v >= med_t:
                    level = ResolutionLevel.LEVEL_1 if r < 20 else ResolutionLevel.LEVEL_2
                else:
                    level = ResolutionLevel.LEVEL_3

                tile.resolution_level = level

            # Record feedback loop telemetry
            frame.timing["S6_feedback"] = {
                "motion_changes": motion_changes,
                "uncertainty_changes": unc_changes,
                "n_dynamic_tiles": sum(1 for t in frame.tiles if t.score_dynamic > 0.1),
                "threshold_high": high_t if len(sel_occ_V) >= 4 else thresh.get("high", 0.45),
                "threshold_med":  med_t  if len(sel_occ_V) >= 4 else thresh.get("medium", 0.22),
            }
