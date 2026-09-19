"""
S6 — Final Allocation (Refined After S4 + S5).

Re-runs allocation using:
  - S4 semantic uncertainty
  - S5 motion probability
  - Updated score_uncertainty / score_dynamic per tile

S3: preliminary — geometry only
S6: refined — geometry + semantics + motion

This feedback loop is one of the key things we demonstrate.
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
            wg = weights.get("geometry", 0.35)
            ws = weights.get("semantic", 0.20)
            wu = weights.get("uncertainty", 0.25)
            wd = weights.get("dynamic", 0.20)

            res_cfg = self.cfg.get("resolution", {})
            thresh = res_cfg.get("thresholds", {})
            high_t = thresh.get("high", 0.65)
            med_t = thresh.get("medium", 0.35)

            for tile in frame.tiles:
                # Refresh dynamic score from S5
                tile.score_dynamic = tile.motion_probability

                # Refresh uncertainty score from S4
                if tile.semantic_uncertainty < 1.0:
                    tile.score_uncertainty = max(
                        tile.score_uncertainty, tile.semantic_uncertainty)

                # Recompute combined V
                v = (wg * tile.score_geometry +
                     ws * tile.score_semantic +
                     wu * tile.score_uncertainty +
                     wd * tile.score_dynamic)
                tile.info_value = float(np.clip(v, 0.0, 1.0))

                # Safety re-check: motion → safety pin
                if tile.motion_probability > self.cfg.get(
                        "thresholds", {}).get("motion", {}).get("probability", 0.55):
                    tile.safety_pinned = True
                    tile.safety_reason = (
                        tile.safety_reason + ", dynamic" if tile.safety_reason
                        else "dynamic object"
                    )

            # Re-apply budget with refined scores
            budget = ctx.budget
            n_total = len(frame.tiles)
            n_budget = max(1, int(n_total * budget))

            priorities = [(t.priority(), i) for i, t in enumerate(frame.tiles)]
            priorities.sort(reverse=True)
            top_indices = set(i for _, i in priorities[:n_budget])

            for i, tile in enumerate(frame.tiles):
                selected = (i in top_indices) or tile.safety_pinned
                tile.selected = selected

                if not selected:
                    tile.resolution_level = ResolutionLevel.LEVEL_4
                    continue

                v = tile.info_value
                r = tile.range_mean

                if v >= high_t:
                    level = ResolutionLevel.LEVEL_0
                elif v >= med_t:
                    level = ResolutionLevel.LEVEL_1 if r < 20 else ResolutionLevel.LEVEL_2
                else:
                    level = ResolutionLevel.LEVEL_3

                tile.resolution_level = level
