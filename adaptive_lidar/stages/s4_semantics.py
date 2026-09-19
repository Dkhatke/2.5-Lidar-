"""
S4 — Per-Point Semantic Perception.

Produces ``frame.sem_evidence`` (N, 6), ``frame.sem_class`` (N,) and
``frame.sem_entropy`` (N,) — one classification per POINT.

This is the fix for BROKEN 2.  The previous implementation classified whole
2 m x 2 m tiles, so a pedestrian standing on a road shared one label with the
road and one of the two was erased.  Per-point evidence makes the headline
claim demonstrable and per-point accuracy computable.

Tile-level ``semantic_class`` survives only as a DERIVED display field
(the argmax of the tile's summed point evidence).  It is never an input to map
fusion or to allocation.

Also computes the geometry<->semantics disagreement signal (Phase 3.6): two
independent estimators of "is this ground" — the geometric ground mask and the
network's ground-class prediction — disagreeing is free epistemic uncertainty,
with no ensemble to train or run.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from adaptive_lidar.perception.backends import build_backend, evidence_to_class_entropy
from adaptive_lidar.pipeline.timing import stage_timer
from adaptive_lidar.pipeline.types import CLASS_IMPORTANCE, NUM_CLASSES, Frame, PipelineContext
from adaptive_lidar.utils.grouping import segment_reduce

GROUND_CLASSES = (0, 1)


class S4Semantics:
    def __init__(self, config: Dict[str, Any], backend: str = "auto"):
        self.cfg = config
        self._backend = build_backend(mode=backend, config=config)
        self.backend_name = self._backend.name
        self.disagreement_boost = float(
            config.get("semantics", {}).get("disagreement_entropy_boost", 0.35))

    def process(self, frame: Frame, ctx: PipelineContext):
        with stage_timer("S4", frame.timing):
            n = len(frame.points)
            if n == 0:
                frame.sem_evidence = np.zeros((0, NUM_CLASSES), np.float32)
                frame.sem_class = np.zeros(0, np.int8)
                frame.sem_entropy = np.zeros(0, np.float32)
                return

            # ── per-point inference ───────────────────────────
            ev = self._backend.predict(frame).astype(np.float32)
            cls, ent = evidence_to_class_entropy(ev)

            # ── geometry <-> semantics disagreement ───────────
            # Two independent estimators of "is this ground".
            gmask = frame.ground_mask
            if gmask is not None:
                sem_ground = np.isin(cls, GROUND_CLASSES)
                disagree = sem_ground != gmask
                ent = np.clip(ent + self.disagreement_boost * disagree, 0.0, 1.0)
                frame._geom_sem_disagree = disagree
            else:
                frame._geom_sem_disagree = np.zeros(n, bool)

            frame.sem_evidence = ev
            frame.sem_class = cls
            frame.sem_entropy = ent
            # legacy aliases
            frame.semantic_labels = cls.astype(np.int32)
            frame.semantic_confidence = ev.max(axis=1)

            # ── tile roll-up: ALLOCATION features only ────────
            self._rollup_tiles(frame)

            frame.timing["S4_diag"] = {
                "backend": self.backend_name,
                "mean_entropy": float(ent.mean()),
                "disagreement_rate": float(frame._geom_sem_disagree.mean()),
                "class_hist": np.bincount(
                    np.clip(cls, 0, NUM_CLASSES - 1), minlength=NUM_CLASSES).tolist(),
            }

        ctx.semantic_backend_name = self.backend_name

    # ────────────────────────────────────────────────────────
    def _rollup_tiles(self, frame: Frame):
        """Aggregate per-point signals onto tiles for the allocation controller.

        Max-reductions, not means: a tile containing one pedestrian must report
        VRU-level stake even when 400 road points outnumber them 400:1.

        The grouping is the frame's existing Morton ordering — re-sorting the
        cloud here to recover a grouping S2 already computed was costing about
        as much as the semantic inference itself.
        """
        tiles = frame.tiles
        mi = getattr(frame, "morton", None)
        if not tiles or mi is None or mi.n_tiles == 0:
            return

        order = mi.order
        starts = mi.tile_starts
        m = mi.n_tiles

        importance = CLASS_IMPORTANCE[np.clip(frame.sem_class, 0, NUM_CLASSES - 1)]
        max_imp = segment_reduce(importance[order], starts, "max")
        max_ent = segment_reduce(frame.sem_entropy[order], starts, "max")
        ev_sum = segment_reduce(frame.sem_evidence[order], starts, "sum")

        run = getattr(frame, "_vertical_run", None)
        if run is not None:
            max_run = segment_reduce(run[order].astype(np.float32), starts, "max")
            hag_max = segment_reduce(frame.height_above_gnd[order], starts, "max")
            nbr = getattr(frame, "_features", None)
            nbr = (nbr[:, 9] if nbr is not None
                   else np.zeros(len(frame.points), np.float32))
            min_nbr = segment_reduce(nbr[order], starts, "min")
        else:
            max_run = np.zeros(m, np.float32)
            hag_max = np.zeros(m, np.float32)
            min_nbr = np.zeros(m, np.float32)

        # THE GEOMETRIC SAFETY PIN: a run of >= 3 consecutive rings climbing
        # at one azimuth, standing above the ground, in a sparse
        # neighbourhood. "Small, isolated, vertically extended" - without
        # knowing what the object is.
        #
        # Three rings, not four: that is the whole point. At 70 m a standing
        # adult subtends exactly three beams, so a four-ring threshold excludes
        # precisely the case the pin exists to protect. Near structure returns
        # runs of 15-60 and clears any threshold; the far pedestrian is the
        # marginal case, and the threshold has to be set for it.
        pin = (max_run >= 3) & (hag_max > 0.4) & (min_nbr < 40)

        # Occlusion-aware density: how many real beams reached this tile.
        # Point count alone conflates "nothing is there" with "the view was
        # blocked"; a tile hit by many beams that returned nothing is genuinely
        # empty, one hit by three beams is merely unobserved.
        ring = frame.ring
        valid_px = (segment_reduce((ring[order] >= 0).astype(np.float32),
                                   starts, "sum").astype(np.int32)
                    if ring is not None else np.zeros(m, np.int32))

        probs = ev_sum / np.maximum(ev_sum.sum(axis=1, keepdims=True), 1e-9)
        cls = np.argmax(probs, axis=1)
        conf = probs.max(axis=1)

        for j in range(m):
            t = tiles[j]
            t.max_class_importance = float(max_imp[j])
            t.max_entropy = float(max_ent[j])
            t.has_vertical_run = bool(pin[j])
            t.valid_pixel_count = int(valid_px[j])
            t.semantic_probs = probs[j].astype(np.float32)
            t.semantic_class = int(cls[j])
            t.semantic_confidence = float(conf[j])
            t.semantic_uncertainty = float(max_ent[j])
            t.score_uncertainty = max(t.score_uncertainty, float(max_ent[j]))
