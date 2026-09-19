"""
Semantic backends — three implementations behind one interface.

    predict(frame) -> (N, 6) float32 evidence, rows summing to 1

``geometry_rules``   Vectorised per-point rules. Always available, no training,
                     no PyTorch. The honest floor the network must beat.
``pointfeature_net`` The trained MLP (M1). Default.
``oracle``           Returns the ground-truth labels directly. EVALUATION ONLY.

WHY AN ORACLE BACKEND EXISTS
----------------------------
It separates two questions that are otherwise tangled: "is the map good?" and
"is the segmenter good?".  Running the whole allocation-and-mapping pipeline on
perfect semantics measures the ceiling of the map design alone.  If adaptive
allocation beats uniform under the oracle, the mechanism works and any shortfall
in the real numbers is the segmenter's; if it does not, the mechanism is at
fault.  That is a genuinely useful experiment, and it is only honest if the
mode is impossible to run by accident — hence the loud warning, and the
``backend`` column recorded in every metrics row.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

import numpy as np

from adaptive_lidar.pipeline.types import NUM_CLASSES

logger = logging.getLogger(__name__)

_EPS = 1e-9
_LOG_NC = float(np.log(NUM_CLASSES))

#: Ground points are separated geometrically before any classifier runs, and
#: the only question left about them is drivable vs rough — a two-way decision
#: that surface roughness and corrected intensity answer directly.  Spending
#: network inference on the ~60% of a scan that is road surface costs most of
#: the frame budget and buys almost nothing, so the default is to answer it
#: with the rule below and give the network the points where the answer is
#: genuinely uncertain.  Set semantics.classify_ground: net to override; the
#: evaluation reports both.
GROUND_RULE_CLASSES = (0, 1)

#: Corrected-intensity boundary between drivable and rough ground, and the
#: softness of the transition. Both read off the measured distributions rather
#: than chosen: see classify_ground_points.
GROUND_INTENSITY_BOUNDARY = 0.20
GROUND_INTENSITY_WIDTH = 0.06


def classify_ground_points(feats: np.ndarray) -> np.ndarray:
    """(K, 6) evidence for points already known to be ground.

    The discriminator is CORRECTED intensity, and it is nearly clean.
    Measured on the synthetic scan (median, interquartile range):

        drivable (asphalt)  0.152   [0.135, 0.169]
        rough (grass/verge) 0.361   [0.217, 0.446]

    which is the material reflectance the LiDAR equation predicts — 0.15 for
    asphalt against 0.45 for vegetation in the near infrared — recovered by
    the range and incidence correction in `features.normalise_intensity`.
    Without that correction the same two surfaces are indistinguishable,
    because a bright surface at 60 m returns less than a dark one at 5 m.
    This rule is the clearest evidence that the correction earns its cost.

    Geometric roughness is kept as secondary evidence rather than primary: on
    a real verge it helps, but on smooth grass it says nothing, and an earlier
    version of this rule that leaned on it (with an intensity threshold set by
    intuition at 0.6 rather than measured at 0.20) classified 97% of all
    ground as drivable.

    Returns evidence rather than a hard label, so the entropy of a marginal
    case still reaches the allocation controller.
    """
    if len(feats) == 0:
        return np.zeros((0, NUM_CLASSES), np.float32)
    inorm = feats[:, 2]
    zvar = feats[:, 3]
    planar = feats[:, 10]

    # Logistic in corrected intensity about the measured boundary.
    t = (inorm - GROUND_INTENSITY_BOUNDARY) / GROUND_INTENSITY_WIDTH
    s = np.zeros((len(feats), NUM_CLASSES), np.float32)
    s[:, 0] = -2.6 * t + 0.7 * planar - 4.0 * np.clip(zvar * 20.0, 0, 3)
    s[:, 1] = 2.6 * t - 0.7 * planar + 4.0 * np.clip(zvar * 20.0, 0, 3)
    s[:, 2:] = -6.0
    s -= s.max(axis=1, keepdims=True)
    ev = np.exp(s, dtype=np.float32)
    return ev / np.maximum(ev.sum(axis=1, keepdims=True), _EPS)


def evidence_to_class_entropy(ev: np.ndarray):
    """(N,6) evidence → (sem_class int8, sem_entropy float32 in [0,1])."""
    if len(ev) == 0:
        return np.zeros(0, np.int8), np.zeros(0, np.float32)
    p = ev / np.maximum(ev.sum(axis=1, keepdims=True), _EPS)
    cls = np.argmax(p, axis=1).astype(np.int8)
    ent = (-(p * np.log(p + _EPS)).sum(axis=1) / _LOG_NC).astype(np.float32)
    return cls, np.clip(ent, 0.0, 1.0)


# ────────────────────────────────────────────────────────────────
# 1. Geometry rules — per-point, vectorised
# ────────────────────────────────────────────────────────────────
class GeometryRulesBackend:
    """Rule-based per-point classification. No network, no training.

    The rules are the obvious geometric ones — height above ground, vertical
    extent, footprint, intensity, penetration — evaluated as soft scores over
    the whole cloud at once and then softmaxed.  Being soft rather than a
    decision tree matters: the entropy of the result is then a usable
    uncertainty signal instead of a constant.
    """

    name = "geometry_rules"
    requires_gt = False

    def __init__(self, config: Dict[str, Any] | None = None):
        self.cfg = config or {}

    def predict(self, frame) -> np.ndarray:
        from adaptive_lidar.perception.features import extract_features

        n = len(frame.points)
        if n == 0:
            return np.zeros((0, NUM_CLASSES), np.float32)

        f = getattr(frame, "_features", None)
        if f is None:
            f = extract_features(frame)
            frame._features = f

        hag = f[:, 0]
        inorm = f[:, 2]
        zvar = f[:, 3]
        run = f[:, 5]
        pen = f[:, 8]
        nbr = f[:, 9]
        planar = f[:, 10]
        vert = f[:, 11]

        inc = f[:, 7]
        s = np.zeros((n, NUM_CLASSES), np.float32)

        # Coefficients calibrated against MEASURED per-class feature medians
        # on the synthetic scan, not chosen by intuition. The medians that
        # matter (ground / static / vehicle / vru / vegetation):
        #
        #   height_above_ground   0.0  1.08  0.86  1.02  1.96
        #   intensity_norm       0.15  0.29  0.53  0.37  0.53
        #   vertical_run          1.0  23.0  12.0  19.0   1.0
        #   incidence_cos        0.28  0.76  0.36  0.57  0.56
        #   voxel_neighbours      104    38    27    86     5
        #   penetration_ratio     0.0   0.0   0.0   0.0   0.5
        #
        # An earlier version assumed a pedestrian is geometrically SPARSE and
        # has a SHORT ring run. The data says the opposite for anything but
        # the far field, and the rules scored 0.02 IoU on the class they were
        # written to protect. These are the same features read the right way
        # round; the three object classes remain genuinely hard to separate
        # geometrically, which is the honest reason the network exists.
        near_ground = (hag < 0.25).astype(np.float32)
        zv = np.clip(zvar * 20.0, 0.0, 3.0)
        t_int = (inorm - GROUND_INTENSITY_BOUNDARY) / GROUND_INTENSITY_WIDTH
        run_n = np.clip(run / 24.0, 0.0, 1.0)
        nbr_n = np.clip(nbr / 80.0, 0.0, 1.5)
        tall = (hag > 0.45).astype(np.float32)

        # 0/1 ground: the same corrected-intensity boundary as above.
        s[:, 0] = 3.0 * near_ground - 2.2 * t_int + 0.6 * planar - 3.0 * zv
        s[:, 1] = 3.0 * near_ground + 2.2 * t_int - 0.6 * planar + 3.0 * zv

        # 5 vegetation: multi-return is almost a sufficient statistic.
        s[:, 5] = (5.0 * pen + 1.2 * (hag > 1.2) + 1.0 * zv
                   - 1.5 * planar - 1.2 * run_n)

        # 2 static_obstacle: the longest ring runs, the flattest faces, seen
        #    closest to head-on.
        s[:, 2] = (2.0 * tall + 1.6 * vert + 2.0 * run_n
                   + 1.4 * planar + 1.2 * inc - 4.0 * pen)

        # 3 vehicle: bright metal, oblique panels, shorter runs than a wall.
        s[:, 3] = (2.0 * tall + 1.4 * vert + 1.3 * np.clip(t_int, 0, 6) / 6.0 * 3.0
                   - 1.6 * inc - 1.2 * run_n - 3.0 * pen)

        # 4 vru: human height, dense at short range, mid-length runs, oblique.
        s[:, 4] = (3.0 * ((hag > 0.5) & (hag < 2.1)).astype(np.float32)
                   + 1.3 * nbr_n + 1.0 * run_n - 0.9 * inc - 3.0 * pen)

        s -= s.max(axis=1, keepdims=True)
        ev = np.exp(s, dtype=np.float32)
        return ev / np.maximum(ev.sum(axis=1, keepdims=True), _EPS)


# ────────────────────────────────────────────────────────────────
# 2. PointFeatureNet — the trained MLP
# ────────────────────────────────────────────────────────────────
class PointFeatureNetBackend:
    name = "pointfeature_net"
    requires_gt = False

    def __init__(self, config: Dict[str, Any] | None = None, checkpoint=None):
        from adaptive_lidar.perception.pointfeature_net import PointFeatureInference
        cfg = (config or {}).get("semantics", {})
        self._inf = PointFeatureInference(checkpoint)
        self.temperature = self._inf.temperature
        self.n_params = self._inf.n_params
        self.meta = self._inf.meta
        self.classify_ground = cfg.get("classify_ground", "rule")

    def predict(self, frame) -> np.ndarray:
        from adaptive_lidar.perception.features import extract_features

        f = getattr(frame, "_features", None)
        if f is None:
            f = extract_features(frame)
            frame._features = f

        n = len(f)
        gmask = frame.ground_mask
        if self.classify_ground == "net" or gmask is None or not gmask.any():
            return self._inf.predict(f)

        out = np.zeros((n, NUM_CLASSES), np.float32)
        ng = np.flatnonzero(~gmask)
        g = np.flatnonzero(gmask)
        if ng.size:
            out[ng] = self._inf.predict(f[ng])
        if g.size:
            out[g] = classify_ground_points(f[g])
        return out


# ────────────────────────────────────────────────────────────────
# 3. Oracle — EVALUATION ONLY
# ────────────────────────────────────────────────────────────────
class OracleBackend:
    """Returns ground truth. Reads ``frame.gt_label`` — one of only two places
    in the codebase permitted to (the other is ``evaluation/``)."""

    name = "oracle"
    requires_gt = True

    def __init__(self, config: Dict[str, Any] | None = None, confidence: float = 0.98):
        self.confidence = float(confidence)
        logger.warning(
            "=" * 68 + "\n"
            "  ORACLE SEMANTIC BACKEND ACTIVE — ground-truth labels, NOT a\n"
            "  prediction. Any accuracy number produced in this mode measures\n"
            "  the MAP, never the segmenter. Never use for reported accuracy.\n"
            + "=" * 68)

    def predict(self, frame) -> np.ndarray:
        n = len(frame.points)
        gt = frame.gt_label
        spread = (1.0 - self.confidence) / (NUM_CLASSES - 1)
        ev = np.full((n, NUM_CLASSES), spread, np.float32)
        if gt is None:
            return np.full((n, NUM_CLASSES), 1.0 / NUM_CLASSES, np.float32)
        gt = np.asarray(gt)
        known = gt >= 0
        ev[known, gt[known]] = self.confidence
        ev[~known] = 1.0 / NUM_CLASSES
        return ev


# ────────────────────────────────────────────────────────────────
# Factory
# ────────────────────────────────────────────────────────────────
_REGISTRY = {
    "geometry_rules": GeometryRulesBackend,
    "pointfeature_net": PointFeatureNetBackend,
    "oracle": OracleBackend,
}

BACKEND_NAMES = tuple(_REGISTRY)


def build_backend(mode: str = "auto", config: Dict[str, Any] | None = None):
    """Build a semantic backend by name.

    ``auto`` prefers the trained network and falls back to geometry rules with
    a clearly logged reason (a missing checkpoint on a fresh clone, typically).
    ``oracle`` is never reachable from ``auto``.
    """
    config = config or {}
    mode = (mode or "auto").lower()
    # Legacy names from the previous CLI.
    mode = {"geometry": "geometry_rules", "prototype": "pointfeature_net",
            "minkowski": "auto"}.get(mode, mode)

    if mode in _REGISTRY and mode != "auto":
        backend = _REGISTRY[mode](config)
        logger.info("Semantic backend: %s", backend.name)
        return backend

    try:
        backend = PointFeatureNetBackend(config)
        logger.info("Semantic backend: pointfeature_net (%d params, T=%.3f)",
                    backend.n_params, backend.temperature)
        return backend
    except Exception as e:
        logger.info("pointfeature_net unavailable (%s: %s) — using geometry_rules.",
                    type(e).__name__, e)
        return GeometryRulesBackend(config)
