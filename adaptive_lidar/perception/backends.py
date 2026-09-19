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


def classify_ground_points(feats: np.ndarray) -> np.ndarray:
    """(K, 6) evidence for points already known to be ground.

    Drivable road is smooth, planar and low-reflectance; rough ground (grass,
    verge, mud) is none of those.  Returns real evidence, not a hard label, so
    the entropy of a marginal case still reaches the allocation controller.
    """
    if len(feats) == 0:
        return np.zeros((0, NUM_CLASSES), np.float32)
    zvar = feats[:, 3]
    inorm = feats[:, 2]
    planar = feats[:, 10]
    s = np.zeros((len(feats), NUM_CLASSES), np.float32)
    s[:, 0] = 2.0 + 1.5 * planar - 5.0 * np.clip(zvar / 0.05, 0, 3) \
        - 1.2 * np.clip(inorm - 0.6, 0, 2)
    s[:, 1] = 1.6 + 3.0 * np.clip(zvar / 0.05, 0, 2) \
        + 1.0 * np.clip(inorm - 0.4, 0, 2) - 1.0 * planar
    s[:, 2:] = -3.0
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

        s = np.zeros((n, NUM_CLASSES), np.float32)

        # Precompute the shared clipped terms once: each of these appeared in
        # three or four class scores and was being recomputed every time,
        # allocating a fresh 128k array each go.
        near_ground = (hag < 0.25).astype(np.float32)
        zv2 = np.clip(zvar * 20.0, 0.0, 2.0)          # zvar / 0.05, capped
        zv3 = np.clip(zvar * 20.0, 0.0, 3.0)
        zv8 = np.clip(zvar * 12.5, 0.0, 2.0)          # zvar / 0.08
        i_hi = np.clip(inorm - 0.6, 0.0, 2.0)
        i_md = np.clip(inorm - 0.45, 0.0, 2.0)
        i_lo = np.clip(inorm - 0.4, 0.0, 2.0)
        run8 = np.clip(run * 0.125, 0.0, 2.0)
        run5 = np.clip(run * 0.2, 0.0, 1.5)
        run12 = np.clip(run * (1.0 / 12.0), 0.0, 2.0)
        nbr14 = np.clip(nbr * (1.0 / 14.0), 0.0, 2.0)
        nbr_sparse = np.clip((10.0 - nbr) * 0.125, 0.0, 1.5)
        tall = (hag > 0.5).astype(np.float32)
        mid = ((hag > 0.3) & (hag < 2.2)).astype(np.float32)
        human = ((hag > 0.4) & (hag < 2.1)).astype(np.float32)
        bushy = (hag > 0.6).astype(np.float32)

        # 0 ground_drivable - flat, smooth, planar, low reflectance
        s[:, 0] = 2.4 * near_ground + 1.6 * planar - 6.0 * zv3 - 1.2 * i_hi
        # 1 ground_rough - near ground but not smooth
        s[:, 1] = 2.0 * near_ground + 3.0 * zv2 + 1.0 * i_lo - 1.0 * planar
        # 2 static_obstacle - tall, vertical, long ring run, solid
        s[:, 2] = 2.2 * tall + 2.0 * vert + 1.4 * run8 + 1.2 * planar - 3.0 * pen
        # 3 vehicle - 0.3-2.2 m, wide (dense voxel), planar panels, solid
        s[:, 3] = 2.6 * mid + 1.6 * nbr14 + 1.0 * planar - 2.5 * pen - 1.5 * run12
        # 4 vru - human height, narrow, short vertical run, not planar.
        #   Deliberately generous: a false VRU costs budget, a missed one costs
        #   a person.
        s[:, 4] = (3.0 * human + 1.8 * nbr_sparse + 1.4 * run5
                   - 1.6 * planar - 2.0 * pen)
        # 5 vegetation - multi-return, high NIR intensity, rough, non-planar
        s[:, 5] = 3.2 * pen + 1.8 * i_md + 1.6 * zv8 + 1.2 * bushy - 2.0 * planar

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
