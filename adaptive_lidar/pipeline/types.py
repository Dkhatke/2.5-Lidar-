"""
SIH26 — Pipeline shared data types.

All stages read/write through Frame and PipelineContext.
No global variables. No duplicated huge arrays — use masks/indices.

THE DATA CONTRACT
-----------------
The point-level fields on :class:`Frame` are a frozen schema.  Everything
downstream (allocation, map, evaluation, dashboard) depends on them being
present, index-aligned to ``frame.points``, and of the declared dtype.
:func:`validate_frame_contract` asserts exactly that and is called at the end
of S5.  Add fields; do not rename or re-type them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

NUM_CLASSES = 6

CLASS_NAMES = [
    "ground_drivable",
    "ground_rough",
    "static_obstacle",
    "vehicle",
    "vru",
    "vegetation",
]

#: Safety stake per class — used by the allocation controller's S term.
#: VRU highest because a missed pedestrian is the failure that matters.
CLASS_IMPORTANCE = np.array(
    [0.20,   # 0 ground_drivable
     0.40,   # 1 ground_rough
     0.70,   # 2 static_obstacle
     0.90,   # 3 vehicle
     1.00,   # 4 vru
     0.30],  # 5 vegetation
    dtype=np.float32,
)


# ────────────────────────────────────────────────────────────────
# Resolution levels
# ────────────────────────────────────────────────────────────────
class ResolutionLevel:
    """Power-of-two cell-size hierarchy sharing one fixed global origin.

    Powers of two (not the PS's illustrative 50 cm) are what make the
    level-l cell id equal to the level-0 Morton code shifted right by 2l bits,
    so a fine cell is always wholly inside exactly one coarse cell.  See
    DECISIONS.md 4.1.
    """
    LEVEL_0 = 0   # 5 cm  — finest
    LEVEL_1 = 1   # 10 cm
    LEVEL_2 = 2   # 20 cm
    LEVEL_3 = 3   # 40 cm
    LEVEL_4 = 4   # 80 cm — coarsest

    N_LEVELS = 5
    BASE = 0.05

    SIZES = {0: 0.05, 1: 0.10, 2: 0.20, 3: 0.40, 4: 0.80}
    NAMES = {0: "5cm", 1: "10cm", 2: "20cm", 3: "40cm", 4: "80cm"}

    @staticmethod
    def size(level: int) -> float:
        return ResolutionLevel.SIZES.get(int(level), 0.80)

    @staticmethod
    def name(level: int) -> str:
        return ResolutionLevel.NAMES.get(int(level), "80cm")

    @staticmethod
    def sizes_array() -> np.ndarray:
        return np.array([0.05, 0.10, 0.20, 0.40, 0.80], dtype=np.float32)


# ────────────────────────────────────────────────────────────────
# Tile — an allocation unit, NOT a semantic unit
# ────────────────────────────────────────────────────────────────
@dataclass
class Tile:
    """A fixed 2 m × 2 m allocation unit.

    A tile carries *allocation features* — aggregate statistics used to decide
    how finely to resolve the area it covers.  It deliberately does NOT carry
    semantic content: a pedestrian standing on a road shares a tile with that
    road, and a single tile-level class would erase one of them.  Semantics are
    per-point (``Frame.sem_evidence``) and per-cell (the map).
    """
    tile_id: int
    ix: int                         # grid index x
    iy: int                         # grid index y
    cx: float                       # centre x (m)
    cy: float                       # centre y (m)
    point_indices: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.int32))

    # ── S2 geometry features ──────────────────────────────────
    point_count: int = 0
    density: float = 0.0
    height_variance: float = 0.0
    height_mean: float = 0.0
    verticality: float = 0.0
    range_mean: float = 0.0
    roughness: float = 0.0
    boundary_score: float = 0.0
    intensity_mean: float = 0.0

    # ── Allocation features (the data contract, Phase 1.3) ────
    valid_pixel_count: int = 0       # occlusion-aware density from range image
    max_class_importance: float = 0.0
    max_entropy: float = 0.0
    max_moving_prob: float = 0.0
    has_vertical_run: bool = False   # geometric pole/VRU detector — the safety pin
    z_spread: float = 0.0
    histogram_gap: float = 0.0       # largest empty vertical band (m)

    # ── S3/S6 scoring ─────────────────────────────────────────
    score_geometry: float = 0.0
    score_semantic: float = 0.0
    score_uncertainty: float = 0.0
    score_dynamic: float = 0.0
    info_value: float = 0.0          # combined V(tile)
    estimated_cost: float = 1.0      # normalised

    # ── Allocation result ─────────────────────────────────────
    resolution_level: int = ResolutionLevel.LEVEL_4
    selected: bool = False
    safety_pinned: bool = False
    safety_reason: str = ""

    # ── Derived display fields (argmax of contained points) ───
    # Present for the dashboard only. Never an input to map fusion.
    semantic_class: int = -1
    semantic_confidence: float = 0.0
    semantic_uncertainty: float = 1.0
    semantic_probs: Optional[np.ndarray] = None

    # ── S5 motion ─────────────────────────────────────────────
    motion_probability: float = 0.0
    instance_id: int = -1

    def priority(self) -> float:
        """info_value / estimated_cost — used for ranking within budget."""
        return self.info_value / max(self.estimated_cost, 1e-6)


# ────────────────────────────────────────────────────────────────
# Frame — raw + derived per-frame data
# ────────────────────────────────────────────────────────────────
@dataclass
class Frame:
    """One LiDAR scan and everything the pipeline derives from it.

    Every array documented as "(N,)" is index-aligned to ``points``.
    """
    frame_id: int
    timestamp: float
    points: np.ndarray              # (N, 3) float32 xyz, sensor frame
    intensity: np.ndarray           # (N,)   float32 raw

    # ── Point-level, all length N, index-aligned to points ────
    intensity_norm: Optional[np.ndarray] = None    # (N,)  float32 range+incidence corrected
    ring: Optional[np.ndarray] = None              # (N,)  int16   -1 if unavailable
    azimuth_bin: Optional[np.ndarray] = None       # (N,)  int32   -1 if unavailable
    height_above_gnd: Optional[np.ndarray] = None  # (N,)  float32
    sem_evidence: Optional[np.ndarray] = None      # (N,6) float32 per-point class evidence
    sem_class: Optional[np.ndarray] = None         # (N,)  int8    argmax of evidence
    sem_entropy: Optional[np.ndarray] = None       # (N,)  float32 normalised, 0..1
    moving_prob: Optional[np.ndarray] = None       # (N,)  float32
    instance_id: Optional[np.ndarray] = None       # (N,)  int32   -1 = none

    # EVALUATION ONLY.  ------------------------------------------------------
    # gt_label is ground truth.  It is read by adaptive_lidar/evaluation/ and by
    # the `oracle` semantic backend, and by NOTHING ELSE.  If any code in S2
    # through S8 reads this field the entire evaluation is invalid, because the
    # system would then be scoring itself against information it was given.
    # tests/test_gt_label_isolation.py enforces this mechanically.
    # ------------------------------------------------------------------------
    gt_label: Optional[np.ndarray] = None          # (N,) int8, -1 if unavailable
    gt_instance: Optional[np.ndarray] = None       # (N,) int32, -1 if unavailable
    gt_moving: Optional[np.ndarray] = None         # (N,) bool

    # Multi-echo (from the sensor or the synthetic generator)
    return_number: Optional[np.ndarray] = None     # (N,) int8, 1-based
    return_count: Optional[np.ndarray] = None      # (N,) int8

    # ── Index structures ──────────────────────────────────────
    range_image: Optional[np.ndarray] = None       # (H,W) int32  POINT INDEX, -1 = empty
    range_image_valid: Optional[np.ndarray] = None # (H,W) bool
    range_image_range: Optional[np.ndarray] = None # (H,W) float32 convenience: r of that point
    voxel_hash: Optional[Any] = None               # utils.voxel_hash.VoxelHash (CSR)
    # The frame's single spatial ordering: one argsort of the level-0 Morton
    # codes, reused for tile grouping, the allocation cost table and every map
    # level. See utils/spatial_index.py.
    morton: Optional[Any] = None                   # utils.spatial_index.MortonIndex

    # ── S2 derived ────────────────────────────────────────────
    ground_mask: Optional[np.ndarray] = None       # (N,) bool
    non_ground_mask: Optional[np.ndarray] = None   # (N,) bool
    ground_z: Optional[np.ndarray] = None          # (N,) float32 smooth ground height field
    noise_mask: Optional[np.ndarray] = None        # (N,) bool  dust/rain returns
    tiles: Optional[List[Tile]] = None
    tile_of_point: Optional[np.ndarray] = None     # (N,) int32 index into tiles, -1 = outside

    # ── S6 allocation result ──────────────────────────────────
    # point_level is THE connection that used to be missing: the resolution
    # level the allocation controller chose, travelling with the point into
    # the map so that cells really do end up different physical sizes.
    point_level: Optional[np.ndarray] = None       # (N,) int8 resolution level
    point_pinned: Optional[np.ndarray] = None      # (N,) bool safety-pinned
    tile_levels: Optional[np.ndarray] = None       # (n_tiles,) int8

    # ── Pose / ego motion ─────────────────────────────────────
    pose: Optional[np.ndarray] = None              # (4,4) sensor→world

    # ── S5/S6 derived ─────────────────────────────────────────
    instances: Optional[List["Instance"]] = None

    # ── Legacy aliases kept so older callers keep working ─────
    range_image_xyz: Optional[np.ndarray] = None
    semantic_labels: Optional[np.ndarray] = None
    semantic_confidence: Optional[np.ndarray] = None
    instance_ids: Optional[np.ndarray] = None

    # Timing per stage
    timing: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(len(self.points))


# ────────────────────────────────────────────────────────────────
# Data contract validation
# ────────────────────────────────────────────────────────────────
#: field name → (expected dtype kind(s), trailing shape)
_CONTRACT: Tuple[Tuple[str, str, Tuple[int, ...]], ...] = (
    ("intensity_norm",   "f", ()),
    ("ring",             "i", ()),
    ("azimuth_bin",      "i", ()),
    ("height_above_gnd", "f", ()),
    ("sem_evidence",     "f", (NUM_CLASSES,)),
    ("sem_class",        "i", ()),
    ("sem_entropy",      "f", ()),
    ("moving_prob",      "f", ()),
    ("instance_id",      "i", ()),
    ("gt_label",         "i", ()),
)


def validate_frame_contract(frame: Frame, strict: bool = True) -> List[str]:
    """Assert every point-level contract field exists, is length N, right dtype.

    Called at the end of S5.  Returns the list of problems found; raises
    ``AssertionError`` when ``strict`` (the default) and the list is non-empty.
    """
    n = frame.n_points
    problems: List[str] = []

    for name, kind, tail in _CONTRACT:
        arr = getattr(frame, name, None)
        if arr is None:
            problems.append(f"{name}: missing (expected shape ({n},)+{tail})")
            continue
        arr = np.asarray(arr)
        expected = (n,) + tail
        if arr.shape != expected:
            problems.append(f"{name}: shape {arr.shape}, expected {expected}")
        if arr.dtype.kind != kind:
            problems.append(
                f"{name}: dtype {arr.dtype} (kind {arr.dtype.kind!r}), "
                f"expected kind {kind!r}")

    if frame.range_image is not None:
        ri = np.asarray(frame.range_image)
        if ri.dtype.kind != "i":
            problems.append(
                f"range_image: dtype {ri.dtype} — must hold POINT INDICES "
                f"(int32), not range values")
        if frame.range_image_valid is None:
            problems.append("range_image_valid: missing alongside range_image")

    if strict and problems:
        raise AssertionError(
            "Frame data contract violated:\n  " + "\n  ".join(problems))
    return problems


# ────────────────────────────────────────────────────────────────
# Instance — tracked object
# ────────────────────────────────────────────────────────────────
class TrackState:
    """A parked car is movable but static.  Conflating these gives you either
    permanent holes in car parks or trails behind pedestrians."""
    STATIC = 0
    MOVING = 1
    MOVABLE_BUT_STATIONARY = 2
    NAMES = {0: "STATIC", 1: "MOVING", 2: "MOVABLE_BUT_STATIONARY"}


@dataclass
class Instance:
    instance_id: int
    centroid: np.ndarray            # (3,)
    bbox_min: np.ndarray            # (3,)
    bbox_max: np.ndarray            # (3,)
    point_count: int
    semantic_class: int
    velocity: Optional[np.ndarray] = None   # (3,) m/s — lives HERE, never in cells
    covariance: Optional[np.ndarray] = None # (6,6) Kalman state covariance
    motion_probability: float = 0.0
    is_dynamic: bool = False
    state: int = TrackState.STATIC
    frames_seen: int = 1
    age: int = 1
    n_obs: int = 1

    @property
    def speed(self) -> float:
        return 0.0 if self.velocity is None else float(np.linalg.norm(self.velocity))

    @property
    def extent(self) -> np.ndarray:
        return self.bbox_max - self.bbox_min

    @property
    def state_name(self) -> str:
        return TrackState.NAMES.get(self.state, "?")


# ────────────────────────────────────────────────────────────────
# MapCell — one cell of the adaptive 2.5D map
# ────────────────────────────────────────────────────────────────
class CellFlags:
    MULTI_SURFACE = 1 << 0
    REFINED = 1 << 1
    STALE = 1 << 2
    GEOM_SEM_DISAGREEMENT = 1 << 3
    SAFETY_PINNED = 1 << 4
    NAMES = {
        1 << 0: "multi_surface",
        1 << 1: "refined",
        1 << 2: "stale",
        1 << 3: "geom_sem_disagreement",
        1 << 4: "safety_pinned",
    }


class Occupancy:
    FREE = 0
    OCCUPIED = 1
    UNKNOWN = 2
    NAMES = {0: "FREE", 1: "OCCUPIED", 2: "UNKNOWN"}


#: The 22-byte on-disk / in-memory cell layout.  One NumPy structured array per
#: level, never a dict of Python objects — see DECISIONS.md 4.3.
CELL_DTYPE = np.dtype([
    ("ground_z",            np.float16),   # 2
    ("z_max",               np.float16),   # 2
    ("overhead_clearance",  np.float16),   # 2
    ("z_var",               np.float16),   # 2
    ("n_points",            np.uint16),    # 2
    ("evidence_top3_cls",   np.uint8, 3),  # 3   class ids
    ("evidence_top3_val",   np.uint8, 3),  # 3   quantised mass
    ("evidence_residual",   np.uint8),     # 1   mass outside the top 3
    ("entropy",             np.uint8),     # 1
    ("occupancy_logodds",   np.int8),      # 1
    ("dynamic_prob",        np.uint8),     # 1
    ("last_seen",           np.uint16),    # 2
    ("intensity_mean",      np.uint8),     # 1
    ("intensity_var",       np.uint8),     # 1
    ("penetration",         np.uint8),     # 1
    ("observability",       np.uint8),     # 1
    ("flags",               np.uint8),     # 1
])                                          # = 27 B packed; see BYTES_PER_CELL


#: Payload size actually occupied per cell, including the int64 Morton key that
#: addresses it.  Reported alongside the tracemalloc measurement so the two can
#: be sanity-checked against each other.
BYTES_PER_CELL = CELL_DTYPE.itemsize + 8


class Traversability:
    DRIVABLE = 0
    CAUTION = 1
    BLOCKED = 2
    NAMES = {0: "DRIVABLE", 1: "CAUTION", 2: "BLOCKED"}


@dataclass
class VehicleProfile:
    """Drivability is a property of the vehicle; slope and step are properties
    of the terrain.  The map stores the terrain; this selects the verdict."""
    name: str = "wheeled"
    max_slope_deg: float = 15.0
    max_step_m: float = 0.15
    min_clearance_m: float = 2.5
    max_roughness: float = 0.08

    @staticmethod
    def wheeled() -> "VehicleProfile":
        return VehicleProfile("wheeled", 15.0, 0.15, 2.5, 0.08)

    @staticmethod
    def tracked() -> "VehicleProfile":
        return VehicleProfile("tracked", 30.0, 0.40, 2.8, 0.20)


@dataclass
class MapCell:
    """Python view of one cell — produced on demand by the cell inspector.

    The map does NOT store these; it stores ``CELL_DTYPE`` structured arrays.
    This class exists so the dashboard and tests can talk about a single cell
    without knowing the packing.
    """
    cx: float
    cy: float
    resolution: float
    level: int = ResolutionLevel.LEVEL_2

    ground_z: float = 0.0
    z_max: float = 0.0
    overhead_clearance: float = np.inf
    z_var: float = 0.0
    n_points: int = 0

    evidence: Optional[np.ndarray] = None    # (6,) reconstructed, sums to ~1
    semantic_class: int = -1
    entropy: float = 1.0

    occupancy_logodds: float = 0.0
    occupancy_state: int = Occupancy.UNKNOWN
    dynamic_probability: float = 0.0
    last_seen: int = 0

    intensity_mean: float = 0.0
    intensity_var: float = 0.0
    penetration: float = 0.0
    observability: float = 0.0
    flags: int = 0

    # ── legacy aliases used by older visualisation code ───────
    @property
    def height_variance(self) -> float:
        return self.z_var

    @property
    def semantic_probs(self) -> Optional[np.ndarray]:
        return self.evidence

    @property
    def semantic_confidence(self) -> float:
        return 0.0 if self.evidence is None else float(self.evidence.max())

    @property
    def is_unknown(self) -> bool:
        return self.occupancy_state == Occupancy.UNKNOWN

    @property
    def log_odds(self) -> float:
        return self.occupancy_logodds

    @property
    def obstacle_height(self) -> float:
        """Derived at query time, never stored."""
        return max(self.z_max - self.ground_z, 0.0)

    def flag_names(self) -> List[str]:
        return [v for k, v in CellFlags.NAMES.items() if self.flags & k]


# ────────────────────────────────────────────────────────────────
# PipelineContext — mutable state shared across stages
# ────────────────────────────────────────────────────────────────
@dataclass
class PipelineContext:
    config: Dict[str, Any]
    frame_history: List[Frame] = field(default_factory=list)
    instance_history: List[Dict[int, Instance]] = field(default_factory=list)
    tracks: Dict[int, Instance] = field(default_factory=dict)
    frame_count: int = 0
    budget: float = 0.80
    semantic_backend_name: str = "geometry_rules"
    allocation_policy: str = "full"
    data_source: str = "synthetic"
    cumulative_timing: Dict[str, List[float]] = field(default_factory=dict)
    amap: Optional[Any] = None           # mapping.adaptive_map.AdaptiveMap
    uniform_ref: Optional[Any] = None    # mapping.adaptive_map.UniformReference
    dynamic_overlay: Optional[Any] = None
    telemetry: Dict[str, Any] = field(default_factory=dict)

    # Legacy: some visualisation code still reads ctx.map_cells.
    @property
    def map_cells(self):
        return {} if self.amap is None else self.amap.cells_view()

    def add_frame(self, frame: Frame):
        max_hist = self.config.get("pipeline", {}).get("max_frames_history", 5)
        self.frame_history.append(frame)
        if len(self.frame_history) > max_hist:
            self.frame_history.pop(0)
        self.frame_count += 1

    def prev_frame(self) -> Optional[Frame]:
        if len(self.frame_history) >= 2:
            return self.frame_history[-2]
        if len(self.frame_history) == 1:
            return self.frame_history[-1]
        return None

    def record_timing(self, stage: str, duration_ms: float):
        self.cumulative_timing.setdefault(stage, []).append(duration_ms)

    def _pct(self, stage: str, q: float) -> float:
        vals = self.cumulative_timing.get(stage) or [0.0]
        return float(np.percentile(vals, q))

    def p50(self, stage: str) -> float:
        return self._pct(stage, 50)

    def p95(self, stage: str) -> float:
        return self._pct(stage, 95)

    def p99(self, stage: str) -> float:
        return self._pct(stage, 99)
