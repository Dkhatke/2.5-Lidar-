"""
SIH26 — Pipeline shared data types.

All stages read/write through Frame and PipelineContext.
No global variables. No duplicated huge arrays — use masks/indices.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import numpy as np


# ────────────────────────────────────────────────────────────────
# Resolution levels
# ────────────────────────────────────────────────────────────────
class ResolutionLevel:
    LEVEL_0 = 0   # 5 cm  — finest
    LEVEL_1 = 1   # 10 cm
    LEVEL_2 = 2   # 20 cm
    LEVEL_3 = 3   # 40 cm
    LEVEL_4 = 4   # 80 cm — coarsest

    SIZES = {0: 0.05, 1: 0.10, 2: 0.20, 3: 0.40, 4: 0.80}
    NAMES = {0: "5cm", 1: "10cm", 2: "20cm", 3: "40cm", 4: "80cm"}

    @staticmethod
    def size(level: int) -> float:
        return ResolutionLevel.SIZES.get(level, 0.80)

    @staticmethod
    def name(level: int) -> str:
        return ResolutionLevel.NAMES.get(level, "80cm")


# ────────────────────────────────────────────────────────────────
# Tile
# ────────────────────────────────────────────────────────────────
@dataclass
class Tile:
    tile_id: int
    ix: int                         # grid index x
    iy: int                         # grid index y
    cx: float                       # centre x (m)
    cy: float                       # centre y (m)
    point_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    # S2 geometry features
    point_count: int = 0
    density: float = 0.0
    height_variance: float = 0.0
    height_mean: float = 0.0
    verticality: float = 0.0
    range_mean: float = 0.0
    roughness: float = 0.0
    boundary_score: float = 0.0
    intensity_mean: float = 0.0

    # S3/S6 scoring
    score_geometry: float = 0.0
    score_semantic: float = 0.0
    score_uncertainty: float = 0.0
    score_dynamic: float = 0.0
    info_value: float = 0.0         # combined V(tile)
    estimated_cost: float = 1.0    # normalised

    # Allocation
    resolution_level: int = ResolutionLevel.LEVEL_4
    selected: bool = False
    safety_pinned: bool = False
    safety_reason: str = ""

    # S4 semantic
    semantic_probs: Optional[np.ndarray] = None   # shape (6,)
    semantic_class: int = -1
    semantic_confidence: float = 0.0
    semantic_uncertainty: float = 1.0

    # S5 motion
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
    frame_id: int
    timestamp: float
    points: np.ndarray              # (N, 3) xyz
    intensity: np.ndarray           # (N,)

    # S1 derived
    range_image: Optional[np.ndarray] = None        # (H, W)
    range_image_xyz: Optional[np.ndarray] = None    # (H, W, 3)
    voxel_hash: Optional[Dict] = None               # key → list[int]

    # S2 derived
    ground_mask: Optional[np.ndarray] = None        # (N,) bool
    non_ground_mask: Optional[np.ndarray] = None    # (N,) bool
    tiles: Optional[List[Tile]] = None

    # S4 derived — point-level labels
    semantic_labels: Optional[np.ndarray] = None    # (N,) int
    semantic_confidence: Optional[np.ndarray] = None  # (N,) float

    # S5 derived
    instance_ids: Optional[np.ndarray] = None       # (N,) int

    # Timing per stage
    timing: Dict[str, float] = field(default_factory=dict)


# ────────────────────────────────────────────────────────────────
# Instance — tracked object
# ────────────────────────────────────────────────────────────────
@dataclass
class Instance:
    instance_id: int
    centroid: np.ndarray            # (3,)
    bbox_min: np.ndarray            # (3,)
    bbox_max: np.ndarray            # (3,)
    point_count: int
    semantic_class: int
    velocity: Optional[np.ndarray] = None   # (3,) m/s estimate
    motion_probability: float = 0.0
    is_dynamic: bool = False
    frames_seen: int = 1


# ────────────────────────────────────────────────────────────────
# MapCell — one cell of the adaptive 2.5D map
# ────────────────────────────────────────────────────────────────
@dataclass
class MapCell:
    cx: float
    cy: float
    resolution: float

    # Elevation
    ground_z: float = 0.0
    z_max: float = 0.0
    height_variance: float = 0.0

    # Occupancy (log-odds)
    log_odds: float = 0.0
    occupancy_confidence: float = 0.0

    # Semantic
    semantic_probs: Optional[np.ndarray] = None   # (6,)
    semantic_class: int = -1
    semantic_confidence: float = 0.0

    # Dynamic
    dynamic_probability: float = 0.0
    is_dynamic: bool = False

    # Provenance
    observation_count: int = 0
    last_timestamp: float = 0.0
    resolution_level: int = ResolutionLevel.LEVEL_2
    is_unknown: bool = True


# ────────────────────────────────────────────────────────────────
# PipelineContext — mutable state shared across stages
# ────────────────────────────────────────────────────────────────
@dataclass
class PipelineContext:
    config: Dict[str, Any]
    frame_history: List[Frame] = field(default_factory=list)
    instance_history: List[Dict[int, Instance]] = field(default_factory=list)
    map_cells: Dict[tuple, MapCell] = field(default_factory=dict)
    frame_count: int = 0
    budget: float = 0.80
    semantic_backend_name: str = "geometry_fallback"
    cumulative_timing: Dict[str, List[float]] = field(default_factory=dict)

    def add_frame(self, frame: Frame):
        max_hist = self.config.get("pipeline", {}).get("max_frames_history", 5)
        self.frame_history.append(frame)
        if len(self.frame_history) > max_hist:
            self.frame_history.pop(0)
        self.frame_count += 1

    def prev_frame(self) -> Optional[Frame]:
        if len(self.frame_history) >= 2:
            return self.frame_history[-2]
        return None

    def record_timing(self, stage: str, duration_ms: float):
        if stage not in self.cumulative_timing:
            self.cumulative_timing[stage] = []
        self.cumulative_timing[stage].append(duration_ms)

    def p50(self, stage: str) -> float:
        vals = self.cumulative_timing.get(stage, [0.0])
        return float(np.percentile(vals, 50))

    def p95(self, stage: str) -> float:
        vals = self.cumulative_timing.get(stage, [0.0])
        return float(np.percentile(vals, 95))
