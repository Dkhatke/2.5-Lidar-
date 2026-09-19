"""
Dataset label id → our 6-class taxonomy.

THE 6 CLASSES — used everywhere in this project::

    0 ground_drivable    road, parking, lane markings
    1 ground_rough       sidewalk, terrain, grass, mud
    2 static_obstacle    building, wall, fence, pole, trunk, sign
    3 vehicle            car, truck, bus, bicycle, motorcycle
    4 vru                person, cyclist, rider          ← safety-critical
    5 vegetation         vegetation, foliage
   -1 ignore             unlabelled / outside the taxonomy

WHY LOOKUP ARRAYS AND NOT DICTS
-------------------------------
The previous implementation was a dict literal in which id 70 appeared twice
(`70: 2` in the static_obstacle row and `70: 5` in the vegetation row).  Python
silently keeps the last binding, so every "vegetation" point was relabelled
building — a corruption that produced no error and no warning.

A dict literal simply cannot express "this key was assigned twice".  Here each
mapping is built through :func:`_assign`, which refuses to overwrite an id that
has already been set, so a collision is a hard ``ValueError`` at import time.
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

GROUND_DRIVABLE = 0
GROUND_ROUGH = 1
STATIC_OBSTACLE = 2
VEHICLE = 3
VRU = 4
VEGETATION = 5
IGNORE = -1

NUM_CLASSES = 6
CLASS_NAMES = [
    "ground_drivable",
    "ground_rough",
    "static_obstacle",
    "vehicle",
    "vru",
    "vegetation",
]

#: ids that denote a *moving* instance of their class, where the dataset
#: encodes motion in the label itself (SemanticKITTI 252-259).
MOVING_IDS = frozenset({252, 253, 254, 255, 256, 257, 258, 259})


def _assign(table: np.ndarray, ids: Iterable[int], cls: int, name: str) -> None:
    """Set ``table[id] = cls``, refusing to silently overwrite."""
    for i in ids:
        prev = int(table[i])
        if prev != IGNORE and prev != cls:
            raise ValueError(
                f"label id {i} assigned twice in {name}: "
                f"already {CLASS_NAMES[prev]} ({prev}), now {CLASS_NAMES[cls]} ({cls})")
        table[i] = cls


def _build(spec: Dict[int, Iterable[int]], size: int, name: str) -> np.ndarray:
    table = np.full(size, IGNORE, dtype=np.int8)
    for cls, ids in spec.items():
        _assign(table, ids, cls, name)
    return table


# ────────────────────────────────────────────────────────────────
# SemanticKITTI (28 used classes; raw ids run to 259)
# ────────────────────────────────────────────────────────────────
_SEMANTICKITTI_SPEC: Dict[int, Iterable[int]] = {
    GROUND_DRIVABLE: (
        40,   # road
        44,   # parking
        60,   # lane-marking
    ),
    GROUND_ROUGH: (
        48,   # sidewalk
        49,   # other-ground
        72,   # terrain
    ),
    STATIC_OBSTACLE: (
        50,   # building
        51,   # fence
        52,   # other-structure
        80,   # pole
        81,   # traffic-sign
        99,   # other-object
        71,   # trunk
    ),
    VEHICLE: (
        10,   # car
        11,   # bicycle
        13,   # bus
        15,   # motorcycle
        16,   # on-rails
        18,   # truck
        20,   # other-vehicle
        252,  # moving-car
        256,  # moving-on-rails
        257,  # moving-bus
        258,  # moving-truck
        259,  # moving-other-vehicle
    ),
    VRU: (
        30,   # person
        31,   # bicyclist
        32,   # motorcyclist
        253,  # moving-bicyclist
        254,  # moving-person
        255,  # moving-motorcyclist
    ),
    VEGETATION: (
        70,   # vegetation
    ),
}

#: 260 entries so every raw SemanticKITTI id (max 259) indexes directly.
SEMANTICKITTI_TO_6 = _build(_SEMANTICKITTI_SPEC, 260, "SemanticKITTI")


# ────────────────────────────────────────────────────────────────
# RELLIS-3D (20 classes, off-road)
# ────────────────────────────────────────────────────────────────
_RELLIS_SPEC: Dict[int, Iterable[int]] = {
    GROUND_DRIVABLE: (
        3,    # asphalt
        19,   # rubble / hard surface
    ),
    GROUND_ROUGH: (
        1,    # dirt
        4,    # grass
        17,   # mud
        18,   # puddle
    ),
    STATIC_OBSTACLE: (
        7,    # pole
        8,    # water-tank / object
        10,   # building
        12,   # fence
        15,   # barrier
        31,   # sign / post
        33,   # concrete
    ),
    VEHICLE: (
        13,   # vehicle
        14,   # log / large object treated as vehicle-scale obstacle
    ),
    VRU: (
        5,    # person
    ),
    VEGETATION: (
        6,    # tree
        9,    # bush
        27,   # deep vegetation
        34,   # grass-tall
    ),
}

#: 40 entries covers RELLIS' id range (max 34).
RELLIS_TO_6 = _build(_RELLIS_SPEC, 40, "RELLIS-3D")


# ────────────────────────────────────────────────────────────────
# Application
# ────────────────────────────────────────────────────────────────
def remap_labels(raw_labels: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Map raw dataset ids to the 6-class taxonomy.

    SemanticKITTI stores the semantic id in the low 16 bits and the instance id
    in the high 16 bits.  Ids outside the table become ``IGNORE`` (-1), which
    evaluation excludes from every denominator — never a real class, because
    defaulting unknowns to ``static_obstacle`` inflates that class's accuracy.
    """
    sem = np.asarray(raw_labels).astype(np.int64) & 0xFFFF
    out = np.full(sem.shape, IGNORE, dtype=np.int8)
    inside = sem < len(table)
    out[inside] = table[sem[inside]]
    return out


def extract_instance_ids(raw_labels: np.ndarray) -> np.ndarray:
    """SemanticKITTI packs the instance id into the upper 16 bits."""
    return (np.asarray(raw_labels).astype(np.int64) >> 16).astype(np.int32)


def is_moving_label(raw_labels: np.ndarray) -> np.ndarray:
    """True where the dataset's own id marks the point as a moving instance."""
    sem = np.asarray(raw_labels).astype(np.int64) & 0xFFFF
    return np.isin(sem, np.fromiter(MOVING_IDS, dtype=np.int64))


def detect_label_convention(sample_raw: np.ndarray) -> str:
    """Guess which dataset a `.label` file came from.

    RELLIS-3D ships in SemanticKITTI *layout* but uses ids < 35, whereas
    SemanticKITTI reserves 252-259 for moving classes and uses 40-99 heavily.
    Sniffing the id range is more reliable than a path-name heuristic.
    """
    sem = np.asarray(sample_raw).astype(np.int64) & 0xFFFF
    if sem.size == 0:
        return "semantickitti"
    hi = int(sem.max())
    return "rellis" if hi < 40 else "semantickitti"


def table_for(convention: str) -> np.ndarray:
    return RELLIS_TO_6 if convention == "rellis" else SEMANTICKITTI_TO_6
