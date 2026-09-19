"""
Unified dataset loader with auto-detection.

    get_dataset(name="auto", root=None, max_frames=None) -> Dataset

A ``Dataset`` is iterable and yields dicts with:

    points        (N, 4)  x y z intensity
    gt_label      (N,) or None       — 6-class taxonomy, -1 = ignore
    gt_instance   (N,) or None
    gt_moving     (N,) or None
    ring          (N,) or None
    azimuth_bin   (N,) or None
    return_number / return_count     (N,) or None
    pose          (4,4) or None
    frame_id, timestamp

AUTO-DETECTION ORDER
--------------------
1. A SemanticKITTI-style tree (``sequences/NN/velodyne/*.bin``) — this also
   covers RELLIS-3D, which ships in the same layout; the two are told apart by
   sniffing the label id range rather than by path name.
2. A flat directory of ``.bin`` or ``.npy`` files.
3. Synthetic — always available, so the project never fails for want of data.

The selected source is printed at startup and recorded in every metrics row.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from adaptive_lidar.data.label_maps import (
    detect_label_convention,
    extract_instance_ids,
    is_moving_label,
    remap_labels,
    table_for,
)
from adaptive_lidar.data.synthetic_scene import SCENARIOS, generate_scenario


# ════════════════════════════════════════════════════════════
class Dataset:
    """Base class. Subclasses implement ``__len__`` and ``__iter__``."""

    name: str = "dataset"
    source: str = "unknown"
    has_labels: bool = False
    has_poses: bool = False
    has_rings: bool = False

    def describe(self) -> str:
        bits = [f"{len(self)} frames"]
        bits.append("labels" if self.has_labels else "no labels")
        bits.append("poses" if self.has_poses else "no poses")
        bits.append("native rings" if self.has_rings else "reconstructed rings")
        return f"{self.name} ({self.source}): " + ", ".join(bits)


# ════════════════════════════════════════════════════════════
class SyntheticDataset(Dataset):
    """Raycast spinning-LiDAR simulation. The guaranteed path."""

    has_labels = True
    has_poses = True
    has_rings = True

    def __init__(self, scenario: str = "mixed_urban", num_frames: int = 20,
                 seed: int = 42, **kw):
        if scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {scenario!r}; choose from {SCENARIOS}")
        self.scenario = scenario
        self.num_frames = int(num_frames)
        self.seed = int(seed)
        self.kw = kw
        self.name = f"synthetic/{scenario}"
        self.source = "synthetic"

    def __len__(self) -> int:
        return self.num_frames

    def frame(self, i: int) -> Dict[str, Any]:
        s = generate_scenario(self.scenario, i, self.num_frames,
                              seed=self.seed, **self.kw)
        return {
            "points": np.column_stack([s["points"], s["intensity"]]).astype(np.float32),
            "intensity": s["intensity"],
            "gt_label": s["gt_label"],
            "gt_instance": s["gt_instance"],
            "gt_moving": s["gt_moving"],
            "ring": s["ring"],
            "azimuth_bin": s["azimuth_bin"],
            "return_number": s["return_number"],
            "return_count": s["return_count"],
            "pose": s["pose"],
            "frame_id": i,
            "timestamp": s["timestamp"],
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i in range(self.num_frames):
            yield self.frame(i)


# ════════════════════════════════════════════════════════════
class SemanticKITTIDataset(Dataset):
    """SemanticKITTI-layout sequence. Also reads RELLIS-3D."""

    def __init__(self, seq_dir: str, max_frames: Optional[int] = None):
        self.seq_dir = seq_dir
        self.velodyne = os.path.join(seq_dir, "velodyne")
        self.labels = os.path.join(seq_dir, "labels")
        if not os.path.isdir(self.velodyne):
            raise FileNotFoundError(f"velodyne directory not found: {self.velodyne}")

        self.files = sorted(f for f in os.listdir(self.velodyne) if f.endswith(".bin"))
        if max_frames:
            self.files = self.files[:max_frames]
        self.has_labels = os.path.isdir(self.labels)

        self.convention = "semantickitti"
        if self.has_labels and self.files:
            probe = os.path.join(self.labels, self.files[0].replace(".bin", ".label"))
            if os.path.isfile(probe):
                self.convention = detect_label_convention(
                    np.fromfile(probe, dtype=np.uint32))
        self.table = table_for(self.convention)

        self.poses = self._load_poses()
        self.has_poses = self.poses is not None
        self.has_rings = False        # reconstructed by scan unfolding
        self.name = f"{self.convention}/{os.path.basename(os.path.normpath(seq_dir))}"
        self.source = self.convention

    def _load_poses(self) -> Optional[np.ndarray]:
        path = os.path.join(self.seq_dir, "poses.txt")
        if not os.path.isfile(path):
            return None
        raw = np.loadtxt(path).reshape(-1, 3, 4)
        out = np.tile(np.eye(4), (len(raw), 1, 1))
        out[:, :3, :4] = raw
        return out

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i, fname in enumerate(self.files):
            cloud = np.fromfile(os.path.join(self.velodyne, fname),
                                dtype=np.float32).reshape(-1, 4)
            gt = gt_inst = gt_mov = None
            if self.has_labels:
                lpath = os.path.join(self.labels, fname.replace(".bin", ".label"))
                if os.path.isfile(lpath):
                    raw = np.fromfile(lpath, dtype=np.uint32)
                    if len(raw) == len(cloud):
                        gt = remap_labels(raw, self.table)
                        gt_inst = extract_instance_ids(raw)
                        gt_mov = is_moving_label(raw)
            pose = self.poses[i] if self.poses is not None and i < len(self.poses) else None
            yield {
                "points": cloud,
                "intensity": cloud[:, 3],
                "gt_label": gt, "gt_instance": gt_inst, "gt_moving": gt_mov,
                "ring": None, "azimuth_bin": None,
                "return_number": None, "return_count": None,
                "pose": pose, "frame_id": i, "timestamp": i * 0.1,
            }


# ════════════════════════════════════════════════════════════
class FlatFileDataset(Dataset):
    """A directory of .bin / .npy scans, or a single file."""

    def __init__(self, root: str, max_frames: Optional[int] = None):
        if os.path.isfile(root):
            self.files = [root]
        else:
            self.files = sorted(glob.glob(os.path.join(root, "*.bin"))
                                + glob.glob(os.path.join(root, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"no .bin/.npy files under {root}")
        if max_frames:
            self.files = self.files[:max_frames]
        self.name = f"files/{os.path.basename(os.path.normpath(root))}"
        self.source = "flat_files"

    def __len__(self) -> int:
        return len(self.files)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for i, path in enumerate(self.files):
            if path.endswith(".npy"):
                a = np.load(path)
            else:
                a = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
            if a.shape[1] == 3:
                a = np.column_stack([a, np.zeros(len(a), np.float32)])
            yield {
                "points": a.astype(np.float32), "intensity": a[:, 3],
                "gt_label": None, "gt_instance": None, "gt_moving": None,
                "ring": None, "azimuth_bin": None,
                "return_number": None, "return_count": None,
                "pose": None, "frame_id": i, "timestamp": i * 0.1,
            }


# ════════════════════════════════════════════════════════════
def _find_sequence(root: str) -> Optional[str]:
    """Locate a SemanticKITTI-style sequence directory under ``root``."""
    if os.path.isdir(os.path.join(root, "velodyne")):
        return root
    seqs = os.path.join(root, "sequences")
    if os.path.isdir(seqs):
        for d in sorted(os.listdir(seqs)):
            cand = os.path.join(seqs, d)
            if os.path.isdir(os.path.join(cand, "velodyne")):
                return cand
    return None


def get_dataset(
    name: str = "auto",
    root: Optional[str] = None,
    max_frames: Optional[int] = None,
    scenario: str = "mixed_urban",
    seed: int = 42,
    quiet: bool = False,
    **kw,
) -> Dataset:
    """Select a data source. Never fails because data is missing."""
    ds: Optional[Dataset] = None

    if name == "synthetic" or (name == "auto" and root is None):
        ds = SyntheticDataset(scenario, max_frames or 20, seed, **kw)

    elif name in ("semantickitti", "rellis") or name == "auto":
        if root and os.path.exists(root):
            seq = _find_sequence(root)
            try:
                if seq is not None:
                    ds = SemanticKITTIDataset(seq, max_frames)
                else:
                    ds = FlatFileDataset(root, max_frames)
            except Exception as e:
                if not quiet:
                    print(f"[Loader] Could not read {root} ({type(e).__name__}: {e})")
                ds = None
        if ds is None:
            if not quiet and root:
                print(f"[Loader] {root!r} unusable — falling back to synthetic.")
            ds = SyntheticDataset(scenario, max_frames or 20, seed, **kw)

    elif name == "files":
        ds = FlatFileDataset(root or ".", max_frames)

    else:
        raise ValueError(f"unknown dataset name {name!r}")

    if not quiet:
        print(f"[Loader] DATA SOURCE: {ds.describe()}")
    return ds


# ════════════════════════════════════════════════════════════
# Legacy shim — main.py's old call signature
# ════════════════════════════════════════════════════════════
def get_input_source(input_path: Optional[str] = None, max_frames: int = 20,
                     num_synthetic_frames: int = 20):
    """Yields (cloud, labels, frame_id, timestamp) like the previous API."""
    ds = get_dataset("auto", input_path, max_frames or num_synthetic_frames)
    return [(f["points"], f["gt_label"], f["frame_id"], f["timestamp"]) for f in ds]
