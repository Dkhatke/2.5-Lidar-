"""
SemanticKITTI loader + synthetic fallback selector.

PRIMARY: SemanticKITTI .bin velodyne scans + .label files
FALLBACK: synthetic_scene.generate_scene()

SemanticKITTI → our 6-class remapping defined in config.yaml.
Labels used ONLY for evaluation — not fed into the allocation controller.
"""
from __future__ import annotations
import os
import numpy as np
from typing import Optional, List, Tuple, Dict

# SemanticKITTI 34-class → our 6-class mapping
# (source: semantic-kitti.yaml learning_map)
KITTI_TO_6 = {
    # ground_drivable (0)
    40: 0, 44: 0, 48: 0, 49: 0,
    # ground_rough (1)
    60: 1, 72: 1,
    # static_obstacle (2)
    50: 2, 51: 2, 52: 2, 70: 2, 71: 2, 80: 2, 81: 2, 99: 2,
    # vehicle (3)
    10: 3, 11: 3, 13: 3, 15: 3, 16: 3, 18: 3, 20: 3,
    252: 3, 256: 3, 257: 3, 258: 3, 259: 3,  # moving vehicles
    # vru (4)
    30: 4, 31: 4, 32: 4,
    253: 4, 254: 4, 255: 4,                   # moving persons
    # vegetation (5)
    70: 5,   # overridden below
}
# Fix overlaps manually
KITTI_TO_6[70] = 2   # building → static_obstacle
KITTI_TO_6[72] = 1   # ground (other)
KITTI_TO_6[71] = 2   # fence
KITTI_TO_6[80] = 2   # trunk (overrides vegetation above)
KITTI_VEGETATION = {21, 22}
for k in KITTI_VEGETATION:
    KITTI_TO_6[k] = 5

DEFAULT_CLASS = 2  # unknown → static_obstacle


def remap_labels(raw_labels: np.ndarray) -> np.ndarray:
    """Map SemanticKITTI uint32 labels to our 6 classes."""
    semantic_ids = raw_labels & 0xFFFF  # lower 16 bits = semantic class
    mapped = np.full_like(semantic_ids, DEFAULT_CLASS, dtype=np.int32)
    for kitti_id, our_id in KITTI_TO_6.items():
        mapped[semantic_ids == kitti_id] = our_id
    return mapped


def load_kitti_bin(path: str) -> np.ndarray:
    """Load KITTI .bin → (N, 4) float32 [x y z intensity]."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_kitti_label(path: str) -> np.ndarray:
    """Load SemanticKITTI .label → (N,) uint32."""
    return np.fromfile(path, dtype=np.uint32)


class SemanticKITTISequence:
    """
    Iterate over a SemanticKITTI sequence directory.
    Yields (cloud_np, label_np, frame_id, timestamp) tuples.
    """

    def __init__(self, sequence_dir: str, max_frames: Optional[int] = None):
        self.velodyne_dir = os.path.join(sequence_dir, "velodyne")
        self.label_dir = os.path.join(sequence_dir, "labels")
        self.max_frames = max_frames

        if not os.path.isdir(self.velodyne_dir):
            raise FileNotFoundError(
                f"velodyne directory not found: {self.velodyne_dir}")

        self.bin_files = sorted(
            [f for f in os.listdir(self.velodyne_dir) if f.endswith(".bin")])
        if max_frames:
            self.bin_files = self.bin_files[:max_frames]

    def __len__(self):
        return len(self.bin_files)

    def __iter__(self):
        for i, fname in enumerate(self.bin_files):
            bin_path = os.path.join(self.velodyne_dir, fname)
            cloud = load_kitti_bin(bin_path)

            label_fname = fname.replace(".bin", ".label")
            label_path = os.path.join(self.label_dir, label_fname)
            if os.path.isfile(label_path):
                raw_labels = load_kitti_label(label_path)
                labels = remap_labels(raw_labels)
            else:
                labels = None

            yield cloud, labels, i, float(i) * 0.1   # 10 Hz assumed


def get_input_source(
    input_path: Optional[str] = None,
    max_frames: int = 200,
    num_synthetic_frames: int = 8,
):
    """
    Returns an iterable of (cloud_np, labels_np_or_None, frame_id, timestamp).

    Priority:
      1. SemanticKITTI sequence directory (if path given and valid)
      2. Single .bin / .npy file
      3. Synthetic scene (fallback)
    """
    if input_path is not None:
        if os.path.isdir(input_path):
            try:
                seq = SemanticKITTISequence(input_path, max_frames=max_frames)
                print(f"[Loader] SemanticKITTI sequence: {len(seq)} frames from {input_path}")
                return seq
            except FileNotFoundError as e:
                print(f"[Loader] WARNING: {e} — falling back to synthetic.")
        elif os.path.isfile(input_path):
            cloud = np.fromfile(input_path, dtype=np.float32).reshape(-1, 4)
            return [(cloud, None, 0, 0.0)]

    # Synthetic fallback
    print("[Loader] Using synthetic scene (no SemanticKITTI path provided)")
    from adaptive_lidar.data.synthetic_scene import generate_multi_frame
    frames = generate_multi_frame(num_synthetic_frames)
    return [(f, None, i, float(i) * 0.1) for i, f in enumerate(frames)]
