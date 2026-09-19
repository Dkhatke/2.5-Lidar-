"""
The one rule that makes every reported number meaningful.

``Frame.gt_label``, ``Frame.gt_instance`` and ``Frame.gt_moving`` are GROUND
TRUTH.  They may be read by:

  * ``adaptive_lidar/evaluation/`` — that is what evaluation is,
  * the ``oracle`` semantic backend — deliberately, loudly, and recorded in
    every metrics row,
  * ``scripts/`` — which measure and plot,
  * ``tests/`` — which check.

If anything in ``stages/s2`` through ``stages/s8``, ``mapping/``, or the rest
of ``perception/`` reads them, the system is being handed the answer and
every accuracy number is meaningless.  Convention does not survive a busy
afternoon, so this is enforced mechanically.
"""
from __future__ import annotations

import ast
import os
import sys

import pytest

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GT_FIELDS = ("gt_label", "gt_instance", "gt_moving")

#: Modules that form the per-frame perception and mapping path. None of these
#: may touch ground truth.
FORBIDDEN = [
    "stages/s1_indices.py",
    "stages/s2_geometry.py",
    "stages/s3_preallocation.py",
    "stages/s4_semantics.py",
    "stages/s5_motion.py",
    "stages/s6_allocation.py",
    "stages/s7_map.py",
    "stages/s8_fusion.py",
    "mapping/adaptive_map.py",
    "mapping/allocation.py",
    "perception/features.py",
    "perception/ground.py",
    "perception/motion.py",
    "perception/tracking.py",
    "perception/pointfeature_net.py",
    "utils/range_image.py",
    "utils/spatial_index.py",
    "utils/grouping.py",
    "utils/voxel_hash.py",
]

#: The two places that are allowed to, and why.
ALLOWED = {
    "perception/backends.py": "the oracle backend, which warns loudly",
    "stages/s0_ingest.py": "ingest copies the loader's arrays onto the Frame",
    "stages/s9_output.py": "telemetry only",
}


def _reads_gt(path: str):
    """Attribute and subscript reads of a ground-truth field, by line."""
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    tree = ast.parse(src, filename=path)
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in GT_FIELDS:
            hits.append((node.lineno, node.attr))
        elif (isinstance(node, ast.Constant) and isinstance(node.value, str)
              and node.value in GT_FIELDS):
            hits.append((node.lineno, node.value))
    return hits


@pytest.mark.parametrize("rel", FORBIDDEN)
def test_perception_and_mapping_never_read_ground_truth(rel):
    path = os.path.join(_PKG, rel)
    if not os.path.isfile(path):
        pytest.skip(f"{rel} not present")
    hits = _reads_gt(path)
    assert not hits, (
        f"{rel} reads ground truth at "
        + ", ".join(f"line {ln} ({name})" for ln, name in hits)
        + ". If a stage can see the labels, every accuracy number this "
          "project reports is measuring a system that was told the answer.")


def test_the_allowed_readers_are_still_the_only_ones():
    """Catch a new module quietly starting to read ground truth."""
    offenders = {}
    for root, _, files in os.walk(_PKG):
        if any(p in root for p in ("__pycache__", "tests", "scripts",
                                   "evaluation", ".venv", "docs")):
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, _PKG).replace("\\", "/")
            if rel in ALLOWED or rel in ("data/loader.py",
                                         "data/synthetic_scene.py",
                                         "data/label_maps.py",
                                         "pipeline/types.py",
                                         "app.py"):
                continue
            hits = _reads_gt(path)
            if hits:
                offenders[rel] = hits
    assert not offenders, (
        "ground truth is read outside evaluation/, scripts/, tests/ and the "
        f"documented exceptions: {offenders}. Add a justification to ALLOWED "
        "only if the reader genuinely is evaluation code.")


def test_oracle_backend_announces_itself():
    """Oracle mode must be impossible to run by accident."""
    path = os.path.join(_PKG, "perception", "backends.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    assert "ORACLE" in src and "logger.warning" in src
    # And it must not be reachable from "auto".
    assert '"oracle"' not in src.split("def build_backend")[1].split(
        "try:")[0].replace('_REGISTRY[mode](config)', '')


def test_the_field_carries_its_warning():
    path = os.path.join(_PKG, "pipeline", "types.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    assert "EVALUATION ONLY" in src
