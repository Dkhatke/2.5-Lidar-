"""
Evaluation harness.

THE ONLY PLACE (besides the `oracle` semantic backend) PERMITTED TO READ
``Frame.gt_label``.  If any code in S2 through S8 reads it, the evaluation is
measuring a system that was handed the answer.
``tests/test_gt_label_isolation.py`` enforces that mechanically.
"""
from adaptive_lidar.evaluation.metrics import (  # noqa: F401
    RANGE_BANDS,
    boundary_consistency,
    elevation_error,
    latency_percentiles,
    map_completeness,
    moving_object_path,
    measure_memory,
    object_retention_rate,
    range_stratified_iou,
    trail_profile,
    vehicle_half_length,
)
from adaptive_lidar.evaluation.reference_map import (  # noqa: F401
    ReferenceMap,
    build_reference_map,
)
