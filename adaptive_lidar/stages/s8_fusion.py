"""
S8 temporal fusion — separate file for cleanliness.
Currently a thin wrapper; real logic is inside AdaptiveMap.
"""
# Re-export from s7_map.py for clean imports
from adaptive_lidar.stages.s7_map import S8Fusion  # noqa: F401
