"""Unit tests for SIH26 prototype — Phase 4."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


# ── Fixtures ────────────────────────────────────────────────
@pytest.fixture
def small_cloud():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-10, 10, (500, 3)).astype(np.float32)
    pts[:, 2] = np.abs(pts[:, 2]) * 0.5  # mostly above ground
    intensity = rng.uniform(0, 1, 500).astype(np.float32)
    return np.concatenate([pts, intensity[:, None]], axis=1)


@pytest.fixture
def config():
    from adaptive_lidar.utils.config import load_config
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
    return load_config(cfg_path)


# ── S0 — Loader ──────────────────────────────────────────────
class TestLoader:
    def test_ingest_synthetic(self, small_cloud, config):
        from adaptive_lidar.stages.s0_ingest import ingest
        frame = ingest(None, 0, 0.0, config, synthetic_cloud=small_cloud)
        assert frame.points.shape[1] == 3
        assert len(frame.points) == len(frame.intensity)
        assert np.all(np.isfinite(frame.points))

    def test_ingest_filters_nan(self, config):
        from adaptive_lidar.stages.s0_ingest import ingest
        cloud = np.array([[1, 2, 3, 0.5],
                          [np.nan, 0, 0, 0.1],
                          [5, 5, 5, 0.8]], dtype=np.float32)
        frame = ingest(None, 0, 0.0, config, synthetic_cloud=cloud)
        assert len(frame.points) == 2

    def test_ingest_filters_range(self, config):
        from adaptive_lidar.stages.s0_ingest import ingest
        cloud = np.array([[1, 0, 0, 0.5],   # in range
                          [200, 0, 0, 0.5],  # too far
                          [0.1, 0, 0, 0.5],  # too close
                          ], dtype=np.float32)
        frame = ingest(None, 0, 0.0, config, synthetic_cloud=cloud)
        assert len(frame.points) == 1


# ── Voxel hash ────────────────────────────────────────────────
class TestVoxelHash:
    def test_build(self, small_cloud):
        from adaptive_lidar.utils.voxel_hash import build_voxel_hash
        pts = small_cloud[:, :3]
        vh = build_voxel_hash(pts, vsize=0.5)
        assert len(vh) > 0
        # All indices must be valid
        all_idx = [i for lst in vh.values() for i in lst]
        assert max(all_idx) < len(pts)

    def test_query_coverage(self, small_cloud):
        from adaptive_lidar.utils.voxel_hash import build_voxel_hash
        pts = small_cloud[:, :3]
        vh = build_voxel_hash(pts, vsize=1.0)
        total = sum(len(v) for v in vh.values())
        assert total == len(pts)  # every point assigned


# ── Range image ───────────────────────────────────────────────
class TestRangeImage:
    def test_shape(self, small_cloud):
        from adaptive_lidar.utils.range_image import build_range_image
        pts = small_cloud[:, :3]
        ri, ri_xyz = build_range_image(pts, num_rings=32, h_res_deg=0.5)
        assert ri.shape == (32, 720)
        assert ri_xyz.shape == (32, 720, 3)

    def test_no_negative(self, small_cloud):
        from adaptive_lidar.utils.range_image import build_range_image
        pts = small_cloud[:, :3]
        ri, _ = build_range_image(pts, num_rings=32, h_res_deg=0.5)
        assert np.all(ri >= 0)


# ── Tile scoring + allocation ─────────────────────────────────
class TestAllocation:
    def _make_frame_with_tiles(self, small_cloud, config):
        from adaptive_lidar.stages.s0_ingest import ingest
        from adaptive_lidar.pipeline.types import PipelineContext
        from adaptive_lidar.stages.s1_indices import S1Indices
        from adaptive_lidar.stages.s2_geometry import S2Geometry
        from adaptive_lidar.stages.s3_preallocation import S3Preallocation

        ctx = PipelineContext(config=config, budget=0.8)
        frame = ingest(None, 0, 0.0, config, synthetic_cloud=small_cloud)
        S1Indices(config).process(frame, ctx)
        S2Geometry(config).process(frame, ctx)
        S3Preallocation(config).process(frame, ctx)
        return frame, ctx

    def test_tiles_created(self, small_cloud, config):
        frame, ctx = self._make_frame_with_tiles(small_cloud, config)
        assert frame.tiles is not None
        assert len(frame.tiles) > 0

    def test_info_values_in_range(self, small_cloud, config):
        frame, ctx = self._make_frame_with_tiles(small_cloud, config)
        for t in frame.tiles:
            assert 0.0 <= t.info_value <= 1.0

    def test_budget_limits_high_res(self, small_cloud, config):
        import copy
        from adaptive_lidar.stages.s0_ingest import ingest
        from adaptive_lidar.pipeline.types import PipelineContext
        from adaptive_lidar.stages.s1_indices import S1Indices
        from adaptive_lidar.stages.s2_geometry import S2Geometry
        from adaptive_lidar.stages.s3_preallocation import S3Preallocation

        # At 10% budget, fewer selected tiles
        ctx_low = PipelineContext(config=config, budget=0.10)
        frame_low = ingest(None, 0, 0.0, config, synthetic_cloud=small_cloud)
        S1Indices(config).process(frame_low, ctx_low)
        S2Geometry(config).process(frame_low, ctx_low)
        S3Preallocation(config).process(frame_low, ctx_low)
        n_sel_low = sum(1 for t in frame_low.tiles if t.selected)

        ctx_high = PipelineContext(config=config, budget=1.0)
        frame_high = ingest(None, 0, 0.0, config, synthetic_cloud=small_cloud)
        S1Indices(config).process(frame_high, ctx_high)
        S2Geometry(config).process(frame_high, ctx_high)
        S3Preallocation(config).process(frame_high, ctx_high)
        n_sel_high = sum(1 for t in frame_high.tiles if t.selected)

        assert n_sel_low <= n_sel_high

    def test_safety_pins_survive_low_budget(self, config):
        """Safety-pinned tiles must remain selected even at 10% budget."""
        from adaptive_lidar.data.synthetic_scene import generate_scene
        cloud = generate_scene(0, 8)
        from adaptive_lidar.stages.s0_ingest import ingest
        from adaptive_lidar.pipeline.types import PipelineContext
        from adaptive_lidar.stages.s1_indices import S1Indices
        from adaptive_lidar.stages.s2_geometry import S2Geometry
        from adaptive_lidar.stages.s3_preallocation import S3Preallocation

        ctx = PipelineContext(config=config, budget=0.10)
        frame = ingest(None, 0, 0.0, config, synthetic_cloud=cloud)
        S1Indices(config).process(frame, ctx)
        S2Geometry(config).process(frame, ctx)
        S3Preallocation(config).process(frame, ctx)

        for tile in frame.tiles:
            if tile.safety_pinned:
                assert tile.selected, (
                    f"Safety-pinned tile {tile.tile_id} must remain selected!")


# ── Map update ────────────────────────────────────────────────
class TestMap:
    def test_map_updates(self, small_cloud, config):
        from adaptive_lidar.mapping.adaptive_map import AdaptiveMap
        amap = AdaptiveMap(config)
        pts = small_cloud[:, :3]
        gnd = pts[:, 2] < 0.3
        labels = np.zeros(len(pts), dtype=np.int32)
        amap.update_from_points(pts, gnd, labels, {}, [], 0.0)
        assert len(amap.cells) > 0

    def test_unknown_not_free(self, config):
        """Cells with no observation must stay unknown (not free)."""
        from adaptive_lidar.mapping.adaptive_map import AdaptiveMap
        amap = AdaptiveMap(config)
        # Don't update any points
        assert len(amap.cells) == 0  # nothing allocated = unknown by default
