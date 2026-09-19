"""
End-to-end and per-stage tests.

    python -m pytest tests/ -q
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adaptive_lidar.data.loader import get_dataset                  # noqa: E402
from adaptive_lidar.pipeline.pipeline import Pipeline               # noqa: E402
from adaptive_lidar.pipeline.types import (                         # noqa: E402
    CELL_DTYPE,
    NUM_CLASSES,
    ResolutionLevel,
    Traversability,
    VehicleProfile,
    validate_frame_contract,
)
from adaptive_lidar.utils.config import load_config                 # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def scan():
    """One real synthetic scan — raycast, with ground truth."""
    return next(iter(get_dataset("synthetic", None, 1, quiet=True)))


@pytest.fixture(scope="session")
def run(config, scan):
    """A three-frame run of the default configuration."""
    frames = list(get_dataset("synthetic", None, 3, quiet=True))
    pipe = Pipeline(config)
    pipe.build_stages(backend="auto", policy="full")
    pipe.set_budget(0.5)
    last = None
    for f in frames:
        last = pipe.run(cloud_np=f, frame_id=f["frame_id"],
                        timestamp=f["timestamp"])
    return pipe, last, frames


# ════════════════════════════════════════════════════════════
class TestPipeline:
    def test_runs_end_to_end(self, run):
        _, frame, _ = run
        assert len(frame.points) > 10_000

    def test_every_stage_reported_a_time(self, run):
        _, frame, _ = run
        for s in ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"):
            assert isinstance(frame.timing.get(s), float), f"{s} missing"

    def test_frame_contract_holds(self, run):
        _, frame, _ = run
        assert validate_frame_contract(frame, strict=False) == []

    def test_telemetry_is_populated(self, run):
        _, frame, _ = run
        t = frame.timing["telemetry"]
        assert t["map_cells"] > 0 and t["total_latency_ms"] > 0

    def test_empty_cloud_does_not_crash(self, config):
        pipe = Pipeline(config)
        pipe.build_stages(backend="geometry_rules")
        frame = pipe.run(cloud_np=np.zeros((0, 4), np.float32), frame_id=0,
                         timestamp=0.0)
        assert len(frame.points) == 0


# ════════════════════════════════════════════════════════════
class TestDataContract:
    def test_point_level_arrays_are_index_aligned(self, run):
        _, f, _ = run
        n = len(f.points)
        for name in ("intensity", "intensity_norm", "ring", "azimuth_bin",
                     "height_above_gnd", "sem_class", "sem_entropy",
                     "moving_prob", "instance_id", "gt_label"):
            assert len(getattr(f, name)) == n, name
        assert f.sem_evidence.shape == (n, NUM_CLASSES)

    def test_evidence_rows_are_normalised(self, run):
        _, f, _ = run
        assert np.allclose(f.sem_evidence.sum(axis=1), 1.0, atol=1e-3)

    def test_range_image_holds_indices_not_ranges(self, run):
        _, f, _ = run
        assert f.range_image.dtype.kind == "i"
        idx = f.range_image[f.range_image_valid]
        assert idx.min() >= 0 and idx.max() < len(f.points)

    def test_violation_is_detected(self, run):
        import copy
        _, f, _ = run
        bad = copy.copy(f)
        bad.sem_class = np.zeros(3, np.int8)     # wrong length
        with pytest.raises(AssertionError):
            validate_frame_contract(bad)


# ════════════════════════════════════════════════════════════
class TestVariableResolution:
    """BROKEN 1 — the allocation must actually reach the map."""

    def test_multiple_levels_are_populated(self, run):
        pipe, _, _ = run
        counts = pipe.context.amap.cell_counts()
        assert sum(1 for c in counts.values() if c > 0) >= 3, counts

    def test_cells_have_different_physical_sizes(self, run):
        pipe, _, _ = run
        a = pipe.context.amap.all_cells_arrays()
        assert len(np.unique(a["size"])) >= 3

    def test_point_level_reaches_the_map(self, run):
        _, f, _ = run
        assert f.point_level is not None
        assert len(np.unique(f.point_level)) >= 2

    def test_budget_changes_the_map_size(self, config):
        frames = list(get_dataset("synthetic", None, 2, quiet=True))
        sizes = []
        for b in (1.0, 0.1):
            p = Pipeline(config)
            p.build_stages(backend="geometry_rules", policy="full")
            p.set_budget(b)
            for f in frames:
                p.run(cloud_np=f, frame_id=f["frame_id"],
                      timestamp=f["timestamp"])
            sizes.append(p.context.amap.n_cells)
        assert sizes[0] > sizes[1], f"budget had no effect: {sizes}"


# ════════════════════════════════════════════════════════════
class TestMapCorrectness:
    def test_cell_record_is_small(self):
        assert CELL_DTYPE.itemsize <= 32, CELL_DTYPE.itemsize

    def test_coarsening_preserves_point_count_exactly(self, run):
        pipe, _, _ = run
        fine = pipe.context.amap
        coarse = fine.coarsen_to(4)
        assert (int(coarse.levels[4].cells["n_points"].sum())
                == sum(int(s.cells["n_points"].sum()) for s in fine.levels))

    def test_no_point_lands_in_two_cells(self, run):
        from adaptive_lidar.utils.grouping import morton_2d
        _, f, _ = run
        pose = f.pose
        pw = f.points @ pose[:3, :3].T + pose[:3, 3]
        ix = np.floor(pw[:, 0] / 0.05).astype(np.int64)
        iy = np.floor(pw[:, 1] / 0.05).astype(np.int64)
        c0 = morton_2d(ix, iy)
        for l in range(1, 5):
            assert np.array_equal(c0 >> (2 * l),
                                  morton_2d(ix >> l, iy >> l, level=l))

    def test_map_is_smaller_than_uniform_5cm(self, run):
        pipe, _, _ = run
        assert pipe.context.amap.nbytes() < pipe.context.uniform_ref.nbytes()

    def test_traversability_depends_on_the_vehicle(self, run):
        pipe, _, _ = run
        m = pipe.context.amap
        w = m.traversability_arrays(VehicleProfile.wheeled())["verdict"]
        t = m.traversability_arrays(VehicleProfile.tracked())["verdict"]
        assert (t == Traversability.DRIVABLE).sum() >= \
               (w == Traversability.DRIVABLE).sum()

    def test_overhead_clearance_exists_and_is_used(self, run):
        pipe, _, _ = run
        a = pipe.context.amap.all_cells_arrays()
        assert np.isfinite(a["overhead_clearance"]).any()


# ════════════════════════════════════════════════════════════
class TestAllocation:
    def test_every_policy_runs(self, config):
        from adaptive_lidar.mapping.allocation import AllocationController
        frames = list(get_dataset("synthetic", None, 1, quiet=True))
        for policy in AllocationController.POLICIES:
            p = Pipeline(config)
            p.build_stages(backend="geometry_rules", policy=policy)
            p.set_budget(0.4)
            f = p.run(cloud_np=frames[0], frame_id=0, timestamp=0.0)
            assert f.point_level is not None, policy

    def test_resolution_law_hits_the_ps_example(self):
        from adaptive_lidar.mapping.allocation import resolution_law
        assert abs(float(resolution_law(np.array([10.0]))[0]) - 0.05) < 1e-6
        assert abs(float(resolution_law(np.array([100.0]))[0]) - 0.50) < 1e-3

    def test_continuity_is_enforced(self, run):
        from adaptive_lidar.mapping.allocation import AllocationController
        from adaptive_lidar.utils.grouping import pack_keys_2d
        _, f, _ = run
        lv = f.tile_levels
        tiles = f.tiles
        ix = np.array([t.ix for t in tiles])
        iy = np.array([t.iy for t in tiles])
        keys = pack_keys_2d(ix, iy)
        order = np.argsort(keys)
        sk = keys[order]
        worst = 0
        for dx, dy in ((1, 0), (0, 1)):
            nb = pack_keys_2d(ix + dx, iy + dy)
            pos = np.clip(np.searchsorted(sk, nb), 0, len(sk) - 1)
            hit = sk[pos] == nb
            if hit.any():
                worst = max(worst, int(np.abs(
                    lv[hit].astype(int)
                    - lv[order[pos[hit]]].astype(int)).max()))
        assert worst <= 1, f"2:1 continuity violated by {worst} levels"

    def test_safety_pins_survive_the_lowest_budget(self, config):
        frames = list(get_dataset("synthetic", None, 2,
                                  scenario="pedestrian_far", quiet=True))
        p = Pipeline(config)
        p.build_stages(backend="auto", policy="full")
        p.set_budget(0.05)
        for f in frames:
            fr = p.run(cloud_np=f, frame_id=f["frame_id"],
                       timestamp=f["timestamp"])
        assert fr.timing["S6_diag"]["n_pinned"] > 0
        assert any(t.has_vertical_run for t in fr.tiles), \
            "the geometric pin never fired"


# ════════════════════════════════════════════════════════════
class TestTemporal:
    def test_gate_holds_moving_points_out(self, config):
        frames = list(get_dataset("synthetic", None, 6,
                                  scenario="moving_vehicle", quiet=True))
        p = Pipeline(config)
        p.build_stages(backend="auto", policy="full")
        p.set_budget(0.5)
        gated = 0
        for f in frames:
            fr = p.run(cloud_np=f, frame_id=f["frame_id"],
                       timestamp=f["timestamp"])
            gated += fr.timing["S8_diag"]["n_moving"]
        assert gated > 0

    def test_tracker_emits_the_object_table(self, run):
        _, f, _ = run
        assert f.instances is not None
        for i in f.instances:
            assert i.velocity is not None and i.covariance is not None
            assert i.cluster_id >= 0 and i.instance_id >= 1

    def test_instance_ids_are_track_ids(self, run):
        """The per-point ids and the object table must share one id space."""
        _, f, _ = run
        if not f.instances:
            pytest.skip("no instances")
        table = {i.instance_id for i in f.instances}
        present = set(np.unique(f.instance_id[f.instance_id >= 0]).tolist())
        assert present <= table, present - table


# ════════════════════════════════════════════════════════════
class TestPerception:
    def test_all_backends_produce_normalised_evidence(self, config, scan):
        from adaptive_lidar.perception.backends import build_backend
        p = Pipeline(config)
        p.build_stages(backend="geometry_rules")
        frame = p.run(cloud_np=scan, frame_id=0, timestamp=0.0)
        for name in ("geometry_rules", "oracle"):
            ev = build_backend(name, config).predict(frame)
            assert ev.shape == (len(frame.points), NUM_CLASSES)
            assert np.allclose(ev.sum(axis=1), 1.0, atol=1e-3), name

    def test_checkpoint_is_committed_and_loads(self):
        from adaptive_lidar.perception.pointfeature_net import (
            DEFAULT_CHECKPOINT,
            PointFeatureInference,
        )
        assert os.path.isfile(DEFAULT_CHECKPOINT), \
            "the trained weights must be committed so the demo works on a clone"
        inf = PointFeatureInference()
        assert inf.temperature > 0
        out = inf.predict(np.zeros((16, 12), np.float32))
        assert out.shape == (16, NUM_CLASSES)

    def test_vertical_run_separates_objects_from_ground(self, run):
        _, f, _ = run
        run_len = f._vertical_run
        gt = np.asarray(f.gt_label)
        ground = np.isin(gt, (0, 1))
        objects = np.isin(gt, (2, 3, 4))
        assert (run_len[objects] >= 3).mean() > 0.8
        assert (run_len[ground] >= 3).mean() < 0.1

    def test_intensity_normalisation_separates_terrain(self, run):
        """The asphalt/grass split is the reason the correction exists."""
        _, f, _ = run
        gt = np.asarray(f.gt_label)
        g = f.ground_mask
        d = f.intensity_norm[(gt == 0) & g]
        r = f.intensity_norm[(gt == 1) & g]
        assert np.median(d) < np.median(r) - 0.1


# ════════════════════════════════════════════════════════════
class TestLabelMapping:
    def test_no_id_maps_to_two_classes(self):
        """The bug that silently relabelled every vegetation point."""
        from adaptive_lidar.data.label_maps import (
            _RELLIS_SPEC,
            _SEMANTICKITTI_SPEC,
        )
        for name, spec in (("SemanticKITTI", _SEMANTICKITTI_SPEC),
                           ("RELLIS", _RELLIS_SPEC)):
            seen = {}
            for cls, ids in spec.items():
                for i in ids:
                    assert i not in seen, \
                        f"{name}: id {i} in both {seen.get(i)} and {cls}"
                    seen[i] = cls

    def test_a_duplicate_raises_at_build_time(self):
        from adaptive_lidar.data.label_maps import _build
        with pytest.raises(ValueError):
            _build({2: (70,), 5: (70,)}, 260, "deliberate-collision")

    def test_unknown_ids_are_ignore_not_a_real_class(self):
        from adaptive_lidar.data.label_maps import (
            IGNORE,
            SEMANTICKITTI_TO_6,
            remap_labels,
        )
        assert remap_labels(np.array([7]), SEMANTICKITTI_TO_6)[0] == IGNORE


# ════════════════════════════════════════════════════════════
class TestGrouping:
    def test_morton_shift_identity(self):
        rng = np.random.default_rng(0)
        from adaptive_lidar.utils.grouping import morton_2d
        ix = rng.integers(-5000, 5000, 20_000).astype(np.int32)
        iy = rng.integers(-5000, 5000, 20_000).astype(np.int32)
        c0 = morton_2d(ix, iy)
        for l in range(1, 6):
            assert np.array_equal(c0 >> (2 * l),
                                  morton_2d(ix >> l, iy >> l, level=l))

    def test_segment_reductions_match_a_reference(self):
        import collections
        from adaptive_lidar.utils.grouping import group_by_key, segment_reduce
        rng = np.random.default_rng(1)
        keys = rng.integers(0, 300, 4000)
        vals = rng.normal(size=4000).astype(np.float32)
        uq, starts, order = group_by_key(keys)
        ref = collections.defaultdict(list)
        for k, v in zip(keys, vals):
            ref[k].append(v)
        sv = vals[order]
        for op, fn in (("sum", np.sum), ("mean", np.mean),
                       ("min", np.min), ("max", np.max), ("var", np.var)):
            assert np.allclose(segment_reduce(sv, starts, op),
                               [fn(ref[u]) for u in uq], atol=1e-4), op

    def test_merge_matches_union1d(self):
        from adaptive_lidar.utils.grouping import merge_sorted_unique
        rng = np.random.default_rng(2)
        for _ in range(100):
            a = np.unique(rng.integers(0, 200, 40).astype(np.int64))
            b = np.unique(rng.integers(0, 200, 40).astype(np.int64))
            m, _, _ = merge_sorted_unique(a, b)
            assert np.array_equal(m, np.union1d(a, b))


# ════════════════════════════════════════════════════════════
class TestSyntheticScene:
    def test_deterministic_for_a_seed(self):
        from adaptive_lidar.data.synthetic_scene import generate_scenario
        a = generate_scenario("mixed_urban", 0)
        b = generate_scenario("mixed_urban", 0)
        assert np.array_equal(a["points"], b["points"])
        assert np.array_equal(a["gt_label"], b["gt_label"])

    def test_every_scenario_generates(self):
        from adaptive_lidar.data.synthetic_scene import (
            SCENARIOS,
            generate_scenario,
        )
        for s in SCENARIOS:
            sc = generate_scenario(s, 0)
            assert len(sc["points"]) > 0 or s == "empty_road"

    def test_far_pedestrian_is_observable(self):
        from adaptive_lidar.data.synthetic_scene import (
            FAR_PEDESTRIAN_X,
            FAR_PEDESTRIAN_Y,
            generate_scenario,
        )
        s = generate_scenario("mixed_urban", 0)
        p = s["points"]
        d = np.hypot(p[:, 0] - FAR_PEDESTRIAN_X, p[:, 1] - FAR_PEDESTRIAN_Y)
        near = d < 1.2
        assert near.sum() > 5
        assert (np.asarray(s["gt_label"])[near] == 4).all()

    def test_ground_density_falls_off_with_range(self):
        """The premise of the whole resolution law."""
        from adaptive_lidar.data.synthetic_scene import generate_scenario
        s = generate_scenario("mixed_urban", 0)
        p, gt = s["points"], np.asarray(s["gt_label"])
        r = np.linalg.norm(p, axis=1)
        g = np.isin(gt, (0, 1))
        near = (g & (r < 10)).sum() / (np.pi * 100)
        far = (g & (r >= 60) & (r < 100)).sum() / (np.pi * (10000 - 3600))
        assert near / max(far, 1e-9) > 50

    def test_multi_echo_and_rings_are_present(self):
        from adaptive_lidar.data.synthetic_scene import generate_scenario
        s = generate_scenario("mixed_urban", 0)
        assert (s["return_count"] == 2).any()
        assert len(np.unique(s["ring"])) > 30
