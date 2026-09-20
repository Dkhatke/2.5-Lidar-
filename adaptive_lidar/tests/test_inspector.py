"""
The inspector: coverage, and the two places it would be easy to mislead.

Coverage first — every field ``MapCell`` carries must reach a section, or
the inspector quietly stops being an inspector. Then the two honesty checks:
the traversability verdict must agree with the map's own rules rather than a
lookalike reimplementation, and the field the map calls ``penetration`` must
not be presented as if a low number meant an obstructed beam.
"""
from __future__ import annotations

import inspect as _inspect

import numpy as np
import pytest

from adaptive_lidar.pipeline.types import (CELL_DTYPE, CellFlags,
                                           ResolutionLevel, Traversability,
                                           VehicleProfile)
from adaptive_lidar.visualization import inspector as INS
from adaptive_lidar.visualization import playback as PB
from adaptive_lidar.visualization import selection as SEL

#: Every MapCell field, as the dashboard names it. `resolution` and `level`
#: are the same decision seen two ways, and both are shown.
MAPCELL_FIELDS = (
    "cx", "cy", "resolution", "level", "ground_z", "z_max",
    "overhead_clearance", "z_var", "n_points", "evidence", "sem_class",
    "entropy", "occupancy_logodds", "occupancy_state", "dynamic_prob",
    "last_seen", "intensity_mean", "intensity_var", "penetration",
    "observability", "flags")


def _cells(entries, **overrides):
    n = len(entries)
    out = {
        "cx": np.array([e[0] for e in entries], np.float32),
        "cy": np.array([e[1] for e in entries], np.float32),
        "level": np.array([e[2] for e in entries], np.int8),
    }
    for k in PB.CELL_FIELDS:
        if k in out:
            continue
        out[k] = (np.full(n, 128, np.uint8) if k in PB._CELL_U8
                  else np.zeros(n, PB._CELL_DTYPE.get(k, np.float32)))
    out[PB.EVIDENCE_FIELD] = np.full((n, 6), 42, np.uint8)
    out["size"] = ResolutionLevel.sizes_array()[
        np.clip(out["level"].astype(np.int64), 0, 4)]
    for k, v in overrides.items():
        out[k][:] = v
    return out


def _centre(ix, iy, level):
    s = ResolutionLevel.size(level)
    return ((ix + 0.5) * s, (iy + 0.5) * s)


def _one(level=1, **overrides):
    c = _centre(40, 12, level)
    cells = _cells([c + (level,)], **overrides)
    return SEL.cell_at_world(cells, *c), cells


# ════════════════════════════════════════════════════════════
# Coverage
# ════════════════════════════════════════════════════════════
def test_every_mapcell_field_reaches_the_inspector():
    """The record the inspector is handed must be complete.

    Checked against the cache's field list AND against the cell dtype, so
    adding a field to the map without adding it to the cache fails here
    rather than showing as a blank row.
    """
    cell, _cells_d = _one()
    for f in MAPCELL_FIELDS:
        assert f in cell, f"inspector never sees {f}"


def test_the_cache_carries_every_stored_cell_field():
    cached = set(PB.CELL_FIELDS) | {PB.EVIDENCE_FIELD}
    # The three `evidence_top3_*` fields ARE the top-3 packing. The cache
    # stores the decoded 6-class distribution instead, which is strictly
    # more information, so they are covered rather than missing.
    packing = {"evidence_top3_cls", "evidence_top3_val", "evidence_residual"}
    missing = sorted(set(CELL_DTYPE.names) - cached - packing)
    assert not missing, f"not cached, so not inspectable: {missing}"
    assert PB.EVIDENCE_FIELD in cached


def test_every_section_is_rendered():
    """All eight sections must actually be emitted, in order."""
    src = _inspect.getsource(INS.render_cell)
    for i, name in enumerate(INS.SECTIONS, start=1):
        assert f"{i} ·" in src, f"section {i} ({name}) is not rendered"
    positions = [src.index(f"{i} ·") for i in range(1, 9)]
    assert positions == sorted(positions), "sections are out of order"


def test_no_raw_array_dump():
    """The brief's explicit prohibition, enforced.

    A ``st.write`` of the record, or a repr of the evidence array, is what
    this is here to stop.
    """
    src = _inspect.getsource(INS)
    assert "st.write(" not in src
    assert "st.json(" not in src


# ════════════════════════════════════════════════════════════
# Derived values
# ════════════════════════════════════════════════════════════
def test_a_flat_well_observed_ground_cell_is_drivable():
    cell, cells = _one(level=1, sem_class=0, n_points=40, z_var=0.0,
                       overhead_clearance=9.0, ground_z=0.0, z_max=0.0)
    d = INS.derive(cell, cells)
    assert d.verdict == Traversability.DRIVABLE, d.reasons


def test_an_obstacle_blocks_and_says_why():
    cell, cells = _one(level=1, sem_class=0, n_points=40,
                       overhead_clearance=9.0, ground_z=0.0, z_max=1.4)
    d = INS.derive(cell, cells)
    assert d.verdict == Traversability.BLOCKED
    assert any("obstacle" in r for r in d.reasons)
    assert d.obstacle_height == pytest.approx(1.4, abs=1e-2)


def test_low_overhead_clearance_blocks():
    cell, cells = _one(level=1, sem_class=0, n_points=40,
                       overhead_clearance=1.2, ground_z=0.0, z_max=0.0)
    d = INS.derive(cell, cells)
    assert d.verdict == Traversability.BLOCKED
    assert any("clearance" in r for r in d.reasons)


def test_a_vru_cell_is_blocked_whatever_its_geometry():
    """The existential rule: class alone is enough."""
    cell, cells = _one(level=1, sem_class=4, n_points=40,
                       overhead_clearance=9.0, ground_z=0.0, z_max=0.0)
    assert INS.derive(cell, cells).verdict == Traversability.BLOCKED


def test_a_sparsely_observed_cell_is_caution_not_drivable():
    cell, cells = _one(level=1, sem_class=0, n_points=1,
                       overhead_clearance=9.0, ground_z=0.0, z_max=0.0)
    d = INS.derive(cell, cells)
    assert d.verdict == Traversability.CAUTION
    assert any("fewer than 2 points" in r for r in d.reasons)


def test_slope_comes_from_neighbours_and_is_zero_without_them():
    cell, cells = _one(level=1, sem_class=0, n_points=40,
                       overhead_clearance=9.0)
    d = INS.derive(cell, cells)
    assert d.neighbours == 0 and d.slope_deg == 0.0


def test_a_step_between_neighbours_is_found():
    lvl = 1
    s = ResolutionLevel.size(lvl)
    a = _centre(40, 12, lvl)
    b = _centre(41, 12, lvl)
    cells = _cells([a + (lvl,), b + (lvl,)], sem_class=0, n_points=40,
                   overhead_clearance=9.0)
    cells["ground_z"][1] = 0.9          # a wall-height jump next door
    cell = SEL.cell_at_world(cells, *a)
    d = INS.derive(cell, cells)
    assert d.neighbours == 1
    assert d.step_m == pytest.approx(0.9, abs=1e-2)
    assert d.slope_deg == pytest.approx(
        float(np.degrees(np.arctan(0.9 / s))), abs=0.5)
    assert d.verdict == Traversability.BLOCKED


def test_the_verdict_matches_the_maps_own_thresholds():
    """Not a lookalike: the same profile numbers decide it."""
    p = VehicleProfile.wheeled()
    just_under, just_over = p.max_step_m * 2.0 - 0.01, p.max_step_m * 2.0 + 0.01
    for z, want in ((just_under, Traversability.DRIVABLE),
                    (just_over, Traversability.BLOCKED)):
        cell, cells = _one(level=1, sem_class=0, n_points=40,
                           overhead_clearance=9.0, ground_z=0.0, z_max=z)
        assert INS.derive(cell, cells, p).verdict == want, z


def test_the_tracked_profile_forgives_what_the_wheeled_one_does_not():
    """One map, two answers — which is the point of deriving, not storing."""
    cell, cells = _one(level=1, sem_class=0, n_points=40,
                       overhead_clearance=9.0, ground_z=0.0, z_max=0.36)
    assert INS.derive(cell, cells, VehicleProfile.wheeled()).verdict \
        == Traversability.BLOCKED
    assert INS.derive(cell, cells, VehicleProfile.tracked()).verdict \
        != Traversability.BLOCKED


# ════════════════════════════════════════════════════════════
# Formatting helpers
# ════════════════════════════════════════════════════════════
def test_morton_code_is_the_prefix_of_the_level_zero_code():
    """The inspector's address must be the map's address.

    ``level-l id == level-0 id >> 2l`` is the identity the whole hierarchy
    rests on; showing a code that did not satisfy it would be worse than
    showing none.
    """
    from adaptive_lidar.utils.grouping import morton_2d
    ix0, iy0 = 641, 233
    for lvl in range(ResolutionLevel.N_LEVELS):
        cx, cy = _centre(ix0 >> lvl, iy0 >> lvl, lvl)
        cell = {"cx": cx, "cy": cy, "level": lvl}
        code0 = int(morton_2d(np.array([ix0]), np.array([iy0]))[0])
        assert INS.cell_morton(cell) == code0 >> (2 * lvl)


def test_flag_names_decode_the_bitfield():
    assert INS.flag_names(0) == []
    assert INS.flag_names(CellFlags.SAFETY_PINNED) == ["safety_pinned"]
    both = CellFlags.REFINED | CellFlags.SAFETY_PINNED
    assert set(INS.flag_names(both)) == {"refined", "safety_pinned"}


@pytest.mark.parametrize("lo,p", [(0.0, 0.5), (10.0, 0.99995),
                                  (-10.0, 4.5398e-5)])
def test_occupancy_probability_inverts_the_log_odds(lo, p):
    assert INS.occupancy_probability(lo) == pytest.approx(p, rel=1e-3)


def test_semantic_bars_show_all_six_classes_and_normalise():
    html = INS.semantic_bars(np.array([3, 1, 0, 0, 0, 1], np.float32))
    from adaptive_lidar.pipeline.types import CLASS_NAMES
    for name in CLASS_NAMES:
        assert name in html
    assert "60.0%" in html          # 3 of 5


def test_semantic_bars_survive_an_all_zero_posterior():
    html = INS.semantic_bars(np.zeros(6, np.float32))
    assert "16.7%" in html


def test_bars_clamp_rather_than_overflow():
    assert "width:100.0%" in INS._bar(5.0, "#000")
    assert "width:0.0%" in INS._bar(-5.0, "#000")


# ════════════════════════════════════════════════════════════
# Honesty
# ════════════════════════════════════════════════════════════
def test_penetration_is_labelled_as_a_return_position_ratio():
    """``penetration`` is return_number / return_count.

    It is 1.0 for a single return — the OPPOSITE of what the word suggests —
    so the inspector must not print it under that name without saying what
    it is.
    """
    src = _inspect.getsource(INS.render_cell)
    assert "return_number / " in src
    assert "1.0 means a single return" in src
    # And the bare word must not be used as a user-facing label.
    assert '("penetration"' not in src


def test_cached_precision_is_disclosed():
    """Byte-packed fields are not float-exact, and the panel says so."""
    src = _inspect.getsource(INS.render_cell)
    assert "1/255" in src


def test_derived_values_are_marked_as_derived():
    src = _inspect.getsource(INS.render_cell)
    assert src.count("derived") >= 3
