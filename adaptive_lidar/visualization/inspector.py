"""
The cell inspector: one selected cell, told properly.

Two rules shape this module.

**No raw arrays.** Dumping a NumPy record is a debugging aid, not an
inspector; a reader cannot tell which of twenty numbers matters. Everything
here is grouped into eight sections, quantities that are probabilities are
drawn as bars, and derived quantities are labelled as derived.

**Nothing is renamed to sound better than it is.** In particular the field
the map calls ``penetration`` is ``return_number / return_count``, which is
**1.0 for a single return** and smaller when the beam got through to
something behind. Displaying it under the word "penetration" reads exactly
backwards, so it is shown as a return-position ratio with the meaning
spelled out.

Derived values (slope, step, obstacle height, traversability) are computed
here from the CACHED frame rather than read from the live map, because the
live map only exists for the last frame and the inspector must answer for
whichever frame the user is looking at. They agree with
``AdaptiveMap.traversability`` on its inputs; the only difference is that
neighbours come from the cache.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adaptive_lidar.pipeline.types import (CLASS_NAMES, CellFlags, Occupancy,
                                           ResolutionLevel, Traversability,
                                           TrackState, VehicleProfile)
from adaptive_lidar.utils.grouping import morton_2d
from adaptive_lidar.visualization import selection as SEL
from adaptive_lidar.visualization.render import CLASS_COLOURS, TRAV_COLOURS

#: The eight sections, in the order they are shown.
SECTIONS = ("Identity", "Terrain & geometry", "Semantics", "Occupancy",
            "Dynamics", "Sensor & intensity", "Flags", "Raw / debug")


# ════════════════════════════════════════════════════════════
# Derived quantities, from the cache
# ════════════════════════════════════════════════════════════
@dataclass
class Derived:
    """What the map computes at query time rather than storing."""
    obstacle_height: float
    roughness: float
    slope_deg: float
    step_m: float
    verdict: int
    reasons: List[str]
    neighbours: int


def _neighbour_ground(cells: Dict[str, np.ndarray], cx: float, cy: float,
                      level: int) -> List[Tuple[float, float]]:
    """(distance, ground_z) for the four same-level neighbours that exist."""
    s = ResolutionLevel.size(level)
    out = []
    for dx, dy in ((s, 0.0), (-s, 0.0), (0.0, s), (0.0, -s)):
        hit = SEL.find_cell_row(cells, cx + dx, cy + dy)
        if hit is None:
            continue
        row, lvl = hit
        gz = float(np.asarray(cells["ground_z"][row], np.float32))
        d = float(np.hypot(dx, dy))
        if lvl != level:
            # A coarser neighbour: its centre is further away than one cell.
            ncx = float(cells["cx"][row])
            ncy = float(cells["cy"][row])
            d = max(float(np.hypot(ncx - cx, ncy - cy)), 1e-3)
        out.append((d, gz))
    return out


def derive(cell: Dict[str, Any], cells: Dict[str, np.ndarray],
           profile: Optional[VehicleProfile] = None) -> Derived:
    """Slope, step, obstacle height and the traversability verdict.

    Same rules as ``AdaptiveMap.traversability_arrays``; the neighbours come
    from the cached frame instead of the live map.
    """
    profile = profile or VehicleProfile.wheeled()
    gz = float(cell["ground_z"])
    obst = max(float(cell["z_max"]) - gz, 0.0)
    rough = float(np.sqrt(max(float(cell["z_var"]), 0.0)))
    clear = float(cell["overhead_clearance"])
    cls = int(cell["sem_class"])

    nb = _neighbour_ground(cells, float(cell["cx"]), float(cell["cy"]),
                           int(cell["level"]))
    if nb:
        grads = [abs(g - gz) / d for d, g in nb]
        steps = [abs(g - gz) for _d, g in nb]
        slope = float(np.degrees(np.arctan(max(grads))))
        step = float(max(steps))
    else:
        slope, step = 0.0, 0.0

    reasons: List[str] = []
    if obst > profile.max_step_m * 2.0:
        reasons.append(f"obstacle {obst:.2f} m > "
                       f"{profile.max_step_m * 2:.2f} m")
    if slope > profile.max_slope_deg:
        reasons.append(f"slope {slope:.1f}° > {profile.max_slope_deg:.0f}°")
    if step > profile.max_step_m:
        reasons.append(f"step {step:.2f} m > {profile.max_step_m:.2f} m")
    if clear < profile.min_clearance_m:
        reasons.append(f"clearance {clear:.2f} m < "
                       f"{profile.min_clearance_m:.2f} m")
    if cls in (2, 3, 4):
        reasons.append(f"class {CLASS_NAMES[cls]}")
    if reasons:
        return Derived(obst, rough, slope, step, Traversability.BLOCKED,
                       reasons, len(nb))

    soft: List[str] = []
    if rough > profile.max_roughness:
        soft.append(f"roughness {rough:.3f} m > {profile.max_roughness:.2f} m")
    if slope > profile.max_slope_deg * 0.6:
        soft.append(f"slope {slope:.1f}° near the limit")
    if step > profile.max_step_m * 0.6:
        soft.append(f"step {step:.2f} m near the limit")
    if cls in (1, 5):
        soft.append(f"class {CLASS_NAMES[cls]}")
    if int(cell["n_points"]) < 2:
        soft.append("observed by fewer than 2 points")
    if soft:
        return Derived(obst, rough, slope, step, Traversability.CAUTION,
                       soft, len(nb))
    return Derived(obst, rough, slope, step, Traversability.DRIVABLE,
                   ["within every limit for the wheeled profile"], len(nb))


def cell_morton(cell: Dict[str, Any]) -> int:
    """The cell's address in the hierarchy — its Morton code at its level."""
    s = ResolutionLevel.size(int(cell["level"]))
    ix = int(np.floor(float(cell["cx"]) / s))
    iy = int(np.floor(float(cell["cy"]) / s))
    # `level=` matters: the index bias coarsens with the grid, which is what
    # makes the prefix identity hold at every level and for negative
    # coordinates. Calling it at level 0 with level-l indices returns a code
    # that is not this cell's address in the map.
    return int(morton_2d(np.array([ix]), np.array([iy]),
                         level=int(cell["level"]))[0])


def flag_names(flags: int) -> List[str]:
    return [name for bit, name in sorted(CellFlags.NAMES.items())
            if int(flags) & bit]


def occupancy_probability(logodds: float) -> float:
    lo = float(np.clip(logodds, -30.0, 30.0))
    return float(1.0 / (1.0 + np.exp(-lo)))


# ════════════════════════════════════════════════════════════
# Small HTML pieces — kept out of the section functions
# ════════════════════════════════════════════════════════════
def _bar(value: float, colour: str, *, lo: float = 0.0, hi: float = 1.0,
         width_px: int = 120) -> str:
    frac = float(np.clip((value - lo) / max(hi - lo, 1e-9), 0.0, 1.0))
    return (f'<span style="display:inline-block;width:{width_px}px;height:8px;'
            f'background:#e6e7ea;border-radius:4px;vertical-align:middle;'
            f'overflow:hidden"><span style="display:block;height:100%;'
            f'width:{frac * 100:.1f}%;background:{colour}"></span></span>')


def _chip(text: str, colour: str = "#444", bg: str = "#eceef2") -> str:
    return (f'<span style="display:inline-block;padding:1px 7px;margin:1px 3px '
            f'1px 0;border-radius:9px;background:{bg};color:{colour};'
            f'font-size:11px;font-weight:600">{text}</span>')


def _kv(rows: List[Tuple[str, str]]) -> str:
    body = "".join(
        f'<tr><td style="padding:1px 10px 1px 0;opacity:.62;'
        f'white-space:nowrap">{k}</td>'
        f'<td style="padding:1px 0;font-variant-numeric:tabular-nums">{v}</td>'
        f'</tr>' for k, v in rows)
    return (f'<table style="font-size:12.5px;border-collapse:collapse;'
            f'width:100%">{body}</table>')


def semantic_bars(evidence: np.ndarray) -> str:
    """The full posterior, not just the argmax.

    The argmax alone hides the case the calibration work exists for: a cell
    at 0.34/0.31/0.30 is a different object from one at 0.97.
    """
    ev = np.asarray(evidence, np.float32)
    tot = float(ev.sum())
    p = ev / tot if tot > 1e-9 else np.full(len(ev), 1.0 / max(len(ev), 1))
    order = np.argsort(-p)
    rows = []
    for i in order:
        c = CLASS_COLOURS[int(i)]
        rows.append(
            f'<tr><td style="padding:1px 8px 1px 0;font-size:11.5px;'
            f'white-space:nowrap">{CLASS_NAMES[int(i)]}</td>'
            f'<td style="padding:1px 6px 1px 0">{_bar(p[i], c, width_px=90)}'
            f'</td><td style="font-size:11.5px;font-variant-numeric:'
            f'tabular-nums;opacity:.75">{p[i] * 100:4.1f}%</td></tr>')
    return (f'<table style="border-collapse:collapse;width:100%">'
            f'{"".join(rows)}</table>')


# ════════════════════════════════════════════════════════════
# The eight sections
# ════════════════════════════════════════════════════════════
def render_cell(cell: Dict[str, Any], cells: Dict[str, np.ndarray], *,
                frame_idx: int, n_frames: int, scene_time: float,
                world_xy: Optional[Tuple[float, float]] = None,
                profile: Optional[VehicleProfile] = None) -> None:
    """Draw all eight sections for one cell."""
    import streamlit as st

    d = derive(cell, cells, profile)
    lvl = int(cell["level"])
    res_cm = float(cell["resolution"]) * 100.0

    # ── 1 Identity ───────────────────────────────────────────
    st.markdown("**1 · Identity**")
    ident = [
        ("cell centre", f'{cell["cx"]:.3f}, {cell["cy"]:.3f} m'),
        ("resolution", f'{res_cm:.0f} cm  (level {lvl} — '
                       f'{ResolutionLevel.name(lvl)})'),
        ("footprint", f'{cell["resolution"]:.2f} × {cell["resolution"]:.2f} m'),
        ("Morton code", f'{cell_morton(cell)}  <span style="opacity:.55">'
                        f'(level-0 code &gt;&gt; {2 * lvl})</span>'),
        ("frame", f'{frame_idx + 1} / {n_frames}  ·  t = {scene_time:.2f} s'),
    ]
    if world_xy is not None:
        ident.insert(0, ("clicked at",
                         f'{world_xy[0]:.3f}, {world_xy[1]:.3f} m (world)'))
    st.markdown(_kv(ident), unsafe_allow_html=True)
    st.caption("Cell size is a decision the allocation controller made for "
               "this tile on this frame, not a function of range alone.")

    # ── 2 Terrain & geometry ─────────────────────────────────
    st.markdown("**2 · Terrain & geometry**")
    tcol = TRAV_COLOURS[int(d.verdict)]
    st.markdown(
        f'<div style="margin:2px 0 5px 0">{_chip(Traversability.NAMES[int(d.verdict)], "#fff", tcol)}'
        f'<span style="font-size:11.5px;opacity:.7">'
        f'{"; ".join(d.reasons)}</span></div>', unsafe_allow_html=True)
    st.markdown(_kv([
        ("ground height", f'{cell["ground_z"]:+.3f} m'),
        ("z max (in band)", f'{cell["z_max"]:+.3f} m'),
        ("obstacle height", f'{d.obstacle_height:.3f} m '
                            f'<span style="opacity:.55">derived</span>'),
        ("overhead clearance", f'{cell["overhead_clearance"]:.2f} m'),
        ("height variance", f'{cell["z_var"]:.4f} m²'),
        ("roughness (σ)", f'{d.roughness:.3f} m '
                          f'<span style="opacity:.55">derived</span>'),
        ("local slope", f'{d.slope_deg:.1f}° '
                        f'<span style="opacity:.55">from {d.neighbours} '
                        f'cached neighbour(s)</span>'),
        ("local step", f'{d.step_m:.3f} m '
                       f'<span style="opacity:.55">derived</span>'),
    ]), unsafe_allow_html=True)
    st.caption("Slope, step, obstacle height and the verdict are computed on "
               "demand from stored terrain properties — which is why the same "
               "map answers differently for a wheeled and a tracked vehicle.")

    # ── 3 Semantics ──────────────────────────────────────────
    st.markdown("**3 · Semantics**")
    cls = int(cell["sem_class"])
    ent = float(cell["entropy"])
    st.markdown(
        f'<div style="margin-bottom:4px">'
        f'{_chip(CLASS_NAMES[cls], "#fff", CLASS_COLOURS[cls])}'
        f'<span style="font-size:11.5px;opacity:.7">argmax of the accumulated '
        f'evidence</span></div>', unsafe_allow_html=True)
    ev = cell.get("evidence")
    if ev is not None:
        st.markdown(semantic_bars(ev), unsafe_allow_html=True)
    st.markdown(_kv([
        ("entropy", f'{_bar(ent, "#8a6fd4")} &nbsp;{ent:.3f} '
                    f'<span style="opacity:.55">0 = certain, 1 = uniform'
                    f'</span>'),
    ]), unsafe_allow_html=True)
    st.caption("Evidence is ADDED across frames (the conjugate categorical "
               "update), never priority-weighted — weighting would corrupt "
               "the posterior and make the accuracy numbers meaningless.")

    # ── 4 Occupancy ──────────────────────────────────────────
    st.markdown("**4 · Occupancy**")
    lo = float(cell["occupancy_logodds"])
    p_occ = occupancy_probability(lo)
    st.markdown(_kv([
        ("state", _chip(Occupancy.NAMES.get(int(cell["occupancy_state"]),
                                            "UNKNOWN"))),
        ("log-odds", f'{lo:+.2f}'),
        ("P(occupied)", f'{_bar(p_occ, "#2f7fd1")} &nbsp;{p_occ * 100:.1f}%'),
    ]), unsafe_allow_html=True)
    st.caption("Three states, not two: UNKNOWN is space never swept, which is "
               "not the same claim as FREE.")

    # ── 5 Dynamics ───────────────────────────────────────────
    st.markdown("**5 · Dynamics**")
    dyn = float(cell["dynamic_prob"])
    age = max(frame_idx - int(cell["last_seen"]), 0)
    st.markdown(_kv([
        ("P(dynamic)", f'{_bar(dyn, "#d9483f")} &nbsp;{dyn * 100:.1f}%'),
        ("last seen", f'frame {int(cell["last_seen"])} '
                      f'<span style="opacity:.55">({age} frame(s) ago)'
                      f'</span>'),
    ]), unsafe_allow_html=True)
    st.caption("Moving points are gated OUT of the persistent map before it "
               "is written, so a high value here means the cell was written "
               "while the evidence still looked static — not that a moving "
               "object is stored in it.")

    # ── 6 Sensor & intensity ─────────────────────────────────
    st.markdown("**6 · Sensor & intensity**")
    ret = float(cell["penetration"])
    st.markdown(_kv([
        ("points fused", f'{int(cell["n_points"]):,}'),
        ("intensity (mean)", f'{_bar(float(cell["intensity_mean"]), "#c98a1e")}'
                             f' &nbsp;{float(cell["intensity_mean"]):.3f}'),
        ("intensity (var)", f'{float(cell["intensity_var"]):.4f}'),
        ("return position", f'{ret:.3f} '
                            f'<span style="opacity:.55">return_number / '
                            f'return_count — 1.0 means a single return, '
                            f'lower means the beam carried on past this '
                            f'surface</span>'),
        ("observability", f'{_bar(float(cell["observability"]), "#3d9a6f")}'
                          f' &nbsp;{float(cell["observability"]):.3f}'),
    ]), unsafe_allow_html=True)
    st.caption("Intensity is range- and incidence-corrected before it reaches "
               "the map; uncorrected, a bright surface at 60 m returns less "
               "than a dark one at 5 m.")

    # ── 7 Flags ──────────────────────────────────────────────
    st.markdown("**7 · Flags**")
    names = flag_names(int(cell["flags"]))
    if names:
        colours = {"safety_pinned": ("#fff", "#b8342b"),
                   "refined": ("#fff", "#2f7fd1"),
                   "stale": ("#333", "#d8d8d8"),
                   "multi_surface": ("#fff", "#8a6fd4"),
                   "geom_sem_disagreement": ("#fff", "#c98a1e")}
        st.markdown("".join(_chip(n, *colours.get(n, ("#444", "#eceef2")))
                            for n in names), unsafe_allow_html=True)
    else:
        st.markdown('<span style="font-size:12px;opacity:.6">none set</span>',
                    unsafe_allow_html=True)
    st.caption("`safety_pinned` is the geometric pin — a vertically extended "
               "cluster — which holds this cell at its resolution regardless "
               "of budget, and regardless of what the classifier thinks.")

    # ── 8 Raw / debug ────────────────────────────────────────
    with st.expander("8 · Raw / debug", expanded=False):
        st.caption("The stored record, field by field. Shown as a table "
                   "rather than an array dump so each number is named.")
        rows = [(k, _fmt_raw(cell[k])) for k in sorted(cell)
                if k not in ("evidence",) and not isinstance(
                    cell[k], np.ndarray)]
        st.markdown(_kv(rows), unsafe_allow_html=True)
        if ev is not None:
            st.markdown(
                "**evidence** (6-class, normalised): " +
                ", ".join(f"{CLASS_NAMES[i]} {float(v):.3f}"
                          for i, v in enumerate(np.asarray(ev))),
                unsafe_allow_html=True)
        st.caption("Values are read back from the per-frame cache, which "
                   "packs several fields to a byte — so they are the map's "
                   "numbers to about 1/255 of their range, not to float "
                   "precision.")


def _fmt_raw(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


# ════════════════════════════════════════════════════════════
# The object inspector
# ════════════════════════════════════════════════════════════
def render_object(obj: Dict[str, Any], *, ego_xy, has_cell: bool = True
                  ) -> bool:
    """Draw a tracked object. Returns True if "inspect cell" was pressed."""
    import streamlit as st

    c = obj["centroid_world"]
    ext = obj["extent"]
    v = obj["vel_world"]
    ocls = int(obj["cls"])
    d = float(np.hypot(c[0] - ego_xy[0], c[1] - ego_xy[1]))
    state_col = {TrackState.STATIC: "#5b6470",
                 TrackState.MOVING: "#d9483f",
                 TrackState.MOVABLE_BUT_STATIONARY: "#c98a1e"}

    st.markdown(
        f'<div style="margin-bottom:4px">'
        f'{_chip(CLASS_NAMES[ocls], "#fff", CLASS_COLOURS[ocls])}'
        f'{_chip(obj["state_name"], "#fff", state_col.get(int(obj["state"]), "#5b6470"))}'
        f'</div>', unsafe_allow_html=True)
    st.markdown(_kv([
        ("track id", f'#{int(obj["id"])}'),
        ("centroid", f'{c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f} m (world)'),
        ("range from ego", f'{d:.1f} m'),
        ("extent", f'{ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f} m'),
        ("speed (world)", f'{obj["speed_world"]:.2f} m/s'),
        ("velocity", f'{v[0]:+.2f}, {v[1]:+.2f} m/s'),
        ("points", f'{int(obj["points"]):,}'),
        ("track age", f'{int(obj["age"])} frame(s)'),
    ]), unsafe_allow_html=True)
    st.caption("Speed is world-frame. A car matching the ego speed has "
               "near-zero RELATIVE speed and is still correctly MOVING; a "
               "parked one is MOVABLE_BUT_STATIONARY rather than STATIC, "
               "because it may drive off.")
    return st.button("Inspect underlying cell", width="stretch",
                     key="obj_inspect_cell", disabled=not has_cell,
                     help="Switch to the map cell nearest this object's "
                          "centroid." if has_cell else
                          "No mapped surface near this centroid.")
