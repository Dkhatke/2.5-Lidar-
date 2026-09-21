"""
The Scene demo tab — what the 2.5D map represents in the physical world.

Live demo answers "what does the adaptive map look like?". This answers
"what is that map a map OF?", using nothing but data the pipeline already
produced: cached cells give the surfaces and their heights, tracked
instances give the objects, the stored pose gives the vehicle. No new
model, no new clustering, no invented geometry — and where a box is drawn
larger than the sensor measured, the panel says which and by how much.

SHARED WITH LIVE DEMO
---------------------
Frame index, playback speed and the current selection all live in
``session.py``. Select a cell here, switch tab, and the 2D map has the same
cell selected with the same inspector open. The transport widgets are keyed
per tab because Streamlit widgets cannot share a key; the state behind them
is one object.

The inspector is not reimplemented. It is the same ``inspector.render_cell``
the Live demo calls, driven from the same cached frame, because two
inspectors is how a dashboard ends up showing two different answers.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import streamlit as st

from adaptive_lidar.visualization import inspector as INS
from adaptive_lidar.visualization import scene_component as SC
from adaptive_lidar.visualization import scene_data as SD
from adaptive_lidar.visualization import selection as SEL
from adaptive_lidar.visualization import session as SESSION
from adaptive_lidar.visualization import ui_layout as UI
from adaptive_lidar.visualization import profile as PROFILE
from adaptive_lidar.visualization.playback_controller import PlaybackController

from adaptive_lidar.visualization.live_demo import (  # noqa: E402
    DEFAULT_FRAMES, MAX_FRAMES, MIN_FRAMES)

RAIL_H = 620
CANVAS_H = 620

#: Measured floor on this tab's redraw, per draw budget: building the
#: payload is ~55 ms and shipping it to the browser is the rest, so the
#: floor tracks how much is being shipped. Scheduling below it queues
#: reruns and the browser then receives them out of order.
#:
#: A LOWER BOUND only. The interval actually used is whichever is larger
#: of this, the requested speed, and 1.5x the measured redraw — see
#: `session.paced_interval`. These numbers only stop the pacing being
#: absurdly fast before the first measurement lands.
REDRAW_FLOOR_S = {"Light": 0.12, "Balanced": 0.18, "Full": 0.30}
_CLICK_KEY = "_scene_last_click"

#: What the browser was last sent. While this matches, the run payload is
#: not re-sent and the component redraws from its own copy.
_RUN_SENT = "_scene_run_sent"

#: Draw-budget presets: (cells drawn at most, radius they are taken from).
#:
#: These are DISPLAY budgets. The map still holds every cell it computed
#: and the inspector still reads them; `scene_data.display_sample` decides
#: which are worth a draw call, foveated so detail stays near the vehicle.
#: The thinning normally lands under the cap on its own — the cap is the
#: guarantee, not the mechanism.
DETAIL = {
    "Light": (2500, 45.0),
    "Balanced": (5000, SD.DEFAULT_RADIUS_M),
    "Full": (9000, 75.0),
}


@st.cache_resource(show_spinner=False, max_entries=4)
def _run_payload(_run, key, max_cells: int, radius: float,
                 show_points: bool):
    """Every frame's drawable geometry, built once per run.

    `cache_resource`, not `cache_data`: this is a megabyte or two and
    `cache_data` would deep-copy it on every hit, which is most of what
    the caching is there to avoid. `_run` is hash-excluded by the leading
    underscore; `key` carries the identity.
    """
    return SD.build_run_payload(_run, budget=max_cells, radius_m=radius,
                                show_points=show_points)


def _state(defaults: Dict[str, Any]) -> None:
    import streamlit as st
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ════════════════════════════════════════════════════════════
# Controls
# ════════════════════════════════════════════════════════════
def _left_rail() -> Dict[str, Any]:
    import streamlit as st
    from adaptive_lidar.data.synthetic_scene import SCENARIOS

    UI.rail_heading("Scene")
    # Shared with the Live demo: changing the scenario here changes it
    # there, because they are one run. Each tab keys its own widget and
    # both sync to the one session slot — Streamlit refuses a duplicate
    # widget key even across tabs.
    SESSION.sync("scene_scenario", SESSION.SCENARIO_KEY, "mixed_urban")
    scenario = st.selectbox("Scenario", SCENARIOS, key="scene_scenario",
                            label_visibility="collapsed")

    SESSION.sync("scene_n_frames", SESSION.N_FRAMES_KEY, DEFAULT_FRAMES)
    n_frames = st.slider("Frames", MIN_FRAMES, MAX_FRAMES,
                         key="scene_n_frames")

    UI.rail_heading("Colour by")
    colour_by = st.selectbox("Colour by", SD.COLOUR_MODES,
                             key="scene_colour", label_visibility="collapsed")

    UI.rail_heading("Height")
    height_mode = st.radio(
        "Height", list(SD.HEIGHT_MODES), key="scene_height",
        horizontal=True, label_visibility="collapsed",
        help="How much of a cell's real base-to-top extent is drawn. "
             "'extruded' is 1:1 with the stored z_max; 'subtle' compresses "
             "it so a 12 m facade does not hide the road behind it. Neither "
             "changes a stored number, and the inspector always shows the "
             "unscaled metres.")

    UI.rail_heading("Show")
    # One column: the rail is ~200 px and a two-column layout truncated
    # every label to its first letter.
    st.checkbox("Adaptive grid", key="scene_grid",
                help="Draws each cell at 88% of its true footprint, so the "
                     "seam between neighbours is visible and the size step "
                     "from 5 cm near the vehicle to 80 cm far away is "
                     "obvious. A drawing device — the footprint the "
                     "inspector reports is the true one.")
    st.checkbox("Objects", key="scene_objects")
    st.checkbox("Labels", key="scene_labels")
    st.checkbox("Distance rings", key="scene_rings",
                help="At the evaluation's own range-band edges (10, 30, 60, "
                     "100 m), so the rings match every stratified table in "
                     "docs/RESULTS.md.")
    st.checkbox("Velocity", key="scene_velocity",
                help="One metre of arrow per m/s, world frame — the same "
                     "convention as the 2D overlay.")
    st.checkbox("Moving points", key="scene_points",
                help="The dynamic overlay: the points the MOS gate held out "
                     "of the persistent map this frame. The full cloud is "
                     "not cached, so this is the only real point data "
                     "available here.")

    with st.expander("Advanced", expanded=False):
        detail = st.select_slider("Draw budget", list(DETAIL),
                                  key="scene_detail",
                                  help="How many cells are sent to the "
                                       "browser, nearest first. Anything "
                                       "dropped is counted under the canvas "
                                       "rather than quietly thinned out.")
        SESSION.sync("scene_gate", SESSION.GATE_KEY, True)
        st.checkbox("MOS gate", key="scene_gate",
                    help="Shared with the Live demo. OFF writes moving "
                         "points into the persistent map — the trail "
                         "ablation, pre-computed either way.")
    return {"scenario": scenario, "n_frames": n_frames,
            "colour_by": colour_by, "height_mode": height_mode,
            "detail": detail}


# ════════════════════════════════════════════════════════════
# Right rail — the SAME inspector the Live demo uses
# ════════════════════════════════════════════════════════════
def _right_rail(sel: Optional[SEL.Selection], snap, *, frame_idx: int,
                n_frames: int) -> None:
    import streamlit as st

    UI.rail_heading("Inspector")
    if sel is None or sel.is_empty:
        st.info("Click a surface or an object in the scene.\n\n"
                "Every shape you can click is a real map cell or a real "
                "tracked instance — the inspector below is the same one the "
                "Live demo opens, reading the same cached frame.")
        if sel is not None and sel.is_empty:
            st.caption(
                f"Last click at ({sel.world_xy[0]:.2f}, "
                f"{sel.world_xy[1]:.2f}) m found no cell. Absence of a cell "
                f"is not free space — it is unobserved ground.")
        return

    if sel.kind == "object" and sel.obj is not None:
        if INS.render_object(sel.obj, ego_xy=snap.ego_xy,
                             has_cell=sel.cell is not None,
                             key="scene_inspect_cell"):
            SESSION.set_selection(SEL.Selection(
                "cell", sel.world_xy, frame_idx, cell=sel.cell))
            SESSION.refresh()
        _geometry_note(sel.obj)
        return

    if sel.cell is not None:
        _headline(sel.cell)
        INS.render_cell(sel.cell, snap.cells, frame_idx=frame_idx,
                        n_frames=n_frames, scene_time=snap.timestamp,
                        world_xy=sel.world_xy)


def _headline(cell: Dict[str, Any]) -> None:
    """One line tying the solid-looking shape back to the 2.5D record.

    The whole point of this tab is that the scene is a drawing OF the map,
    so the first thing the inspector says is which cell you just clicked.
    """
    import streamlit as st
    from adaptive_lidar.pipeline.types import CLASS_NAMES
    st.markdown(
        f'<div style="font-size:13px;margin:.1rem 0 .4rem 0">'
        f'<b>{cell["resolution"] * 100:.0f} cm cell</b> · '
        f'{CLASS_NAMES[int(cell["sem_class"])]} · '
        f'{int(cell["n_points"])} return(s)<br>'
        f'<span style="opacity:.62">the shape you clicked is one row of '
        f'the 2.5D map, drawn from ground '
        f'{cell["ground_z"]:+.2f} m to top {cell["z_max"]:+.2f} m</span>'
        f'</div>', unsafe_allow_html=True)


def _geometry_note(obj: Dict[str, Any]) -> None:
    """Say plainly when a drawn box is bigger than what was measured."""
    import streamlit as st
    scene_obj = SD.object_to_scene_object(obj, (0.0, 0.0))
    if scene_obj["geomSource"] != "padded":
        st.caption("The box drawn in the scene is the tracker's measured "
                   "extent, unmodified.")
        return
    m, d = scene_obj["measured"], scene_obj["size"]
    st.caption(
        f"**Visualisation geometry.** The sensor measured "
        f"{m[0]:.2f} × {m[1]:.2f} × {m[2]:.2f} m — the returns from one "
        f"side of the object. The scene draws {d[0]:.2f} × {d[1]:.2f} × "
        f"{d[2]:.2f} m, raised to this class's floor so the object is "
        f"visible at all, and shaded fainter than a fully measured box. "
        f"The measured numbers above are the real ones.")


# ════════════════════════════════════════════════════════════
# The tab
# ════════════════════════════════════════════════════════════
def render_scene_demo(precompute_fn: Callable[..., Any],
                      n_frames_default: int = DEFAULT_FRAMES) -> None:
    """Draw the Scene demo tab.

    ``precompute_fn(scenario, n_frames, gate)`` must be cached by the
    caller: it is the only expensive call here, and a frame advance must
    never reach it.
    """
    import streamlit as st

    _state({
        SESSION.SCENARIO_KEY: "mixed_urban",
        SESSION.N_FRAMES_KEY: n_frames_default,
        SESSION.GATE_KEY: True,
        "scene_colour": "semantic",
        "scene_height": "subtle",
        "scene_grid": True,
        "scene_objects": True,
        "scene_labels": True,
        "scene_rings": True,
        "scene_velocity": True,
        "scene_points": False,
        "scene_detail": "Balanced",
        SESSION.SELECTION_KEY: None,
    })

    if not SC.available():
        st.error(f"The scene renderer could not be loaded: "
                 f"{SC.unavailable_reason()}. The Live demo is unaffected.")
        return

    interval = None
    pb0 = st.session_state.get(SESSION.PLAYBACK_STATE_KEY)
    if pb0 is not None and getattr(pb0, "playing", False):
        # See MIN_REDRAW_S: this view costs more per redraw than the 2D
        # one — a scene payload is ~400 kB — so it is scheduled slower and
        # simply skips frames. The frame shown is still whatever the
        # shared clock says at the moment it draws, so the two tabs never
        # disagree about where playback is.
        if SESSION.owns_ticker("scene"):
            # Paced by what a redraw HERE actually costs, not by a
            # constant measured on the development machine. On a weaker
            # laptop a fixed interval schedules reruns faster than they
            # can be served; they queue, and playback stutters and looks
            # like it restarts.
            interval = SESSION.paced_interval(
                "scene", pb0.interval_s(),
                floor=REDRAW_FLOOR_S.get(
                    st.session_state.get("scene_detail", "Balanced"), 0.25))
    PROFILE.log("arm", f"scene interval={interval} "
                       f"playing={getattr(pb0, 'playing', None)}")

    if PlaybackController.supports_fragments():
        st.fragment(run_every=interval)(_workspace)(precompute_fn)
    else:
        st.caption("Installed Streamlit has no fragment support; playback "
                   "steps manually.")
        _workspace(precompute_fn)


def _workspace(precompute_fn: Callable[..., Any]) -> None:
    import streamlit as st
    import time as _t
    _t0 = _t.perf_counter()

    left, centre, right = UI.workspace((1.05, 3.25, 1.5))

    with left:
        with st.container(height=RAIL_H, border=False):
            opt = _left_rail()

    with st.spinner(f"Pre-computing {opt['scenario']} "
                    f"({opt['n_frames']} frames, both MOS variants)…"):
        run_on = precompute_fn(opt["scenario"], opt["n_frames"], True)
        run_off = precompute_fn(opt["scenario"], opt["n_frames"], False)
    run = run_on if st.session_state[SESSION.GATE_KEY] else run_off
    n = len(run)
    if n == 0:
        with centre:
            st.warning("No frames for this scenario.")
        return

    pb = PlaybackController.bind(n, key="scene",
                                 state_key=SESSION.PLAYBACK_STATE_KEY)
    # Advance BEFORE choosing the frame, not after drawing it.
    # The index is a function of the clock, so each tab renders
    # whatever is current at its own render moment. Doing it last
    # meant rendering the value the other tab's ticker happened to
    # leave behind, which made the displayed frame jump backwards
    # whenever the two fragments got out of step.
    # Only the tab driving playback advances the frame; the other shows
    # whatever the shared state holds. Two tickers on one index play the
    # run at the sum of their render rates.
    if SESSION.owns_ticker("scene"):
        pb.tick_if_playing()
    idx = min(pb.state.frame_idx, n - 1)
    def _report(paced: bool = True) -> None:
        took = _t.perf_counter() - _t0
        # A redraw that also BUILT and sent the run is a load, not a
        # playback frame. Folding seconds of one-off work into the pace
        # stretched the playback interval to several seconds.
        if paced:
            SESSION.record_redraw("scene", took)
        PROFILE.log("scene", f"{took * 1000:7.1f} ms  frame {idx}  "
                             f"pace {SESSION.redraw_cost('scene'):.3f}")
    snap = run.frames[idx]

    sel: Optional[SEL.Selection] = SESSION.get_selection()
    if sel is not None and sel.frame_idx != idx:
        sel = SEL.reselect_on_frame(sel, snap, idx)
        SESSION.set_selection(sel)

    max_cells, radius = DETAIL[opt["detail"]]

    # The run's geometry is built once and cached in the browser. The
    # identity is everything that changes what was built; the display
    # options are deliberately NOT in it, so changing colour or height
    # costs nothing.
    run_id = (opt["scenario"], opt["n_frames"],
              bool(st.session_state[SESSION.GATE_KEY]), opt["detail"],
              bool(st.session_state["scene_points"]))
    # Built (or fetched from cache) every run; SENT only when it changed.
    payload = _run_payload(run, run_id, max_cells, radius,
                           bool(st.session_state["scene_points"]))
    run_payload = None
    if st.session_state.get(_RUN_SENT) != run_id:
        run_payload = payload
        st.session_state[_RUN_SENT] = run_id

    state = {
        "frame": int(idx),
        "nFrames": int(n),
        "colourBy": opt["colour_by"],
        "heightScale": SD.HEIGHT_MODES.get(opt["height_mode"], 0.45),
        "heightMode": opt["height_mode"],
        "selected": SD.selection_marker(sel),
    }

    with centre:
        UI.disclaimer(
            "DEMO DATA · Synthetic scenario. Every surface, height and box "
            "below is drawn from the map and tracks the pipeline already "
            "produced — this is a stylised view of the 2.5D map, not a 3D "
            "reconstruction, and no second perception run stands behind it.")
        raw = SC.render(
            run=run_payload, state=state, key="fovea_scene", height=CANVAS_H,
            show_grid=st.session_state["scene_grid"],
            show_objects=st.session_state["scene_objects"],
            show_labels=st.session_state["scene_labels"],
            show_rings=st.session_state["scene_rings"],
            show_velocity=st.session_state["scene_velocity"])

        if SC.needs_run(raw):
            # A page reload drops the browser's copy while this session
            # still believes it was sent. Forget, and send it again.
            st.session_state.pop(_RUN_SENT, None)
            SESSION.refresh("app")

        click = SC.consume_click(raw, _CLICK_KEY)
        if click is not None:
            _apply_click(click, snap, idx)

        _legend(opt["colour_by"])
        UI.metrics_strip(list(SD.scene_summary(
            payload["frames"][min(idx, len(payload["frames"]) - 1)]).items()))
        st.caption(
            f"Heights are drawn at {state['heightScale']:.0%} of the stored "
            f"base-to-top extent ({opt['height_mode']}). The map keeps "
            f"every cell; the scene draws a foveated sample of them, and "
            f"a drawn tile covers the area it stands for — so colour by "
            f"**resolution level** to read what the map actually chose. "
            f"The run's geometry is sent once and a frame change sends "
            f"only its index.")

    with right:
        with st.container(height=RAIL_H, border=False):
            _right_rail(sel, snap, frame_idx=idx, n_frames=n)

    st.markdown("")
    pb.render_transport(scene_time_s=snap.timestamp,
                        driving=SESSION.owns_ticker("scene"))
    sp1, _sp2 = st.columns([1, 5])
    with sp1:
        pb.speed_selector()
    _report(paced=run_payload is None)


def _apply_click(click: Dict[str, Any], snap, idx: int) -> None:
    """Turn a browser click into a Selection, through the real lookup.

    The renderer reports a world point and, if it hit one, a track id. It
    never reports a cell: resolving the point is ``selection.py``'s job
    against the cached frame, so the scene and the 2D map cannot disagree
    about which cell is under a given place.
    """
    kind = click.get("kind")
    xy = SC.click_to_world(click)

    if kind == "object":
        oid = int(click.get("id", -1))
        obj = next((o for o in (snap.objects or []) if o["id"] == oid), None)
        if obj is not None:
            c = obj["centroid_world"]
            cell, off = SEL.nearest_cell(snap.cells, float(c[0]), float(c[1]))
            SESSION.set_selection(SEL.Selection(
                "object", (float(c[0]), float(c[1])), idx,
                cell=cell, obj=obj, cell_offset_m=off))
            SESSION.refresh()
            return

    if xy is None:
        return
    cell = SEL.cell_at_world(snap.cells, xy[0], xy[1])
    SESSION.set_selection(
        SEL.Selection("cell", xy, idx, cell=cell) if cell is not None
        else SEL.Selection("empty", xy, idx))
    SESSION.refresh()


def _legend(colour_by: str) -> None:
    """Compact legend, in the project's own vocabulary."""
    import streamlit as st
    from adaptive_lidar.pipeline.types import (CLASS_NAMES, ResolutionLevel,
                                               Traversability)
    from adaptive_lidar.visualization.render import (CLASS_COLOURS,
                                                     LEVEL_COLOURS,
                                                     TRAV_COLOURS)
    if colour_by == "traversability":
        items = [(Traversability.NAMES[i], TRAV_COLOURS[i]) for i in range(3)]
    elif colour_by == "resolution level":
        items = [(ResolutionLevel.name(i), LEVEL_COLOURS[i])
                 for i in range(ResolutionLevel.N_LEVELS)]
    elif colour_by == "height":
        items = [("low", "#2b4a8c"), ("mid", "#2e8b57"), ("high", "#e8a33d")]
    else:
        items = [(CLASS_NAMES[i], CLASS_COLOURS[i])
                 for i in range(len(CLASS_NAMES))]

    chips = "".join(
        f'<span style="display:inline-block;margin:0 12px 2px 0;'
        f'font-size:11.5px"><span style="display:inline-block;width:10px;'
        f'height:10px;border-radius:2px;background:{c};'
        f'margin-right:5px;vertical-align:-1px"></span>{n}</span>'
        for n, c in items)
    states = (
        '<span style="opacity:.55;font-size:11.5px">tracks:</span> '
        '<span style="color:#5b6470;font-size:11.5px;font-weight:600">'
        'STATIC</span> · '
        '<span style="color:#d9483f;font-size:11.5px;font-weight:600">'
        'MOVING</span> · '
        '<span style="color:#c98a1e;font-size:11.5px;font-weight:600">'
        'MOVABLE_BUT_STATIONARY</span>')
    st.markdown(f'<div style="margin:.35rem 0">{chips}</div>'
                f'<div style="margin:.1rem 0 .3rem 0">{states}</div>',
                unsafe_allow_html=True)
