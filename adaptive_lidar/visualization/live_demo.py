"""
The Live Demo tab — the thing a judge sees first.

LAYOUT
------
    ┌──────────┬────────────────────────────┬──────────────┐
    │ controls │  the scene (clickable)     │  inspector   │
    └──────────┴────────────────────────────┴──────────────┘
    ┌──────────────────────────────────────────────────────┐
    │  ◀ Previous   ▶ Play   ⏭ Next   ↺ Reset   timeline   │
    └──────────────────────────────────────────────────────┘

Controls on the left because they are secondary; the scene in the middle
because it is the subject; the inspector on the right because it answers a
question the scene raised.

WHY THE WHOLE WORKSPACE IS ONE FRAGMENT
---------------------------------------
Everything in here reads from the playback cache. Wrapping the workspace in
a single ``st.fragment`` means a frame advance, a layer change or a click
re-executes only this function — the pipeline, which lives above it behind
``st.cache_resource``, is never re-entered. With ``run_every`` set while
playing, that is also what animates the scene: no ``time.sleep``, so the
controls stay live and a click during playback is not swallowed.

The pipeline runs once per scenario, when it is loaded. Nothing below that
point re-runs perception — which is also why the number beside the transport
is called a replay rate and never FPS.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import numpy as np

from adaptive_lidar.visualization import camera as CAM
from adaptive_lidar.visualization import inspector as INS
from adaptive_lidar.visualization import interactive_map as IM
from adaptive_lidar.visualization import overlays as OV
from adaptive_lidar.visualization import render as R
from adaptive_lidar.visualization import selection as SEL
from adaptive_lidar.visualization import session as SESSION
from adaptive_lidar.visualization import ui_layout as UI
from adaptive_lidar.visualization.drive_tab import PRESETS, scene_diagnostics
from adaptive_lidar.visualization import profile as PROFILE
from adaptive_lidar.visualization.playback_controller import PlaybackController

#: How long a run can be. The scans are raycast once per scenario and
#: memoised to `.scan_cache/`, so a longer run costs generation time on the
#: first load of a scenario and nothing after that. 24 frames is 2.4 s of
#: scene at 10 Hz, which is long enough for the vehicle to drive past the
#: 70 m pedestrian and for the resolution island to visibly travel with it.
MIN_FRAMES, MAX_FRAMES, DEFAULT_FRAMES = 8, 40, 24

#: Fixed height for the two side rails, in pixels. Chosen to match the
#: canvas so the workspace fits one screen and the transport stays visible.
RAIL_H = 640

#: A LOWER BOUND only. The interval used is whichever is larger of this,
#: the requested speed, and 1.5x the measured redraw — see
#: `session.paced_interval`. Set FOVEA_PROFILE=1 to watch it settle.
MIN_REDRAW_S = 0.12



#: Selection and playback are SHARED with the Scene demo tab, so that
#: switching tabs keeps the same frame and the same selected cell. Only the
#: widget keys stay per-tab.
_SEL_KEY = SESSION.SELECTION_KEY
_CANVAS_KEY = "live_canvas"


def _refresh() -> None:
    """Redraw. See :func:`session.refresh` for why this is app-scoped.

    In short: the selection is shared with the Scene demo tab, and a
    fragment-scoped rerun would leave that tab showing the previous one.
    """
    SESSION.refresh()


def _state(defaults: Dict[str, Any]) -> None:
    import streamlit as st
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _apply_preset(key: str) -> None:
    import streamlit as st
    for k, v in PRESETS[key]["state"].items():
        st.session_state[k] = v
    pb = st.session_state.get(SESSION.PLAYBACK_STATE_KEY)
    if pb is not None:
        pb.frame_idx = 0
        pb.playing = True
    st.session_state[_SEL_KEY] = None


# ════════════════════════════════════════════════════════════
# The left rail
# ════════════════════════════════════════════════════════════
def _left_rail(scenarios, n_frames_default: int) -> Dict[str, Any]:
    """The controls, compact.

    Held to a fixed height with its own scroll. Streamlit stretches every
    column in a row to the tallest one, so a rail that grew past the canvas
    would open a band of dead space under the scene and push the transport
    off the screen.
    """
    import streamlit as st

    UI.rail_heading("Scene")
    # Scenario, frame count and the MOS gate decide WHICH RUN is loaded, so
    # they are shared with the Scene demo tab. Each tab keys its own widget
    # — Streamlit refuses a duplicate key even across tabs — and both read
    # and write one session slot.
    SESSION.sync("drive_scenario", SESSION.SCENARIO_KEY, "mixed_urban")
    scenario = st.selectbox("Scenario", scenarios, key="drive_scenario",
                            label_visibility="collapsed")
    c1, c2 = st.columns(2)
    with c1:
        SESSION.sync("drive_n_frames", SESSION.N_FRAMES_KEY,
                     n_frames_default)
        n_frames = st.slider("Frames", MIN_FRAMES, MAX_FRAMES,
                             key="drive_n_frames")
    with c2:
        zoom = st.slider("Zoom", 0.5, 2.5, 1.0, 0.1, key="drive_zoom")
    frame_ref = st.radio("View frame", CAM.FRAME_CHOICES,
                         key="drive_frame_ref", horizontal=True,
                         help=" · ".join(f"{k}: {v}" for k, v
                                         in CAM.FRAME_CAPTION.items()))
    cam = st.selectbox("Camera", CAM.CAMERA_PRESETS, key="drive_camera",
                       help=CAM.PRESET_CAPTION[
                           st.session_state.get("drive_camera",
                                                "top-follow")])

    UI.rail_heading("Map layer")
    layer = st.selectbox("Layer", R.LAYERS, key="drive_layer",
                         label_visibility="collapsed")

    UI.rail_heading("Display")
    d1, d2 = st.columns(2)
    with d1:
        st.checkbox("Cell edges", key="drive_edges")
        st.checkbox("Objects", key="drive_objects")
    with d2:
        st.checkbox("Age tint", key="drive_age_tint",
                    help="Brighten cells refined in the last few frames, so "
                         "the leading edge of the refinement front is "
                         "visible rather than merely inferable.")
        st.checkbox("Corridor", key="drive_corridor",
                    help="Every location ever resolved at 5 or 10 cm, kept "
                         "in a separate accumulator because the map's "
                         "sliding window has already evicted the cells "
                         "behind the vehicle.")

    UI.rail_heading("Scene presets")
    for key, p in PRESETS.items():
        if st.button(p["label"], help=p["help"], width="stretch",
                     key=f"live_preset_{key}"):
            _apply_preset(key)
            _refresh()

    with st.expander("Advanced", expanded=False):
        SESSION.sync("drive_gate", SESSION.GATE_KEY, True)
        st.checkbox("MOS gate", key="drive_gate",
                    help="OFF writes moving points into the persistent map. "
                         "Both variants are pre-computed, so this is "
                         "instant — it is an ablation, not a setting.")
        st.checkbox("Predicted path", key="drive_predict",
                    help="2 s constant-velocity extrapolation of a MOVING "
                         "track. Not a prediction model.")
        st.checkbox("Dynamic overlay", key="drive_overlay",
                    help="The moving points held out of the persistent map "
                         "and rebuilt every frame.")
        st.checkbox("Objects are clickable", key="drive_click_objects",
                    help="Off makes every click select the map cell "
                         "underneath, including where an object sits.")
        st.slider("Render resolution (px/m)", 4, 14, 7, key="drive_ppm",
                  help="Canvas detail. Higher costs more per displayed "
                       "frame and changes nothing the pipeline measured.")
    return {"scenario": scenario, "n_frames": n_frames,
            "frame_ref": frame_ref, "camera": cam, "layer": layer,
            "zoom": zoom}


# ════════════════════════════════════════════════════════════
# The right rail
# ════════════════════════════════════════════════════════════
def _right_rail(sel: Optional[SEL.Selection], snap, *, frame_idx: int,
                n_frames: int) -> None:
    import streamlit as st

    UI.rail_heading("Inspector")
    if sel is None or sel.is_empty:
        st.info("Click anywhere on the map.\n\n"
                "Clicking a **cell** opens every stored layer for it. "
                "Clicking a **tracked object** opens the track, with a link "
                "down to the cell beneath it.")
        if sel is not None and sel.is_empty:
            st.caption(
                f"Last click at ({sel.world_xy[0]:.2f}, "
                f"{sel.world_xy[1]:.2f}) m found no cell — the sensor never "
                f"observed that location on this frame. Absence of a cell is "
                f"not free space.")
        return

    if sel.kind == "object" and sel.obj is not None:
        if INS.render_object(sel.obj, ego_xy=snap.ego_xy,
                             has_cell=sel.cell is not None):
            # "Inspect underlying cell" — same place, cell view.
            st.session_state[_SEL_KEY] = SEL.Selection(
                "cell", sel.world_xy, frame_idx, cell=sel.cell)
            _refresh()
        if sel.cell is None:
            st.caption(
                f"No mapped surface within "
                f"{SEL.nearest_cell.__defaults__[0]:.1f} m of this "
                f"centroid. For a MOVING track that is the MOS gate doing "
                f"its job — moving points are held out of the persistent "
                f"map. For a stationary one it means the area really was "
                f"not observed this frame.")
        elif sel.cell_offset_m > 0:
            st.caption(
                f"The cell shown is the nearest mapped surface, "
                f"{sel.cell_offset_m:.2f} m from the centroid. A centroid "
                f"is a centre of mass and a LiDAR only sees surfaces, so "
                f"the middle of a car is usually its own occlusion "
                f"shadow — there is genuinely no cell exactly there.")
        return

    if sel.cell is not None:
        INS.render_cell(sel.cell, snap.cells, frame_idx=frame_idx,
                        n_frames=n_frames, scene_time=snap.timestamp,
                        world_xy=sel.world_xy)


# ════════════════════════════════════════════════════════════
# The tab
# ════════════════════════════════════════════════════════════
def render_live_demo(precompute_fn: Callable[..., Any],
                     n_frames_default: int = DEFAULT_FRAMES) -> None:
    """Draw the Live Demo tab. ``precompute_fn(scenario, n_frames, gate)``
    is expected to be cached by the caller — it is the only expensive call
    in this tab, and it must not be reachable from a frame advance."""
    import streamlit as st

    _state({
        "drive_scenario": "mixed_urban",
        "drive_frame_ref": "World",
        "drive_camera": "top-follow",
        "drive_layer": "resolution level",
        "drive_edges": True,
        "drive_age_tint": True,
        "drive_corridor": False,
        "drive_objects": True,
        "drive_gate": True,
        "drive_predict": False,
        "drive_overlay": True,
        "drive_zoom": 1.0,
        "drive_ppm": 7,
        "drive_click_objects": True,
        _SEL_KEY: None,
    })

    interval = None
    pb0 = st.session_state.get(SESSION.PLAYBACK_STATE_KEY)
    if pb0 is not None and getattr(pb0, "playing", False):
        # Never schedule faster than this tab can actually redraw. A
        # `run_every` shorter than the render queues reruns behind each
        # other, and the browser then receives payloads out of order — the
        # frame counter appears to walk backwards even though the state
        # machine is strictly monotonic. The frame is derived from the
        # clock, so a slower timer skips frames rather than slowing the
        # scene down.
        if SESSION.owns_ticker("live"):
            interval = SESSION.paced_interval("live", pb0.interval_s(),
                                          floor=MIN_REDRAW_S)
    PROFILE.log("arm", f"live interval={interval} "
                       f"playing={getattr(pb0, 'playing', None)}")

    if PlaybackController.supports_fragments():
        frag = st.fragment(run_every=interval)(_workspace)
        frag(precompute_fn, n_frames_default)
    else:
        # Older Streamlit: everything works except auto-advance.
        st.caption("Installed Streamlit has no fragment support; playback "
                   "steps manually. Everything else is unaffected.")
        _workspace(precompute_fn, n_frames_default)


def _workspace(precompute_fn: Callable[..., Any],
               n_frames_default: int) -> None:
    import streamlit as st
    from adaptive_lidar.data.synthetic_scene import SCENARIOS
    import time as _t
    _t0 = _t.perf_counter()

    left, centre, right = UI.workspace()

    with left:
        with st.container(height=RAIL_H, border=False):
            opt = _left_rail(SCENARIOS, n_frames_default)

    # ── load: the ONLY expensive call, and it is cached ──────
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

    pb = PlaybackController.bind(n, key="live",
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
    if SESSION.owns_ticker("live"):
        pb.tick_if_playing()
    idx = min(pb.state.frame_idx, n - 1)
    snap = run.frames[idx]

    # A selection made on an earlier frame is re-resolved against this one,
    # so the inspector shows what the map knows HERE NOW rather than stale
    # numbers from whenever the click happened.
    sel: Optional[SEL.Selection] = st.session_state.get(_SEL_KEY)
    if sel is not None and sel.frame_idx != idx:
        sel = SEL.reselect_on_frame(sel, snap, idx)
        st.session_state[_SEL_KEY] = sel

    # ── centre: the scene ────────────────────────────────────
    with centre:
        UI.disclaimer(UI.DEMO_DISCLAIMER)
        result = IM.render_canvas(
            snap, key=_CANVAS_KEY, layer=opt["layer"],
            preset=opt["camera"], frame_of_reference=opt["frame_ref"],
            zoom=opt["zoom"],
            show_edges=st.session_state["drive_edges"],
            age_tint=st.session_state["drive_age_tint"],
            show_objects=st.session_state["drive_objects"],
            show_overlay=st.session_state["drive_overlay"],
            show_prediction=st.session_state["drive_predict"],
            corridor=run.corridor if st.session_state["drive_corridor"]
            else None,
            corridor_upto=idx if st.session_state["drive_corridor"] else None,
            selected=sel, px_per_m=int(st.session_state["drive_ppm"]))

        if result.click is not None:
            new_sel = SEL.select_at(
                result.click[0], result.click[1], result.transform, snap, idx,
                allow_objects=bool(st.session_state["drive_click_objects"]))
            st.session_state[_SEL_KEY] = new_sel
            # Redraw immediately so the highlight appears on the click that
            # produced it rather than one interaction later.
            _refresh()

        st.markdown(R.legend_html(opt["layer"]), unsafe_allow_html=True)
        if st.session_state["drive_objects"]:
            st.markdown(OV.legend_states_html(), unsafe_allow_html=True)
        st.caption(
            "Click the map to inspect. Priority is tracked object → map "
            "cell → empty space, and the finest cell covering the point "
            "wins.")

        _metrics_strip(run, snap, result, idx)

    # ── right: inspector ─────────────────────────────────────
    with right:
        with st.container(height=RAIL_H, border=False):
            _right_rail(sel, snap, frame_idx=idx, n_frames=n)

    # ── bottom: the transport, full width ────────────────────
    # Outside the three columns deliberately: inside the centre column the
    # four buttons are narrow enough that their labels truncate to "◀…".
    st.markdown("")
    pb.render_transport(scene_time_s=snap.timestamp,
                        driving=SESSION.owns_ticker("live"))
    sp1, _sp2 = st.columns([1, 5])
    with sp1:
        pb.speed_selector()

    with st.expander("Scene diagnostics — MOS ablation, static sharpness, "
                     "swept corridor, tracked objects", expanded=False):
        scene_diagnostics(run, run_on, run_off, snap, idx,
                          result.corridor_area, result.drawn_objects)

    _took = _t.perf_counter() - _t0
    SESSION.record_redraw("live", _took)
    PROFILE.log("frag", f"{_took * 1000:7.1f} ms  frame {idx}  "
                        f"pace {SESSION.redraw_cost('live'):.3f}")


def _metrics_strip(run, snap, result, idx: int) -> None:
    """The numbers, on one line, below the scene rather than above it."""
    cells = snap.cells
    n_cells = int(len(cells.get("cx", ())))
    fine = int((np.asarray(cells["level"]) <= 1).sum()) if n_cells else 0
    movers = sum(1 for o in (snap.objects or []) if o["state_name"] == "MOVING")
    tel = snap.telemetry or {}
    items = [
        ("cells in view", f"{n_cells:,}"),
        ("at 5–10 cm", f"{100 * fine / max(n_cells, 1):.0f}%"),
        ("points this frame", f"{snap.n_points:,}"),
        ("tracked objects", f"{len(snap.objects or [])}"),
        ("moving", f"{movers}"),
    ]
    if "map_bytes" in tel:
        items.append(("map memory", f"{tel['map_bytes'] / 1e6:.2f} MB"))
    if result.corridor_area:
        items.append(("corridor swept", f"{result.corridor_area:,.0f} m²"))
    UI.metrics_strip(items)
    UI.disclaimer()
