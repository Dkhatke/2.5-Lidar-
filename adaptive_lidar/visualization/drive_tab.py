"""
The Drive tab: ego motion, cameras and the dynamic scene.

Everything here reads from the playback cache.  Changing the camera, the
frame of reference, the layer or the displayed frame never re-runs the
pipeline — that is the rule the whole module is arranged around, and it is
also why the replay rate on screen is labelled a replay rate and not FPS.
"""
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import streamlit as st

from adaptive_lidar.pipeline.types import CLASS_NAMES, TrackState
from adaptive_lidar.visualization import camera as CAM
from adaptive_lidar.visualization import overlays as OV
from adaptive_lidar.visualization import render as R

SPEEDS = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "2x": 2.0}
TRANSITION_S = 0.5


# ════════════════════════════════════════════════════════════
# Scenario presets — one click, full intended state
# ════════════════════════════════════════════════════════════
PRESETS = {
    "foveation": {
        "label": "★ Foveation follows the vehicle",
        "help": ("Top-follow camera, resolution-level colours, world frame, "
                 "half speed. The fine region travels with the car."),
        "state": {
            "drive_scenario": "mixed_urban",
            "drive_frame_ref": "World",
            "drive_camera": "top-follow",
            "drive_layer": "resolution level",
            "drive_speed": "0.5x",
            "drive_edges": True,
            "drive_age_tint": True,
            "drive_corridor": False,
            "drive_objects": True,
        },
    },
    "relative": {
        "label": "Relative motion",
        "help": ("The convoy scenario in the VEHICLE frame, where the car "
                 "travelling at your speed looks parked and the parked one "
                 "looks like it is reversing. The object list shows what the "
                 "pipeline actually concluded."),
        "state": {
            "drive_scenario": "convoy",
            "drive_frame_ref": "Vehicle",
            "drive_camera": "chase",
            "drive_layer": "semantic",
            "drive_speed": "0.5x",
            "drive_edges": False,
            "drive_age_tint": False,
            "drive_corridor": False,
            "drive_objects": True,
        },
    },
    "corridor": {
        "label": "Swept corridor",
        "help": ("World-fixed camera with the breadcrumb overlay: every "
                 "location ever resolved finely over the whole route."),
        "state": {
            "drive_scenario": "mixed_urban",
            "drive_frame_ref": "World",
            "drive_camera": "world-fixed",
            "drive_layer": "resolution level",
            "drive_speed": "1x",
            "drive_edges": False,
            "drive_age_tint": False,
            "drive_corridor": True,
            "drive_objects": False,
        },
    },
}


def _apply_preset(key: str):
    for k, v in PRESETS[key]["state"].items():
        st.session_state[k] = v
    st.session_state["drive_frame_idx"] = 0
    st.session_state["drive_playing"] = True
    st.session_state["drive_cam_changed_at"] = time.time()


def _init_state(defaults: Dict[str, Any]):
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


# ════════════════════════════════════════════════════════════
def render_drive_tab(precompute_fn, n_frames_default: int = 12):
    """Draw the tab. ``precompute_fn(scenario, n_frames, gate)`` is cached."""
    _init_state({
        "drive_scenario": "mixed_urban",
        "drive_frame_ref": "World",
        "drive_camera": "top-follow",
        "drive_layer": "resolution level",
        "drive_speed": "0.5x",
        "drive_edges": True,
        "drive_age_tint": True,
        "drive_corridor": False,
        "drive_objects": True,
        "drive_gate": True,
        "drive_frame_idx": 0,
        "drive_playing": False,
        "drive_prev_camera": "top-follow",
        "drive_cam_changed_at": 0.0,
        "drive_zoom": 1.0,
        "drive_predict": False,
    })

    st.markdown("##### Drive the scene")
    st.caption(
        "The pipeline runs once per frame when a scenario loads. Everything "
        "below — camera, frame of reference, layer, scrubbing — reads from "
        "that cache, so none of it re-runs perception.")

    # ── one-click presets ────────────────────────────────────
    cols = st.columns(len(PRESETS))
    for col, (key, p) in zip(cols, PRESETS.items()):
        with col:
            if st.button(p["label"], help=p["help"], width='stretch'):
                _apply_preset(key)
                st.rerun()

    # ── controls ─────────────────────────────────────────────
    from adaptive_lidar.data.synthetic_scene import SCENARIOS
    c1, c2, c3, c4 = st.columns([1.1, 1.1, 1.1, 1.0])
    with c1:
        st.selectbox("Scenario", SCENARIOS, key="drive_scenario")
    with c2:
        st.radio("View frame", CAM.FRAME_CHOICES, key="drive_frame_ref",
                 horizontal=True)
    with c3:
        st.selectbox("Camera", CAM.CAMERA_PRESETS, key="drive_camera")
    with c4:
        n_frames = st.slider("Frames", 6, 24, n_frames_default,
                             key="drive_n_frames")

    st.caption(CAM.FRAME_CAPTION[st.session_state["drive_frame_ref"]])
    st.caption(f"**{st.session_state['drive_camera']}** — "
               f"{CAM.PRESET_CAPTION[st.session_state['drive_camera']]}")

    d1, d2, d3, d4, d5 = st.columns([1.2, 1, 1, 1, 1])
    with d1:
        st.selectbox("Layer", R.LAYERS, key="drive_layer")
    with d2:
        st.checkbox("Cell edges", key="drive_edges")
        st.checkbox("Age tint", key="drive_age_tint",
                    help="Brighten cells refined in the last few frames, so "
                         "the leading edge of the refinement front is "
                         "visible rather than merely inferable.")
    with d3:
        st.checkbox("Swept corridor", key="drive_corridor",
                    help="Every location ever resolved at 5 or 10 cm. Kept "
                         "in a separate accumulator, because the map's "
                         "sliding window has already evicted the cells "
                         "behind the vehicle.")
        st.checkbox("Objects", key="drive_objects")
    with d4:
        st.checkbox("MOS gate", key="drive_gate",
                    help="OFF writes moving points into the persistent map. "
                         "Both variants are pre-computed, so this is instant.")
        st.checkbox("Predicted path", key="drive_predict",
                    help="2 s constant-velocity extrapolation. Not a "
                         "prediction model.")
    with d5:
        st.selectbox("Replay speed", list(SPEEDS), key="drive_speed")
        st.slider("Zoom", 0.5, 2.5, 1.0, 0.1, key="drive_zoom")

    # ── load (pipeline runs here, once) ──────────────────────
    scenario = st.session_state["drive_scenario"]
    with st.spinner(f"Pre-computing {scenario} ({n_frames} frames, both MOS "
                    f"variants)..."):
        run_on = precompute_fn(scenario, n_frames, True)
        run_off = precompute_fn(scenario, n_frames, False)
    run = run_on if st.session_state["drive_gate"] else run_off
    n = len(run)
    if n == 0:
        st.warning("No frames.")
        return

    # ── transport ────────────────────────────────────────────
    t1, t2, t3, t4 = st.columns([0.6, 0.6, 3.2, 1.4])
    with t1:
        if st.button("▶ Play" if not st.session_state["drive_playing"]
                     else "⏸ Pause", width='stretch'):
            st.session_state["drive_playing"] = \
                not st.session_state["drive_playing"]
            st.rerun()
    with t2:
        if st.button("⏭ Step", width='stretch',
                     help="Advance one frame. For pointing at things."):
            st.session_state["drive_playing"] = False
            st.session_state["drive_frame_idx"] = \
                (st.session_state["drive_frame_idx"] + 1) % n
            st.rerun()
    with t3:
        idx = st.slider("Frame", 0, n - 1,
                        min(st.session_state["drive_frame_idx"], n - 1),
                        key="drive_scrub", label_visibility="collapsed")
        st.session_state["drive_frame_idx"] = idx
    snap = run.frames[idx]
    with t4:
        st.markdown(
            f'<div style="font-size:12px;line-height:1.35">'
            f'frame <b>{idx + 1}/{n}</b><br>'
            f'scene time <b>{snap.timestamp:.2f} s</b><br>'
            f'<span style="opacity:.62">replayed at '
            f'{st.session_state["drive_speed"]} — a replay rate, not a '
            f'latency measurement</span></div>', unsafe_allow_html=True)

    # ── camera, with a smooth transition between presets ─────
    preset = st.session_state["drive_camera"]
    if preset != st.session_state["drive_prev_camera"]:
        st.session_state["drive_cam_from"] = st.session_state["drive_prev_camera"]
        st.session_state["drive_cam_changed_at"] = time.time()
        st.session_state["drive_prev_camera"] = preset

    fr_ref = st.session_state["drive_frame_ref"]
    zoom = st.session_state["drive_zoom"]
    ego, head = snap.ego_xy, snap.heading

    def view_for(p):
        v = CAM.compute_view(p, fr_ref, ego, head, zoom=zoom)
        return CAM.frame_view(v, fr_ref, ego, head)

    view = view_for(preset)
    since = time.time() - st.session_state.get("drive_cam_changed_at", 0.0)
    prev = st.session_state.get("drive_cam_from")
    if prev and prev != preset and since < TRANSITION_S:
        # Ease rather than cut: a hard jump mid-playback disorients.
        view = CAM.interpolate(view_for(prev), view, since / TRANSITION_S)

    # ── sensor preset renders the range image instead ────────
    if preset == "sensor":
        st.info("The sensor's own view is the range image — one pixel per "
                "beam, which is the projection every geometric feature in "
                "this system is computed on. There is no 3D scene renderer "
                "in this repository, so this is the honest answer rather "
                "than a synthesised perspective.")
        st.caption("Range image is not cached per frame; showing the map "
                   "view instead. Switch camera to see the scene.")

    # ── draw ─────────────────────────────────────────────────
    img = R.render_view(
        snap.cells, view, st.session_state["drive_layer"],
        frame_of_reference=fr_ref, ego_xy=ego, heading=head,
        show_cell_edges=st.session_state["drive_edges"],
        age_tint_frame=(snap.frame_id if st.session_state["drive_age_tint"]
                        else None))

    project = OV.make_projector(view, fr_ref, ego, head)
    corridor_area = 0.0
    if st.session_state["drive_corridor"]:
        img, corridor_area = OV.draw_corridor(img, project, run.corridor, idx)

    drawn = []
    if st.session_state["drive_objects"]:
        img, drawn = OV.draw_objects(
            img, project, snap.objects,
            show_prediction=st.session_state["drive_predict"])
    if len(snap.overlay):
        img = OV.draw_overlay_points(img, project, snap.overlay)
    img = OV.draw_ego(img, project, ego, head)

    st.image(np.flipud(img), width='stretch')
    st.markdown(R.legend_html(st.session_state["drive_layer"]),
                unsafe_allow_html=True)
    if st.session_state["drive_objects"]:
        st.markdown(OV.legend_states_html(), unsafe_allow_html=True)

    # ── readouts ─────────────────────────────────────────────
    _readouts(run, run_on, run_off, snap, idx, corridor_area, drawn)

    # ── auto-advance ─────────────────────────────────────────
    if st.session_state["drive_playing"]:
        dt = 0.1 / SPEEDS[st.session_state["drive_speed"]]
        time.sleep(min(dt, 0.8))
        st.session_state["drive_frame_idx"] = (idx + 1) % n
        st.rerun()


# ════════════════════════════════════════════════════════════
def _readouts(run, run_on, run_off, snap, idx, corridor_area, drawn):
    a, b, c = st.columns(3)

    # E — the MOS ablation, both variants already computed
    with a:
        st.markdown("**MOS gate — our own ablation**")
        t_on = (run_on.trail or {}).get("trail_m")
        t_off = (run_off.trail or {}).get("trail_m")
        c_on = (run_on.trail or {}).get("cells", 0)
        c_off = (run_off.trail or {}).get("cells", 0)
        active = "ON" if st.session_state["drive_gate"] else "OFF"
        st.markdown(
            f'<div style="font-size:13px">'
            f'gate <b>{active}</b><br>'
            f'trail with gate ON&nbsp;: <b>'
            f'{"—" if t_on is None else f"{t_on:.2f} m"}</b> '
            f'<span style="opacity:.6">({c_on} cells)</span><br>'
            f'trail with gate OFF: <b>'
            f'{"—" if t_off is None else f"{t_off:.2f} m"}</b> '
            f'<span style="opacity:.6">({c_off} cells)</span>'
            f'</div>', unsafe_allow_html=True)
        st.caption("Gating at the input is what removes trails; decay alone "
                   "does not. Measured by the same code as "
                   "`scripts/test_temporal.py`.")

    # G — the correctness proof
    with b:
        st.markdown("**Static sharpness**")
        w = run.wall
        if w and w.get("ok"):
            driven = abs(run.frames[-1].ego_xy[0] - run.frames[0].ego_xy[0])
            st.markdown(
                f'<div style="font-size:13px">'
                f'wall bulk <b>{w["thickness_cells_iqr"]:.2f} cells</b> '
                f'({w["iqr_m"] * 100:.0f} cm at {w["median_cell_m"] * 100:.0f} cm)'
                f'<br>over {w["n_cells"]:,} cells, {driven:.0f} m driven'
                f'<br><span style="opacity:.6">5-95 spread '
                f'{w["spread_m"] * 100:.0f} cm — the tail is facade seen at '
                f'grazing incidence</span></div>', unsafe_allow_html=True)
            st.caption("If the pose transform were wrong, accumulating this "
                       "wall over the whole run would smear it across "
                       "several cells. It does not.")
        else:
            st.caption(f"No wall probe for this scenario"
                       f"{'' if not w else ': ' + w.get('reason', '')}.")

    # D — the corridor
    with c:
        st.markdown("**Swept corridor**")
        if run.corridor is None:
            st.caption("No corridor accumulated.")
        else:
            total = run.corridor.area_upto(len(run) - 1)
            now = run.corridor.area_upto(idx)
            st.markdown(
                f'<div style="font-size:13px">'
                f'resolved finely by this frame: <b>{now:,.0f} m²</b><br>'
                f'over the whole route: <b>{total:,.0f} m²</b></div>',
                unsafe_allow_html=True)
            st.caption("Total detail spent over a route, not per frame. Held "
                       "separately from the map, which has already evicted "
                       "the cells behind the vehicle.")

    # F — the three states, surfaced distinctly
    st.markdown("**Tracked objects — world-frame velocity, not relative**")
    rows = sorted(drawn or snap.objects, key=lambda o: -o["points"])[:12]
    if not rows:
        st.caption("No tracked objects in this frame.")
        return
    try:
        import pandas as pd
        ego_v = np.array([10.0, 0.0, 0.0])      # the scenarios' ego speed
        df = pd.DataFrame([{
            "id": o["id"],
            "class": CLASS_NAMES[o["cls"]],
            "state": o["state_name"],
            "speed world (m/s)": round(o["speed_world"], 2),
            "speed relative (m/s)": round(
                float(np.linalg.norm(o["vel_world"] - ego_v)), 2),
            "points": o["points"],
            "x": round(float(o["centroid_world"][0]), 1),
            "y": round(float(o["centroid_world"][1]), 1),
        } for o in rows])
        st.dataframe(df, hide_index=True, width='stretch')
    except Exception:
        st.write(rows[:6])
    st.caption(
        "In the vehicle frame a car matching your speed looks parked and a "
        "parked car looks like it is reversing. The pipeline transforms to "
        "world coordinates before tracking, so it reports what is actually "
        "moving — which is why MOVABLE_BUT_STATIONARY is its own state and "
        "not folded into either neighbour.")
