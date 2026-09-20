"""
Scene presets and the scene diagnostics panel.

This module used to own the Drive tab, including a playback loop built from
``time.sleep`` + ``st.rerun``. That loop is gone: it blocked the server
thread for the length of every frame, re-executed the whole script on each
advance, and dropped any click that arrived mid-sleep. Playback now lives in
``playback_controller.py`` and the tab itself in ``live_demo.py``, which runs
the workspace inside an ``st.fragment``.

What stays here is what was always independent of that loop and is still
used by the new tab:

  * ``PRESETS`` — one click, a full intended demonstration state;
  * ``scene_diagnostics`` — the MOS ablation, the static-sharpness proof,
    the swept corridor and the tracked-object table.

Everything still reads from the playback cache. Changing the camera, the
frame of reference, the layer or the displayed frame never re-runs the
pipeline — which is also why the replay rate on screen is labelled a replay
rate and not FPS.
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from adaptive_lidar.pipeline.types import CLASS_NAMES



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
            "drive_edges": False,
            "drive_age_tint": False,
            "drive_corridor": True,
            "drive_objects": False,
        },
    },
}


# ════════════════════════════════════════════════════════════
def scene_diagnostics(run, run_on, run_off, snap, idx,
                      corridor_area, drawn):
    """The four readouts that make the demo checkable.

    Kept out of the main hierarchy — they answer questions a
    reviewer asks second, not the ones the scene answers first.
    """
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
