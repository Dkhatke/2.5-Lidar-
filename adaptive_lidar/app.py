"""
app.py — Streamlit dashboard for SIH26 Adaptive Foveated 2.5D LiDAR Perception.

Run: streamlit run app.py

Tabs:
  1. Raw LiDAR
  2. Range Image
  3. Geometry
  4. Adaptive Allocation   ← KEY DEMO
  5. Semantic Perception
  6. Motion
  7. 2.5D Map              ← KEY DEMO
  8. Telemetry
"""
from __future__ import annotations
import sys
import os
import time
import numpy as np
import streamlit as st

# ── Make package importable from app.py location ────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="SIH26 — Adaptive LiDAR",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark theme CSS injection ──────────────────────────────────
st.markdown("""
<style>
body, .stApp { background-color: #0d1117; color: #e6edf3; }
.metric-container { background: #161b22; border-radius: 8px; padding: 10px; }
.stTabs [data-baseweb="tab"] { background: #161b22; border-radius: 6px 6px 0 0; }
.stTabs [aria-selected="true"] { background: #1f6feb; }
div[data-testid="stMetricValue"] { font-size: 1.6rem; color: #58a6ff; }
div[data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.75rem; }
.safety-banner {
    background: linear-gradient(90deg,#b45309,#92400e);
    color:#fef3c7; padding:8px 16px;
    border-radius:6px; font-weight:bold;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Session state helpers
# ════════════════════════════════════════════════════════════
def _init_state():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None
    if "frames" not in st.session_state:
        st.session_state.frames = []
    if "current_frame_idx" not in st.session_state:
        st.session_state.current_frame_idx = 0
    if "loaded" not in st.session_state:
        st.session_state.loaded = False


_init_state()


# ════════════════════════════════════════════════════════════
# Load config
# ════════════════════════════════════════════════════════════
@st.cache_resource
def get_config():
    from adaptive_lidar.utils.config import load_config
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    return load_config(cfg_path)


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("🔭 SIH26 Controls")
    st.markdown("---")

    data_mode = st.radio(
        "Input source",
        ["Synthetic demo", "SemanticKITTI sequence", "Single .bin/.npy file"],
        index=0,
    )

    kitti_path = ""
    single_path = ""
    if data_mode == "SemanticKITTI sequence":
        kitti_path = st.text_input("Sequence directory",
                                   "dataset/sequences/00")
    elif data_mode == "Single .bin/.npy file":
        single_path = st.text_input("File path", "")

    n_frames = st.slider("Frames to load", 2, 50, 8)

    st.markdown("---")
    backend_choice = st.selectbox(
        "Semantic backend",
        ["auto", "prototype", "geometry", "minkowski"],
        index=0,
    )

    budget_pct = st.slider(
        "⚡ Computation budget (%)", 10, 100, 80, step=5,
        help="Lower = fewer tiles processed at high resolution. Safety pins always protected.")

    st.markdown("---")
    max_range = st.slider("Max LiDAR range (m)", 20, 80, 60)

    st.markdown("---")
    show_raw = st.checkbox("Show raw LiDAR", True)
    show_alloc = st.checkbox("Show allocation map", True)
    show_sem = st.checkbox("Show semantics", True)
    show_map = st.checkbox("Show 2.5D map", True)

    st.markdown("---")
    load_btn = st.button("▶  Load & Process", type="primary", use_container_width=True)


# ════════════════════════════════════════════════════════════
# LOAD + PROCESS
# ════════════════════════════════════════════════════════════
if load_btn:
    config = get_config()
    config.setdefault("sensor", {})["max_range"] = float(max_range)

    from adaptive_lidar.pipeline.pipeline import Pipeline
    from adaptive_lidar.data.loader import get_input_source

    with st.spinner("Building pipeline and processing frames…"):
        pipe = Pipeline(config)
        pipe.build_stages(backend=backend_choice)
        pipe.set_budget(budget_pct / 100.0)

        # Choose data source
        if data_mode == "Synthetic demo":
            source = get_input_source(max_frames=n_frames,
                                      num_synthetic_frames=n_frames)
        elif data_mode == "SemanticKITTI sequence":
            source = get_input_source(input_path=kitti_path,
                                      max_frames=n_frames)
        else:
            source = get_input_source(input_path=single_path or None,
                                      max_frames=n_frames)

        frames = []
        prog = st.progress(0.0, text="Processing…")
        items = list(source)
        for step_i, (cloud, labels, fid, ts) in enumerate(items):
            frame = pipe.run(cloud_np=cloud, frame_id=fid, timestamp=ts)
            frames.append(frame)
            prog.progress((step_i + 1) / len(items),
                          text=f"Frame {fid} — {len(cloud):,} points")

        prog.empty()
        st.session_state.pipeline = pipe
        st.session_state.frames = frames
        st.session_state.current_frame_idx = len(frames) - 1
        st.session_state.loaded = True
        st.session_state.backend_name = pipe.context.semantic_backend_name

    st.success(f"✅ Loaded {len(frames)} frames.")


# ════════════════════════════════════════════════════════════
# HEADER
# ════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='text-align:center;
   background:linear-gradient(90deg,#1f6feb,#388bfd);
   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
   font-size:2rem;margin-bottom:0'>
   🔭 Adaptive Foveated 2.5D LiDAR Perception
</h1>
<p style='text-align:center;color:#8b949e;margin-top:4px'>
SIH26 — Intelligent computation allocation for autonomous perception
</p>
""", unsafe_allow_html=True)

if not st.session_state.loaded:
    st.info("👈  Configure options in the sidebar, then click **▶ Load & Process**.")
    st.markdown("""
    ### What this prototype demonstrates
    ```
    ENTIRE LiDAR SCENE
          │
          ▼
    CHEAP GLOBAL ANALYSIS      (S1 + S2)
          │
          ▼
    INFORMATION VALUE SCORE    (S3)
          │
          ▼
    ADAPTIVE BUDGET CONTROL
     COARSE ── MEDIUM ── FINE
          │
          ▼
    SPARSE PERCEPTION          (S4) ← only selected tiles
          │
     ┌────┴────┐
     ▼         ▼
    SEMANTIC  MOTION           (S4 + S5)
          │
          ▼
    UNCERTAINTY FEEDBACK       (S6)
          │
          ▼
    ADAPTIVE 2.5D MAP          (S7 + S8)
    ```
    """)
    st.stop()


# ════════════════════════════════════════════════════════════
# FRAME SELECTOR
# ════════════════════════════════════════════════════════════
frames = st.session_state.frames
pipe = st.session_state.pipeline

fi = st.slider("Frame", 0, len(frames) - 1,
               st.session_state.current_frame_idx, key="frame_sel")
frame = frames[fi]
telem = frame.timing.get("telemetry", {})


# ════════════════════════════════════════════════════════════
# TELEMETRY ROW
# ════════════════════════════════════════════════════════════
n_tiles = telem.get("total_tiles", 0)
n_hi = telem.get("high_res_tiles", 0)
n_med = telem.get("medium_tiles", 0)
n_low = telem.get("low_res_tiles", 0)
n_safety = telem.get("safety_pinned", 0)
total_ms = telem.get("total_latency_ms", 0.0)
n_pts = telem.get("point_count", 0)
backend_name = telem.get("backend", "—")

pct_hi = 100 * n_hi / max(n_tiles, 1)
pct_med = 100 * n_med / max(n_tiles, 1)
pct_low = 100 * n_low / max(n_tiles, 1)

cols = st.columns(8)
cols[0].metric("Points", f"{n_pts:,}")
cols[1].metric("Latency", f"{total_ms:.0f} ms")
cols[2].metric("🔴 HIGH", f"{n_hi} ({pct_hi:.0f}%)")
cols[3].metric("🟡 MED", f"{n_med} ({pct_med:.0f}%)")
cols[4].metric("🔵 LOW", f"{n_low} ({pct_low:.0f}%)")
cols[5].metric("⚠ Safety pins", n_safety)
cols[6].metric("Map cells", f"{len(pipe.map_cells):,}")
cols[7].metric("Frame", f"{fi}/{len(frames)-1}")

# Safety pin banner
if n_safety > 0 and budget_pct < 60:
    st.markdown(
        f'<div class="safety-banner">⚠️ SAFETY PIN ACTIVE — '
        f'{n_safety} tile(s) protected at HIGH resolution despite {budget_pct}% budget</div>',
        unsafe_allow_html=True,
    )

# Backend badge
st.markdown(
    f'<div style="background:#161b22;border-radius:6px;padding:4px 12px;'
    f'display:inline-block;margin:4px 0;font-size:0.8rem;color:#58a6ff">'
    f'🧠 Semantic backend: <b>{backend_name}</b></div>',
    unsafe_allow_html=True,
)

st.markdown("---")

# ════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════
from adaptive_lidar.visualization.dashboard import (
    pointcloud_figure, semantic_pointcloud_figure,
    range_image_figure, allocation_heatmap, map_figure, telemetry_bar,
    CLASS_NAMES,
)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "1 Raw LiDAR", "2 Range Image", "3 Geometry",
    "4 🎯 Adaptive Allocation", "5 Semantics",
    "6 Motion", "7 🗺️ 2.5D Map", "8 Telemetry",
])


# ── Tab 1: Raw LiDAR ─────────────────────────────────────────
with tab1:
    if show_raw and frame.points is not None:
        fig = pointcloud_figure(
            frame.points,
            color_arr=frame.intensity,
            colorscale="Plasma",
            title=f"Raw LiDAR — Frame {frame.frame_id} ({len(frame.points):,} pts)",
        )
        st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    col1.metric("Points after filter", f"{len(frame.points):,}")
    col2.metric("S0 latency", f"{frame.timing.get('S0',0):.1f} ms")


# ── Tab 2: Range Image ────────────────────────────────────────
with tab2:
    if frame.range_image is not None:
        fig = range_image_figure(frame.range_image)
        st.plotly_chart(fig, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.metric("Range image size",
                  f"{frame.range_image.shape[0]} × {frame.range_image.shape[1]}")
        c2.metric("S1 latency", f"{frame.timing.get('S1',0):.1f} ms")
    else:
        st.warning("Range image not available.")


# ── Tab 3: Geometry ───────────────────────────────────────────
with tab3:
    st.markdown("**Ground / Non-ground separation**  "
                "*(Prototype: height-grid method — future: Patchwork++)*")
    if frame.ground_mask is not None:
        n_gnd = int(frame.ground_mask.sum())
        n_ng = int((~frame.ground_mask).sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Ground pts", f"{n_gnd:,}")
        c2.metric("Non-ground pts", f"{n_ng:,}")
        c3.metric("S2 latency", f"{frame.timing.get('S2',0):.1f} ms")

        # Colour points by ground/non-ground
        color_arr = frame.ground_mask.astype(float)
        fig = pointcloud_figure(
            frame.points, color_arr=color_arr,
            colorscale="RdYlGn",
            title="Ground (green) vs Non-ground (red)",
        )
        st.plotly_chart(fig, use_container_width=True)

    if frame.tiles:
        st.markdown("**Tile geometry features**")
        import pandas as pd
        df = pd.DataFrame([{
            "tile_id": t.tile_id,
            "cx": round(t.cx, 1), "cy": round(t.cy, 1),
            "pts": t.point_count,
            "density": round(t.density, 2),
            "height_var": round(t.height_variance, 3),
            "verticality": round(t.verticality, 2),
            "roughness": round(t.roughness, 3),
            "boundary": round(t.boundary_score, 2),
        } for t in frame.tiles if t.point_count > 0])
        st.dataframe(df, use_container_width=True, height=300)


# ── Tab 4: Adaptive Allocation ────────────────────────────────
with tab4:
    if show_alloc and frame.tiles:
        st.markdown(
            "**Adaptive computation allocation** — RED = HIGH resolution (expensive), "
            "BLUE = LOW (cheap). ⭐ = Safety pin (protected at any budget)."
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 HIGH", f"{n_hi} ({pct_hi:.0f}%)")
        c2.metric("🟡 MEDIUM", f"{n_med} ({pct_med:.0f}%)")
        c3.metric("🔵 LOW", f"{n_low} ({pct_low:.0f}%)")
        c4.metric("⚠ Safety", n_safety)

        st.markdown(
            f"*Budget: **{budget_pct}%** — "
            f"estimated sparse inference on **{pct_hi + pct_med:.0f}%** of tiles*"
        )

        fig = allocation_heatmap(frame.tiles)
        st.plotly_chart(fig, use_container_width=True)

        # Explain panel — top 5 tiles by info value
        st.markdown("---\n**🔍 Top tiles by information value**")
        top_tiles = sorted(frame.tiles, key=lambda t: t.info_value, reverse=True)[:8]
        for tile in top_tiles:
            with st.expander(
                f"Tile {tile.tile_id} ({tile.cx:.0f},{tile.cy:.0f}) — "
                f"V={tile.info_value:.3f} → {['5cm','10cm','20cm','40cm','80cm'][tile.resolution_level]}"
                f" {'⚠ SAFETY PIN' if tile.safety_pinned else ''}",
                expanded=False,
            ):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Geometry G", f"{tile.score_geometry:.3f}")
                c2.metric("Semantic S", f"{tile.score_semantic:.3f}")
                c3.metric("Uncertainty U", f"{tile.score_uncertainty:.3f}")
                c4.metric("Dynamic D", f"{tile.score_dynamic:.3f}")
                st.caption(
                    f"Priority = V/cost = {tile.priority():.3f} | "
                    f"Points: {tile.point_count} | "
                    f"Range: {tile.range_mean:.1f} m"
                )
                if tile.safety_pinned:
                    st.warning(f"⚠ Safety reason: {tile.safety_reason}")
    else:
        st.info("No allocation data available.")


# ── Tab 5: Semantics ──────────────────────────────────────────
with tab5:
    if show_sem and frame.tiles:
        sem_labels = np.zeros(len(frame.points), dtype=np.int32)
        for tile in frame.tiles:
            if tile.semantic_class >= 0 and len(tile.point_indices) > 0:
                sem_labels[tile.point_indices] = tile.semantic_class

        fig = semantic_pointcloud_figure(
            frame.points, sem_labels,
            title="Semantic Perception (selected tiles only)",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Per-class summary
        import pandas as pd
        class_counts = {cls: int((sem_labels == i).sum())
                        for i, cls in enumerate(CLASS_NAMES)}
        df = pd.DataFrame(list(class_counts.items()), columns=["class", "points"])
        st.dataframe(df, use_container_width=True)

        avg_conf = np.mean([t.semantic_confidence for t in frame.tiles if t.selected])
        avg_unc = np.mean([t.semantic_uncertainty for t in frame.tiles if t.selected])
        c1, c2, c3 = st.columns(3)
        c1.metric("S4 latency", f"{frame.timing.get('S4',0):.1f} ms")
        c2.metric("Avg confidence", f"{avg_conf:.2f}")
        c3.metric("Avg uncertainty", f"{avg_unc:.2f}")

        st.caption(
            f"⚠ Backend: **{backend_name}** — "
            "only selected tiles were processed by the semantic stage."
        )


# ── Tab 6: Motion ─────────────────────────────────────────────
with tab6:
    if pipe.context.instance_history:
        inst_dict = pipe.context.instance_history[-1]
        if inst_dict:
            import pandas as pd
            rows = []
            for inst in inst_dict.values():
                speed = float(np.linalg.norm(inst.velocity)) if inst.velocity is not None else 0.0
                rows.append({
                    "id": inst.instance_id,
                    "class": CLASS_NAMES[inst.semantic_class] if inst.semantic_class < 6 else "?",
                    "cx": round(float(inst.centroid[0]), 1),
                    "cy": round(float(inst.centroid[1]), 1),
                    "cz": round(float(inst.centroid[2]), 1),
                    "pts": inst.point_count,
                    "speed (m/s)": round(speed, 2),
                    "motion_prob": round(inst.motion_probability, 2),
                    "dynamic": "🔴 YES" if inst.is_dynamic else "🟢 NO",
                })
            df = pd.DataFrame(rows)
            st.dataframe(df, use_container_width=True)

            n_dyn = sum(1 for i in inst_dict.values() if i.is_dynamic)
            c1, c2, c3 = st.columns(3)
            c1.metric("Instances", len(inst_dict))
            c2.metric("Dynamic", n_dyn)
            c3.metric("S5 latency", f"{frame.timing.get('S5',0):.1f} ms")

            st.caption("PROTOTYPE: motion = centroid displacement / Δt | Future: 4DMOS")
        else:
            st.info("No instances detected in this frame.")
    else:
        st.info("Process multiple frames to see motion tracking.")


# ── Tab 7: 2.5D Map ───────────────────────────────────────────
with tab7:
    if show_map and pipe.map_cells:
        map_mode = st.radio(
            "Map view mode",
            ["semantic", "elevation", "occupancy"],
            horizontal=True,
        )
        amap = getattr(pipe.context, "_amap", None)
        if amap:
            arrs = amap.all_cells_as_arrays()
            cx_a, cy_a, gz_a, zmx_a, occ_a, scl_a, scnf_a, unk_a, dyn_a = arrs
            if len(cx_a) > 0:
                fig = map_figure(
                    cx_a, cy_a, scl_a, occ_a, unk_a.astype(bool),
                    mode=map_mode,
                    z_arr=zmx_a,
                )
                st.plotly_chart(fig, use_container_width=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Map cells", len(cx_a))
                c2.metric("Unknown", f"{unk_a.mean()*100:.0f}%")
                c3.metric("Avg occupancy", f"{occ_a.mean():.2f}")
                c4.metric("S7 latency", f"{frame.timing.get('S7',0):.1f} ms")
                st.caption(
                    "IMPORTANT: Unknown cells ≠ Free space. "
                    "No LiDAR return does NOT mean the area is drivable."
                )
    else:
        st.info("Map data will appear after frames are processed.")


# ── Tab 8: Telemetry ──────────────────────────────────────────
with tab8:
    st.markdown("**Per-stage latency (current frame)**")
    fig = telemetry_bar(frame.timing)
    st.plotly_chart(fig, use_container_width=True)

    # Stage timing table across all frames
    import pandas as pd
    stages = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
    rows = []
    for s in stages:
        p50 = pipe.context.p50(s)
        p95 = pipe.context.p95(s)
        latest = frame.timing.get(s, 0.0)
        rows.append({"Stage": s, "Latest (ms)": round(latest, 2),
                     "P50 (ms)": round(p50, 2), "P95 (ms)": round(p95, 2)})
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    total_latest = sum(frame.timing.get(s, 0) for s in stages)
    st.metric(
        "Total pipeline latency",
        f"{total_latest:.1f} ms",
        help="Actual measured. TARGET: ~40 ms (future C++/CUDA implementation)."
    )
    st.caption(
        "⚠ TARGET: ~40 ms/frame is the **future** C++/CUDA goal. "
        "The current Python prototype measures actual latency without claims."
    )
