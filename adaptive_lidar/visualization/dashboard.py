"""
Plotly visualisation helpers — point cloud, range image,
allocation heatmap, 2.5D map, telemetry.

All functions return Plotly figure objects so they can be
embedded directly into Streamlit with st.plotly_chart().
"""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Optional, List

from adaptive_lidar.pipeline.types import Tile, ResolutionLevel

# ── Colour maps ──────────────────────────────────────────────
CLASS_COLORS = {
    0: "#4CAF50",   # ground_drivable — green
    1: "#8BC34A",   # ground_rough — light green
    2: "#607D8B",   # static_obstacle — blue-grey
    3: "#FF5722",   # vehicle — deep orange
    4: "#E91E63",   # vru — pink
    5: "#2E7D32",   # vegetation — dark green
    -1: "#9E9E9E",  # unknown — grey
}

CLASS_NAMES = [
    "ground_drivable", "ground_rough",
    "static_obstacle", "vehicle", "vru", "vegetation",
]

RES_COLORS = {
    0: "#F44336",   # HIGH — red
    1: "#FF9800",   # — orange
    2: "#FFC107",   # MEDIUM — amber
    3: "#8BC34A",   # — light green
    4: "#2196F3",   # LOW — blue
}

RES_LABELS = {0: "HIGH (5cm)", 1: "10cm", 2: "MEDIUM (20cm)",
              3: "40cm", 4: "LOW (80cm)"}


# ── Point cloud ───────────────────────────────────────────────
def pointcloud_figure(
    points: np.ndarray,
    color_arr: Optional[np.ndarray] = None,
    colorscale: str = "Viridis",
    title: str = "Raw LiDAR",
    max_pts: int = 15000,
) -> go.Figure:
    """3-D scatter of point cloud, coloured by height or custom array."""
    if len(points) > max_pts:
        idx = np.random.choice(len(points), max_pts, replace=False)
        points = points[idx]
        if color_arr is not None:
            color_arr = color_arr[idx]

    color = color_arr if color_arr is not None else points[:, 2]

    fig = go.Figure(go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="markers",
        marker=dict(size=1.2, color=color,
                    colorscale=colorscale, opacity=0.75),
    ))
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data",
            bgcolor="#0d1117",
        ),
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
    )
    return fig


def semantic_pointcloud_figure(
    points: np.ndarray,
    semantic_labels: np.ndarray,
    title: str = "Semantic Perception",
    max_pts: int = 15000,
) -> go.Figure:
    """Point cloud coloured by semantic class."""
    if len(points) > max_pts:
        idx = np.random.choice(len(points), max_pts, replace=False)
        points = points[idx]
        semantic_labels = semantic_labels[idx]

    traces = []
    for cls_id, color in CLASS_COLORS.items():
        if cls_id < 0:
            continue
        mask = semantic_labels == cls_id
        if not mask.any():
            continue
        pts = points[mask]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=1.5, color=color, opacity=0.85),
            name=CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else "unknown",
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m)", yaxis_title="Y (m)", zaxis_title="Z (m)",
            aspectmode="data", bgcolor="#0d1117",
        ),
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        margin=dict(l=0, r=0, t=30, b=0),
        height=400,
    )
    return fig


# ── Range image ───────────────────────────────────────────────
def range_image_figure(range_image: np.ndarray) -> go.Figure:
    ri = np.where(range_image > 0, range_image, np.nan)
    fig = px.imshow(
        ri,
        color_continuous_scale="plasma",
        labels=dict(color="Range (m)"),
        title="Range Image (ring × azimuth)",
        aspect="auto",
    )
    fig.update_layout(
        paper_bgcolor="#0d1117",
        font_color="#e6edf3",
        margin=dict(l=0, r=0, t=40, b=0),
        height=200,
    )
    return fig


# ── Allocation heatmap ────────────────────────────────────────
def allocation_heatmap(tiles: List[Tile]) -> go.Figure:
    """
    Top-down tile grid showing resolution levels.
    HIGH = red, MEDIUM = amber, LOW = blue, safety pins outlined.
    """
    if not tiles:
        return go.Figure()

    xs = [t.cx for t in tiles]
    ys = [t.cy for t in tiles]
    colors = [RES_COLORS.get(t.resolution_level, "#9E9E9E") for t in tiles]
    labels = [RES_LABELS.get(t.resolution_level, "?") for t in tiles]
    scores = [f"{t.info_value:.2f}" for t in tiles]
    safety = ["⚠ SAFETY PIN" if t.safety_pinned else "" for t in tiles]

    hover = [
        f"Tile {t.tile_id}<br>"
        f"Resolution: {RES_LABELS.get(t.resolution_level,'?')}<br>"
        f"V={t.info_value:.2f} (G={t.score_geometry:.2f} S={t.score_semantic:.2f} "
        f"U={t.score_uncertainty:.2f} D={t.score_dynamic:.2f})<br>"
        f"Points: {t.point_count}<br>"
        f"{'⚠ SAFETY PIN: ' + t.safety_reason if t.safety_pinned else ''}"
        for t in tiles
    ]

    fig = go.Figure(go.Scatter(
        x=xs, y=ys,
        mode="markers",
        marker=dict(
            size=12,
            color=colors,
            symbol=["star" if t.safety_pinned else "square" for t in tiles],
            line=dict(
                color=["#FFD700" if t.safety_pinned else "rgba(0,0,0,0)" for t in tiles],
                width=[2 if t.safety_pinned else 0 for t in tiles],
            ),
        ),
        text=hover,
        hoverinfo="text",
        name="Tiles",
    ))

    # Legend annotation
    for res_id, col in RES_COLORS.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=10, color=col, symbol="square"),
            name=RES_LABELS.get(res_id, "?"),
        ))

    fig.update_layout(
        title="Adaptive Computation Allocation",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=420,
    )
    return fig


# ── 2.5D Map ──────────────────────────────────────────────────
def map_figure(
    cx_arr: np.ndarray, cy_arr: np.ndarray,
    sem_class: np.ndarray,
    occupancy: np.ndarray,
    is_unknown: np.ndarray,
    mode: str = "semantic",  # "semantic" | "elevation" | "occupancy"
    z_arr: Optional[np.ndarray] = None,
) -> go.Figure:
    """Top-down adaptive 2.5D map."""
    if len(cx_arr) == 0:
        fig = go.Figure()
        fig.update_layout(title="2.5D Map (no data yet)",
                          paper_bgcolor="#0d1117", font_color="#e6edf3")
        return fig

    if mode == "semantic":
        color = [CLASS_COLORS.get(int(c), "#9E9E9E") for c in sem_class]
    elif mode == "elevation" and z_arr is not None:
        color = z_arr
    else:
        color = occupancy

    hover = [
        f"({cx_arr[i]:.1f}, {cy_arr[i]:.1f})<br>"
        f"Class: {CLASS_NAMES[int(sem_class[i])] if 0<=int(sem_class[i])<6 else 'unknown'}<br>"
        f"Occupancy: {occupancy[i]:.2f}<br>"
        f"{'UNKNOWN' if is_unknown[i] else ''}"
        for i in range(len(cx_arr))
    ]

    fig = go.Figure(go.Scatter(
        x=cx_arr, y=cy_arr,
        mode="markers",
        marker=dict(
            size=5,
            color=color,
            colorscale="Viridis" if mode != "semantic" else None,
            opacity=0.85,
        ),
        text=hover,
        hoverinfo="text",
    ))

    fig.update_layout(
        title=f"Adaptive 2.5D Map — {mode}",
        xaxis_title="X (m)", yaxis_title="Y (m)",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        margin=dict(l=0, r=0, t=40, b=0),
        height=420,
        yaxis=dict(scaleanchor="x"),
    )
    return fig


# ── Telemetry bar chart ────────────────────────────────────────
def telemetry_bar(timing: dict) -> go.Figure:
    stages = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]
    vals = [timing.get(s, 0.0) for s in stages]

    fig = go.Figure(go.Bar(
        x=stages, y=vals,
        marker_color=["#00BCD4"] * len(stages),
        text=[f"{v:.1f}" for v in vals],
        textposition="outside",
    ))
    fig.update_layout(
        title="Per-Stage Latency (ms)",
        xaxis_title="Stage", yaxis_title="ms",
        paper_bgcolor="#0d1117",
        plot_bgcolor="#161b22",
        font_color="#e6edf3",
        margin=dict(l=0, r=0, t=40, b=0),
        height=280,
    )
    return fig
