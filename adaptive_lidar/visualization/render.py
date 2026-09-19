"""
Map rendering for the dashboard.

Turns the map's per-level cell arrays into an image, preserving the one thing
that matters most about this map: **cells have different physical sizes, and
you must be able to see that.**  A naive scatter plot of cell centres hides it
entirely — every cell becomes the same dot — so cells are painted as filled
rectangles at their true extent, coarse first so fine cells land on top.

COLOUR CODING (M5 asks for "distinct colour coding for terrain and objects")
---------------------------------------------------------------------------
Terrain is cool and desaturated; objects are warm and saturated; VRU is a
single unmissable colour used for nothing else.  The palette is checked for
deuteranopia/protanopia separation — the terrain/object split survives as a
temperature difference even when hue is lost.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from adaptive_lidar.pipeline.types import (
    CLASS_NAMES,
    Occupancy,
    ResolutionLevel,
    Traversability,
)

# ── palettes ────────────────────────────────────────────────
#: Terrain cool + desaturated, objects warm + saturated, VRU unmissable.
CLASS_COLOURS = {
    0: "#5B7DB1",   # ground_drivable  — cool slate blue
    1: "#7FA05A",   # ground_rough     — muted olive
    2: "#9A8C82",   # static_obstacle  — warm grey
    3: "#E8A33D",   # vehicle          — amber
    4: "#E8112D",   # vru              — one colour, used for nothing else
    5: "#2E8B57",   # vegetation       — sea green
}
CLASS_RGB = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5))
             for k, v in CLASS_COLOURS.items()}

LEVEL_COLOURS = {
    0: "#7F1D1D",   # 5 cm  — finest
    1: "#C2410C",
    2: "#D97706",
    3: "#65A30D",
    4: "#0E7490",   # 80 cm — coarsest
}
LEVEL_RGB = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5))
             for k, v in LEVEL_COLOURS.items()}

TRAV_COLOURS = {
    Traversability.DRIVABLE: "#2E7D32",
    Traversability.CAUTION: "#F9A825",
    Traversability.BLOCKED: "#C62828",
}
TRAV_RGB = {k: tuple(int(v[i:i + 2], 16) for i in (1, 3, 5))
            for k, v in TRAV_COLOURS.items()}

OCC_RGB = {
    Occupancy.FREE: (232, 236, 240),
    Occupancy.OCCUPIED: (70, 70, 78),
    Occupancy.UNKNOWN: (170, 172, 178),
}

LAYERS = ("height", "semantic", "traversability", "occupancy",
          "dynamic", "uncertainty", "resolution level", "intensity")


def _ramp(v: np.ndarray, lo: float, hi: float, stops) -> np.ndarray:
    """Interpolate a value array through a list of RGB stops."""
    t = np.clip((v - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    stops = np.asarray(stops, np.float32)
    n = len(stops) - 1
    idx = np.clip((t * n).astype(np.int32), 0, n - 1)
    frac = (t * n - idx)[:, None]
    return (stops[idx] * (1 - frac) + stops[idx + 1] * frac).astype(np.uint8)


VIRIDIS = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98),
           (253, 231, 37)]
TERRAIN = [(48, 62, 94), (70, 116, 136), (139, 168, 130), (214, 201, 155),
           (246, 245, 240)]
MAGMA = [(0, 0, 4), (81, 18, 124), (183, 55, 121), (252, 137, 97),
         (252, 253, 191)]


#: A cell that holds no points exists only because free-space carving put it
#: there. It is a statement about emptiness, not a surface, and painting it
#: like one fills the view with a solid block of "ground at 0 m" that hides
#: the actual map.
FREE_RGB = (238, 240, 244)


def cell_colours(a: Dict[str, np.ndarray], layer: str,
                 trav: Optional[np.ndarray] = None) -> np.ndarray:
    """(M, 3) uint8 colour per cell for the chosen layer."""
    n = len(a["cx"])
    if n == 0:
        return np.zeros((0, 3), np.uint8)
    empty = a["n_points"] == 0

    if layer == "semantic":
        out = np.zeros((n, 3), np.uint8)
        for c, rgb in CLASS_RGB.items():
            out[a["sem_class"] == c] = rgb
        out[empty] = FREE_RGB
        return out

    if layer == "resolution level":
        # The one layer where carved cells SHOULD be shown: their size is the
        # point, and hiding them would misrepresent what the map holds.
        out = np.zeros((n, 3), np.uint8)
        for l, rgb in LEVEL_RGB.items():
            out[a["level"] == l] = rgb
        return out

    if layer == "traversability":
        out = np.full((n, 3), 200, np.uint8)
        if trav is not None:
            for v, rgb in TRAV_RGB.items():
                out[trav == v] = rgb
        out[empty] = FREE_RGB
        return out

    if layer == "occupancy":
        out = np.zeros((n, 3), np.uint8)
        for v, rgb in OCC_RGB.items():
            out[a["occupancy_state"] == v] = rgb
        return out

    if layer == "dynamic":
        out = _ramp(a["dynamic_prob"], 0.0, 1.0,
                    [(235, 235, 235), (230, 140, 40), (200, 20, 40)])
        out[empty] = FREE_RGB
        return out

    if layer == "uncertainty":
        out = _ramp(a["entropy"], 0.0, 1.0, MAGMA)
        out[empty] = FREE_RGB
        return out

    if layer == "intensity":
        out = _ramp(a["intensity_mean"], 0.0, 0.8, VIRIDIS)
        out[empty] = FREE_RGB
        return out

    # height (default): shade by obstacle height above local ground
    h = np.clip(a["z_max"] - a["ground_z"], 0.0, 3.0)
    out = _ramp(h, 0.0, 3.0, TERRAIN)
    out[empty] = FREE_RGB
    return out


def render(
    a: Dict[str, np.ndarray],
    layer: str = "height",
    extent: Tuple[float, float, float, float] = (-20, 100, -30, 30),
    px_per_m: float = 6.0,
    show_cell_edges: bool = False,
    trav: Optional[np.ndarray] = None,
    background: Tuple[int, int, int] = (250, 250, 252),
) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Paint the map to an RGB image.

    Cells are drawn as FILLED RECTANGLES AT THEIR TRUE SIZE, coarse first so
    fine cells overwrite them.  That ordering plus the optional edge overlay
    is what makes the variable resolution visible rather than merely claimed.
    """
    x0, x1, y0, y1 = extent
    W = max(int((x1 - x0) * px_per_m), 16)
    H = max(int((y1 - y0) * px_per_m), 16)
    img = np.empty((H, W, 3), np.uint8)
    img[:, :] = background
    if len(a["cx"]) == 0:
        return img, extent

    cols = cell_colours(a, layer, trav)
    size = a["size"]
    cx, cy = a["cx"], a["cy"]

    # Coarse to fine: a fine cell must be able to overwrite the coarse cell
    # that contains it, otherwise the refinement is invisible.
    for lvl in range(ResolutionLevel.N_LEVELS - 1, -1, -1):
        m = a["level"] == lvl
        if not m.any():
            continue
        s = float(ResolutionLevel.size(lvl))
        # Pixel span of one cell at this level, at least one pixel.
        sp = max(int(round(s * px_per_m)), 1)
        px = ((cx[m] - s / 2 - x0) * px_per_m).astype(np.int32)
        py = ((cy[m] - s / 2 - y0) * px_per_m).astype(np.int32)
        c = cols[m]
        keep = (px > -sp) & (px < W) & (py > -sp) & (py < H)
        px, py, c = px[keep], py[keep], c[keep]
        if sp == 1:
            gx = np.clip(px, 0, W - 1)
            gy = np.clip(py, 0, H - 1)
            img[gy, gx] = c
            continue
        # Stamp the block by writing each of its pixel offsets.
        for dy in range(sp):
            gy = py + dy
            ok = (gy >= 0) & (gy < H)
            if not ok.any():
                continue
            for dx in range(sp):
                gx = px + dx
                o = ok & (gx >= 0) & (gx < W)
                if o.any():
                    img[gy[o], gx[o]] = c[o]
        if show_cell_edges and sp >= 3:
            edge = (np.asarray(c, np.int16) * 0.55).astype(np.uint8)
            for dy in (0, sp - 1):
                gy = py + dy
                o = (gy >= 0) & (gy < H)
                for dx in range(sp):
                    gx = px + dx
                    oo = o & (gx >= 0) & (gx < W)
                    if oo.any():
                        img[gy[oo], gx[oo]] = edge[oo]
            for dx in (0, sp - 1):
                gx = px + dx
                o = (gx >= 0) & (gx < W)
                for dy in range(sp):
                    gy = py + dy
                    oo = o & (gy >= 0) & (gy < H)
                    if oo.any():
                        img[gy[oo], gx[oo]] = edge[oo]

    return img, extent


def overlay_points(img, pts, extent, px_per_m, rgb=(230, 20, 60), radius=1):
    """Stamp the dynamic overlay (moving objects) on top of the map."""
    if len(pts) == 0:
        return img
    x0, x1, y0, y1 = extent
    H, W = img.shape[:2]
    px = ((pts[:, 0] - x0) * px_per_m).astype(np.int32)
    py = ((pts[:, 1] - y0) * px_per_m).astype(np.int32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            gx, gy = px + dx, py + dy
            ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
            if ok.any():
                img[gy[ok], gx[ok]] = rgb
    return img


def pixel_to_world(ix, iy, extent, px_per_m):
    x0, x1, y0, y1 = extent
    return x0 + ix / px_per_m, y0 + iy / px_per_m


def legend_html(layer: str) -> str:
    """Always-visible legend for the active layer."""
    def swatch(colour, label):
        return (f'<span style="display:inline-flex;align-items:center;'
                f'margin-right:14px;margin-bottom:3px;font-size:12px">'
                f'<span style="width:13px;height:13px;background:{colour};'
                f'border:1px solid rgba(0,0,0,.25);border-radius:2px;'
                f'margin-right:5px"></span>{label}</span>')

    if layer == "semantic":
        terrain = "".join(swatch(CLASS_COLOURS[c], CLASS_NAMES[c]) for c in (0, 1))
        objects = "".join(swatch(CLASS_COLOURS[c], CLASS_NAMES[c])
                          for c in (2, 3, 4, 5))
        return (f'<div><b style="font-size:11px">TERRAIN</b> (cool) '
                f'{terrain}</div>'
                f'<div><b style="font-size:11px">OBJECTS</b> (warm) '
                f'{objects}</div>')
    if layer == "resolution level":
        return "".join(swatch(LEVEL_COLOURS[l], ResolutionLevel.name(l))
                       for l in range(5))
    if layer == "traversability":
        return "".join(swatch(TRAV_COLOURS[v], Traversability.NAMES[v])
                       for v in (0, 1, 2))
    if layer == "occupancy":
        return "".join(swatch("#%02x%02x%02x" % OCC_RGB[v], Occupancy.NAMES[v])
                       for v in (0, 1, 2))
    if layer == "height":
        return ("<span style='font-size:12px'>obstacle height above local "
                "ground: dark 0 m &rarr; light 3 m &nbsp;·&nbsp; pale grey = "
                "free space carved by a beam, no surface</span>")
    if layer == "uncertainty":
        return ("<span style='font-size:12px'>semantic entropy: "
                "black 0 (certain) &rarr; pale 1 (uncertain)</span>")
    if layer == "dynamic":
        return ("<span style='font-size:12px'>dynamic probability: "
                "grey 0 &rarr; red 1</span>")
    return ("<span style='font-size:12px'>corrected intensity "
            "(reflectance): dark 0 &rarr; yellow 0.8</span>")


# ════════════════════════════════════════════════════════════
# Playback rendering: a frame of reference and a rotated camera
# ════════════════════════════════════════════════════════════
def render_view(
    cells: Dict[str, np.ndarray],
    view,
    layer: str = "semantic",
    *,
    frame_of_reference: str = "World",
    ego_xy=(0.0, 0.0),
    heading: float = 0.0,
    show_cell_edges: bool = False,
    trav: Optional[np.ndarray] = None,
    age_tint_frame: Optional[int] = None,
    age_window: int = 3,
    background=(250, 250, 252),
) -> np.ndarray:
    """Paint a cached frame's cells through a camera.

    Same rules as :func:`render` — cells are filled rectangles at their TRUE
    size, painted coarse-first so fine cells overwrite them — with two
    additions the playback needs: the view may be rotated (the chase camera),
    and the whole scene may be expressed in the vehicle's frame.

    Neither changes what is stored. The map is world-anchored in both frames;
    this is a transform on the way to pixels.
    """
    from adaptive_lidar.visualization.camera import to_frame
    from adaptive_lidar.visualization.overlays import apply_age_tint

    x0, x1, y0, y1 = view.extent
    ppm = view.px_per_m
    W = max(int((x1 - x0) * ppm), 16)
    H = max(int((y1 - y0) * ppm), 16)
    img = np.empty((H, W, 3), np.uint8)
    img[:, :] = background
    if cells is None or len(cells.get("cx", ())) == 0:
        return img

    from adaptive_lidar.visualization.playback import decode_cells
    a = decode_cells(cells)
    colours = cell_colours(a, layer, trav)
    if age_tint_frame is not None and "last_seen" in a:
        colours = apply_age_tint(colours, a["last_seen"], age_tint_frame,
                                 age_window)

    xy = np.stack([a["cx"], a["cy"]], axis=1).astype(np.float32)
    xy = to_frame(xy, frame_of_reference, ego_xy, heading)
    rot = view.rotation
    if abs(rot) > 1e-6:
        c, sn = np.cos(rot), np.sin(rot)
        d = xy - np.array(view.centre, np.float32)
        xy = np.stack([d[:, 0] * c - d[:, 1] * sn,
                       d[:, 0] * sn + d[:, 1] * c], axis=1) \
            + np.array(view.centre, np.float32)

    lvl = a["level"]
    # Coarse to fine, so a refinement is never hidden by its own parent.
    for L in range(ResolutionLevel.N_LEVELS - 1, -1, -1):
        m = lvl == L
        if not m.any():
            continue
        s_m = float(ResolutionLevel.size(L))
        sp = max(int(round(s_m * ppm)), 1)
        px = ((xy[m, 0] - s_m / 2 - x0) * ppm).astype(np.int32)
        py = ((xy[m, 1] - s_m / 2 - y0) * ppm).astype(np.int32)
        c = colours[m]
        keep = (px > -sp) & (px < W) & (py > -sp) & (py < H)
        px, py, c = px[keep], py[keep], c[keep]
        if px.size == 0:
            continue
        if sp == 1:
            img[np.clip(py, 0, H - 1), np.clip(px, 0, W - 1)] = c
            continue
        for dy in range(sp):
            gy = py + dy
            ok = (gy >= 0) & (gy < H)
            if not ok.any():
                continue
            for dx in range(sp):
                gx = px + dx
                o = ok & (gx >= 0) & (gx < W)
                if o.any():
                    img[gy[o], gx[o]] = c[o]
        if show_cell_edges and sp >= 3:
            edge = (np.asarray(c, np.int16) * 0.55).astype(np.uint8)
            for dy in (0, sp - 1):
                gy = py + dy
                o = (gy >= 0) & (gy < H)
                for dx in range(sp):
                    gx = px + dx
                    oo = o & (gx >= 0) & (gx < W)
                    if oo.any():
                        img[gy[oo], gx[oo]] = edge[oo]
            for dx in (0, sp - 1):
                gx = px + dx
                o = (gx >= 0) & (gx < W)
                for dy in range(sp):
                    gy = py + dy
                    oo = o & (gy >= 0) & (gy < H)
                    if oo.any():
                        img[gy[oo], gx[oo]] = edge[oo]
    return img
