"""
Cameras, frames of reference, and the things that get drawn on top.

FRAME OF REFERENCE IS A CAMERA CHOICE, NOT A STORAGE CHOICE
-----------------------------------------------------------
The map is world-anchored in both frames.  Nothing here touches storage; the
frame selector changes which transform is applied on the way to pixels.

WORLD is the default, and that is deliberate.  In the vehicle frame the entire
static scene slides past, which a first-time viewer reads as "everything is
moving" — the exact opposite of the point.  The vehicle frame earns its place
once the concept has landed, and for the `convoy` scenario, where the illusion
it creates is the thing being demonstrated.

WHAT THE PRESETS MEAN IN A TOP-DOWN VIEW
----------------------------------------
This repository renders a top-down raster; there is no 3D scene view here, so
the five presets are realised as the top-down cameras they correspond to:

  top-follow   centred on the vehicle, world-north up      — exactly itself
  chase        centred on the vehicle, vehicle heading up  — a rotated follow
  world-fixed  a fixed window; the vehicle drives through  — exactly itself
  sensor       the range image, which IS the sensor's own view
  free orbit   a manually panned and zoomed window

Only `sensor` is a reinterpretation rather than a projection of the same
scene, and it is labelled as such on screen.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

CAMERA_PRESETS = ("top-follow", "chase", "world-fixed", "sensor", "free orbit")

FRAME_CHOICES = ("World", "Vehicle")

FRAME_CAPTION = {
    "World": ("Buildings and poles hold still. The car and other traffic "
              "move. **This is what the map actually stores.**"),
    "Vehicle": ("What the sensor sees. The static world appears to slide "
                "past because the vehicle is the thing that is moving."),
}

PRESET_CAPTION = {
    "top-follow": ("Straight down, centred on the vehicle, world-north fixed. "
                   "Watch the fine-resolution region travel with the car."),
    "chase": ("Centred on the vehicle with its heading pointing up — the "
              "intuitive driving view."),
    "world-fixed": ("The camera does not move; the vehicle drives through "
                    "frame. Best for seeing that static geometry does not "
                    "smear as it accumulates."),
    "sensor": ("The range image — the sensor's own view, one pixel per beam. "
               "This is the projection every geometric feature is computed "
               "on."),
    "free orbit": ("Manual pan and zoom. Nothing follows anything."),
}


@dataclass
class View:
    """A window on the world, plus an optional rotation about its centre."""
    extent: Tuple[float, float, float, float]      # x0, x1, y0, y1
    rotation: float = 0.0                           # radians, about the centre
    px_per_m: float = 7.0
    centre: Tuple[float, float] = (0.0, 0.0)
    label: str = ""

    @property
    def span(self) -> Tuple[float, float]:
        x0, x1, y0, y1 = self.extent
        return x1 - x0, y1 - y0


def _window(centre, half_x, half_y, rotation=0.0, px_per_m=7.0, label=""):
    cx, cy = centre
    return View((cx - half_x, cx + half_x, cy - half_y, cy + half_y),
                rotation, px_per_m, (cx, cy), label)


def compute_view(
    preset: str,
    frame_of_reference: str,
    ego_xy: Tuple[float, float],
    heading: float,
    *,
    half_x: float = 60.0,
    half_y: float = 32.0,
    px_per_m: float = 7.0,
    fixed_centre: Tuple[float, float] = (30.0, 0.0),
    pan: Tuple[float, float] = (0.0, 0.0),
    zoom: float = 1.0,
) -> View:
    """The window and rotation for one preset, in WORLD coordinates.

    The frame of reference decides whether the window tracks the vehicle.
    Storage is world-anchored regardless; this only moves the camera.
    """
    hx, hy = half_x / max(zoom, 1e-3), half_y / max(zoom, 1e-3)

    if preset == "world-fixed":
        return _window(fixed_centre, hx, hy, 0.0, px_per_m, "world-fixed")

    if preset == "free orbit":
        return _window((fixed_centre[0] + pan[0], fixed_centre[1] + pan[1]),
                       hx, hy, 0.0, px_per_m, "free orbit")

    if preset == "chase":
        # Vehicle heading points up; the window sits slightly ahead of the car
        # so most of the view is the road in front rather than behind.
        ahead = 0.35 * hx
        c = (ego_xy[0] + ahead * np.cos(heading),
             ego_xy[1] + ahead * np.sin(heading))
        return _window(c, hx, hy, -heading, px_per_m, "chase")

    # top-follow (and the fallback for `sensor`, which renders separately)
    # World-north stays up: the vehicle moves within a stable frame, which is
    # what makes the travelling resolution island legible.
    return _window(ego_xy, hx, hy, 0.0, px_per_m, "top-follow")


def interpolate(a: View, b: View, t: float) -> View:
    """Ease between two views. A hard cut mid-playback disorients the viewer.

    ``t`` in [0, 1]; smoothstep rather than linear so the motion starts and
    stops gently.
    """
    t = float(np.clip(t, 0.0, 1.0))
    s = t * t * (3.0 - 2.0 * t)

    def lerp(u, v):
        return u + (v - u) * s

    # Rotation takes the short way round.
    d = (b.rotation - a.rotation + np.pi) % (2 * np.pi) - np.pi
    return View(
        tuple(lerp(np.array(a.extent), np.array(b.extent)).tolist()),
        a.rotation + d * s,
        lerp(a.px_per_m, b.px_per_m),
        tuple(lerp(np.array(a.centre), np.array(b.centre)).tolist()),
        b.label,
    )


# ════════════════════════════════════════════════════════════
# Frame-of-reference transform
# ════════════════════════════════════════════════════════════
def to_frame(xy: np.ndarray, frame_of_reference: str,
             ego_xy: Tuple[float, float], heading: float) -> np.ndarray:
    """Map world xy into the chosen frame of reference.

    World: identity. Vehicle: translate and rotate so the ego sits at the
    origin pointing along +x — which is exactly the illusion the `convoy`
    scenario exists to expose.
    """
    xy = np.asarray(xy, np.float32)
    if frame_of_reference != "Vehicle" or len(xy) == 0:
        return xy
    d = xy - np.asarray(ego_xy, np.float32)
    c, s = np.cos(-heading), np.sin(-heading)
    return np.stack([d[:, 0] * c - d[:, 1] * s,
                     d[:, 0] * s + d[:, 1] * c], axis=1).astype(np.float32)


def frame_view(view: View, frame_of_reference: str,
               ego_xy: Tuple[float, float], heading: float) -> View:
    """The same window, expressed in the chosen frame of reference."""
    if frame_of_reference != "Vehicle":
        return view
    hx = (view.extent[1] - view.extent[0]) / 2.0
    hy = (view.extent[3] - view.extent[2]) / 2.0
    c = to_frame(np.array([view.centre], np.float32),
                 "Vehicle", ego_xy, heading)[0]
    # In the vehicle frame the ego is at the origin, so a following camera is
    # simply centred there and needs no rotation of its own.
    rot = 0.0 if view.label in ("top-follow", "chase") else view.rotation
    centre = (0.0, 0.0) if view.label in ("top-follow", "chase") else tuple(c)
    return View((centre[0] - hx, centre[0] + hx, centre[1] - hy, centre[1] + hy),
                rot, view.px_per_m, centre, view.label)
