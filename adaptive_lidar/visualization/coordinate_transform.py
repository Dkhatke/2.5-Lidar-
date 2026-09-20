"""
Screen <-> world coordinate transforms.

The forward path (world -> pixel) lives in ``overlays.make_projector`` and is
what the renderer draws through.  This module is its exact inverse, and it has
to be exact: a click that lands one cell to the left silently inspects the
wrong cell, and nothing on screen would reveal it.

THE CHAIN, AND WHY EACH STEP IS THERE
-------------------------------------
    screen pixel (as displayed)
      |  the image is shown flipped vertically, because the renderer draws
      |  with +y upward and the browser draws with +y downward
      v
    image pixel
      |  divide by px_per_m, add the view origin
      v
    view coordinates
      |  undo the camera rotation about the view centre (the chase camera
      |  rotates the world so the vehicle points up)
      v
    frame coordinates
      |  if the view frame is Vehicle, undo the ego translation + heading
      v
    WORLD x, y
      |
      v
    the adaptive cell containing that point

Getting any one of those backwards produces a plausible-looking answer, which
is why :func:`round_trip_error` exists and the tests use it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class ViewTransform:
    """Everything needed to map between the screen and the world.

    This is deliberately a plain value object rather than a reference to the
    live camera: a click arrives on the *next* Streamlit run, by which time
    the camera may have moved. Capturing the transform that actually produced
    the pixels the user clicked keeps the answer correct.
    """

    extent: Tuple[float, float, float, float]   # x0, x1, y0, y1 (view coords)
    px_per_m: float
    rotation: float                              # radians, about `centre`
    centre: Tuple[float, float]
    frame_of_reference: str                      # "World" | "Vehicle"
    ego_xy: Tuple[float, float]
    heading: float
    image_hw: Tuple[int, int]                    # (H, W) of the rendered image
    displayed_flipped: bool = True               # st.image gets np.flipud(img)

    # ── sizes ────────────────────────────────────────────────
    @property
    def width_px(self) -> int:
        return int(self.image_hw[1])

    @property
    def height_px(self) -> int:
        return int(self.image_hw[0])

    @classmethod
    def from_view(cls, view, frame_of_reference: str, ego_xy, heading,
                  image_hw, displayed_flipped: bool = True) -> "ViewTransform":
        return cls(
            extent=tuple(float(v) for v in view.extent),
            px_per_m=float(view.px_per_m),
            rotation=float(view.rotation),
            centre=(float(view.centre[0]), float(view.centre[1])),
            frame_of_reference=str(frame_of_reference),
            ego_xy=(float(ego_xy[0]), float(ego_xy[1])),
            heading=float(heading),
            image_hw=(int(image_hw[0]), int(image_hw[1])),
            displayed_flipped=bool(displayed_flipped),
        )

    # ── display flip ─────────────────────────────────────────
    def display_to_image_px(self, px: float, py: float) -> Tuple[float, float]:
        """Undo the vertical flip applied on the way to ``st.image``."""
        if not self.displayed_flipped:
            return float(px), float(py)
        return float(px), float(self.height_px - 1 - py)

    def image_to_display_px(self, px: float, py: float) -> Tuple[float, float]:
        return self.display_to_image_px(px, py)     # the flip is its own inverse

    # ── the inverse chain ────────────────────────────────────
    def screen_to_world(self, display_x: float, display_y: float
                        ) -> Tuple[float, float]:
        """A pixel the user clicked -> the world point under it."""
        ix, iy = self.display_to_image_px(display_x, display_y)

        # image pixel -> view coordinates
        x0, _x1, y0, _y1 = self.extent
        vx = x0 + ix / self.px_per_m
        vy = y0 + iy / self.px_per_m

        # undo the camera rotation about the view centre
        if abs(self.rotation) > 1e-9:
            cx, cy = self.centre
            dx, dy = vx - cx, vy - cy
            c, s = np.cos(-self.rotation), np.sin(-self.rotation)
            vx = cx + dx * c - dy * s
            vy = cy + dx * s + dy * c

        # undo the frame of reference
        if self.frame_of_reference == "Vehicle":
            # forward was: d = xy - ego; q = R(-heading) @ d
            # so:          xy = ego + R(+heading) @ q
            c, s = np.cos(self.heading), np.sin(self.heading)
            wx = self.ego_xy[0] + vx * c - vy * s
            wy = self.ego_xy[1] + vx * s + vy * c
            return float(wx), float(wy)
        return float(vx), float(vy)

    # ── the forward chain, kept here so the two cannot drift ──
    def world_to_screen(self, world_x: float, world_y: float
                        ) -> Tuple[float, float]:
        """The inverse of :meth:`screen_to_world`, for drawing highlights."""
        vx, vy = float(world_x), float(world_y)

        if self.frame_of_reference == "Vehicle":
            dx = vx - self.ego_xy[0]
            dy = vy - self.ego_xy[1]
            c, s = np.cos(-self.heading), np.sin(-self.heading)
            vx, vy = dx * c - dy * s, dx * s + dy * c

        if abs(self.rotation) > 1e-9:
            cx, cy = self.centre
            dx, dy = vx - cx, vy - cy
            c, s = np.cos(self.rotation), np.sin(self.rotation)
            vx = cx + dx * c - dy * s
            vy = cy + dx * s + dy * c

        x0, _x1, y0, _y1 = self.extent
        ix = (vx - x0) * self.px_per_m
        iy = (vy - y0) * self.px_per_m
        return self.image_to_display_px(ix, iy)

    # ── misc ─────────────────────────────────────────────────
    def contains_display_px(self, px: float, py: float) -> bool:
        return 0 <= px < self.width_px and 0 <= py < self.height_px

    def metres_per_pixel(self) -> float:
        return 1.0 / max(self.px_per_m, 1e-9)


def round_trip_error(t: ViewTransform, world_xy) -> float:
    """Max |error| in metres of world -> screen -> world.

    Used by the tests. A transform that is wrong in one step usually still
    round-trips through its own inverse, so the tests ALSO check the forward
    path against ``overlays.make_projector`` — the renderer's own projector —
    which is the thing the pixels actually came from.
    """
    worst = 0.0
    for wx, wy in np.atleast_2d(world_xy):
        sx, sy = t.world_to_screen(float(wx), float(wy))
        bx, by = t.screen_to_world(sx, sy)
        worst = max(worst, float(np.hypot(bx - wx, by - wy)))
    return worst
