"""
The inverse projection, checked against the thing it is the inverse OF.

A transform that is wrong in one step almost always still round-trips through
its own inverse — both halves make the same mistake and it cancels. So these
tests do two separate things:

  1. round-trip world -> screen -> world, which catches algebra slips;
  2. compare the FORWARD half against ``overlays.make_projector``, which is
     the projector the renderer actually draws the pixels with.

Only the second one can catch "the whole transform is self-consistently
wrong", which is the failure that silently inspects the neighbouring cell.
"""
from __future__ import annotations

import numpy as np
import pytest

from adaptive_lidar.visualization import camera as CAM
from adaptive_lidar.visualization import overlays as OV
from adaptive_lidar.visualization.coordinate_transform import (
    ViewTransform, round_trip_error)

PRESETS = list(CAM.CAMERA_PRESETS)
FRAMES = list(CAM.FRAME_CHOICES)

EGO = (18.5, -4.25)
HEADING = 0.7853981633974483          # 45 deg, so sin and cos differ in sign


def _transform(preset: str, frame: str, zoom: float = 1.0) -> ViewTransform:
    view = CAM.compute_view(preset, frame, EGO, HEADING, zoom=zoom)
    view = CAM.frame_view(view, frame, EGO, HEADING)
    x0, x1, y0, y1 = view.extent
    W = max(int((x1 - x0) * view.px_per_m), 16)
    H = max(int((y1 - y0) * view.px_per_m), 16)
    return ViewTransform.from_view(view, frame, EGO, HEADING, (H, W))


def _sample_world(t: ViewTransform, n: int = 200) -> np.ndarray:
    """World points that land inside the image, whatever the camera is."""
    rng = np.random.default_rng(7)
    px = rng.uniform(0, t.width_px, n)
    py = rng.uniform(0, t.height_px, n)
    return np.array([t.screen_to_world(a, b) for a, b in zip(px, py)])


# ════════════════════════════════════════════════════════════
@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("frame", FRAMES)
def test_round_trip_subpixel(preset, frame):
    """world -> screen -> world returns the same point, everywhere."""
    t = _transform(preset, frame)
    err = round_trip_error(t, _sample_world(t))
    # A pixel is 1/px_per_m metres; the transform should be exact to float,
    # so anything above a hundredth of a pixel is an algebra bug.
    assert err < 0.01 * t.metres_per_pixel(), f"{preset}/{frame}: {err} m"


@pytest.mark.parametrize("zoom", [0.5, 1.0, 2.0, 4.0])
def test_round_trip_survives_zoom(zoom):
    t = _transform("chase", "World", zoom=zoom)
    assert round_trip_error(t, _sample_world(t)) < 0.01 * t.metres_per_pixel()


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("frame", FRAMES)
def test_forward_matches_the_renderers_own_projector(preset, frame):
    """The forward half must agree with ``overlays.make_projector``.

    That projector is what draws the object boxes and the ego, so if the two
    disagree the highlight lands somewhere the user did not click.
    """
    view = CAM.compute_view(preset, frame, EGO, HEADING)
    view = CAM.frame_view(view, frame, EGO, HEADING)
    t = _transform(preset, frame)
    project = OV.make_projector(view, frame, EGO, HEADING)

    world = _sample_world(t, 64)
    px, py = project(world)
    for (wx, wy), ex, ey in zip(world, px, py):
        sx, sy = t.world_to_screen(wx, wy)
        # make_projector returns IMAGE pixels; undo the display flip.
        ix, iy = t.display_to_image_px(sx, sy)
        assert abs(ix - ex) <= 1.0 and abs(iy - ey) <= 1.0, (
            f"{preset}/{frame}: transform {ix:.2f},{iy:.2f} "
            f"vs projector {ex},{ey}")


def test_display_flip_is_accounted_for():
    """A click at the TOP of the displayed image is the FAR side of the map.

    ``st.image(np.flipud(img))`` is easy to forget, and forgetting it mirrors
    every selection about the horizontal axis — which looks plausible.
    """
    t = _transform("world-fixed", "World")
    _x_top, y_top = t.screen_to_world(t.width_px / 2, 0.0)
    _x_bot, y_bot = t.screen_to_world(t.width_px / 2, t.height_px - 1)
    assert y_top > y_bot, "display flip not inverted: +y must be UP on screen"


def test_flip_can_be_turned_off():
    t = _transform("world-fixed", "World")
    unflipped = ViewTransform(**{**t.__dict__, "displayed_flipped": False})
    _x, y_top = unflipped.screen_to_world(0.0, 0.0)
    _x, y_bot = unflipped.screen_to_world(0.0, unflipped.height_px - 1)
    assert y_top < y_bot


def test_vehicle_frame_puts_the_ego_at_the_view_centre():
    """In the vehicle frame a following camera is centred on the ego."""
    t = _transform("top-follow", "Vehicle")
    sx, sy = t.world_to_screen(*EGO)
    assert abs(sx - t.width_px / 2) < 2.0
    assert abs(sy - t.height_px / 2) < 2.0


def test_world_and_vehicle_frames_agree_on_the_same_world_point():
    """Two cameras, one world point: the inverse must return it in both."""
    target = (EGO[0] + 11.0, EGO[1] - 3.0)
    for preset in PRESETS:
        for frame in FRAMES:
            t = _transform(preset, frame)
            sx, sy = t.world_to_screen(*target)
            bx, by = t.screen_to_world(sx, sy)
            assert np.hypot(bx - target[0], by - target[1]) < 1e-3
