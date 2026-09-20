"""
The transport, and the one property that matters most about it.

Playback must never re-enter the LiDAR pipeline. The old implementation did
not guarantee that: it advanced with ``time.sleep`` plus a whole-app
``st.rerun``, so every displayed frame re-executed the entire script and only
``st.cache_resource`` stood between the animation and perception running
again. A cache miss — a changed scenario, a cold start, an eviction — would
have run the pipeline once per frame.

So the state machine here has no Streamlit in it at all, and
``test_ticking_never_touches_the_pipeline`` hands it a pipeline that raises.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

from adaptive_lidar.visualization.playback_controller import (
    BASE_FRAME_S, MIN_INTERVAL_S, SPEEDS, PlaybackController, PlaybackState)

VIS = pathlib.Path(__file__).resolve().parents[1] / "visualization"


# ════════════════════════════════════════════════════════════
# Frame stepping
# ════════════════════════════════════════════════════════════
def test_next_and_previous_move_one_frame():
    s = PlaybackState(10, frame_idx=4)
    s.next()
    assert s.frame_idx == 5
    s.previous()
    assert s.frame_idx == 4


def test_stepping_pauses_but_ticking_does_not():
    """Stepping is for pointing at something; a timer tick is not."""
    s = PlaybackState(10, playing=True)
    s.tick()
    assert s.playing and s.frame_idx == 1
    s.next()
    assert not s.playing, "stepping must pause"


def test_stepping_wraps_at_both_ends():
    s = PlaybackState(5, frame_idx=4)
    s.next()
    assert s.frame_idx == 0
    s.previous()
    assert s.frame_idx == 4


def test_reset_goes_to_the_first_frame_and_pauses():
    s = PlaybackState(8, frame_idx=6, playing=True)
    s.reset()
    assert s.frame_idx == 0 and not s.playing


def test_seek_does_not_pause():
    """The timeline is a position, not a stop button.

    Scrubbing while playing should move the playhead and keep playing, the
    way a video scrubber does.
    """
    s = PlaybackState(20, playing=True)
    s.seek(13)
    assert s.frame_idx == 13 and s.playing


def test_seek_is_clamped():
    s = PlaybackState(6)
    s.seek(999)
    assert s.frame_idx == 5
    s.seek(-4)
    assert s.frame_idx == 0


def test_a_single_frame_run_cannot_advance():
    s = PlaybackState(1, playing=True)
    s.tick()
    assert s.frame_idx == 0


def test_an_empty_run_does_not_crash():
    s = PlaybackState(0)
    s.tick()
    s.next()
    s.seek(3)
    assert s.frame_idx == 0


# ════════════════════════════════════════════════════════════
# Playback state
# ════════════════════════════════════════════════════════════
def test_toggle_flips_playing():
    s = PlaybackState(10)
    s.toggle()
    assert s.playing
    s.toggle()
    assert not s.playing


def test_tick_is_a_no_op_when_paused():
    s = PlaybackState(10, frame_idx=3, playing=False)
    s.tick()
    assert s.frame_idx == 3


def test_playback_loops_by_default():
    s = PlaybackState(4, frame_idx=3, playing=True)
    s.tick()
    assert s.frame_idx == 0 and s.playing


def test_without_loop_it_stops_at_the_end():
    s = PlaybackState(4, frame_idx=3, playing=True, loop=False)
    s.tick()
    assert s.frame_idx == 3 and not s.playing


@pytest.mark.parametrize("speed,mult", list(SPEEDS.items()))
def test_interval_tracks_the_speed(speed, mult):
    s = PlaybackState(10, speed=speed)
    assert s.interval_s() == pytest.approx(
        max(BASE_FRAME_S / mult, MIN_INTERVAL_S))


def test_interval_never_goes_below_the_floor():
    """Scheduling faster than the server can serve only queues reruns."""
    s = PlaybackState(10, speed="4x")
    assert s.interval_s() >= MIN_INTERVAL_S


def test_resize_keeps_the_playhead_in_range():
    s = PlaybackState(20, frame_idx=18)
    s.resize(6)
    assert s.n_frames == 6 and s.frame_idx == 5


def test_caption_reports_frame_time_and_speed():
    s = PlaybackState(100, frame_idx=11, speed="2x")
    cap = s.caption(1.23)
    assert "Frame 12/100" in cap
    assert "Scene time 1.23 s" in cap
    assert "Speed 2x" in cap


def test_run_every_is_none_unless_playing():
    c = PlaybackController(state=PlaybackState(10))
    assert c.run_every() is None
    c.state.playing = True
    assert c.run_every() == pytest.approx(c.state.interval_s())


def test_run_every_is_none_for_a_single_frame():
    c = PlaybackController(state=PlaybackState(1, playing=True))
    assert c.run_every() is None


def test_a_slow_renderer_still_shows_every_frame():
    """The bug this contract exists to prevent.

    A redraw costs far more than the 100 ms a frame is meant to be held
    for. Deriving the index from elapsed time — frame = elapsed /
    interval — therefore jumped three or four frames per redraw, and a
    12-frame run cycled through the same three of them: "it plays only 3
    frames and jumps around randomly". One frame per tick, however late
    the tick is.
    """
    c = PlaybackController(state=PlaybackState(12, playing=True,
                                               speed="1x"))
    c.tick_if_playing(now=0.0)              # starts the interval
    seen = [c.state.frame_idx]
    for k in range(1, 13):
        c.tick_if_playing(now=k * 0.45)     # a redraw 4.5x the interval
        seen.append(c.state.frame_idx)
    assert seen == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]


def test_redraws_inside_the_interval_do_not_advance():
    """A fragment reruns for reasons other than its timer."""
    c = PlaybackController(state=PlaybackState(100, playing=True,
                                               speed="1x"))
    c.tick_if_playing(now=100.0)
    for _ in range(10):
        assert c.tick_if_playing(now=100.04) is False
    assert c.state.frame_idx == 0
    assert c.tick_if_playing(now=100.11) is True
    assert c.state.frame_idx == 1


@pytest.mark.parametrize("speed", list(SPEEDS))
def test_the_interval_is_respected_at_every_speed(speed):
    c = PlaybackController(state=PlaybackState(100, playing=True,
                                               speed=speed))
    step = c.state.interval_s()
    c.tick_if_playing(now=0.0)
    assert c.tick_if_playing(now=step * 0.9) is False
    assert c.tick_if_playing(now=step * 1.1) is True


def test_scrubbing_restarts_the_interval_from_where_it_was_dropped():
    c = PlaybackController(state=PlaybackState(100, playing=True))
    c.tick_if_playing(now=10.0)
    c.tick_if_playing(now=10.5)
    assert c.state.frame_idx == 1
    c.state.seek(60)
    c.tick_if_playing(now=10.6)        # re-arms, does not advance
    assert c.state.frame_idx == 60
    c.tick_if_playing(now=10.9)
    assert c.state.frame_idx == 61


# ════════════════════════════════════════════════════════════
# The property the whole module exists for
# ════════════════════════════════════════════════════════════
class _ExplodingPipeline:
    """Anything that would run perception. Touching it fails the test."""

    def __getattr__(self, name):
        raise AssertionError(f"playback invoked the pipeline: .{name}")

    def __call__(self, *a, **k):
        raise AssertionError("playback invoked the pipeline")


def test_ticking_never_touches_the_pipeline():
    pipeline = _ExplodingPipeline()
    s = PlaybackState(24, playing=True)
    for _ in range(200):          # several loops around the run
        s.tick()
    assert isinstance(pipeline, _ExplodingPipeline)   # never called
    assert 0 <= s.frame_idx < 24


def test_the_controller_module_imports_no_pipeline_code():
    """Static guarantee, not just a behavioural one.

    A future edit could reach for the pipeline from inside the transport and
    every behavioural test above would still pass, because they never build
    a Streamlit page. This reads the imports instead.
    """
    src = (VIS / "playback_controller.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    banned = ("pipeline", "stages", "perception", "mapping", "evaluation",
              "data.")
    for mod in imported:
        assert not any(b in mod for b in banned), \
            f"the transport must not import {mod}"


def test_no_sleep_driven_playback_survives_anywhere_in_the_ui():
    """The anti-pattern being removed must not come back by another route."""
    offenders = []
    for path in VIS.glob("*.py"):
        src = path.read_text(encoding="utf-8")
        code = "\n".join(l for l in src.splitlines()
                         if not l.lstrip().startswith("#"))
        # Strip docstrings so the explanation of the old approach does not
        # trip its own test.
        body = ast.parse(src)
        docstrings = {ast.get_docstring(n) for n in ast.walk(body)
                      if isinstance(n, (ast.Module, ast.FunctionDef,
                                        ast.AsyncFunctionDef, ast.ClassDef))}
        for d in docstrings:
            if d:
                code = code.replace(d, "")
        if "time.sleep" in code:
            offenders.append(path.name)
    assert not offenders, f"time.sleep back in the UI: {offenders}"


def test_fragment_support_is_probed_not_assumed():
    """The binding must check the installed Streamlit rather than trust it."""
    assert isinstance(PlaybackController.supports_fragments(), bool)
    assert isinstance(PlaybackController.supports_fragment_scope(), bool)
