"""
Playback transport: the frame index, and how it advances.

WHY THIS IS NOT ``time.sleep`` + ``st.rerun``
---------------------------------------------
The obvious way to animate in Streamlit is to sleep at the bottom of the
script and rerun. It works, and it is wrong for this application:

  * ``time.sleep`` blocks the server thread, so every control on the page is
    dead for the duration of each frame;
  * a whole-app ``st.rerun`` re-executes the entire script, which means every
    widget is rebuilt and anything not memoised is recomputed — here that is
    the LiDAR pipeline, and a cache miss (a changed scenario, a cold start)
    would run perception once per displayed frame;
  * a click captured mid-sleep is simply lost.

So the state machine lives here as plain Python, with no Streamlit in it at
all, and the Streamlit binding drives it from inside an ``st.fragment`` with
``run_every``. A fragment rerun re-executes only the fragment, so advancing a
frame redraws the canvas and nothing else. ``st.fragment`` and
``st.rerun(scope=...)`` are both present in the installed Streamlit (1.64);
the binding checks and degrades to manual stepping rather than calling an API
that may not be there.

The pipeline is never invoked from anything in this module. That is the
property ``tests/test_playback_controller.py`` asserts directly, by handing
the controller a callable that raises.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

#: Replay speed multipliers offered in the transport.
SPEEDS: Dict[str, float] = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0,
                            "2x": 2.0, "4x": 4.0}

#: Wall-clock seconds per displayed frame at 1x. The scenes are 10 Hz, so
#: this is real time; it is a REPLAY rate and is labelled as one, because it
#: says nothing about how long the pipeline took to produce a frame.
BASE_FRAME_S = 0.10

#: Streamlit will not schedule a fragment faster than this usefully, and
#: trying only queues reruns faster than they can be served.
MIN_INTERVAL_S = 0.05


# ════════════════════════════════════════════════════════════
# The state machine — no Streamlit, so it is directly testable
# ════════════════════════════════════════════════════════════
@dataclass
class PlaybackState:
    """Where playback is, and where it goes next."""

    n_frames: int
    frame_idx: int = 0
    playing: bool = False
    speed: str = "1x"
    loop: bool = True
    #: Wall clock of the last timer-driven advance. Lives on the state
    #: rather than the controller because the controller is rebuilt on every
    #: script run while the state persists in session state.
    last_tick: Optional[float] = None

    def __post_init__(self) -> None:
        self.n_frames = max(int(self.n_frames), 0)
        self.frame_idx = self._clamp(self.frame_idx)

    # ── bounds ───────────────────────────────────────────────
    def _clamp(self, i: int) -> int:
        if self.n_frames <= 0:
            return 0
        return int(min(max(i, 0), self.n_frames - 1))

    @property
    def last(self) -> int:
        return max(self.n_frames - 1, 0)

    @property
    def at_end(self) -> bool:
        return self.frame_idx >= self.last

    # ── transitions ──────────────────────────────────────────
    def seek(self, i: int) -> None:
        """Scrub. Scrubbing does NOT pause: the timeline is a live position."""
        self.frame_idx = self._clamp(i)

    def next(self) -> None:
        """One frame forward. Stepping pauses — it is for pointing at things."""
        self.playing = False
        self.frame_idx = self._wrap(self.frame_idx + 1)

    def previous(self) -> None:
        self.playing = False
        self.frame_idx = self._wrap(self.frame_idx - 1)

    def reset(self) -> None:
        self.playing = False
        self.frame_idx = 0

    def toggle(self) -> None:
        if not self.playing and self.at_end and not self.loop:
            self.frame_idx = 0          # replay rather than sit at the end
        self.playing = not self.playing

    def tick(self) -> None:
        """Advance because the timer fired. A no-op when paused.

        Kept separate from :meth:`next` because the timer must not be able to
        clear ``playing`` — that is what stepping means, not what a tick
        means.
        """
        if not self.playing or self.n_frames == 0:
            return
        if self.at_end and not self.loop:
            self.playing = False
            return
        self.frame_idx = self._wrap(self.frame_idx + 1)

    def _wrap(self, i: int) -> int:
        if self.n_frames <= 0:
            return 0
        return int(i % self.n_frames) if self.loop else self._clamp(i)

    # ── derived ──────────────────────────────────────────────
    def interval_s(self) -> float:
        """Wall-clock seconds the current frame should be held for."""
        return max(BASE_FRAME_S / SPEEDS.get(self.speed, 1.0), MIN_INTERVAL_S)

    def resize(self, n_frames: int) -> None:
        """A different scenario or frame count arrived."""
        n = max(int(n_frames), 0)
        if n != self.n_frames:
            self.n_frames = n
            self.frame_idx = self._clamp(self.frame_idx)

    def caption(self, scene_time_s: Optional[float] = None) -> str:
        """The one-line readout beside the transport."""
        parts = [f"Frame {self.frame_idx + 1}/{max(self.n_frames, 1)}"]
        if scene_time_s is not None:
            parts.append(f"Scene time {scene_time_s:.2f} s")
        parts.append(f"Speed {self.speed}")
        return "  ·  ".join(parts)


# ════════════════════════════════════════════════════════════
# The Streamlit binding
# ════════════════════════════════════════════════════════════
@dataclass
class PlaybackController:
    """Binds a :class:`PlaybackState` to ``st.session_state`` and widgets."""

    key: str = "pb"
    state: PlaybackState = field(default_factory=lambda: PlaybackState(1))

    # ── session-state persistence ────────────────────────────
    @classmethod
    def bind(cls, n_frames: int, key: str = "pb") -> "PlaybackController":
        import streamlit as st
        st_key = f"_{key}_state"
        s = st.session_state.get(st_key)
        if not isinstance(s, PlaybackState):
            s = PlaybackState(n_frames)
            st.session_state[st_key] = s
        s.resize(n_frames)
        return cls(key=key, state=s)

    # ── capability probe ─────────────────────────────────────
    @staticmethod
    def supports_fragments() -> bool:
        """Whether the installed Streamlit can auto-advance without a sleep.

        Checked rather than assumed: on an older Streamlit the transport
        still works, it just does not animate on its own, which is a far
        better failure than an AttributeError on load.
        """
        import streamlit as st
        return hasattr(st, "fragment")

    @staticmethod
    def supports_fragment_scope() -> bool:
        import inspect
        import streamlit as st
        try:
            return "scope" in inspect.signature(st.rerun).parameters
        except (TypeError, ValueError):
            return False

    def run_every(self) -> Optional[float]:
        """The ``st.fragment(run_every=...)`` value for the current state."""
        if not self.state.playing or self.state.n_frames <= 1:
            return None
        return self.state.interval_s()

    # ── the transport row ────────────────────────────────────
    def render_transport(self, scene_time_s: Optional[float] = None,
                         *, on_change: Optional[Callable[[], None]] = None
                         ) -> int:
        """Draw [◀ Previous][▶ Play][⏭ Next][Reset] + timeline + readout.

        Returns the frame index to display. Nothing here loads or computes
        anything; it only moves an integer.
        """
        import streamlit as st

        s = self.state
        scoped = self.supports_fragment_scope()

        def _rerun(scope: str = "fragment"):
            """Redraw.

            ``scope`` matters here in a way that is easy to miss.
            ``run_every`` is fixed when the fragment is DECORATED, which
            happens in the enclosing script run — so starting or stopping
            playback has to re-run the app, or the timer keeps whatever
            schedule it had and Play does nothing visible. Every other
            control only changes what is drawn, so it stays fragment-scoped
            and never re-enters the pipeline.
            """
            if on_change is not None:
                on_change()
            if scoped:
                st.rerun(scope=scope)
            else:
                st.rerun()

        c_prev, c_play, c_next, c_reset, c_bar, c_read = st.columns(
            [1.0, 1.0, 1.0, 1.0, 6.0, 2.6], vertical_alignment="center")

        with c_prev:
            if st.button("◀ Previous", key=f"{self.key}_prev",
                         width="stretch", disabled=s.n_frames <= 1,
                         help="One frame back. Pauses."):
                s.previous()
                _rerun("app")          # stepping pauses, so the timer stops
        with c_play:
            label = "⏸ Pause" if s.playing else "▶ Play"
            if st.button(label, key=f"{self.key}_play", width="stretch",
                         type="primary" if not s.playing else "secondary",
                         disabled=s.n_frames <= 1):
                s.toggle()
                s.last_tick = None
                _rerun("app")          # the timer schedule changes
        with c_next:
            if st.button("⏭ Next", key=f"{self.key}_next", width="stretch",
                         disabled=s.n_frames <= 1,
                         help="One frame forward. Pauses."):
                s.next()
                _rerun("app")          # stepping pauses, so the timer stops
        with c_reset:
            if st.button("↺ Reset", key=f"{self.key}_reset", width="stretch",
                         help="Back to frame 1 and pause."):
                s.reset()
                _rerun("app")          # reset pauses, so the timer stops

        with c_bar:
            if s.n_frames > 1:
                # Deliberately KEYLESS. A keyed slider latches its value in
                # session state and would then ignore `value` on every later
                # run, so the handle would sit still while playback advanced
                # underneath it. Without a key the passed value wins, and a
                # drag still returns the dragged position on its own rerun.
                i = st.slider("Timeline", 0, s.last, s.frame_idx,
                              label_visibility="collapsed")
                if i != s.frame_idx:
                    s.seek(i)
                    if on_change is not None:
                        on_change()
            else:
                st.caption("Single frame — nothing to play.")

        with c_read:
            st.markdown(
                f'<div style="font-size:12px;line-height:1.4">'
                f'{s.caption(scene_time_s)}'
                f'<br><span style="opacity:.6">a replay rate, not a latency '
                f'measurement. Measured end to end it settles near 2 frames '
                f'per second whatever this is set to — the limit is drawing '
                f'and shipping the canvas to the browser, not the '
                f'pipeline</span></div>',
                unsafe_allow_html=True)

        return s.frame_idx

    def speed_selector(self, label: str = "Replay speed") -> str:
        import streamlit as st
        s = self.state
        opts = list(SPEEDS)
        before = s.speed
        chosen = st.selectbox(label, opts, index=opts.index(s.speed),
                              key=f"{self.key}_speed",
                              help="Playback rate only. It changes nothing "
                                   "the pipeline measured. The achieved "
                                   "rate is bounded by the redraw: ~95 ms "
                                   "of server time per displayed frame at "
                                   "the default canvas on this machine, and "
                                   "~460 ms end to end once the browser "
                                   "round trip is included. Set "
                                   "FOVEA_PROFILE=1 to print the server "
                                   "half to the console.")
        s.speed = chosen
        if chosen != before and s.playing:
            # The interval feeds run_every, which is set on the enclosing
            # script run, so the new speed needs an app rerun to take hold.
            s.last_tick = None
            st.rerun()
        return chosen

    def tick_if_playing(self, now: Optional[float] = None) -> bool:
        """Called at the END of the fragment body, once it has drawn.

        Gated on wall clock rather than simply advancing, because a fragment
        reruns for two reasons — the ``run_every`` timer, and a widget inside
        it changing — and Streamlit does not say which. Without the gate,
        dragging the timeline during playback would also steal a frame.

        Returns True if the frame index moved.
        """
        import time as _time
        s = self.state
        if not s.playing or s.n_frames <= 1:
            s.last_tick = None
            return False
        t = _time.monotonic() if now is None else float(now)
        if s.last_tick is not None and (t - s.last_tick) < 0.75 * s.interval_s():
            return False
        s.last_tick = t
        before = s.frame_idx
        s.tick()
        return s.frame_idx != before
