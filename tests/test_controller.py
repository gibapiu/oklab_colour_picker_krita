import numpy as np

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.app.controller import ChangeKind, ColourPickerController, normalize_oklab_for_krita
from oklab_colour_picker.infrastructure.krita_adapter import KritaForegroundAdapter


def _observe(observed):
    return lambda snapshot: observed.append((snapshot.intent, snapshot.kind))


def _krita_bgra_with_alpha(rgb):
    return [float(rgb[2]), float(rgb[1]), float(rgb[0]), 1.0]


def test_foreground_commits_are_coalesced_to_latest_colour():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)

    first = np.array([0.45, 0.01, 0.02])
    latest = np.array([0.62, -0.03, 0.04])
    controller.request_foreground_commit(first)
    controller.request_foreground_commit(latest)

    assert adapter.set_foreground_calls == []
    assert scheduler.pending_count == 1

    scheduler.run_pending()

    assert len(adapter.set_foreground_calls) == 1
    np.testing.assert_allclose(adapter.set_foreground_calls[0], latest)
    np.testing.assert_allclose(controller.selected_colour, latest)


def test_duplicate_commits_are_suppressed_after_normalization():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    colour = np.array([0.55, 0.02, -0.03])
    same_quantized_colour = normalize_oklab_for_krita(colour)

    controller.request_foreground_commit(colour)
    scheduler.run_pending()
    controller.request_foreground_commit(same_quantized_colour)
    scheduler.run_pending()

    assert len(adapter.set_foreground_calls) == 1


def test_duplicate_achromatic_commit_still_broadcasts_updated_hue_intent():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    observed = []
    controller.add_colour_listener(_observe(observed))

    controller.request_foreground_commit(ColourIntent.from_lch(0.5, 0.0, 0.0))
    scheduler.run_pending()
    controller.request_foreground_commit(
        ColourIntent.from_lch(0.5, 0.0, np.radians(210.0))
    )
    scheduler.run_pending()

    assert len(adapter.set_foreground_calls) == 1
    assert [kind for _colour, kind in observed] == [ChangeKind.COMMIT, ChangeKind.COMMIT]
    assert observed[-1][0].hue == np.radians(210.0)


def test_duplicate_suppression_normalizes_adapter_readback_before_storage():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter(readback_offset=np.array([1e-10, -1e-10, 1e-10]))
    controller = ColourPickerController(adapter, scheduler=scheduler)
    colour = np.array([0.55, 0.02, -0.03])
    same_quantized_colour = normalize_oklab_for_krita(colour)

    controller.request_foreground_commit(colour)
    scheduler.run_pending()
    controller.request_foreground_commit(same_quantized_colour)
    scheduler.run_pending()

    assert len(adapter.set_foreground_calls) == 1


def test_missing_active_krita_view_does_not_record_a_commit():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter(available=False)
    controller = ColourPickerController(adapter, scheduler=scheduler)
    colour = np.array([0.5, 0.01, 0.02])

    controller.request_foreground_commit(colour)
    scheduler.run_pending()

    assert len(adapter.set_foreground_calls) == 1
    np.testing.assert_allclose(adapter.set_foreground_calls[0], colour)
    assert controller.selected_colour is None


def test_failed_commit_rolls_back_to_previous_selected_colour():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter(available=False)
    controller = ColourPickerController(adapter, scheduler=scheduler)
    previous = np.array([0.45, -0.01, 0.03])
    requested = np.array([0.5, 0.01, 0.02])

    controller.set_preview_colour(previous)
    controller.request_foreground_commit(requested)
    scheduler.run_pending()

    np.testing.assert_allclose(controller.selected_colour, previous)


def test_external_foreground_sync_updates_selected_colour_once():
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler())
    external = np.array([0.4, -0.03, 0.07])
    observed = []
    controller.add_colour_listener(_observe(observed))

    adapter.foreground_colour = external
    assert controller.sync_external_foreground() is True
    assert controller.sync_external_foreground() is False

    np.testing.assert_allclose(controller.selected_colour, external)
    assert len(observed) == 1
    np.testing.assert_allclose(observed[0][0].paint_oklab, external)
    # The first colour the controller ever holds is the seed (INITIAL);
    # only later changes are EXTERNAL.
    assert observed[0][1] is ChangeKind.INITIAL

    changed = np.array([0.7, 0.01, -0.02])
    adapter.foreground_colour = changed
    assert controller.sync_external_foreground() is True
    assert observed[-1][1] is ChangeKind.EXTERNAL


def test_subscribe_does_not_replay_when_no_colour_is_available():
    controller = ColourPickerController(FakeKritaAdapter(), scheduler=FakeScheduler())
    observed = []

    controller.add_colour_listener(_observe(observed))

    assert observed == []


def test_poll_seeds_initial_when_view_appears_after_construction():
    adapter = FakeKritaAdapter()
    timer = FakeRepeatingTimer()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler(), foreground_timer=timer)
    observed = []
    controller.add_colour_listener(_observe(observed))

    assert observed == []
    assert timer.start_count == 1

    external = np.array([0.4, -0.03, 0.07])
    adapter.foreground_colour = external
    timer.tick()

    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL]
    np.testing.assert_allclose(observed[0][0].paint_oklab, external)
    assert timer.start_count == 1


def test_construction_does_not_read_foreground():
    adapter = FakeKritaAdapter()
    adapter.foreground_colour = np.array([0.4, -0.03, 0.07])

    controller = ColourPickerController(adapter, scheduler=FakeScheduler())

    # Acquisition is event-driven: nothing is read until the dock subscribes,
    # canvasChanged forces a sync, or the poll timer ticks.
    assert controller.selected_colour is None


def test_subscribe_pulls_foreground_on_empty_without_extra_broadcast():
    adapter = FakeKritaAdapter()
    external = np.array([0.4, -0.03, 0.07])
    adapter.foreground_colour = external
    controller = ColourPickerController(adapter, scheduler=FakeScheduler())
    observed = []

    controller.add_colour_listener(_observe(observed))

    np.testing.assert_allclose(controller.selected_colour, external)
    # The subscribe-time pull does not broadcast EXTERNAL; the new listener
    # converges via the single INITIAL replay only.
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL]
    np.testing.assert_allclose(observed[0][0].paint_oklab, external)


def test_forced_sync_acquires_when_view_appears_after_construction():
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler())
    observed = []
    controller.add_colour_listener(_observe(observed))
    external = np.array([0.48, 0.02, 0.01])
    adapter.foreground_colour = external

    # Krita's canvasChanged hook forces a sync once a view exists; this is
    # still the first colour the controller holds, so it seeds (INITIAL).
    assert controller.sync_external_foreground(force=True) is True

    np.testing.assert_allclose(controller.selected_colour, external)
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL]


def test_commit_echo_is_suppressed_by_token_and_normalized_colour_match():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    committed = np.array([0.65, 0.04, -0.02])
    observed = []
    controller.add_colour_listener(_observe(observed))

    controller.request_foreground_commit(committed)
    scheduler.run_pending()
    adapter.foreground_colour = normalize_oklab_for_krita(committed)

    assert controller.sync_external_foreground() is False
    # The commit broadcasts COMMIT; the self-feedback sync adds no EXTERNAL.
    assert [kind for _colour, kind in observed] == [ChangeKind.COMMIT]
    np.testing.assert_allclose(controller.selected_colour, committed)


def test_external_change_clears_stale_self_feedback_suppression():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    committed = np.array([0.65, 0.04, -0.02])
    external = np.array([0.38, -0.02, 0.06])
    observed = []
    controller.add_colour_listener(_observe(observed))

    controller.request_foreground_commit(committed)
    scheduler.run_pending()
    adapter.foreground_colour = external
    assert controller.sync_external_foreground() is True
    adapter.foreground_colour = normalize_oklab_for_krita(committed)

    assert controller.sync_external_foreground() is True
    assert [kind for _colour, kind in observed] == [
        ChangeKind.COMMIT,
        ChangeKind.EXTERNAL,
        ChangeKind.EXTERNAL,
    ]
    np.testing.assert_allclose(
        observed[-1][0].paint_oklab,
        normalize_oklab_for_krita(committed),
    )


def test_external_sync_does_not_refresh_failed_commit_snapshot_during_pending_interaction():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter(available=False)
    controller = ColourPickerController(adapter, scheduler=scheduler)
    first_pending = np.array([0.65, 0.04, -0.02])
    external = np.array([0.38, -0.02, 0.06])
    latest_pending = np.array([0.5, 0.01, 0.02])

    controller.request_foreground_commit(first_pending)
    adapter.foreground_colour = external
    assert controller.sync_external_foreground() is False
    controller.request_foreground_commit(latest_pending)
    scheduler.run_pending()

    assert controller.selected_colour is None


def test_forced_external_sync_handles_one_shot_foreground_switch_during_grace_window():
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler())
    preview = np.array([0.62, 0.02, -0.03])
    switched_foreground = np.array([0.20, -0.04, 0.07])

    controller.set_preview_colour(preview)
    adapter.foreground_colour = switched_foreground

    assert controller.sync_external_foreground() is False
    assert controller.sync_external_foreground(force=True) is True
    np.testing.assert_allclose(controller.selected_colour, switched_foreground)


def test_noop_forced_external_sync_preserves_local_interaction_guard():
    adapter = FakeKritaAdapter()
    clock = FakeClock()
    previous = np.array([0.45, -0.01, 0.02])
    preview = np.array([0.62, 0.02, -0.03])
    switched_foreground = np.array([0.20, -0.04, 0.07])
    adapter.foreground_colour = previous
    controller = ColourPickerController(adapter, scheduler=FakeScheduler(), clock=clock)

    controller.set_preview_colour(preview)
    adapter.foreground_colour = None

    assert controller.sync_external_foreground(force=True) is False

    adapter.foreground_colour = switched_foreground
    assert controller.sync_external_foreground() is False
    np.testing.assert_allclose(controller.selected_colour, preview)

    clock.advance(0.76)
    assert controller.sync_external_foreground() is True
    np.testing.assert_allclose(controller.selected_colour, switched_foreground)


def test_forced_external_sync_does_not_interrupt_pending_commit():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    pending = np.array([0.62, 0.02, -0.03])
    switched_foreground = np.array([0.20, -0.04, 0.07])

    controller.request_foreground_commit(pending)
    adapter.foreground_colour = switched_foreground

    assert controller.sync_external_foreground(force=True) is False
    scheduler.run_pending()
    np.testing.assert_allclose(controller.selected_colour, pending)


def test_poll_timer_starts_at_construction_and_runs_for_lifetime():
    timer = FakeRepeatingTimer()

    ColourPickerController(FakeKritaAdapter(), scheduler=FakeScheduler(), foreground_timer=timer)

    assert timer.start_count == 1
    assert timer.stop_count == 0


def test_timer_tick_syncs_foreground():
    adapter = FakeKritaAdapter()
    timer = FakeRepeatingTimer()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler(), foreground_timer=timer)

    adapter.foreground_colour = np.array([0.48, 0.02, 0.01])
    timer.tick()
    np.testing.assert_allclose(controller.selected_colour, adapter.foreground_colour)

    adapter.foreground_colour = np.array([0.72, -0.02, 0.06])
    timer.tick()
    np.testing.assert_allclose(controller.selected_colour, adapter.foreground_colour)
    assert timer.start_count == 1


def test_removed_foreground_listener_does_not_receive_updates():
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler())
    observed = []
    listener = _observe(observed)
    controller.add_colour_listener(listener)
    controller.remove_colour_listener(listener)

    adapter.foreground_colour = np.array([0.48, 0.02, 0.01])
    assert controller.sync_external_foreground() is True

    assert observed == []


def test_raising_foreground_listener_does_not_block_later_listeners():
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler())
    observed = []

    def raising_listener(_snapshot):
        raise RuntimeError("deleted widget")

    controller.add_colour_listener(raising_listener)
    controller.add_colour_listener(_observe(observed))
    adapter.foreground_colour = np.array([0.48, 0.02, 0.01])

    assert controller.sync_external_foreground() is True
    assert len(observed) == 1


def test_preview_does_not_replace_pending_commit_before_flush():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    committed = np.array([0.45, 0.01, 0.02])
    preview = np.array([0.62, -0.03, 0.04])

    controller.request_foreground_commit(committed)
    controller.set_preview_colour(preview)
    scheduler.run_pending()

    np.testing.assert_allclose(adapter.set_foreground_calls[0], committed)
    np.testing.assert_allclose(controller.selected_colour, committed)


def test_external_sync_does_not_override_local_preview_before_commit():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    clock = FakeClock()
    previous = np.array([0.45, -0.01, 0.02])
    preview = np.array([0.62, 0.03, -0.04])
    adapter.foreground_colour = previous
    controller = ColourPickerController(adapter, scheduler=scheduler, clock=clock)
    observed = []
    controller.add_colour_listener(_observe(observed))

    controller.set_preview_colour(preview)

    assert controller.sync_external_foreground() is False
    np.testing.assert_allclose(controller.selected_colour, preview)
    # Subscribe replays INITIAL (foreground existed at startup); then
    # set_preview_colour broadcasts PREVIEW. The blocked external sync adds
    # no EXTERNAL.
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL, ChangeKind.PREVIEW]
    np.testing.assert_allclose(observed[-1][0].paint_oklab, preview)


def test_external_sync_does_not_override_preview_across_repeated_polls():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    clock = FakeClock()
    previous = np.array([0.45, -0.01, 0.02])
    preview = np.array([0.62, 0.03, -0.04])
    adapter.foreground_colour = previous
    controller = ColourPickerController(adapter, scheduler=scheduler, clock=clock)
    observed = []
    controller.add_colour_listener(_observe(observed))

    controller.set_preview_colour(preview)

    assert controller.sync_external_foreground() is False
    clock.advance(0.25)
    assert controller.sync_external_foreground() is False
    np.testing.assert_allclose(controller.selected_colour, preview)
    # INITIAL replay on subscribe, then one PREVIEW; the repeated blocked
    # syncs add no EXTERNAL.
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL, ChangeKind.PREVIEW]


def test_preview_cancellation_does_not_drop_external_sync_guard():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    clock = FakeClock()
    previous = np.array([0.45, -0.01, 0.02])
    preview = np.array([0.62, 0.03, -0.04])
    adapter.foreground_colour = previous
    controller = ColourPickerController(adapter, scheduler=scheduler, clock=clock)
    observed = []
    controller.add_colour_listener(_observe(observed))

    controller.set_preview_colour(preview)
    controller.set_preview_colour(None)

    assert controller.sync_external_foreground() is False
    # INITIAL replay on subscribe, then PREVIEW for the non-None preview; the
    # None cancel does not broadcast, and the blocked sync adds no EXTERNAL.
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL, ChangeKind.PREVIEW]


def test_external_sync_resumes_after_preview_guard_expires():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    clock = FakeClock()
    previous = np.array([0.45, -0.01, 0.02])
    preview = np.array([0.62, 0.03, -0.04])
    adapter.foreground_colour = previous
    controller = ColourPickerController(adapter, scheduler=scheduler, clock=clock)

    controller.set_preview_colour(preview)
    clock.advance(0.76)

    assert controller.sync_external_foreground() is True
    np.testing.assert_allclose(controller.selected_colour, previous)


def test_external_sync_does_not_override_pending_local_commit_before_flush():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    previous = np.array([0.45, -0.01, 0.02])
    committed = np.array([0.62, 0.03, -0.04])
    adapter.foreground_colour = previous
    controller = ColourPickerController(adapter, scheduler=scheduler)
    observed = []
    controller.add_colour_listener(_observe(observed))

    controller.request_foreground_commit(committed)

    assert controller.sync_external_foreground() is False
    np.testing.assert_allclose(controller.selected_colour, committed)
    # Only the INITIAL replay so far; the commit broadcast happens at flush.
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL]

    scheduler.run_pending()
    np.testing.assert_allclose(controller.selected_colour, committed)
    assert len(adapter.set_foreground_calls) == 1
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL, ChangeKind.COMMIT]


def test_krita_adapter_returns_none_without_active_window():
    adapter = KritaForegroundAdapter(FakeKrita(active_window=None))

    assert adapter.get_foreground() is None
    assert adapter.set_foreground([0.5, 0.0, 0.0]) is None


def test_krita_adapter_returns_none_without_active_view():
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=None)))

    assert adapter.get_foreground() is None
    assert adapter.set_foreground([0.5, 0.0, 0.0]) is None


def test_krita_adapter_reads_srgb_foreground_without_converting():
    managed = FakeManagedColor(components=_krita_bgra_with_alpha([0.25, 0.5, 0.75]))
    view = FakeView(foreground_color=managed)
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=view)))

    np.testing.assert_allclose(
        adapter.get_foreground(),
        color_math.srgb_to_oklab([0.25, 0.5, 0.75]),
    )
    assert managed.set_color_space_calls == []


def test_krita_adapter_converts_non_srgb_foreground_via_krita_converter():
    managed = FakeManagedColor(
        model="CMYK",
        depth="U16",
        profile="Chemical proof",
        components_after_conversion=_krita_bgra_with_alpha([0.25, 0.5, 0.75]),
    )
    view = FakeView(foreground_color=managed)
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=view)))

    np.testing.assert_allclose(
        adapter.get_foreground(),
        color_math.srgb_to_oklab([0.25, 0.5, 0.75]),
    )
    assert managed.set_color_space_calls == [("RGBA", "U8", "sRGB-elle-V2-srgbtrc.icc")]


def test_krita_adapter_converts_linear_srgb_foreground_via_krita_converter():
    managed = FakeManagedColor(
        depth="F32",
        profile="krita-2.5, lcms sRGB built-in with linear gamma TRC",
        components_after_conversion=_krita_bgra_with_alpha([0.25, 0.5, 0.75]),
    )
    view = FakeView(foreground_color=managed)
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=view)))

    np.testing.assert_allclose(
        adapter.get_foreground(),
        color_math.srgb_to_oklab([0.25, 0.5, 0.75]),
    )
    assert managed.set_color_space_calls == [("RGBA", "U8", "sRGB-elle-V2-srgbtrc.icc")]


def test_krita_adapter_returns_none_when_conversion_raises():
    managed = FakeManagedColor(
        model="CMYK",
        profile="Chemical proof",
        set_color_space_raises=RuntimeError("no converter available"),
    )
    view = FakeView(foreground_color=managed)
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=view)))

    assert adapter.get_foreground() is None


def test_krita_adapter_returns_none_when_conversion_reports_failure():
    managed = FakeManagedColor(
        model="CMYK",
        depth="U16",
        profile="Chemical proof",
        components=[0.1, 0.2, 0.3, 0.4],
        set_color_space_result=False,
    )
    view = FakeView(foreground_color=managed)
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=view)))

    assert adapter.get_foreground() is None


def test_krita_adapter_clips_normalized_components_above_one():
    managed = FakeManagedColor(components=_krita_bgra_with_alpha([1.2, 0.5, 0.25]))
    view = FakeView(foreground_color=managed)
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=view)))

    expected = color_math.srgb_to_oklab([1.0, 0.5, 0.25])
    actual = adapter.get_foreground()

    np.testing.assert_allclose(actual, expected)


def test_krita_adapter_writes_components_in_bgra_order():
    managed = FakeManagedColor()
    view = FakeView(foreground_color=managed)

    def factory(*args, **kwargs):
        return managed

    adapter = KritaForegroundAdapter(
        FakeKrita(active_window=FakeWindow(active_view=view)),
        managed_color_factory=factory,
    )

    red_oklab = color_math.srgb_to_oklab([1.0, 0.0, 0.0])
    adapter.set_foreground(red_oklab)

    assert len(view.set_foreground_calls) == 1
    np.testing.assert_allclose(managed.components(), _krita_bgra_with_alpha([1.0, 0.0, 0.0]), atol=1e-6)


def test_krita_adapter_quantizes_white_before_writing_u8_components():
    managed = FakeManagedColor()
    view = FakeView(foreground_color=managed)

    def factory(*args, **kwargs):
        return managed

    adapter = KritaForegroundAdapter(
        FakeKrita(active_window=FakeWindow(active_view=view)),
        managed_color_factory=factory,
    )

    adapter.set_foreground(np.array([1.0, 0.0, 0.0]))

    assert managed.components() == _krita_bgra_with_alpha([1.0, 1.0, 1.0])


def test_krita_adapter_returns_readback_colour_after_setting_foreground():
    readback = FakeManagedColor(components=_krita_bgra_with_alpha([0.25, 0.5, 0.75]))
    view = FakeView(foreground_color=readback)
    adapter = KritaForegroundAdapter(
        FakeKrita(active_window=FakeWindow(active_view=view)),
        managed_color_factory=FakeManagedColor,
    )

    actual = adapter.set_foreground([0.5, 0.0, 0.0])

    assert len(view.set_foreground_calls) == 1
    np.testing.assert_allclose(actual, adapter.get_foreground())


class FakeScheduler:
    def __init__(self):
        self._callbacks = []

    @property
    def pending_count(self):
        return len(self._callbacks)

    def call_soon(self, callback):
        self._callbacks.append(callback)

    def run_pending(self):
        callbacks = self._callbacks
        self._callbacks = []
        for callback in callbacks:
            callback()


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


class FakeRepeatingTimer:
    def __init__(self):
        self.start_count = 0
        self.stop_count = 0
        self._callback = None
        self._running = False

    def start(self, callback):
        self.start_count += 1
        self._callback = callback
        self._running = True

    def stop(self):
        self.stop_count += 1
        self._running = False

    def tick(self):
        if self._running and self._callback is not None:
            self._callback()


class FakeKritaAdapter:
    def __init__(self, *, available=True, readback_offset=None):
        self.available = available
        self.readback_offset = None if readback_offset is None else np.asarray(readback_offset, dtype=float)
        self.foreground_colour = None
        self.set_foreground_calls = []

    def set_foreground(self, oklab):
        colour = np.asarray(oklab, dtype=float)
        self.set_foreground_calls.append(colour.tolist())
        if not self.available:
            return None
        self.foreground_colour = normalize_oklab_for_krita(colour)
        if self.readback_offset is not None:
            return self.foreground_colour + self.readback_offset
        return self.foreground_colour

    def get_foreground(self):
        return self.foreground_colour


class FakeKrita:
    def __init__(self, *, active_window):
        self._active_window = active_window

    def activeWindow(self):
        return self._active_window


class FakeWindow:
    def __init__(self, *, active_view):
        self._active_view = active_view

    def activeView(self):
        return self._active_view


class FakeView:
    def __init__(self, *, foreground_color):
        self._foreground_color = foreground_color
        self.set_foreground_calls = []

    def foregroundColor(self):
        return self._foreground_color

    def setForeGroundColor(self, managed):
        self.set_foreground_calls.append(managed)


class FakeManagedColor:
    def __init__(
        self,
        *managed_color_args,
        components=None,
        model="RGBA",
        depth="U8",
        profile="sRGB-elle-V2-srgbtrc.icc",
        components_after_conversion=None,
        set_color_space_raises=None,
        set_color_space_result=True,
    ):
        self._components = components if components is not None else [0.75, 0.5, 0.25, 1.0]
        self._model = model
        self._depth = depth
        self._profile = profile
        self._components_after_conversion = components_after_conversion
        self._set_color_space_raises = set_color_space_raises
        self._set_color_space_result = set_color_space_result
        self.set_color_space_calls = []

    def setComponents(self, components):
        self._components = components

    def components(self):
        return self._components

    def colorModel(self):
        return self._model

    def colorDepth(self):
        return self._depth

    def colorProfile(self):
        return self._profile

    def setColorSpace(self, model, depth, profile):
        if self._set_color_space_raises is not None:
            raise self._set_color_space_raises
        self.set_color_space_calls.append((model, depth, profile))
        if self._set_color_space_result is False:
            return False
        self._model = model
        self._depth = depth
        self._profile = profile
        if self._components_after_conversion is not None:
            self._components = list(self._components_after_conversion)
        return self._set_color_space_result
