import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")

from oklab_colour_picker.qt import QtCore, QtWidgets

from tests.qt_helpers import send_mouse
from oklab_colour_picker.app.selector_model_cache import SelectorMode
from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_presentation import (
    PresentedColour,
    default_colour_presenter,
)
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.app.controller import (
    ChangeKind,
    ColourPickerController,
    ColourSnapshot,
    normalize_oklab_for_krita,
)
from oklab_colour_picker.ui.dock import ColourPickerDockPanel
from oklab_colour_picker.ui.selectors import HueLightnessSliceDiskWidget
from tests.helpers import presented_colour


def _present(colour):
    return default_colour_presenter().present(colour)


def test_dock_panel_constructs_all_selector_views_and_switches_modes(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    assert [widget.objectName() for widget in panel.selector_widgets] == [
        "lightness-slice-selector",
        "hue-lightness-slice-selector",
        "lightness-chroma-slice-selector",
    ]
    assert panel.mode == SelectorMode.LIGHTNESS_SLICE

    assert isinstance(
        panel.selector_for_mode(SelectorMode.HUE_LIGHTNESS_SLICE),
        HueLightnessSliceDiskWidget,
    )

    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)

    assert panel.mode == SelectorMode.LIGHTNESS_CHROMA_SLICE
    assert panel.active_selector is panel.selector_for_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)


def test_dock_panel_initializes_only_active_selector_view(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    assert tuple(panel._selectors) == (SelectorMode.LIGHTNESS_SLICE,)
    assert panel._tabs.count() == len(SelectorMode)
    assert panel._tabs.widget(0) is panel.selector_for_mode(SelectorMode.LIGHTNESS_SLICE)
    assert panel._tabs.widget(1).objectName() == "hue-lightness-slice-selector-placeholder"


def test_dock_panel_uses_current_foreground_on_construction(qtbot):
    colour = color_math.oklch_to_oklab([0.58, 0.07, math.pi / 3.0])
    controller = FakeController(selected_colour=colour)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    for widget in panel.selector_widgets:
        np.testing.assert_allclose(widget.selected_colour, colour)
    _assert_readout_matches(panel, colour)


def test_dock_panel_construction_does_not_synchronously_resync_foreground(qtbot):
    colour = color_math.oklch_to_oklab([0.58, 0.07, math.pi / 3.0])
    controller = FakeController(selected_colour=colour)

    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    assert controller.sync_count == 0


def test_selector_signals_update_controller_and_sibling_indicators(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)
    active = panel.active_selector
    active.resize(120, 80)

    colour = active.model.color_at_position((40, 20), (active.width(), active.height()))
    assert colour is not None
    active.previewed.emit(colour.copy())
    active.committed.emit(colour.copy())

    np.testing.assert_allclose(controller.previews[-1].paint_oklab, colour)
    np.testing.assert_allclose(controller.commits[-1].paint_oklab, colour)
    for widget in panel.selector_widgets:
        np.testing.assert_allclose(widget.selected_colour, colour)


def test_controller_presents_once_and_dock_fans_same_snapshot_to_views(qtbot, monkeypatch):
    presenter = SpyPresenter()
    controller = ColourPickerController(
        NullForegroundAdapter(),
        scheduler=ImmediateTestScheduler(),
        presenter=presenter,
    )
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    _ = panel.selector_widgets

    view_calls = []

    def spy_view(name, view):
        original = view.show_colour

        def record(colour, kind, **kwargs):
            view_calls.append((name, colour, kind))
            return original(colour, kind, **kwargs)

        monkeypatch.setattr(view, "show_colour", record)

    spy_view("readout", panel._readout_panel)
    for mode, widget in panel._selectors.items():
        spy_view(mode.value, widget)

    intent = ColourIntent.from_lch(0.52, 0.04, math.radians(135.0))
    controller.set_preview_colour(intent)

    assert presenter.presented == [intent]
    assert {name for name, _colour, _kind in view_calls} == {
        "readout",
        "lightness_slice",
        "hue_lightness_slice",
        "lightness_chroma_slice",
    }
    assert {id(colour) for _name, colour, _kind in view_calls} == {
        id(presenter.snapshots[0])
    }
    assert {kind for _name, _colour, kind in view_calls} == {ChangeKind.PREVIEW}


def test_active_slice_coordinate_change_refreshes_the_fallback_plane(qtbot):
    # Changing the active slice's fixed coordinate (here lightness, on the
    # fixed-lightness disk) and going out of gamut in the same step must project
    # onto the new plane - not a slice cached from the previous colour.
    controller = ColourPickerController(
        NullForegroundAdapter(),
        scheduler=ImmediateTestScheduler(),
    )
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_SLICE)
    seen = []
    controller.add_colour_listener(lambda snapshot: seen.append(snapshot.colour))
    hue = math.radians(95.0)

    controller.set_preview_colour(ColourIntent.from_lch(0.5, 0.05, hue))    # in gamut at L=0.5
    controller.set_preview_colour(ColourIntent.from_lch(0.95, 0.18, hue))   # new L, out of gamut

    assert not seen[-1].in_gamut
    assert seen[-1].resolved_lch[0] == pytest.approx(0.95, abs=1e-6)


def test_selector_widget_signals_emit_intent_not_presentation(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)
    active = panel.active_selector
    active.resize(120, 80)

    payloads = []
    active.previewed.connect(lambda colour: payloads.append(colour))
    active.committed.connect(lambda colour: payloads.append(colour))
    point = QtCore.QPoint(40, 20)

    send_mouse(active, "press", point)
    send_mouse(active, "release", point)

    assert payloads
    assert all(isinstance(payload, ColourIntent) for payload in payloads if payload is not None)
    assert not any(isinstance(payload, PresentedColour) for payload in payloads)


def test_real_controller_normalized_commit_echo_keeps_emitter_pinned(qtbot):
    # Regression: with the *real* controller the COMMIT broadcast carries
    # normalize_oklab_for_krita(committed). That 8-bit round trip shifts the
    # fixed slice coordinate enough that the rebuilt model no longer compares
    # equal, so a pre-show_colour set_model() used to knock the emitting
    # selector out of PINNED. The emitter must absorb its own normalized echo.
    controller = ColourPickerController(
        NormalizingForegroundAdapter(),
        scheduler=ImmediateTestScheduler(),
    )
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)
    active = panel.active_selector
    active.resize(120, 80)
    click = QtCore.QPoint(20, 10)
    expected = active.model.color_at_position((click.x(), click.y()), (120, 80))
    assert expected is not None

    send_mouse(active, "press", click)
    send_mouse(active, "release", click)

    assert active.indicator_position() == pytest.approx(
        (float(click.x()), float(click.y()))
    )


def test_real_controller_achromatic_hue_lightness_commit_keeps_emitter_pinned(qtbot):
    controller = ColourPickerController(
        NormalizingForegroundAdapter(
            color_math.oklch_to_oklab([0.5, 0.0, 0.0])
        ),
        scheduler=ImmediateTestScheduler(),
    )
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS_SLICE)
    active = panel.active_selector
    active.resize(121, 121)

    click = QtCore.QPoint(60, 20)
    send_mouse(active, "press", click)
    send_mouse(active, "release", click)

    assert active.indicator_position() == pytest.approx((float(click.x()), float(click.y())))


def test_achromatic_hue_slider_moves_hue_lightness_selector_indicator(qtbot):
    grey = color_math.oklch_to_oklab([0.5, 0.0, 0.0])
    controller = FakeController(selected_colour=grey)
    panel = ColourPickerDockPanel(controller)
    panel.resize(360, 320)
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitExposed(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS_SLICE)
    active = panel.active_selector
    active.resize(121, 121)

    click = QtCore.QPoint(60, 20)
    send_mouse(active, "press", click)
    send_mouse(active, "release", click)

    row = panel._readout_panel._row_h
    slider = row.slider
    track = slider.track_rect()
    target_x = track.left() + QtWidgets.QStyle.sliderPositionFromValue(
        slider.minimum(),
        slider.maximum(),
        row._value_to_slider(210.0),
        max(1, track.width() - 1),
    )
    target = QtCore.QPoint(target_x, track.center().y())
    send_mouse(slider, "press", target)
    send_mouse(slider, "release", target)

    hue = math.radians(panel._readout_panel._row_h.value())
    lightness = panel._readout_panel._row_l.value()
    radius = (1.0 - lightness) * 60.0
    assert active.indicator_position() == pytest.approx(
        (
            60.0 + radius * math.cos(hue),
            60.0 - radius * math.sin(hue),
        ),
        abs=1.0,
    )
    assert panel._readout_panel._row_h.value() == pytest.approx(210.0, abs=0.75)


def test_seeded_chromatic_hue_positions_later_achromatic_hue_lightness_indicator(qtbot):
    hue = math.radians(210.0)
    seed = color_math.oklch_to_oklab([0.5, 0.05, hue])
    grey = color_math.oklch_to_oklab([0.5, 0.0, 0.0])
    controller = FakeController(selected_colour=seed)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS_SLICE)
    active = panel.active_selector
    active.resize(121, 121)

    panel.set_selected_colour(grey, committed=False)

    assert panel._readout_panel.hue_intent == pytest.approx(hue)
    assert active.indicator_position() == pytest.approx(
        (
            60.0 + 30.0 * math.cos(hue),
            60.0 - 30.0 * math.sin(hue),
        )
    )


def test_achromatic_hue_intent_survives_chroma_round_trip(qtbot):
    grey = color_math.oklch_to_oklab([0.5, 0.0, 0.0])
    controller = FakeController(selected_colour=grey)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS_SLICE)
    active = panel.active_selector
    active.resize(121, 121)

    hue_degrees = 210.0
    panel._readout_panel._row_h.set_value(hue_degrees)
    panel._readout_panel._row_h.valueChanged.emit(hue_degrees, True)
    panel._readout_panel._row_c.set_value(0.08)
    panel._readout_panel._row_c.valueChanged.emit(0.08, True)
    panel._readout_panel._row_c.set_value(0.0)
    panel._readout_panel._row_c.valueChanged.emit(0.0, True)

    chromatic = controller.commits[-2]
    neutral = controller.commits[-1]
    assert chromatic.hue == pytest.approx(math.radians(hue_degrees))
    assert chromatic.chroma == pytest.approx(0.08)
    assert neutral.hue == pytest.approx(math.radians(hue_degrees))
    assert neutral.chroma == pytest.approx(0.0, abs=1e-6)
    assert panel._readout_panel.hue_intent == pytest.approx(math.radians(hue_degrees))
    assert active.indicator_position() == pytest.approx(
        (
            60.0 + 30.0 * math.cos(math.radians(hue_degrees)),
            60.0 - 30.0 * math.sin(math.radians(hue_degrees)),
        ),
        abs=1.0,
    )


def test_preview_replaces_only_models_whose_fixed_coordinate_changes(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    for mode in SelectorMode:
        panel.selector_for_mode(mode)

    hue = 1.25
    panel.set_selected_colour(color_math.oklch_to_oklab([0.40, 0.06, hue]), committed=False)
    first_models = {
        mode: panel.selector_for_mode(mode).model
        for mode in SelectorMode
    }

    panel.set_selected_colour(color_math.oklch_to_oklab([0.65, 0.11, hue]), committed=False)

    assert panel.selector_for_mode(SelectorMode.LIGHTNESS_SLICE).model is not first_models[
        SelectorMode.LIGHTNESS_SLICE
    ]
    assert panel.selector_for_mode(SelectorMode.HUE_LIGHTNESS_SLICE).model is not first_models[
        SelectorMode.HUE_LIGHTNESS_SLICE
    ]
    assert panel.selector_for_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE).model is first_models[
        SelectorMode.LIGHTNESS_CHROMA_SLICE
    ]


def test_achromatic_hue_change_rebuilds_lightness_chroma_slice(qtbot):
    # Bug: at the L/C tab with chroma=0, changing hue (via the shared H
    # slider) had no effect on the slice because the cache coordinate
    # collapsed achromatic hue to 0. The slice is parameterized by hue,
    # so a hue change must rebuild the model regardless of chroma.
    grey = color_math.oklch_to_oklab([0.5, 0.0, 0.0])
    controller = FakeController(selected_colour=grey)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)
    active = panel.active_selector
    initial_hue = active.model.hue

    new_hue = math.radians(210.0)
    controller.emit_foreground(ColourIntent.from_lch(0.5, 0.0, new_hue))

    assert new_hue != pytest.approx(initial_hue)
    assert active.model.hue == pytest.approx(new_hue)


def test_external_foreground_sync_updates_all_selector_views(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.resize(120, 80)
    colour = color_math.oklch_to_oklab([0.42, 0.06, math.pi / 4.0])

    controller.emit_foreground(colour)

    for widget in panel.selector_widgets:
        np.testing.assert_allclose(widget.selected_colour, colour)
        assert widget.indicator_position() is not None


def test_lazy_selector_uses_latest_colour_when_first_built(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    colour = color_math.oklch_to_oklab([0.42, 0.06, math.pi / 4.0])

    panel.set_selected_colour(colour)
    assert SelectorMode.LIGHTNESS_CHROMA_SLICE not in panel._selectors

    widget = panel.selector_for_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)

    np.testing.assert_allclose(widget.selected_colour, colour)
    assert widget.indicator_position() is not None


@pytest.mark.parametrize("source", ["selector", "readout"])
def test_commit_signal_preserves_previous_swatch(qtbot, source):
    colour_a = color_math.oklch_to_oklab([0.50, 0.05, math.pi / 6.0])
    colour_b = color_math.oklch_to_oklab([0.60, 0.08, math.pi / 3.0])
    controller = FakeController(selected_colour=colour_a)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    emitter = panel.active_selector if source == "selector" else panel._readout_panel
    if source == "selector":
        emitter.previewed.emit(np.asarray(colour_b, dtype=float).copy())
    emitter.committed.emit(np.asarray(colour_b, dtype=float).copy())
    _assert_readout_matches(panel, colour_b)

    panel._readout_panel._swatch.revert_clicked.emit()

    np.testing.assert_allclose(controller.commits[-1].paint_oklab, colour_a, atol=1e-6)


def test_first_open_seeds_readout_revert_target_to_real_foreground(qtbot):
    from oklab_colour_picker.app.controller import ColourPickerController

    external = np.array([0.4, -0.03, 0.07])

    class _Adapter:
        def get_foreground(self):
            return external

        def set_foreground(self, oklab):
            return None

    panel = ColourPickerDockPanel(ColourPickerController(_Adapter()))
    qtbot.addWidget(panel)

    _assert_readout_matches(panel, external)
    assert panel._readout_panel._swatch._revert_button.isEnabled()
    assert panel._readout_panel._swatch._hex_edit.text() in (
        panel._readout_panel._swatch._revert_button.toolTip()
    )


def test_cold_start_poll_seeds_readout_revert_target_not_placeholder(qtbot):
    from oklab_colour_picker.app.controller import ColourPickerController

    external = np.array([0.4, -0.03, 0.07])

    class _Adapter:
        def __init__(self):
            self.foreground = None

        def get_foreground(self):
            return self.foreground

        def set_foreground(self, oklab):
            return None

    class _Timer:
        def __init__(self):
            self._callback = None

        def start(self, callback):
            self._callback = callback

        def stop(self):
            self._callback = None

        def tick(self):
            if self._callback is not None:
                self._callback()

    adapter = _Adapter()
    timer = _Timer()
    panel = ColourPickerDockPanel(
        ColourPickerController(adapter, foreground_timer=timer)
    )
    qtbot.addWidget(panel)

    adapter.foreground = external
    timer.tick()

    _assert_readout_matches(panel, external)
    assert panel._readout_panel._swatch._revert_button.isEnabled()
    assert panel._readout_panel._swatch._hex_edit.text() in (
        panel._readout_panel._swatch._revert_button.toolTip()
    )


def test_show_event_forces_a_foreground_sync_independent_of_the_poll(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    sync_count = controller.sync_count

    panel.show()
    qtbot.waitUntil(lambda: controller.sync_count > sync_count, timeout=1000)

    assert controller.last_force_sync is True


def test_indicator_survives_slider_commit_on_active_selector(qtbot):
    controller = ColourPickerController(
        NormalizingForegroundAdapter(
            color_math.oklch_to_oklab([0.5, 0.05, math.radians(120.0)])
        ),
        scheduler=ImmediateTestScheduler(),
    )
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_SLICE)
    panel.active_selector.resize(121, 121)

    row_l = panel._readout_panel._row_l
    row_l.set_value(0.6)
    row_l.valueChanged.emit(0.6, True)

    widget = panel.active_selector
    assert widget.indicator_position() is not None


def test_achromatic_hue_lightness_pick_carries_click_hue_to_controller(qtbot):
    seed = ColourIntent.from_lch(0.5, 0.0, math.radians(10.0))
    controller = FakeController(selected_colour=seed)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS_SLICE)
    active = panel.active_selector
    active.resize(121, 121)

    click = QtCore.QPoint(34, 90)
    expected_hue = math.atan2(60.0 - click.y(), click.x() - 60.0)
    send_mouse(active, "press", click)
    send_mouse(active, "release", click)

    assert controller.commits
    committed = controller.commits[-1]
    assert committed.chroma == pytest.approx(0.0, abs=1e-6)
    assert committed.hue == pytest.approx(expected_hue % math.tau, abs=1e-6)


def _assert_readout_matches(panel: ColourPickerDockPanel, colour) -> None:
    lightness, chroma, hue = ColourIntent.from_value(colour).selector_lch
    readout = panel._readout_panel
    assert readout._row_l.value() == pytest.approx(lightness, abs=1e-3)
    assert readout._row_c.value() == pytest.approx(chroma, abs=1e-3)
    assert readout._row_h.value() == pytest.approx(math.degrees(hue), abs=0.1)


class FakeController:
    def __init__(self, selected_colour=None):
        self.previews = []
        self.commits = []
        self._foreground_listeners = []
        self._selected_intent = None if selected_colour is None else ColourIntent.from_value(selected_colour)
        self.sync_count = 0
        self._presenter = default_colour_presenter()
        self._fallback_strategy_provider = None

    @property
    def selected_colour(self):
        return None if self._selected_intent is None else self._selected_intent.paint_oklab

    @property
    def selected_intent(self):
        return self._selected_intent

    def set_preview_colour(self, colour):
        intent = None if colour is None else self._intent_from_value(colour)
        self.previews.append(intent)
        if colour is not None:
            self._broadcast(intent, ChangeKind.PREVIEW)

    def request_foreground_commit(self, colour):
        intent = None if colour is None else self._intent_from_value(colour)
        self.commits.append(intent)
        if colour is not None:
            self._broadcast(intent, ChangeKind.COMMIT)

    def sync_external_foreground(self, *, force=False):
        self.sync_count += 1
        self.last_force_sync = bool(force)
        return False

    def add_colour_listener(self, listener):
        self._foreground_listeners.append(listener)
        if self._selected_intent is not None:
            listener(ColourSnapshot(self._present(self._selected_intent), ChangeKind.INITIAL))

    def remove_colour_listener(self, listener):
        self._foreground_listeners.remove(listener)

    def set_fallback_strategy_provider(self, provider):
        self._fallback_strategy_provider = provider

    def reproject(self):
        if self._selected_intent is not None:
            self._broadcast(self._selected_intent, ChangeKind.PREVIEW)

    def _present(self, intent):
        if self._fallback_strategy_provider is None:
            return self._presenter.present(intent)
        return self._presenter.with_fallback_strategy(
            self._fallback_strategy_provider(intent)
        ).present(intent)

    def _broadcast(self, colour, kind):
        self._selected_intent = self._intent_from_value(colour)
        snapshot = ColourSnapshot(self._present(self._selected_intent), kind)
        for listener in list(self._foreground_listeners):
            listener(snapshot)

    def emit_foreground(self, colour):
        self._broadcast(colour, ChangeKind.EXTERNAL)

    def _intent_from_value(self, colour):
        fallback_hue = 0.0 if self._selected_intent is None else self._selected_intent.hue
        return ColourIntent.from_value(colour, achromatic_hue=fallback_hue)


class SpyPresenter:
    def __init__(self):
        self.presented = []
        self.snapshots = []

    def with_fallback_strategy(self, _strategy):
        return self

    def present(self, colour, *, achromatic_hue=0.0):
        intent = ColourIntent.from_value(colour, achromatic_hue=achromatic_hue)
        snapshot = presented_colour(intent, srgb8=(101, 102, 103))
        self.presented.append(intent)
        self.snapshots.append(snapshot)
        return snapshot


class NullForegroundAdapter:
    def set_foreground(self, oklab):
        return None

    def get_foreground(self):
        return None


class ImmediateTestScheduler:
    def call_soon(self, callback):
        callback()


class NormalizingForegroundAdapter:
    def __init__(self, foreground=None):
        self.foreground = foreground

    def set_foreground(self, oklab):
        self.foreground = normalize_oklab_for_krita(oklab)
        return self.foreground.copy()

    def get_foreground(self):
        return None if self.foreground is None else self.foreground.copy()
