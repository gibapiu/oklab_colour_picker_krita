import math
import sys

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui, QtWidgets

import oklab_colour_picker
from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_presentation import PresentedColour, default_colour_presenter
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.app.controller import ChangeKind, ColourPickerController, ColourSnapshot
from oklab_colour_picker.ui.dock import ColourPickerDockPanel, SelectorMode
from oklab_colour_picker.plugin import (
    DOCK_FACTORY_ID,
    DOCK_TITLE,
    create_dock_widget_class,
    register_plugin,
)
from oklab_colour_picker.ui.selectors import HueLightnessSliceDiskWidget
import oklab_colour_picker.ui.dock as dock_module
import oklab_colour_picker.plugin as plugin_module
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


def test_dock_panel_initializes_only_active_selector_model(qtbot, monkeypatch):
    controller = FakeController()
    original = dock_module._model_for_oklch
    model_calls = []

    def counted_model_for_oklch(mode, oklch):
        model_calls.append(mode)
        return original(mode, oklch)

    monkeypatch.setattr(dock_module, "_model_for_oklch", counted_model_for_oklch)

    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    assert model_calls == [SelectorMode.LIGHTNESS_SLICE]


def test_dock_panel_uses_current_foreground_on_construction(qtbot):
    colour = color_math.oklch_to_oklab([0.58, 0.07, math.pi / 3.0])
    controller = FakeController(selected_colour=colour)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    for widget in panel.selector_widgets:
        np.testing.assert_allclose(widget.selected_colour, colour)
    np.testing.assert_allclose(panel._readout_panel._current_oklab, colour)


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

    _send_mouse(active, QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(active, QtCore.QEvent.MouseButtonRelease, point, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert payloads
    assert all(isinstance(payload, ColourIntent) for payload in payloads if payload is not None)
    assert not any(isinstance(payload, PresentedColour) for payload in payloads)


def test_real_controller_normalized_commit_echo_keeps_emitter_pinned(qtbot):
    # Regression: with the *real* controller the COMMIT broadcast carries
    # normalize_oklab_for_krita(committed). That 8-bit round trip shifts the
    # fixed slice coordinate enough that the rebuilt model no longer compares
    # equal, so a pre-show_colour set_model() used to knock the emitting
    # selector out of PINNED. The emitter must absorb its own normalized echo.
    from oklab_colour_picker.app.controller import ColourPickerController, normalize_oklab_for_krita

    class NormalizingAdapter:
        def __init__(self):
            self.foreground = None

        def set_foreground(self, oklab):
            self.foreground = normalize_oklab_for_krita(oklab)
            return self.foreground.copy()

        def get_foreground(self):
            return None if self.foreground is None else self.foreground.copy()

    class ImmediateScheduler:
        def call_soon(self, callback):
            callback()

    controller = ColourPickerController(NormalizingAdapter(), scheduler=ImmediateScheduler())
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)
    active = panel.active_selector
    active.resize(120, 80)
    click = QtCore.QPoint(20, 10)
    expected = active.model.color_at_position((click.x(), click.y()), (120, 80))
    assert expected is not None

    press = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    )
    release = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.LeftButton,
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
    )
    QtCore.QCoreApplication.sendEvent(active, press)
    QtCore.QCoreApplication.sendEvent(active, release)

    assert active.state == "PINNED"
    assert active.anchor == pytest.approx((float(click.x()), float(click.y())))
    assert active.indicator_position() == pytest.approx(
        (float(click.x()), float(click.y()))
    )


def test_click_on_achromatic_hue_lightness_slice_keeps_indicator_at_click(qtbot):
    # The dock loops set_selected_colour back to the source widget after
    # every previewed/committed signal. On a chroma=0 hue/lightness disk the
    # picked OKLab is greyscale, so model.position_for_intent cannot recover
    # the click angle; the indicator must still report the click point
    # rather than snapping to the model's hue=0 fallback.
    grey = color_math.oklch_to_oklab([0.5, 0.0, 0.0])
    controller = FakeController(selected_colour=grey)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS_SLICE)
    active = panel.active_selector
    active.resize(121, 121)

    click = QtCore.QPoint(60, 20)
    expected_colour = active.model.color_at_position(
        (click.x(), click.y()), (active.width(), active.height())
    )
    assert expected_colour is not None

    press = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    )
    release = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.LeftButton,
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
    )
    QtCore.QCoreApplication.sendEvent(active, press)
    QtCore.QCoreApplication.sendEvent(active, release)

    indicator = active.indicator_position()
    assert indicator is not None
    assert indicator == pytest.approx((float(click.x()), float(click.y())))


def test_real_controller_achromatic_hue_lightness_commit_keeps_emitter_pinned(qtbot):
    from oklab_colour_picker.app.controller import ColourPickerController, normalize_oklab_for_krita

    class NormalizingAdapter:
        def __init__(self):
            self.foreground = color_math.oklch_to_oklab([0.5, 0.0, 0.0])

        def set_foreground(self, oklab):
            self.foreground = normalize_oklab_for_krita(oklab)
            return self.foreground.copy()

        def get_foreground(self):
            return self.foreground.copy()

    class ImmediateScheduler:
        def call_soon(self, callback):
            callback()

    controller = ColourPickerController(NormalizingAdapter(), scheduler=ImmediateScheduler())
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS_SLICE)
    active = panel.active_selector
    active.resize(121, 121)

    click = QtCore.QPoint(60, 20)
    _send_mouse(active, QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(active, QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert active.state == "PINNED"
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
    _send_mouse(active, QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(active, QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

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
    _send_mouse(slider, QtCore.QEvent.MouseButtonPress, target, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(slider, QtCore.QEvent.MouseButtonRelease, target, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    hue = math.radians(panel._readout_panel._row_h.value())
    lightness = panel._readout_panel._row_l.value()
    radius = (1.0 - lightness) * 60.0
    assert active.state == "IDLE"
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


def test_preview_reuses_equal_selector_models(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)
    active = panel.active_selector
    active.resize(120, 80)

    colour = active.model.color_at_position((40, 20), (active.width(), active.height()))
    assert colour is not None
    panel.set_selected_colour(colour)
    models = {mode: panel.selector_for_mode(mode).model for mode in SelectorMode}

    active.previewed.emit(colour.copy())

    for mode, model in models.items():
        assert panel.selector_for_mode(mode).model is model


def test_preview_skips_slice_model_rebuild_when_fixed_coordinate_is_unchanged(
    qtbot, monkeypatch
):
    controller = FakeController()
    original = dock_module._model_for_oklch
    model_calls = []

    def counted_model_for_oklch(mode, oklch):
        model_calls.append(mode)
        return original(mode, oklch)

    monkeypatch.setattr(dock_module, "_model_for_oklch", counted_model_for_oklch)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    for mode in SelectorMode:
        panel.selector_for_mode(mode)

    model_calls.clear()
    hue = 1.25
    panel.set_selected_colour(color_math.oklch_to_oklab([0.40, 0.06, hue]), committed=False)
    panel.set_selected_colour(color_math.oklch_to_oklab([0.65, 0.11, hue]), committed=False)

    assert model_calls.count(SelectorMode.LIGHTNESS_SLICE) == 2
    assert model_calls.count(SelectorMode.HUE_LIGHTNESS_SLICE) == 2
    assert model_calls.count(SelectorMode.LIGHTNESS_CHROMA_SLICE) == 1


def test_drag_rebuilds_only_background_models_whose_fixed_coordinate_changes(
    qtbot, monkeypatch
):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    for mode in SelectorMode:
        panel.selector_for_mode(mode)
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)
    active = panel.active_selector
    active.resize(120, 80)
    initial_coordinates = {
        mode: entry.coordinate
        for mode, entry in panel._selector_model_cache.items()
    }

    original = dock_module._model_for_oklch
    model_calls = []

    def counted_model_for_oklch(mode, oklch):
        model_calls.append(mode)
        return original(mode, oklch)

    monkeypatch.setattr(dock_module, "_model_for_oklch", counted_model_for_oklch)

    press = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPoint(20, 20),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    move = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPoint(20, 40),
        QtCore.Qt.NoButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    QtCore.QCoreApplication.sendEvent(active, press)
    QtCore.QCoreApplication.sendEvent(active, move)

    assert active.state == "DRAGGING"
    assert len(controller.previews) == 2
    expected_counts = _expected_rebuild_counts(initial_coordinates, controller.previews)
    assert model_calls.count(SelectorMode.LIGHTNESS_SLICE) == expected_counts[SelectorMode.LIGHTNESS_SLICE]
    assert model_calls.count(SelectorMode.HUE_LIGHTNESS_SLICE) == expected_counts[SelectorMode.HUE_LIGHTNESS_SLICE]
    assert SelectorMode.LIGHTNESS_CHROMA_SLICE not in model_calls


def test_slice_model_cache_returns_same_instance_for_same_fixed_coordinate(qtbot):
    panel = ColourPickerDockPanel(FakeController())
    qtbot.addWidget(panel)
    hue = 1.25

    first = panel._cached_model_for_colour(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        color_math.oklch_to_oklab([0.40, 0.06, hue]),
    )
    second = panel._cached_model_for_colour(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        color_math.oklch_to_oklab([0.65, 0.11, hue]),
    )

    assert second is first


def test_slice_model_cache_replaces_entry_when_fixed_coordinate_changes(qtbot):
    panel = ColourPickerDockPanel(FakeController())
    qtbot.addWidget(panel)

    first = panel._cached_model_for_colour(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        color_math.oklch_to_oklab([0.50, 0.06, 0.25]),
    )
    second = panel._cached_model_for_colour(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        color_math.oklch_to_oklab([0.50, 0.06, 0.75]),
    )

    assert second is not first


def test_slice_model_cache_treats_hue_seam_as_same_slice(qtbot):
    panel = ColourPickerDockPanel(FakeController())
    qtbot.addWidget(panel)

    first = panel._cached_model_for_colour(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        color_math.oklch_to_oklab([0.50, 0.06, 1e-12]),
    )
    second = panel._cached_model_for_colour(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        color_math.oklch_to_oklab([0.50, 0.06, math.tau - 1e-12]),
    )

    assert second is first


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


def test_slice_model_cache_rebuilds_when_achromatic_hue_changes(qtbot):
    panel = ColourPickerDockPanel(FakeController())
    qtbot.addWidget(panel)

    first = panel._cached_model_for_colour(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        ColourIntent.from_lch(0.40, 0.0, 1.25),
    )
    second = panel._cached_model_for_colour(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        ColourIntent.from_lch(0.40, 0.0, 2.50),
    )

    assert second is not first
    assert first.hue == pytest.approx(1.25)
    assert second.hue == pytest.approx(2.50)


def test_slice_model_cache_uses_explicit_achromatic_hue_intent(qtbot):
    panel = ColourPickerDockPanel(FakeController())
    qtbot.addWidget(panel)

    hue = 1.25
    first = panel._cached_model_for_colour(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        ColourIntent.from_lch(0.40, 0.0, hue),
    )
    second = panel._cached_model_for_colour(
        SelectorMode.LIGHTNESS_CHROMA_SLICE,
        ColourIntent.from_lch(0.70, 0.0, hue),
    )

    assert second is first
    assert first.hue == pytest.approx(hue)


def test_chroma_slice_coordinate_matches_sub_threshold_achromatic_values():
    hue = math.radians(210.0)
    first = dock_module.ChromaSliceCoordinate(0.0, hue)
    second = dock_module.ChromaSliceCoordinate(color_math.ACHROMATIC_CHROMA_EPSILON * 0.9, hue)

    assert first.equivalent_to(second)


@pytest.mark.parametrize(
    "mode,first,second",
    [
        (
            SelectorMode.LIGHTNESS_SLICE,
            [0.42, 0.03, 0.20],
            [0.42, 0.11, 2.40],
        ),
        (
            SelectorMode.HUE_LIGHTNESS_SLICE,
            [0.35, 0.07, 0.20],
            [0.78, 0.07, 2.40],
        ),
        (
            SelectorMode.LIGHTNESS_CHROMA_SLICE,
            [0.35, 0.07, 1.25],
            [0.78, 0.13, 1.25],
        ),
    ],
)
def test_slice_model_specs_depend_only_on_their_fixed_coordinate(mode, first, second):
    first_model = dock_module._model_for_oklch(mode, tuple(first))
    second_model = dock_module._model_for_oklch(mode, tuple(second))

    assert second_model == first_model


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


def test_indicator_maps_to_same_colour_after_resize(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)
    widget = panel.active_selector
    widget.resize(80, 40)
    colour = widget.model.color_at_position((30, 12), (80, 40))
    assert colour is not None
    widget.set_selected_colour(_present(colour))

    widget.resize(160, 90)
    actual = widget.indicator_position()
    expected = widget.model.position_for_intent(color_math.oklab_to_oklch(colour), (160, 90))

    assert actual is not None
    assert expected is not None
    assert actual == pytest.approx(expected, abs=1.0)


def test_selector_preview_then_commit_preserves_previous_swatch(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    colour_a = color_math.oklch_to_oklab([0.50, 0.05, math.pi / 6.0])
    colour_b = color_math.oklch_to_oklab([0.60, 0.08, math.pi / 3.0])
    panel.set_selected_colour(colour_a, committed=True)
    panel._readout_panel.set_previous_colour(_present(colour_a))

    selector = panel.active_selector
    selector.previewed.emit(np.asarray(colour_b, dtype=float).copy())
    selector.committed.emit(np.asarray(colour_b, dtype=float).copy())

    np.testing.assert_allclose(panel._readout_panel._previous_oklab, colour_a, atol=1e-6)
    np.testing.assert_allclose(panel._readout_panel._current_oklab, colour_b, atol=1e-6)


def test_readout_commit_signal_preserves_previous_swatch(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    colour_a = color_math.oklch_to_oklab([0.50, 0.05, math.pi / 6.0])
    colour_b = color_math.oklch_to_oklab([0.60, 0.08, math.pi / 3.0])
    panel.set_selected_colour(colour_a, committed=True)
    panel._readout_panel.set_previous_colour(_present(colour_a))

    panel._readout_panel.committed.emit(np.asarray(colour_b, dtype=float).copy())

    np.testing.assert_allclose(panel._readout_panel._previous_oklab, colour_a, atol=1e-6)
    np.testing.assert_allclose(panel._readout_panel._current_oklab, colour_b, atol=1e-6)


def test_plugin_registers_krita_dock_factory():
    app = FakeKritaApp()
    api = FakeKritaApi(app)

    assert register_plugin(krita_instance=app, api=api) is True

    assert len(app.factories) == 1
    factory = app.factories[0]
    assert factory.identifier == DOCK_FACTORY_ID
    assert factory.area == FakeDockWidgetFactoryBase.DockRight


def test_vendor_site_packages_are_added_before_runtime_imports(tmp_path, monkeypatch):
    vendor_dir = tmp_path / plugin_module.VENDOR_ROOT_DIRECTORY_NAME / plugin_module.VENDOR_SITE_PACKAGES_DIRECTORY_NAME
    vendor_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))

    plugin_module._add_vendor_site_packages(str(tmp_path))

    assert sys.path[0] == str(vendor_dir)


def test_vendor_site_packages_fall_back_next_to_plugin_package(tmp_path, monkeypatch):
    package_dir = tmp_path / "pykrita" / "oklab_colour_picker"
    package_dir.mkdir(parents=True)
    expected = package_dir / plugin_module.VENDOR_SITE_PACKAGES_DIRECTORY_NAME
    monkeypatch.setattr(plugin_module, "__file__", str(package_dir / "plugin.py"))

    assert plugin_module._vendor_site_packages_path() == str(expected)


def test_created_krita_dock_builds_panel(qtbot):
    controller = FakeController()
    dock_class = create_dock_widget_class(FakeDockWidget, controller_factory=lambda: controller)
    dock = dock_class()
    qtbot.addWidget(dock)

    assert dock.windowTitle() == DOCK_TITLE
    assert isinstance(dock.widget(), ColourPickerDockPanel)


def test_created_krita_dock_syncs_foreground_on_canvas_change(qtbot):
    controller = FakeController()
    dock_class = create_dock_widget_class(FakeDockWidget, controller_factory=lambda: controller)
    dock = dock_class()
    qtbot.addWidget(dock)
    sync_count = controller.sync_count

    dock.canvasChanged(object())

    assert controller.sync_count == sync_count + 1
    assert controller.last_force_sync is True


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

    # The subscribe-time pull acquires the real foreground; the INITIAL
    # replay must adopt it as both current and the revert baseline, never
    # leave the DEFAULT_COLOUR placeholder as the previous colour.
    np.testing.assert_allclose(panel._readout_panel._current_oklab, external, atol=1e-6)
    np.testing.assert_allclose(panel._readout_panel._previous_oklab, external, atol=1e-6)


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

    np.testing.assert_allclose(panel._readout_panel._current_oklab, external, atol=1e-6)
    np.testing.assert_allclose(panel._readout_panel._previous_oklab, external, atol=1e-6)


def test_show_event_forces_a_foreground_sync_independent_of_the_poll(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    sync_count = controller.sync_count

    panel.show()
    qtbot.waitUntil(lambda: controller.sync_count > sync_count, timeout=1000)

    assert controller.last_force_sync is True


def test_qt_foreground_timer_runs_at_fixed_interval(qtbot):
    from oklab_colour_picker.infrastructure.krita_adapter import FOREGROUND_POLL_INTERVAL_MS, QtForegroundTimer

    timer = QtForegroundTimer()
    timer.start(lambda: None)
    try:
        assert timer._timer.isActive()
        assert timer._timer.interval() == FOREGROUND_POLL_INTERVAL_MS
    finally:
        timer.stop()

    assert not timer._timer.isActive()


def test_dock_shows_friendly_message_when_numpy_is_missing(qtbot, monkeypatch):
    import types

    fake_dock = types.ModuleType("oklab_colour_picker.ui.dock")

    def _raise_numpy_missing(_name):
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    fake_dock.__getattr__ = _raise_numpy_missing
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.ui.dock", fake_dock)

    dock_class = create_dock_widget_class(FakeDockWidget)
    dock = dock_class()
    qtbot.addWidget(dock)

    widget = dock.widget()
    assert isinstance(widget, QtWidgets.QWidget)
    assert widget.objectName() == "oklab-missing-dependency"
    assert "numpy" in widget.findChild(QtWidgets.QLabel).text().lower()
    assert widget.findChild(QtWidgets.QPushButton, "oklab-install-numpy").text() == "Install NumPy"


def test_dock_shows_installer_when_numpy_binary_is_incompatible(qtbot, monkeypatch):
    import types

    fake_dock = types.ModuleType("oklab_colour_picker.ui.dock")

    def _raise_numpy_incompatible(_name):
        raise ImportError("Importing the numpy C-extensions failed: cpython-314 is incompatible")

    fake_dock.__getattr__ = _raise_numpy_incompatible
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.ui.dock", fake_dock)

    dock_class = create_dock_widget_class(FakeDockWidget)
    dock = dock_class()
    qtbot.addWidget(dock)

    widget = dock.widget()
    assert widget.objectName() == "oklab-missing-dependency"
    assert "numpy" in widget.findChild(QtWidgets.QLabel).text().lower()
    assert widget.findChild(QtWidgets.QPushButton, "oklab-install-numpy").text() == "Install NumPy"


def test_install_numpy_action_requires_confirmation(qtbot, monkeypatch, tmp_path):
    import types

    fake_dock = types.ModuleType("oklab_colour_picker.ui.dock")

    def _raise_numpy_missing(_name):
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    fake_dock.__getattr__ = _raise_numpy_missing
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.ui.dock", fake_dock)
    captured_messages = []

    def reject_install(parent, title, message, *args, **kwargs):
        captured_messages.append(message)
        return QtWidgets.QMessageBox.No

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", reject_install)
    installer_calls = []

    dock_class = create_dock_widget_class(
        FakeDockWidget,
        app_data_location=str(tmp_path),
        dependency_installer=lambda vendor_path: installer_calls.append(vendor_path),
    )
    dock = dock_class()
    qtbot.addWidget(dock)

    button = dock.widget().findChild(QtWidgets.QPushButton, "oklab-install-numpy")
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)

    assert installer_calls == []
    expected_vendor = str(tmp_path / plugin_module.VENDOR_ROOT_DIRECTORY_NAME / plugin_module.VENDOR_SITE_PACKAGES_DIRECTORY_NAME)
    assert captured_messages
    assert expected_vendor not in captured_messages[0]
    assert "private dependency folder" in captured_messages[0]


def test_install_numpy_action_runs_installer_when_confirmed(qtbot, monkeypatch, tmp_path):
    import types

    from oklab_colour_picker.infrastructure.dependency_bootstrap import InstallResult

    fake_dock = types.ModuleType("oklab_colour_picker.ui.dock")

    def _raise_numpy_missing(_name):
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    fake_dock.__getattr__ = _raise_numpy_missing
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.ui.dock", fake_dock)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: QtWidgets.QMessageBox.Yes)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: QtWidgets.QMessageBox.Ok)
    installer_calls = []

    def fake_installer(vendor_path):
        installer_calls.append(vendor_path)
        return InstallResult(True, "NumPy installed.")

    dock_class = create_dock_widget_class(
        FakeDockWidget,
        app_data_location=str(tmp_path),
        dependency_installer=fake_installer,
    )
    dock = dock_class()
    qtbot.addWidget(dock)

    status = dock.widget().findChild(QtWidgets.QLabel, "oklab-install-status")
    button = dock.widget().findChild(QtWidgets.QPushButton, "oklab-install-numpy")
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(lambda: bool(installer_calls) and "installed" in status.text().lower(), timeout=5000)
    expected_vendor = str(tmp_path / plugin_module.VENDOR_ROOT_DIRECTORY_NAME / plugin_module.VENDOR_SITE_PACKAGES_DIRECTORY_NAME)
    assert installer_calls == [expected_vendor]
    assert button.isEnabled()


def test_install_numpy_action_reports_installer_exception(qtbot, monkeypatch, tmp_path):
    import types

    fake_dock = types.ModuleType("oklab_colour_picker.ui.dock")

    def _raise_numpy_missing(_name):
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    fake_dock.__getattr__ = _raise_numpy_missing
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.ui.dock", fake_dock)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: QtWidgets.QMessageBox.Yes)
    captured = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda parent, title, message, *args, **kwargs: captured.append(message) or QtWidgets.QMessageBox.Ok,
    )

    def boom(_vendor_path):
        raise RuntimeError("network is down")

    dock_class = create_dock_widget_class(
        FakeDockWidget,
        app_data_location=str(tmp_path),
        dependency_installer=boom,
    )
    dock = dock_class()
    qtbot.addWidget(dock)

    button = dock.widget().findChild(QtWidgets.QPushButton, "oklab-install-numpy")
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(lambda: bool(captured), timeout=5000)
    assert "network is down" in captured[0]
    assert button.isEnabled()


def test_dock_propagates_unexpected_import_errors(qtbot, monkeypatch):
    import sys
    import types

    fake_dock = types.ModuleType("oklab_colour_picker.ui.dock")

    def _raise_unknown(_name):
        raise ModuleNotFoundError("No module named 'something_else'", name="something_else")

    fake_dock.__getattr__ = _raise_unknown
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.ui.dock", fake_dock)

    dock_class = create_dock_widget_class(FakeDockWidget)
    with pytest.raises(ModuleNotFoundError):
        dock_class()


def test_package_exports_register_plugin():
    assert oklab_colour_picker.__all__ == ["register_plugin"]
    assert oklab_colour_picker.register_plugin is register_plugin


def test_indicator_survives_commit_broadcast_on_every_selector(qtbot):
    from oklab_colour_picker.app.controller import ColourPickerController

    class NormalizingAdapter:
        def __init__(self):
            self.foreground = None

        def set_foreground(self, oklab):
            from oklab_colour_picker.app.controller import normalize_oklab_for_krita
            self.foreground = normalize_oklab_for_krita(oklab)
            return self.foreground.copy()

        def get_foreground(self):
            return None if self.foreground is None else self.foreground.copy()

    class ImmediateScheduler:
        def call_soon(self, callback):
            callback()

    controller = ColourPickerController(NormalizingAdapter(), scheduler=ImmediateScheduler())
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    for mode in SelectorMode:
        panel.set_mode(mode)
        panel.active_selector.resize(121, 121)

    panel.set_mode(SelectorMode.LIGHTNESS_SLICE)
    active = panel.active_selector
    click = QtCore.QPoint(80, 50)
    picked = active.model.color_at_position((click.x(), click.y()), (121, 121))
    assert picked is not None
    _send_mouse(active, QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(active, QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    for mode in SelectorMode:
        widget = panel.selector_for_mode(mode)
        rings = widget._interaction.indicator(widget).rings
        assert rings, f"{mode.value}: indicator vanished after COMMIT broadcast"


def test_indicator_survives_slider_commit_on_active_selector(qtbot):
    from oklab_colour_picker.app.controller import ColourPickerController

    class NormalizingAdapter:
        def __init__(self):
            self.foreground = color_math.oklch_to_oklab([0.5, 0.05, math.radians(120.0)])

        def set_foreground(self, oklab):
            from oklab_colour_picker.app.controller import normalize_oklab_for_krita
            self.foreground = normalize_oklab_for_krita(oklab)
            return self.foreground.copy()

        def get_foreground(self):
            return self.foreground.copy()

    class ImmediateScheduler:
        def call_soon(self, callback):
            callback()

    controller = ColourPickerController(NormalizingAdapter(), scheduler=ImmediateScheduler())
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_SLICE)
    panel.active_selector.resize(121, 121)

    row_l = panel._readout_panel._row_l
    row_l.set_value(0.6)
    row_l.valueChanged.emit(0.6, True)

    widget = panel.active_selector
    rings = widget._interaction.indicator(widget).rings
    assert widget.state == "IDLE"
    assert rings


def test_off_leaf_press_keeps_indicator_visible_with_drag_snap(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_SLICE)
    active = panel.active_selector
    active.resize(121, 121)

    oog = ColourIntent.from_lch(0.5, color_math.SRGB_MAX_CHROMA, 0.0)
    controller._broadcast(oog, ChangeKind.PREVIEW)
    assert len(active._interaction.indicator(active).rings) == 1

    off_leaf = QtCore.QPoint(118, 60)
    _send_mouse(active, QtCore.QEvent.MouseButtonPress, off_leaf, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    assert active._interaction.indicator(active).rings

    _send_mouse(active, QtCore.QEvent.MouseButtonRelease, off_leaf, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
    assert active.state == "PINNED"
    assert active._interaction.indicator(active).rings


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
    _send_mouse(active, QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(active, QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert controller.commits
    committed = controller.commits[-1]
    assert committed.chroma == pytest.approx(0.0, abs=1e-6)
    assert committed.hue == pytest.approx(expected_hue % math.tau, abs=1e-6)


def test_absorbed_echo_with_model_swap_preserves_intent_lch(qtbot):
    from oklab_colour_picker.app.controller import normalize_oklab_for_krita
    from oklab_colour_picker.ui.selectors.selector import SelectorWidget
    from oklab_colour_picker.models import HueLightnessSliceModel

    widget = SelectorWidget(HueLightnessSliceModel(chroma=0.05))
    widget.resize(121, 121)
    qtbot.addWidget(widget)
    widget.show()

    click = QtCore.QPoint(60, 30)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
    assert widget.state == "PINNED"

    # Reuse the pinned paint so PINNED.quantized_equal absorbs the echo; the
    # intent's coords still differ from the paint-recovered LCH.
    pinned_paint = widget.selected_colour
    new_model = HueLightnessSliceModel(chroma=0.051)
    intent = ColourIntent.from_lch(0.5, 0.051, 0.0)
    echo_intent = ColourIntent._create(
        (float(pinned_paint[0]), float(pinned_paint[1]), float(pinned_paint[2])),
        intent.lightness,
        intent.chroma,
        intent.hue,
    )
    assert np.array_equal(
        normalize_oklab_for_krita(echo_intent.paint_oklab),
        normalize_oklab_for_krita(pinned_paint),
    )

    widget.show_colour(_present(echo_intent), model_factory=lambda: new_model)

    assert widget.model is new_model
    assert widget.colour is not None
    assert widget.colour.selector_lch == pytest.approx((0.5, 0.051, 0.0))
    assert widget.indicator_position() is not None


def test_invalid_press_on_warm_idle_is_a_no_op(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_SLICE)
    active = panel.active_selector
    active.resize(121, 121)

    in_leaf = ColourIntent.from_lch(0.5, 0.05, 0.0)
    controller._broadcast(in_leaf, ChangeKind.PREVIEW)
    rings_before = active._interaction.indicator(active).rings
    assert rings_before

    class _InvalidPickModel:
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            return getattr(self._inner, name)

        def color_at_position(self, *args, **kwargs):
            return None

        def selection_at_position(self, *args, **kwargs):
            return None

        def snapped_selector_selection_at_position(self, *args, **kwargs):
            return None

    active._model = _InvalidPickModel(active._model)

    _send_mouse(active, QtCore.QEvent.MouseButtonPress, QtCore.QPoint(60, 60), QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    assert active.state == "IDLE"
    assert active._interaction.indicator(active).rings == rings_before


def _send_mouse(widget, event_type, point, button, buttons):
    event = QtGui.QMouseEvent(
        event_type,
        point,
        button,
        buttons,
        QtCore.Qt.NoModifier,
    )
    QtCore.QCoreApplication.sendEvent(widget, event)
    return event


def _expected_rebuild_counts(initial_coordinates, colours):
    coordinates = dict(initial_coordinates)
    counts = {mode: 0 for mode in SelectorMode}
    for colour in colours:
        oklch = ColourIntent.from_value(colour).selector_lch
        for mode in SelectorMode:
            coordinate = dock_module._fixed_slice_coordinate(mode, oklch)
            if coordinates[mode].equivalent_to(coordinate):
                continue
            counts[mode] += 1
            coordinates[mode] = coordinate
    return counts


class FakeController:
    def __init__(self, selected_colour=None):
        self.previews = []
        self.commits = []
        self._foreground_listeners = []
        self._selected_intent = None if selected_colour is None else ColourIntent.from_value(selected_colour)
        self.sync_count = 0

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
            listener(ColourSnapshot(_present(self._selected_intent), ChangeKind.INITIAL))

    def remove_colour_listener(self, listener):
        self._foreground_listeners.remove(listener)

    def _broadcast(self, colour, kind):
        self._selected_intent = self._intent_from_value(colour)
        snapshot = ColourSnapshot(_present(self._selected_intent), kind)
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


class FakeKritaApp:
    def __init__(self):
        self.factories = []

    def addDockWidgetFactory(self, factory):
        self.factories.append(factory)

    def getAppDataLocation(self):
        return "/tmp/fake-krita-app-data"


class FakeDockWidgetFactoryBase:
    DockRight = object()


class FakeDockWidgetFactory:
    def __init__(self, identifier, area, widget_class):
        self.identifier = identifier
        self.area = area
        self.widget_class = widget_class


class FakeKritaApi:
    def __init__(self, app):
        self.Krita = FakeKrita(app)
        self.DockWidget = FakeDockWidget
        self.DockWidgetFactory = FakeDockWidgetFactory
        self.DockWidgetFactoryBase = FakeDockWidgetFactoryBase


class FakeKrita:
    def __init__(self, app):
        self._app = app

    def instance(self):
        return self._app


class FakeDockWidget(QtWidgets.QDockWidget):
    def canvasChanged(self, canvas):
        pass
