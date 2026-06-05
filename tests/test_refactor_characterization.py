import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker import color_math
from oklab_colour_picker.colour_presentation import default_colour_presenter
from oklab_colour_picker.controller import ChangeKind, ColourSnapshot
from oklab_colour_picker.dock import ColourPickerDockPanel, SelectorMode
from oklab_colour_picker.selector_models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)
from oklab_colour_picker.colour_state import ColourIntent
from oklab_colour_picker.widgets import (
    HueLightnessSliceDiskWidget,
    LightnessSliceDiskWidget,
    ReadoutPanel,
    SelectorWidget,
)


def _selector_widget(model):
    return SelectorWidget(model)


def _lightness_widget(model):
    return LightnessSliceDiskWidget(model)


def _hue_lightness_widget(model):
    return HueLightnessSliceDiskWidget(model)


def _readout_panel():
    return ReadoutPanel()


def _present(colour):
    return default_colour_presenter().present(colour)


def _show(panel, colour, kind):
    panel.show_colour(_present(colour), kind)


SELECTOR_CASES = (
    (
        "lightness-slice",
        lambda: _lightness_widget(LightnessSliceModel(lightness=0.55)),
        (101, 101),
    ),
    (
        "hue-lightness-slice",
        lambda: _hue_lightness_widget(HueLightnessSliceModel(chroma=0.03)),
        (101, 101),
    ),
    (
        "lightness-chroma-slice",
        lambda: _selector_widget(LightnessChromaSliceModel(hue=1.0)),
        (101, 81),
    ),
)

SNAP_CASES = (
    (
        "lightness-slice",
        lambda: _lightness_widget(LightnessSliceModel(lightness=0.5)),
        (101, 101),
        QtCore.QPoint(60, 50),
        QtCore.QPoint(100, 50),
    ),
    (
        "hue-lightness-slice",
        lambda: _hue_lightness_widget(HueLightnessSliceModel(chroma=0.2)),
        (101, 101),
        QtCore.QPoint(75, 50),
        QtCore.QPoint(50, 50),
    ),
    (
        "lightness-chroma-slice",
        lambda: _selector_widget(LightnessChromaSliceModel(hue=0.0)),
        (101, 81),
        QtCore.QPoint(8, 40),
        QtCore.QPoint(100, 40),
    ),
)


@pytest.mark.parametrize(("case_name", "factory", "size"), SELECTOR_CASES, ids=[case[0] for case in SELECTOR_CASES])
def test_core_click_commits_once_and_keeps_indicator_at_clicked_pixel(qtbot, case_name, factory, size):
    widget = _shown_selector(qtbot, factory(), size)
    click = _valid_points(widget.model, size, count=1)[0]
    expected = widget.model.color_at_position((click.x(), click.y()), size)
    assert expected is not None

    previews, commits = _capture_selector_signals(widget)

    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert len(previews) == 1
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], expected)
    assert widget.indicator_position() == pytest.approx((float(click.x()), float(click.y())), abs=1.0)


@pytest.mark.parametrize(("case_name", "factory", "size"), SELECTOR_CASES, ids=[case[0] for case in SELECTOR_CASES])
def test_core_press_drag_emits_ordered_previews_and_single_commit(qtbot, case_name, factory, size):
    widget = _shown_selector(qtbot, factory(), size)
    start, middle, end = _valid_points(widget.model, size, count=3)
    expected = widget.model.color_at_position((end.x(), end.y()), size)
    assert expected is not None

    previews, commits = _capture_selector_signals(widget)

    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, start, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseMove, middle, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, end, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert len(previews) >= 2
    assert len(commits) == 1
    np.testing.assert_allclose(previews[0], widget.model.color_at_position((start.x(), start.y()), size))
    np.testing.assert_allclose(previews[-1], widget.model.color_at_position((middle.x(), middle.y()), size))
    np.testing.assert_allclose(commits[0], expected)


@pytest.mark.parametrize(("case_name", "factory", "size"), SELECTOR_CASES, ids=[case[0] for case in SELECTOR_CASES])
def test_core_keyboard_nudge_previews_then_commits_on_release(qtbot, case_name, factory, size):
    widget = _shown_selector(qtbot, factory(), size)
    start = _keyboard_start_point(widget, size)
    colour = widget.model.color_at_position((start.x(), start.y()), size)
    assert colour is not None
    widget.set_selected_colour(_present(colour))
    previews, commits = _capture_selector_signals(widget)

    press = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    release = QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, press)
    QtWidgets.QApplication.sendEvent(widget, release)

    assert press.isAccepted()
    assert release.isAccepted()
    assert len(previews) == 1
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], widget.selected_colour)


@pytest.mark.parametrize(("case_name", "factory", "size", "valid", "invalid"), SNAP_CASES, ids=[case[0] for case in SNAP_CASES])
def test_secondary_drag_leaving_gamut_snaps_preview_and_commits_boundary(qtbot, case_name, factory, size, valid, invalid):
    widget = _shown_selector(qtbot, factory(), size)
    assert widget.model.color_at_position((valid.x(), valid.y()), size) is not None
    assert widget.model.color_at_position((invalid.x(), invalid.y()), size) is None
    expected = widget.model.snapped_color_at_position((invalid.x(), invalid.y()), size)
    assert expected is not None
    widget.set_selected_colour(_present(widget.model.color_at_position((valid.x(), valid.y()), size)))
    previews, commits = _capture_selector_signals(widget)

    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, valid, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseMove, invalid, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, invalid, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert not any(preview is None for preview in previews)
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], expected)


@pytest.mark.parametrize(("case_name", "factory", "size"), SELECTOR_CASES, ids=[case[0] for case in SELECTOR_CASES])
def test_secondary_drag_outside_leaf_snaps_and_commits(qtbot, case_name, factory, size):
    widget = _shown_selector(qtbot, factory(), size)
    previous_point = _valid_points(widget.model, size, count=1)[0]
    previous = widget.model.color_at_position((previous_point.x(), previous_point.y()), size)
    assert previous is not None
    widget.set_selected_colour(_present(previous))
    invalid = _invalid_point(widget.model, size)
    expected = widget.model.snapped_color_at_position((invalid.x(), invalid.y()), size)
    assert expected is not None
    previews, commits = _capture_selector_signals(widget)

    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, invalid, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseMove, invalid, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, invalid, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], expected)
    np.testing.assert_allclose(widget.selected_colour, expected)
    assert previews and previews[0] is not None


def test_core_tab_switch_lazy_selector_uses_latest_colour_and_preserves_selection(qtbot):
    colour = color_math.oklch_to_oklab([0.42, 0.06, math.pi / 4.0])
    controller = FakeController(selected_colour=colour)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    assert SelectorMode.LIGHTNESS_CHROMA_SLICE not in panel._selectors
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)

    for widget in panel.selector_widgets:
        np.testing.assert_allclose(widget.selected_colour, colour)
        assert widget.indicator_position() is not None


def test_core_external_foreground_sync_updates_idle_selectors_and_readout(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    external = color_math.oklch_to_oklab([0.62, 0.08, math.pi / 3.0])

    controller.emit_foreground(external)

    for widget in panel.selector_widgets:
        np.testing.assert_allclose(widget.selected_colour, external)
    np.testing.assert_allclose(panel._readout_panel._current_oklab, external)


def test_core_controller_krita_round_trip_suppresses_self_feedback():
    from oklab_colour_picker.controller import ColourPickerController, normalize_oklab_for_krita

    adapter = FakeAdapter()
    controller = ColourPickerController(adapter, scheduler=ImmediateTestScheduler())
    observed = []
    controller.add_colour_listener(lambda snapshot: observed.append((snapshot.intent, snapshot.kind)))
    committed = np.array([0.65, 0.04, -0.02])

    controller.request_foreground_commit(committed)
    adapter.foreground_colour = normalize_oklab_for_krita(committed)

    assert controller.sync_external_foreground() is False
    # The commit broadcasts COMMIT once; the self-feedback sync that follows
    # must not produce an EXTERNAL echo.
    assert [kind for _colour, kind in observed] == [ChangeKind.COMMIT]
    np.testing.assert_allclose(controller.selected_colour, committed)


def test_secondary_mouse_press_cancels_pending_keyboard_commit(qtbot):
    widget = _shown_selector(qtbot, _selector_widget(LightnessChromaSliceModel(hue=0.0)), (64, 32))
    start = widget.model.color_at_position((20, 10), (64, 32))
    assert start is not None
    widget.set_selected_colour(_present(start))
    _, commits = _capture_selector_signals(widget)

    key_press = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, key_press)
    assert key_press.isAccepted()
    point = QtCore.QPoint(24, 16)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, point, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
    key_release = QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, key_release)

    assert len(commits) == 1


def test_secondary_focus_loss_flushes_pending_keyboard_commit(qtbot):
    widget = _shown_selector(qtbot, _selector_widget(LightnessChromaSliceModel(hue=0.0)), (64, 32))
    start = widget.model.color_at_position((20, 10), (64, 32))
    assert start is not None
    widget.set_selected_colour(_present(start))
    _, commits = _capture_selector_signals(widget)

    key_press = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, key_press)
    assert key_press.isAccepted()
    focus_out = QtGui.QFocusEvent(QtCore.QEvent.FocusOut)
    QtWidgets.QApplication.sendEvent(widget, focus_out)

    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], widget.selected_colour)


def test_secondary_resize_after_commit_drops_absolute_pixel_override(qtbot):
    widget = _shown_selector(qtbot, _selector_widget(LightnessChromaSliceModel(hue=0.0)), (80, 40))
    point = QtCore.QPoint(30, 12)
    colour = widget.model.color_at_position((point.x(), point.y()), (80, 40))
    assert colour is not None

    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, point, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
    widget.resize(160, 90)

    # Characterizes the desired visible behavior, not the legacy resizeEvent
    # override: PR-2 should keep this green after replacing that implementation.
    expected = widget.model.position_for_intent(color_math.oklab_to_oklch(colour), (160, 90))
    assert expected is not None
    assert widget.indicator_position() == pytest.approx(expected, abs=1.0)


def test_secondary_readout_slider_drag_previews_then_commits(qtbot):
    panel = _readout_panel()
    panel.resize(320, 200)
    qtbot.addWidget(panel)
    panel.show()
    _show(panel, color_math.oklch_to_oklab([0.2, 0.05, 0.0]), ChangeKind.COMMIT)
    previews = []
    commits = []
    panel.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))
    panel.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    slider = panel._row_l.slider
    track = slider._track_rect()
    start = QtCore.QPoint(track.left() + int(round(track.width() * 0.25)), track.center().y())
    end = QtCore.QPoint(track.left() + int(round(track.width() * 0.75)), track.center().y())
    _send_mouse(slider, QtCore.QEvent.MouseButtonPress, start, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(slider, QtCore.QEvent.MouseButtonRelease, end, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert previews
    assert len(commits) == 1


def test_edge_off_slice_colour_clears_indicator_without_raising(qtbot):
    widget = _shown_selector(qtbot, _selector_widget(LightnessChromaSliceModel(hue=0.0)), (80, 40))
    off_slice = color_math.oklch_to_oklab([0.5, 0.05, math.pi / 2.0])

    widget.set_selected_colour(_present(off_slice))

    assert widget.selected_colour is not None
    assert widget.indicator_position() is None


@pytest.mark.parametrize(("case_name", "factory"), [(case[0], case[1]) for case in SELECTOR_CASES])
def test_edge_tiny_widget_size_does_not_raise(qtbot, case_name, factory):
    widget = factory()
    widget.setMinimumSize(0, 0)
    widget = _shown_selector(qtbot, widget, (1, 1))
    widget.set_selected_colour(_present(np.array([0.5, 0.0, 0.0])))

    assert widget.indicator_position() is None
    image = QtGui.QImage(QtCore.QSize(1, 1), QtGui.QImage.Format_RGBA8888)
    painter = QtGui.QPainter(image)
    widget.render(painter)
    painter.end()


def test_edge_achromatic_hue_lightness_click_keeps_indicator_at_click(qtbot):
    grey = color_math.oklch_to_oklab([0.5, 0.0, 0.0])
    controller = FakeController(selected_colour=grey)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS_SLICE)
    active = panel.active_selector
    active.resize(121, 121)
    click = QtCore.QPoint(60, 20)

    _send_mouse(active, QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(active, QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert active.indicator_position() == pytest.approx((float(click.x()), float(click.y())))


def test_edge_achromatic_echo_has_no_pinned_idle_pinned_transition(qtbot):
    widget = _shown_selector(qtbot, _hue_lightness_widget(HueLightnessSliceModel(chroma=0.0)), (121, 121))
    click = QtCore.QPoint(60, 20)

    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)
    widget.set_selected_colour(_present(widget.selected_colour))

    # The achromatic echo must be swallowed by PINNED: no PINNED→IDLE→PINNED
    # round-trip and the indicator stays at the click.
    log = widget.transition_log
    assert not any(
        log[i : i + 3] == ("PINNED", "IDLE", "PINNED") for i in range(len(log) - 2)
    )
    assert widget.state == "PINNED"
    assert widget.indicator_position() == pytest.approx((float(click.x()), float(click.y())))


def test_secondary_readout_external_change_during_edit_is_latched_until_cancel(qtbot):
    panel = _readout_panel()
    qtbot.addWidget(panel)
    original = color_math.oklch_to_oklab([0.4, 0.05, 0.0])
    external = color_math.oklch_to_oklab([0.8, 0.02, 1.0])
    _show(panel, original, ChangeKind.COMMIT)

    panel._swatch._enter_edit_mode()
    _show(panel, external, ChangeKind.EXTERNAL)

    np.testing.assert_allclose(panel._current_oklab, original)
    escape = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(panel._swatch._hex_edit, escape)
    np.testing.assert_allclose(panel._current_oklab, external)


def _shown_selector(qtbot, widget, size):
    widget.resize(*size)
    qtbot.addWidget(widget)
    widget.show()
    return widget


def _capture_selector_signals(widget):
    previews = []
    commits = []
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))
    return previews, commits


def _valid_points(model, size, *, count):
    width, height = size
    points = []
    for y in range(height):
        for x in range(width):
            if model.color_at_position((x, y), size) is not None:
                points.append(QtCore.QPoint(x, y))
    assert len(points) >= count
    if count == 1:
        return (points[len(points) // 2],)
    indices = np.linspace(0, len(points) - 1, count, dtype=int)
    return tuple(points[int(index)] for index in indices)


def _keyboard_start_point(widget, size):
    for point in _valid_points(widget.model, size, count=20):
        colour = widget.model.color_at_position((point.x(), point.y()), size)
        widget.set_selected_colour(_present(colour))
        event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
        QtWidgets.QApplication.sendEvent(widget, event)
        if event.isAccepted():
            release = QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
            QtWidgets.QApplication.sendEvent(widget, release)
            widget.resize(widget.width() + 1, widget.height() + 1)
            widget.resize(*size)
            widget.set_selected_colour(_present(colour))
            return point
    raise AssertionError("could not find keyboard-nudgeable point")


def _invalid_point(model, size):
    width, height = size
    for point in (
        QtCore.QPoint(0, 0),
        QtCore.QPoint(width - 1, 0),
        QtCore.QPoint(width - 1, height - 1),
        QtCore.QPoint(0, height - 1),
    ):
        if model.color_at_position((point.x(), point.y()), size) is None:
            return point
    raise AssertionError("could not find invalid point")


def _send_mouse(widget, event_type, position, button, buttons):
    event = QtGui.QMouseEvent(
        event_type,
        QtCore.QPointF(position),
        button,
        buttons,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(widget, event)
    assert event.isAccepted()


class FakeController:
    def __init__(self, *, selected_colour=None):
        self._selected_intent = None if selected_colour is None else ColourIntent.from_value(selected_colour)
        self.previews = []
        self.commits = []
        self._foreground_listeners = []

    @property
    def selected_colour(self):
        return None if self._selected_intent is None else self._selected_intent.paint_oklab

    @property
    def selected_intent(self):
        return self._selected_intent

    def set_preview_colour(self, colour):
        self.previews.append(_paint_of(colour))
        if colour is not None:
            self._broadcast(colour, ChangeKind.PREVIEW)

    def request_foreground_commit(self, colour):
        self.commits.append(_paint_of(colour))
        if colour is not None:
            self._broadcast(colour, ChangeKind.COMMIT)

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


class ImmediateTestScheduler:
    def call_soon(self, callback):
        callback()


class FakeAdapter:
    def __init__(self):
        self.foreground_colour = None

    def set_foreground(self, oklab):
        from oklab_colour_picker.controller import normalize_oklab_for_krita

        self.foreground_colour = normalize_oklab_for_krita(oklab)
        return self.foreground_colour.copy()

    def get_foreground(self):
        return None if self.foreground_colour is None else self.foreground_colour.copy()


def _paint_of(c):
    if c is None:
        return None
    return c.paint_oklab if isinstance(c, ColourIntent) else c
