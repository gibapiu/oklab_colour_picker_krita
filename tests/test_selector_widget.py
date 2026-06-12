import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytestmark = pytest.mark.qt

from oklab_colour_picker.infrastructure.qt_facade import QtCore, QtWidgets

from tests.qt_helpers import focus_event, key_event, send_mouse
from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
    SelectorModel,
    SelectorSelection,
)
from oklab_colour_picker.ui.selectors import SelectorWidget
from tests.helpers import presented_colour


def _paint_of(c):
    if c is None:
        return None
    return c.paint_oklab if isinstance(c, ColourIntent) else c


def _widget(model):
    return SelectorWidget(model)


def _present(colour):
    return presented_colour(colour)


def test_mouse_drag_emits_previews_and_commit(qtbot):
    widget = _widget(LightnessChromaSliceModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    previews = []
    commits = []
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    start = QtCore.QPoint(8, 12)
    end = QtCore.QPoint(24, 16)
    send_mouse(widget, "press", start)
    send_mouse(widget, "move", end)
    send_mouse(widget, "release", end)

    assert len(previews) >= 2
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], widget.model.color_at_position((end.x(), end.y()), _size(widget)))


def test_cold_start_invalid_release_does_not_commit(qtbot):
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(40, 80)
    qtbot.addWidget(widget)
    widget.show()

    commits = []
    previews = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))

    invalid_corner = QtCore.QPoint(0, 0)
    send_mouse(widget, "press", invalid_corner)
    send_mouse(widget, "release", invalid_corner)

    assert commits == []
    assert previews and all(p is None for p in previews)


def test_warm_off_leaf_press_snaps_and_commits(qtbot):
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(40, 80)
    qtbot.addWidget(widget)
    widget.show()

    previous = widget.model.color_at_position((20, 40), _size(widget))
    assert previous is not None
    widget.set_selected_colour(_present(previous))

    commits = []
    previews = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))

    off_leaf = QtCore.QPoint(0, 0)
    expected = widget.model.snapped_color_at_position((off_leaf.x(), off_leaf.y()), _size(widget))
    assert expected is not None
    send_mouse(widget, "press", off_leaf)
    send_mouse(widget, "move", off_leaf)
    send_mouse(widget, "release", off_leaf)

    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], expected)
    np.testing.assert_allclose(widget.selected_colour, expected)
    assert previews and previews[0] is not None and previews[-1] is not None


def test_hue_chroma_drag_outside_snaps_to_cursor_boundary(qtbot):
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(64, 64)
    qtbot.addWidget(widget)
    widget.show()

    previous = widget.model.color_at_position((32, 32), _size(widget))
    assert previous is not None
    widget.set_selected_colour(_present(previous))

    commits = []
    previews = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))

    first_valid = QtCore.QPoint(42, 32)
    latest_valid = QtCore.QPoint(46, 32)
    invalid_corner = QtCore.QPoint(0, 0)
    expected = widget.model.snapped_color_at_position((invalid_corner.x(), invalid_corner.y()), _size(widget))
    assert expected is not None

    send_mouse(widget, "press", first_valid)
    send_mouse(widget, "move", latest_valid)
    send_mouse(widget, "move", invalid_corner)
    send_mouse(widget, "release", invalid_corner)

    # Invalid movement after a valid hit snaps to the nearest selectable
    # boundary, so the indicator keeps following the drag.
    assert len(previews) == 3
    assert not any(preview is None for preview in previews)
    assert len(commits) == 1
    expected_position = widget.model.position_for_intent(color_math.oklab_to_oklch(expected), _size(widget))
    assert expected_position is not None
    assert widget.indicator_position() == pytest.approx(expected_position, abs=1.0)
    np.testing.assert_allclose(commits[0], expected)
    np.testing.assert_allclose(widget.selected_colour, expected)


def test_lightness_chroma_drag_outside_snaps_to_cursor_boundary(qtbot):
    widget = _widget(LightnessChromaSliceModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    previous = widget.model.color_at_position((8, 16), _size(widget))
    assert previous is not None
    widget.set_selected_colour(_present(previous))

    commits = []
    previews = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))

    first_valid = QtCore.QPoint(12, 16)
    latest_valid = QtCore.QPoint(24, 16)
    invalid_gamut = QtCore.QPoint(63, 16)
    expected = widget.model.snapped_color_at_position((invalid_gamut.x(), invalid_gamut.y()), _size(widget))
    assert expected is not None
    assert widget.model.color_at_position((invalid_gamut.x(), invalid_gamut.y()), _size(widget)) is None

    send_mouse(widget, "press", first_valid)
    send_mouse(widget, "move", latest_valid)
    send_mouse(widget, "move", invalid_gamut)
    send_mouse(widget, "release", invalid_gamut)

    # Invalid movement after a valid hit snaps to the gamut boundary.
    assert len(previews) == 3
    assert not any(preview is None for preview in previews)
    expected_position = widget.model.position_for_intent(color_math.oklab_to_oklch(expected), _size(widget))
    assert expected_position is not None
    assert widget.indicator_position() == pytest.approx(expected_position, abs=1.0)
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], expected)
    np.testing.assert_allclose(widget.selected_colour, expected)


def test_achromatic_hue_lightness_drag_outside_keeps_cursor_hue_anchor(qtbot):
    widget = _widget(HueLightnessSliceModel(chroma=0.0))
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    widget.show()

    start = QtCore.QPoint(50, 20)
    outside = QtCore.QPoint(50, -20)
    snapped = widget.model.snapped_selector_selection_at_position((outside.x(), outside.y()), _size(widget))
    assert snapped is not None
    expected = snapped.paint_oklab
    expected_position = snapped.position

    send_mouse(widget, "press", start)
    send_mouse(widget, "move", outside)
    send_mouse(widget, "release", outside)

    assert widget.indicator_position() == pytest.approx(expected_position, abs=1.0)
    np.testing.assert_allclose(widget.selected_colour, expected)


def test_hue_lightness_out_of_zone_pick_computes_snap_once(qtbot, monkeypatch):
    from oklab_colour_picker.models import hue_lightness_slice

    widget = _widget(HueLightnessSliceModel(chroma=0.2))
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    widget.show()
    point = (50.0, 50.0)
    assert widget.model.color_at_position(point, _size(widget)) is None

    original = hue_lightness_slice._snap_lightness_to_gamut
    calls = []

    def counted_snap(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(hue_lightness_slice, "_snap_lightness_to_gamut", counted_snap)

    pick = widget.pick(point)

    assert pick is not None
    assert len(calls) == 1


def test_snapped_colour_with_unresolvable_position_still_commits(qtbot):
    widget = _widget(UnresolvableSnapModel())
    widget.resize(10, 10)
    qtbot.addWidget(widget)
    widget.show()
    commits = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    send_mouse(widget, "press", QtCore.QPoint(1, 1))
    send_mouse(widget, "move", QtCore.QPoint(8, 1))
    send_mouse(widget, "release", QtCore.QPoint(8, 1))

    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], UnresolvableSnapModel.SNAPPED)
    np.testing.assert_allclose(widget.selected_colour, UnresolvableSnapModel.SNAPPED)
    assert widget.indicator_position() == pytest.approx((8.0, 1.0))


def test_leave_during_drag_does_not_emit_invalid_preview(qtbot):
    widget = _widget(LightnessChromaSliceModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    previews = []
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))

    point = QtCore.QPoint(24, 16)
    send_mouse(widget, "press", point)
    leave = QtCore.QEvent(QtCore.QEvent.Type.Leave)
    QtWidgets.QApplication.sendEvent(widget, leave)

    assert len(previews) == 1
    assert previews[0] is not None

    send_mouse(widget, "release", point)


def test_programmatic_colour_update_blocks_widget_signals(qtbot):
    widget = _widget(LightnessChromaSliceModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)

    previews = []
    commits = []
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    colour = widget.model.color_at_position((20, 10), (64, 32))
    assert colour is not None
    blocker = QtCore.QSignalBlocker(widget)
    widget.set_selected_colour(_present(colour))
    del blocker

    assert previews == []
    assert commits == []
    np.testing.assert_allclose(widget.selected_colour, colour)


def test_keyboard_nudge_previews_then_commits_on_release(qtbot):
    widget = _widget(LightnessChromaSliceModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    start = widget.model.color_at_position((20, 10), _size(widget))
    assert start is not None
    widget.set_selected_colour(_present(start))

    previews = []
    commits = []
    widget.previewed.connect(lambda c, _ps=previews: _ps.append(_paint_of(c)))
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    press = key_event("press", "Right")
    release = key_event("release", "Right")
    QtWidgets.QApplication.sendEvent(widget, press)

    assert press.isAccepted()
    assert len(previews) == 1
    assert commits == []

    QtWidgets.QApplication.sendEvent(widget, release)
    assert release.isAccepted()
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], widget.selected_colour)


def test_signal_payload_mutation_does_not_corrupt_widget_state(qtbot):
    widget = _widget(LightnessChromaSliceModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    commits = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    point = QtCore.QPoint(24, 16)
    send_mouse(widget, "press", point)
    send_mouse(widget, "release", point)

    selected = widget.selected_colour
    assert selected is not None
    commits[0][:] = 0.0
    np.testing.assert_allclose(widget.selected_colour, selected)


def test_keyboard_step_at_boundary_keeps_event_handled(qtbot):
    widget = _widget(LightnessSliceModel(lightness=0.5))
    widget.resize(40, 80)
    qtbot.addWidget(widget)
    widget.show()

    # Near the bottom of the disk (blue hue), where the L=0.5 gamut leaf
    # extends out far enough that this radius is in-gamut.
    start = widget.model.color_at_position((20, 55), _size(widget))
    assert start is not None
    widget.set_selected_colour(_present(start))

    event = key_event("press", "Right")
    QtWidgets.QApplication.sendEvent(widget, event)

    assert event.isAccepted()
    assert widget.selected_colour is not None


def test_mouse_interaction_cancels_pending_keyboard_commit(qtbot):
    widget = _widget(LightnessChromaSliceModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    start = widget.model.color_at_position((20, 10), _size(widget))
    assert start is not None
    widget.set_selected_colour(_present(start))

    commits = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    key_press = key_event("press", "Right")
    QtWidgets.QApplication.sendEvent(widget, key_press)
    assert key_press.isAccepted()

    point = QtCore.QPoint(24, 16)
    send_mouse(widget, "press", point)
    send_mouse(widget, "release", point)

    key_release = key_event("release", "Right")
    QtWidgets.QApplication.sendEvent(widget, key_release)

    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], widget.selected_colour)


def test_focus_loss_flushes_pending_keyboard_commit(qtbot):
    widget = _widget(LightnessChromaSliceModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    start = widget.model.color_at_position((20, 10), _size(widget))
    assert start is not None
    widget.set_selected_colour(_present(start))

    commits = []
    widget.committed.connect(lambda c, _cs=commits: _cs.append(_paint_of(c)))

    key_press = key_event("press", "Right")
    QtWidgets.QApplication.sendEvent(widget, key_press)
    assert key_press.isAccepted()

    focus_out = focus_event("out")
    QtWidgets.QApplication.sendEvent(widget, focus_out)

    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], widget.selected_colour)


def test_pick_uses_selector_domain_snap():
    widget = _widget(UnresolvableSnapModel())
    widget.resize(40, 20)

    picked = widget.pick((8.0, 3.0))

    assert type(picked).__name__ == "SnappedPick"
    np.testing.assert_allclose(picked.colour.paint_oklab, UnresolvableSnapModel.SNAPPED)
    assert picked.position == pytest.approx((8.0, 3.0))


def _size(widget):
    return widget.width(), widget.height()


class UnresolvableSnapModel(SelectorModel):
    SNAPPED = np.array([0.25, 0.04, -0.02], dtype=float)

    def color_at_position(self, position, size):
        if tuple(position) == (1.0, 1.0):
            return np.array([0.20, 0.01, 0.00], dtype=float)
        return None

    def selection_at_position(self, position, size):
        colour = self.color_at_position(position, size)
        if colour is None:
            return None
        lch = color_math.oklab_to_oklch(colour)
        return SelectorSelection(
            (float(lch[0]), float(lch[1]), float(lch[2])),
            (float(position[0]), float(position[1])),
        )

    def colors_at_positions(self, x, y, size):
        x = np.asarray(x)
        return np.zeros(x.shape + (3,), dtype=float), np.zeros_like(x, dtype=bool)

    def position_for_intent(self, lch, size):
        return None

    def snapped_selector_selection_at_position(self, position, size):
        lch = color_math.oklab_to_oklch(self.SNAPPED)
        return SelectorSelection(
            (float(lch[0]), float(lch[1]), float(lch[2])),
            (float(position[0]), float(position[1])),
        )
