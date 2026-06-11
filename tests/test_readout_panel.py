"""Tests for the expanded LCH readout panel."""

import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")

from oklab_colour_picker.infrastructure.qt_facade import QtCore, QtWidgets

from tests.qt_helpers import key_event, send_focus, send_mouse
from oklab_colour_picker.app.controller import ChangeKind
from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.ui.readout.panel import ReadoutPanel
from tests.helpers import presented_colour


# -- pure helpers -----------------------------------------------------------


def _paint_oklab(colour):
    return ColourIntent.from_value(colour).paint_oklab


def _panel() -> ReadoutPanel:
    return ReadoutPanel()


def _present(colour, *, srgb8=None, in_gamut=True, fallback=None):
    intent = ColourIntent.from_value(colour)
    return presented_colour(
        intent,
        srgb8=_srgb8(intent.paint_oklab) if srgb8 is None else srgb8,
        in_gamut=in_gamut,
        fallback=fallback,
    )


def _show(panel: ReadoutPanel, colour, kind: ChangeKind) -> None:
    panel.show_colour(_present(colour), kind)


def _srgb8(oklab) -> tuple[int, int, int]:
    srgb8 = color_math.quantize_srgb8(color_math.oklab_to_srgb(oklab))
    return tuple(int(v) for v in srgb8)


def test_readout_panel_round_trips_through_sliders(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    panel.resize(320, 200)

    target = color_math.oklch_to_oklab([0.62, 0.11, math.radians(245.0)])
    _show(panel, target, ChangeKind.COMMIT)

    assert panel._row_l.value() == pytest.approx(0.62, abs=1e-3)
    assert panel._row_c.value() == pytest.approx(0.11, abs=1e-3)
    assert panel._row_h.value() == pytest.approx(245.0, abs=0.5)


def test_readout_panel_preserves_hue_intent_at_zero_chroma(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    panel.resize(320, 200)
    _show(panel, color_math.oklch_to_oklab([0.5, 0.0, 0.0]), ChangeKind.COMMIT)

    panel._row_h.set_value(210.0)
    panel._row_h.valueChanged.emit(210.0, False)

    assert panel._row_h.value() == pytest.approx(210.0)

    received: list[ColourIntent] = []
    panel.committed.connect(lambda intent: received.append(intent))
    panel._row_h.valueChanged.emit(210.0, True)
    _show(panel, received[-1], ChangeKind.COMMIT)

    assert panel._row_h.value() == pytest.approx(210.0)
    assert panel._row_c.value() == pytest.approx(0.0, abs=1e-6)

    echoed = ColourIntent.from_value(
        received[-1].quantized_paint_oklab,
        achromatic_hue=panel.hue_intent,
    )
    _show(panel, echoed, ChangeKind.COMMIT)

    assert panel._row_h.value() == pytest.approx(210.0)
    assert panel._row_c.value() == pytest.approx(0.0, abs=1e-6)


def test_readout_panel_greyscale_hex_commit_preserves_hue_intent(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    panel.resize(320, 200)
    _show(panel, color_math.oklch_to_oklab([0.5, 0.0, 0.0]), ChangeKind.COMMIT)
    panel._row_h.set_value(210.0)
    panel._row_h.valueChanged.emit(210.0, True)

    received = []
    panel.committed.connect(lambda intent: received.append(intent))

    panel._swatch.hex_committed.emit("#808080")

    assert received
    assert received[-1].hue == pytest.approx(math.radians(210.0))
    assert panel._row_h.value() == pytest.approx(210.0)
    assert panel._row_c.value() == pytest.approx(0.0, abs=1e-6)


def test_readout_panel_slider_edit_updates_handle_fallback(qtbot):
    # Direct slider interaction goes through _emit_from_lch, not
    # show_colour. The handle fallback colour must still update so the
    # OOG handle fill stays in sync with the colour the panel just emitted.
    panel = _panel()
    qtbot.addWidget(panel)
    panel.resize(320, 200)
    _show(panel, color_math.oklch_to_oklab([0.5, 0.05, 0.0]), ChangeKind.COMMIT)

    received: list[ColourIntent] = []
    panel.previewed.connect(
        lambda intent: (
            received.append(intent),
            panel.show_colour(_present(intent), ChangeKind.PREVIEW),
        )
    )
    panel._row_h.set_value(180.0)
    panel._row_h.valueChanged.emit(180.0, False)

    target = color_math.oklch_to_oklab([0.5, 0.05, math.radians(180.0)])
    srgb = color_math.clip_srgb(color_math.oklab_to_srgb(target))
    expected = tuple(int(round(float(c) * 255.0)) for c in srgb)
    fallback = panel._row_h.slider._fallback_colour
    assert received
    assert (
        panel._swatch._colour.red(),
        panel._swatch._colour.green(),
        panel._swatch._colour.blue(),
    ) == expected
    assert fallback is not None
    assert (fallback.red(), fallback.green(), fallback.blue()) == expected


def test_readout_panel_resize_refreshes_tracks_from_current_edit_values(
    qtbot,
    monkeypatch,
):
    panel = _panel()
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitExposed(panel)
    _show(panel, ColourIntent.from_lch(0.2, 0.05, 0.5), ChangeKind.COMMIT)
    panel._rows.set_lch(0.75, 0.10, 2.5)
    displayed_lch = panel._rows.current_lch()
    refreshes = []
    monkeypatch.setattr(
        panel._track_presenter,
        "refresh",
        lambda _rows, lightness, chroma, hue: refreshes.append(
            (lightness, chroma, hue)
        ),
    )

    panel.resize(360, 220)
    QtWidgets.QApplication.sendPostedEvents()

    assert refreshes[-1] == pytest.approx(displayed_lch)
    assert refreshes[-1][0] == pytest.approx(0.75)


def test_readout_slider_click_jumps_to_clicked_position(qtbot):
    panel = _panel()
    panel.resize(320, 200)
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitExposed(panel)

    _show(panel, color_math.oklch_to_oklab([0.2, 0.05, 0.0]), ChangeKind.COMMIT)
    commits: list[np.ndarray] = []
    previews: list[np.ndarray] = []
    panel.previewed.connect(lambda colour: previews.append(_paint_oklab(colour)))
    panel.committed.connect(lambda colour: commits.append(_paint_oklab(colour)))

    slider = panel._row_l.slider
    track = slider.track_rect()
    target_x = track.left() + int(round(track.width() * 0.75))
    target = QtCore.QPoint(target_x, track.center().y())
    send_mouse(slider, "press", target)
    send_mouse(slider, "release", target)

    assert previews
    assert commits
    lightness, _, _ = color_math.oklab_to_oklch(commits[-1])
    assert lightness == pytest.approx(0.75, abs=0.02)


def test_readout_slider_drag_previews_and_commits_release_position(qtbot):
    panel = _panel()
    panel.resize(320, 200)
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitExposed(panel)

    _show(panel, color_math.oklch_to_oklab([0.2, 0.05, 0.0]), ChangeKind.COMMIT)
    commits: list[np.ndarray] = []
    previews: list[np.ndarray] = []
    panel.previewed.connect(lambda colour: previews.append(_paint_oklab(colour)))
    panel.committed.connect(lambda colour: commits.append(_paint_oklab(colour)))

    slider = panel._row_l.slider
    track = slider.track_rect()
    start = QtCore.QPoint(track.left() + int(round(track.width() * 0.25)), track.center().y())
    middle = QtCore.QPoint(track.left() + int(round(track.width() * 0.50)), track.center().y())
    end = QtCore.QPoint(track.left() + int(round(track.width() * 0.75)), track.center().y())

    send_mouse(slider, "press", start)
    send_mouse(slider, "move", middle)
    send_mouse(slider, "release", end)

    assert len(previews) >= 2
    assert len(commits) == 1
    preview_lightness = [float(color_math.oklab_to_oklch(colour)[0]) for colour in previews]
    assert any(value == pytest.approx(0.50, abs=0.02) for value in preview_lightness)
    committed_lightness, _, _ = color_math.oklab_to_oklch(commits[0])
    assert committed_lightness == pytest.approx(0.75, abs=0.02)


def test_readout_panel_hex_field_reflects_current_colour(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)

    oklab = color_math.srgb_to_oklab(np.array([0x4A, 0x8F, 0xB2]) / 255.0)
    _show(panel, oklab, ChangeKind.COMMIT)

    assert panel._swatch._hex_edit.text() == "#4a8fb2"


def test_readout_panel_hex_edit_emits_committed_colour(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    _show(panel, color_math.srgb_to_oklab(np.array([0.5, 0.5, 0.5])), ChangeKind.COMMIT)

    received: list[ColourIntent] = []
    panel.committed.connect(lambda colour: received.append(colour))

    panel._swatch.hex_committed.emit("#4a8fb2")

    assert received
    expected = color_math.srgb_to_oklab(np.array([0x4A, 0x8F, 0xB2]) / 255.0)
    np.testing.assert_allclose(_paint_oklab(received[-1]), expected, atol=1e-4)


def test_readout_panel_revert_button_restores_previous(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)

    first = color_math.srgb_to_oklab(np.array([0.2, 0.4, 0.6]))
    second = color_math.srgb_to_oklab(np.array([0.7, 0.3, 0.1]))
    _show(panel, first, ChangeKind.COMMIT)
    _show(panel, second, ChangeKind.COMMIT)

    received: list[np.ndarray] = []
    panel.committed.connect(lambda colour: received.append(_paint_oklab(colour)))

    assert panel._swatch._revert_button.isEnabled()
    panel._swatch._revert_button.click()

    assert received
    np.testing.assert_allclose(received[-1], first, atol=1e-4)


def test_readout_panel_revert_restores_previous_achromatic_hue_intent(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    _show(panel, color_math.oklch_to_oklab([0.5, 0.0, 0.0]), ChangeKind.COMMIT)

    received: list[ColourIntent] = []
    panel.committed.connect(lambda intent: received.append(intent))
    panel._row_h.set_value(210.0)
    panel._row_h.valueChanged.emit(210.0, True)
    _show(panel, received[-1], ChangeKind.COMMIT)
    panel._row_h.set_value(30.0)
    panel._row_h.valueChanged.emit(30.0, True)
    _show(panel, received[-1], ChangeKind.COMMIT)

    panel._swatch.revert_clicked.emit()

    assert received[-1].selector_lch == pytest.approx(
        (0.5, 0.0, math.radians(210.0))
    )
    _show(panel, received[-1], ChangeKind.COMMIT)
    assert panel._row_h.value() == pytest.approx(210.0)


def test_readout_panel_hex_focus_out_without_edit_does_not_commit(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)

    a = color_math.srgb_to_oklab(np.array([0.2, 0.4, 0.6]))
    b = color_math.srgb_to_oklab(np.array([0.7, 0.3, 0.1]))
    _show(panel, a, ChangeKind.COMMIT)
    _show(panel, b, ChangeKind.COMMIT)
    received: list[np.ndarray] = []
    panel.committed.connect(lambda colour: received.append(_paint_oklab(colour)))

    panel._swatch._enter_edit_mode()
    panel._swatch._hex_edit.editingFinished.emit()

    assert received == []
    _assert_rows_match(panel, b)

    panel._swatch.revert_clicked.emit()

    assert len(received) == 1
    np.testing.assert_allclose(received[-1], a, atol=1e-4)


def test_readout_panel_slider_commit_discards_latched_external_colour(qtbot):
    panel = _panel()
    panel.resize(320, 200)
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitExposed(panel)
    original = color_math.oklch_to_oklab([0.2, 0.05, 0.0])
    external = color_math.oklch_to_oklab([0.8, 0.02, 1.0])
    _show(panel, original, ChangeKind.COMMIT)

    slider = panel._row_l.slider
    track = slider.track_rect()
    start = QtCore.QPoint(track.left() + int(round(track.width() * 0.25)), track.center().y())
    end = QtCore.QPoint(track.left() + int(round(track.width() * 0.75)), track.center().y())
    send_mouse(slider, "press", start)
    editing_lightness = panel._row_l.value()

    _show(panel, external, ChangeKind.EXTERNAL)
    assert panel._row_l.value() == pytest.approx(editing_lightness)

    received: list[ColourIntent] = []
    panel.committed.connect(lambda colour: received.append(colour))
    send_mouse(slider, "release", end)

    assert received[-1].lightness == pytest.approx(0.75, abs=0.02)


def test_readout_panel_spinbox_typing_latches_external_without_clobbering_text(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    original = color_math.oklch_to_oklab([0.2, 0.05, 0.0])
    external = color_math.oklch_to_oklab([0.8, 0.02, 1.0])
    _show(panel, original, ChangeKind.COMMIT)

    spin = panel._row_l.spin
    send_focus(spin, "in")
    spin.lineEdit().selectAll()
    spin.lineEdit().setText("0.750")
    _show(panel, external, ChangeKind.EXTERNAL)

    assert spin.lineEdit().text() == "0.750"

    received: list[ColourIntent] = []
    panel.committed.connect(lambda colour: received.append(colour))

    spin.editingFinished.emit()
    spin.editingFinished.emit()

    assert len(received) == 1
    assert received[-1].lightness == pytest.approx(0.75, abs=1e-3)


def test_readout_panel_spinbox_cancel_applies_latched_external_colour(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    original = color_math.oklch_to_oklab([0.2, 0.05, 0.0])
    external = color_math.oklch_to_oklab([0.8, 0.02, 1.0])
    _show(panel, original, ChangeKind.COMMIT)

    spin = panel._row_l.spin
    send_focus(spin, "in")
    _show(panel, external, ChangeKind.EXTERNAL)
    spin.editingFinished.emit()

    _assert_rows_match(panel, external)


def test_readout_panel_spinbox_escape_applies_latch_without_spurious_commit(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    original = color_math.oklch_to_oklab([0.2, 0.05, 0.0])
    external = color_math.oklch_to_oklab([0.8, 0.02, 1.0])
    _show(panel, original, ChangeKind.COMMIT)
    received: list[np.ndarray] = []
    panel.committed.connect(lambda colour: received.append(_paint_oklab(colour)))

    spin = panel._row_l.spin
    send_focus(spin, "in")
    spin.lineEdit().selectAll()
    spin.lineEdit().setText("0.750")
    _show(panel, external, ChangeKind.EXTERNAL)
    escape = key_event("press", "Escape")
    QtWidgets.QApplication.sendEvent(spin, escape)
    spin.editingFinished.emit()

    assert received == []
    _assert_rows_match(panel, external)


def test_readout_panel_zero_delta_slider_press_applies_latched_external_on_release(qtbot):
    panel = _panel()
    panel.resize(320, 200)
    qtbot.addWidget(panel)
    panel.show()
    qtbot.waitExposed(panel)
    original = color_math.oklch_to_oklab([0.2, 0.05, 0.0])
    external = color_math.oklch_to_oklab([0.8, 0.02, 1.0])
    _show(panel, original, ChangeKind.COMMIT)

    slider = panel._row_l.slider
    track = slider.track_rect()
    target = QtCore.QPoint(slider.handle_x_center(track), track.center().y())
    send_mouse(slider, "press", target)
    original_lightness = panel._row_l.value()

    _show(panel, external, ChangeKind.EXTERNAL)
    assert panel._row_l.value() == pytest.approx(original_lightness)

    send_mouse(slider, "release", target)

    _assert_rows_match(panel, external)


def test_readout_panel_out_of_gamut_warning_visibility(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)

    panel.show_colour(
        presented_colour(
            color_math.srgb_to_oklab(np.array([0.5, 0.5, 0.5])),
            in_gamut=True,
        ),
        ChangeKind.COMMIT,
    )
    assert not panel._swatch._oog_visible

    panel.show_colour(
        presented_colour(
            color_math.oklch_to_oklab([0.6, color_math.SRGB_MAX_CHROMA, 0.0]),
            in_gamut=False,
        ),
        ChangeKind.COMMIT,
    )
    assert panel._swatch._oog_visible


def test_readout_panel_uses_shared_fallback_rgb_for_swatch_and_handles(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    intent = ColourIntent.from_lch(0.6, color_math.SRGB_MAX_CHROMA, 0.0)
    fallback_rgb = (23, 45, 67)

    panel.show_colour(
        presented_colour(intent, srgb8=fallback_rgb, in_gamut=False),
        ChangeKind.COMMIT,
    )

    assert (
        panel._swatch._colour.red(),
        panel._swatch._colour.green(),
        panel._swatch._colour.blue(),
    ) == fallback_rgb
    for row in (panel._row_l, panel._row_c, panel._row_h):
        fallback = row.slider._fallback_colour
        assert fallback is not None
        assert (fallback.red(), fallback.green(), fallback.blue()) == fallback_rgb


def _assert_rows_match(panel: ReadoutPanel, colour) -> None:
    lightness, chroma, hue = ColourIntent.from_value(colour).selector_lch
    assert panel._row_l.value() == pytest.approx(lightness, abs=1e-3)
    assert panel._row_c.value() == pytest.approx(chroma, abs=1e-3)
    assert panel._row_h.value() == pytest.approx(math.degrees(hue), abs=0.1)
