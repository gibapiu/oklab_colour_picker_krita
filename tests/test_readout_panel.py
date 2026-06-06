"""Tests for the expanded LCH readout panel."""

import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker.app.controller import ChangeKind, normalize_oklab_for_krita
from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.render import renderers
from oklab_colour_picker.ui.readout.axis import AxisTrackPresenter, ReadoutAxisRows
from oklab_colour_picker.ui.readout.panel import ReadoutPanel
from oklab_colour_picker.ui.readout.swatch import UnifiedSwatch, hex_to_oklab
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


def oklab_to_hex(oklab) -> str:
    return QtGui.QColor(*_srgb8(oklab)).name(QtGui.QColor.HexRgb)


def _srgb8(oklab) -> tuple[int, int, int]:
    srgb8 = color_math.quantize_srgb8(color_math.oklab_to_srgb(oklab))
    return tuple(int(v) for v in srgb8)


@pytest.mark.parametrize(
    "hex_value",
    ["#000000", "#ffffff", "#4a8fb2", "#7f3322"],
)
def test_hex_round_trip(hex_value):
    oklab = hex_to_oklab(hex_value)
    assert oklab is not None
    assert oklab_to_hex(oklab) == hex_value


def test_hex_accepts_uppercase_and_missing_hash():
    assert oklab_to_hex(hex_to_oklab("4A8FB2")) == "#4a8fb2"
    assert oklab_to_hex(hex_to_oklab("#FF00AA")) == "#ff00aa"


@pytest.mark.parametrize("bad", ["", "not-hex", "#12345", "#1234567", "#zzzzzz"])
def test_hex_rejects_malformed(bad):
    assert hex_to_oklab(bad) is None


def test_in_gamut_detects_displayable_colour():
    assert presented_colour(
        color_math.srgb_to_oklab(np.array([0.5, 0.5, 0.5])),
        in_gamut=True,
    ).in_gamut


def test_in_gamut_flags_super_saturated_oklch():
    super_saturated = color_math.oklch_to_oklab([0.6, color_math.SRGB_MAX_CHROMA, 0.0])
    assert not presented_colour(super_saturated, in_gamut=False).in_gamut


# -- gamut-gap rendering ----------------------------------------------------


def test_axis_track_hue_marks_out_of_gamut_with_checker():
    # At high chroma and L=0.95 most hues are unreachable in sRGB; we expect
    # a substantial fraction of the hue track to be flagged out-of-gamut.
    rgba = renderers.render_axis_track(
        renderers.AXIS_H,
        (0.95, color_math.SRGB_MAX_CHROMA * 0.9),
        color_math.SRGB_MAX_CHROMA,
        (256, 12),
    )
    # In-gamut pixels never use the (200, 200, 200) / (120, 120, 120) tones
    # exclusively for entire pattern columns, so look for the dark-tile colour.
    has_dark_tile = np.any(np.all(rgba[..., :3] == 120, axis=-1))
    assert has_dark_tile


def test_axis_track_chroma_low_l_is_fully_in_gamut_at_zero_chroma_start():
    # At chroma=0 the swept C axis starts in gamut and crosses the cusp once.
    rgba = renderers.render_axis_track(
        renderers.AXIS_C,
        (0.5, 0.0),
        color_math.SRGB_MAX_CHROMA,
        (256, 12),
    )
    # The leftmost column (C=0) must be in gamut (no checker tile colour).
    left = rgba[:, 0, :3]
    assert not np.any(np.all(left == 120, axis=-1))
    assert not np.any(np.all(left == 200, axis=-1))


def test_axis_track_l_at_extremes_is_out_of_gamut_for_nonzero_chroma():
    # At L=0 or L=1 any positive chroma is out of gamut.
    rgba = renderers.render_axis_track(
        renderers.AXIS_L,
        (0.15, 0.0),  # chroma=0.15, hue=0
        color_math.SRGB_MAX_CHROMA,
        (256, 12),
    )
    # First column corresponds to L=0 and last to L=1; both should be flagged.
    for col in (0, -1):
        pixel = rgba[0, col, :3]
        assert tuple(pixel) in {(120, 120, 120), (200, 200, 200)}


def test_axis_track_unknown_axis_raises():
    with pytest.raises(ValueError):
        renderers.render_axis_track("Q", (0.5, 0.0), color_math.SRGB_MAX_CHROMA, (32, 10))


def test_axis_track_presenter_owns_cache_policy(qtbot, monkeypatch):
    parent = QtWidgets.QWidget()
    qtbot.addWidget(parent)
    rows = ReadoutAxisRows.create(parent)
    for row in rows.as_tuple():
        row.slider.resize(120, 24)

    calls: list[tuple[str, tuple[float, float], tuple[int, int], float]] = []

    def fake_render_axis_track(
        axis,
        fixed,
        max_chroma,
        size,
        *,
        hue_chroma_floor=0.0,
    ):
        _ = max_chroma
        calls.append((axis, fixed, size, hue_chroma_floor))
        return np.zeros((size[1], size[0], 4), dtype=np.uint8)

    monkeypatch.setattr(renderers, "render_axis_track", fake_render_axis_track)
    presenter = AxisTrackPresenter()

    presenter.refresh(rows, 0.5, 0.1, 1.0)
    assert [call[0] for call in calls] == [
        renderers.AXIS_L,
        renderers.AXIS_C,
        renderers.AXIS_H,
    ]

    presenter.refresh(rows, 0.5, 0.1, 1.0)
    assert len(calls) == 3

    presenter.refresh(rows, 0.5, 0.2, 1.0)
    assert [call[0] for call in calls[3:]] == [renderers.AXIS_L, renderers.AXIS_H]

    replacement_rows = ReadoutAxisRows.create(parent)
    for row in replacement_rows.as_tuple():
        row.slider.resize(120, 24)
    presenter.refresh(replacement_rows, 0.5, 0.2, 1.0)

    assert [call[0] for call in calls[5:]] == [
        renderers.AXIS_L,
        renderers.AXIS_C,
        renderers.AXIS_H,
    ]


# -- panel round-trips ------------------------------------------------------


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

    assert panel.readout_state == "IDLE"
    assert panel._row_h.value() == pytest.approx(210.0)
    assert panel._row_c.value() == pytest.approx(0.0, abs=1e-6)

    echoed = ColourIntent.from_value(
        normalize_oklab_for_krita(panel._current_oklab),
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
    assert panel.readout_state == "EDITING"
    assert (
        panel._swatch._colour.red(),
        panel._swatch._colour.green(),
        panel._swatch._colour.blue(),
    ) == expected
    assert fallback is not None
    assert (fallback.red(), fallback.green(), fallback.blue()) == expected


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
    _send_mouse(slider, QtCore.QEvent.MouseButtonPress, target)
    _send_mouse(slider, QtCore.QEvent.MouseButtonRelease, target)

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

    _send_mouse(slider, QtCore.QEvent.MouseButtonPress, start)
    _send_mouse(slider, QtCore.QEvent.MouseMove, middle)
    _send_mouse(slider, QtCore.QEvent.MouseButtonRelease, end)

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

    assert panel._swatch.hex_text == "#4a8fb2"


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


def test_readout_panel_hex_edit_mode_commits_via_lineedit(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    _show(panel, color_math.srgb_to_oklab(np.array([0.5, 0.5, 0.5])), ChangeKind.COMMIT)

    received: list[np.ndarray] = []
    panel.committed.connect(lambda colour: received.append(_paint_oklab(colour)))

    swatch = panel._swatch
    swatch._enter_edit_mode()
    assert not swatch._hex_edit.isReadOnly()
    swatch._hex_edit.setText("#4a8fb2")
    swatch._hex_edit.editingFinished.emit()

    assert received
    expected = color_math.srgb_to_oklab(np.array([0x4A, 0x8F, 0xB2]) / 255.0)
    np.testing.assert_allclose(received[-1], expected, atol=1e-4)


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


def test_readout_panel_set_previous_seeds_revert_target(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)

    seed = color_math.srgb_to_oklab(np.array([0.2, 0.4, 0.6]))
    _show(panel, seed, ChangeKind.COMMIT)
    panel.set_previous_colour(_present(seed))

    received: list[np.ndarray] = []
    panel.committed.connect(lambda colour: received.append(_paint_oklab(colour)))

    assert panel._swatch._revert_button.isEnabled()
    panel._swatch.revert_clicked.emit()

    assert received
    np.testing.assert_allclose(received[-1], seed, atol=1e-4)


def test_readout_panel_hex_focus_out_without_edit_does_not_commit(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)

    a = color_math.srgb_to_oklab(np.array([0.2, 0.4, 0.6]))
    b = color_math.srgb_to_oklab(np.array([0.7, 0.3, 0.1]))
    _show(panel, a, ChangeKind.COMMIT)
    _show(panel, b, ChangeKind.COMMIT)
    previous = panel._previous_oklab.copy()

    received: list[np.ndarray] = []
    panel.committed.connect(lambda colour: received.append(_paint_oklab(colour)))

    panel._swatch._enter_edit_mode()
    panel._swatch._hex_edit.editingFinished.emit()

    assert received == []
    np.testing.assert_allclose(panel._previous_oklab, previous, atol=1e-12)
    np.testing.assert_allclose(panel._current_oklab, b, atol=1e-12)


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
    _send_mouse(slider, QtCore.QEvent.MouseButtonPress, start)

    assert panel.readout_state == "EDITING"
    _show(panel, external, ChangeKind.EXTERNAL)
    np.testing.assert_allclose(panel._current_oklab, original, atol=1e-12)

    received: list[ColourIntent] = []
    panel.committed.connect(lambda colour: received.append(colour))
    _send_mouse(slider, QtCore.QEvent.MouseButtonRelease, end)
    _show(panel, received[-1], ChangeKind.COMMIT)

    assert panel.readout_state == "IDLE"
    committed_lightness, _, _ = color_math.oklab_to_oklch(panel._current_oklab)
    assert committed_lightness == pytest.approx(0.75, abs=0.02)


def test_readout_panel_spinbox_typing_latches_external_without_clobbering_text(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    original = color_math.oklch_to_oklab([0.2, 0.05, 0.0])
    external = color_math.oklch_to_oklab([0.8, 0.02, 1.0])
    _show(panel, original, ChangeKind.COMMIT)

    spin = panel._row_l.spin
    _send_focus(spin, QtCore.QEvent.FocusIn)
    spin.lineEdit().selectAll()
    spin.lineEdit().setText("0.750")
    _show(panel, external, ChangeKind.EXTERNAL)

    assert panel.readout_state == "EDITING"
    assert spin.lineEdit().text() == "0.750"
    np.testing.assert_allclose(panel._current_oklab, original, atol=1e-12)

    received: list[ColourIntent] = []
    panel.committed.connect(lambda colour: received.append(colour))

    spin.editingFinished.emit()
    spin.editingFinished.emit()

    assert panel.readout_state == "IDLE"
    assert len(received) == 1
    _show(panel, received[-1], ChangeKind.COMMIT)
    committed_lightness, _, _ = color_math.oklab_to_oklch(panel._current_oklab)
    assert committed_lightness == pytest.approx(0.75, abs=1e-3)


def test_readout_panel_spinbox_cancel_applies_latched_external_colour(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    original = color_math.oklch_to_oklab([0.2, 0.05, 0.0])
    external = color_math.oklch_to_oklab([0.8, 0.02, 1.0])
    _show(panel, original, ChangeKind.COMMIT)

    spin = panel._row_l.spin
    _send_focus(spin, QtCore.QEvent.FocusIn)
    _show(panel, external, ChangeKind.EXTERNAL)
    spin.editingFinished.emit()

    assert panel.readout_state == "IDLE"
    np.testing.assert_allclose(panel._current_oklab, external, atol=1e-12)


def test_readout_panel_spinbox_escape_applies_latch_without_spurious_commit(qtbot):
    panel = _panel()
    qtbot.addWidget(panel)
    original = color_math.oklch_to_oklab([0.2, 0.05, 0.0])
    external = color_math.oklch_to_oklab([0.8, 0.02, 1.0])
    _show(panel, original, ChangeKind.COMMIT)
    received: list[np.ndarray] = []
    panel.committed.connect(lambda colour: received.append(_paint_oklab(colour)))

    spin = panel._row_l.spin
    _send_focus(spin, QtCore.QEvent.FocusIn)
    spin.lineEdit().selectAll()
    spin.lineEdit().setText("0.750")
    _show(panel, external, ChangeKind.EXTERNAL)
    escape = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Escape, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(spin, escape)
    spin.editingFinished.emit()

    assert panel.readout_state == "IDLE"
    assert received == []
    np.testing.assert_allclose(panel._current_oklab, external, atol=1e-12)


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
    _send_mouse(slider, QtCore.QEvent.MouseButtonPress, target)

    assert panel.readout_state == "EDITING"
    _show(panel, external, ChangeKind.EXTERNAL)
    np.testing.assert_allclose(panel._current_oklab, original, atol=1e-12)

    _send_mouse(slider, QtCore.QEvent.MouseButtonRelease, target)

    assert panel.readout_state == "IDLE"
    np.testing.assert_allclose(panel._current_oklab, external, atol=1e-12)


def test_unified_swatch_skips_stylesheet_reassignment_when_ink_is_unchanged(qtbot, monkeypatch):
    swatch = UnifiedSwatch()
    qtbot.addWidget(swatch)
    swatch.set_srgb8((230, 230, 230))

    calls: list[str] = []

    def record_hex_style(style: str) -> None:
        calls.append(style)

    def record_oog_style(style: str) -> None:
        calls.append(style)

    def record_revert_style(style: str) -> None:
        calls.append(style)

    monkeypatch.setattr(swatch._hex_edit, "setStyleSheet", record_hex_style)
    monkeypatch.setattr(swatch._oog_label, "setStyleSheet", record_oog_style)
    monkeypatch.setattr(swatch._revert_button, "setStyleSheet", record_revert_style)

    swatch.set_srgb8((204, 204, 204))
    swatch.set_oog_visible(True)

    assert calls == []


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


def _send_mouse(widget, event_type, position):
    button = QtCore.Qt.LeftButton if event_type != QtCore.QEvent.MouseMove else QtCore.Qt.NoButton
    buttons = QtCore.Qt.NoButton if event_type == QtCore.QEvent.MouseButtonRelease else QtCore.Qt.LeftButton
    event = QtGui.QMouseEvent(
        event_type,
        QtCore.QPointF(position),
        button,
        buttons,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(widget, event)
    assert event.isAccepted()


def _send_focus(widget, event_type):
    event = QtGui.QFocusEvent(event_type, QtCore.Qt.OtherFocusReason)
    QtWidgets.QApplication.sendEvent(widget, event)


def test_axis_track_hue_chroma_floor_lifts_neutral_colors():
    # At chroma=0 every column collapses to grey; the floor must paint a
    # visibly colourful track instead while gamut classification stays at the
    # actual chroma (so no checker should appear here).
    flat = renderers.render_axis_track(
        renderers.AXIS_H,
        (0.5, 0.0),
        color_math.SRGB_MAX_CHROMA,
        (64, 8),
    )
    floored = renderers.render_axis_track(
        renderers.AXIS_H,
        (0.5, 0.0),
        color_math.SRGB_MAX_CHROMA,
        (64, 8),
        hue_chroma_floor=0.08,
    )
    # Flat rail is monochrome: identical RGB across all columns.
    assert np.all(flat[..., 0] == flat[0, 0, 0])
    assert np.all(flat[..., 1] == flat[0, 0, 1])
    assert np.all(flat[..., 2] == flat[0, 0, 2])
    # Floored rail has multiple distinct hues across columns.
    unique_cols = {tuple(floored[0, x, :3]) for x in range(floored.shape[1])}
    assert len(unique_cols) > 8


def test_axis_track_hue_chroma_floor_preserves_actual_gamut_classification():
    # Pick a chroma above the actual cusp for L=0.5 so some columns are OOG
    # without any floor. The floor must not hide those OOG columns.
    rgba = renderers.render_axis_track(
        renderers.AXIS_H,
        (0.5, color_math.SRGB_MAX_CHROMA * 0.95),
        color_math.SRGB_MAX_CHROMA,
        (256, 12),
        hue_chroma_floor=0.001,
    )
    has_dark_tile = np.any(np.all(rgba[..., :3] == 120, axis=-1))
    assert has_dark_tile
