"""Expanded readout panel: unified swatch and L/C/H gradient sliders.

This widget is *additive* to the dock layout. It owns no colour state of its
own — every interaction emits :attr:`previewed` / :attr:`committed` so the
host panel can route through the existing controller path, the same way the
selector widgets do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker import color_math, renderers
from oklab_colour_picker.colour_presentation import (
    PresentedColour,
    require_presented_colour,
)
from oklab_colour_picker.colour_state import ColourIntent
from oklab_colour_picker.controller import ChangeKind


# Per-axis numeric step sizes (Shift x10). H is in degrees, displayed in
# degrees, stored on the model in radians.
_STEP_L = 0.01
_STEP_C = 0.005
_STEP_H = 1.0

# Contrast-aware paint colours (dark vs light) for handle borders and overlay
# text drawn on top of arbitrary gradient pixels.
_DARK_INK = QtGui.QColor("#1e1e1e")
_LIGHT_INK = QtGui.QColor("#f2f2f2")

_HANDLE_WIDTH = 10
_HANDLE_BORDER = 2

_SWATCH_HEIGHT = 48
_CORNER_BUTTON_SIZE = 20

# Minimum chroma used when *rendering* the H slider rail. The hue track sweeps
# all hues at the current (L, C); for near-neutral colours every column
# collapses to the same grey, which makes the rail unreadable as a hue picker.
# We floor the rendering chroma to keep the rail colourful while the OOG
# checker continues to reflect the actual selected chroma.
_H_TRACK_CHROMA_FLOOR = 0.06


class _ReadoutState(Enum):
    IDLE = "IDLE"
    EDITING = "EDITING"


class _EditExit(Enum):
    COMMIT = "COMMIT"
    CANCEL = "CANCEL"


@dataclass(frozen=True)
class _LatchedColour:
    colour: PresentedColour
    committed: bool


def hex_to_oklab(text: str) -> np.ndarray | None:
    """Parse ``#rrggbb`` / ``rrggbb`` to OKLab; ``None`` on malformed input."""

    candidate = (text or "").strip()
    if not candidate.startswith("#"):
        candidate = "#" + candidate
    if len(candidate) != 7:
        return None
    colour = QtGui.QColor(candidate)
    if not colour.isValid():
        return None
    r, g, b, _ = colour.getRgbF()
    return color_math.srgb_to_oklab(np.array([r, g, b], dtype=float))


def _perceived_luminance(r: int, g: int, b: int) -> float:
    """Simple Rec.709 luma on 0-255 sRGB bytes; good enough for ink choice."""

    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _ink_for(r: int, g: int, b: int) -> QtGui.QColor:
    return _DARK_INK if _perceived_luminance(r, g, b) > 0.55 else _LIGHT_INK


def _qcolor_from_srgb8(srgb8: tuple[int, int, int]) -> QtGui.QColor:
    return QtGui.QColor(int(srgb8[0]), int(srgb8[1]), int(srgb8[2]))


class _GradientSlider(QtWidgets.QSlider):
    """Horizontal slider with a custom-painted gradient track and hollow handle.

    Replaces Qt's default groove+handle painting because the track is already a
    cached RGBA image and the handle is a contrast-aware hollow rectangle that
    must reveal the underlying gradient (including the 4 px checkerboard for
    out-of-gamut regions). We subclass ``QSlider`` and override ``paintEvent``
    instead of routing through ``QProxyStyle`` because the renderer pipeline
    already gives us a cached RGBA track.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(QtCore.Qt.Horizontal, parent)
        self.setMinimum(0)
        self.setMaximum(1000)
        self.setSingleStep(1)
        self.setPageStep(10)
        # Match the spinbox sizeHint so slider + spinbox align vertically.
        probe = QtWidgets.QSpinBox()
        self.setFixedHeight(max(20, probe.sizeHint().height()))
        self._track_image: QtGui.QImage | None = None
        self._track_buffer: np.ndarray | None = None
        self._track_cache_key: tuple | None = None
        self._fallback_colour: QtGui.QColor | None = None
        self._pressed_handle = False
        self._moved_since_press = False

    def set_track(self, rgba: np.ndarray) -> None:
        self._track_buffer = rgba
        bytes_per_line = int(rgba.strides[0])
        self._track_image = QtGui.QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            bytes_per_line,
            QtGui.QImage.Format_RGBA8888,
        )
        self.update()

    def cache_key(self) -> tuple | None:
        return self._track_cache_key

    def set_cache_key(self, key: tuple | None) -> None:
        self._track_cache_key = key

    def set_fallback_colour(self, colour: QtGui.QColor | None) -> None:
        # The fallback colour fills the inside of the hollow handle so it
        # always shows the clipped sRGB colour the user will actually paint
        # with. When the handle sits over an in-gamut track pixel the fill
        # matches the track underneath; over the OOG checker it shows the
        # solid clipped colour, making the fallback visible.
        if colour is None:
            self._fallback_colour = None
        else:
            self._fallback_colour = QtGui.QColor(colour)
        self.update()

    def _track_rect(self) -> QtCore.QRect:
        # Reserve a little horizontal padding so the handle never paints past
        # the slider edges; mapping below stays linear inside that band.
        pad = _HANDLE_WIDTH // 2
        return self.rect().adjusted(pad, 2, -pad, -2)

    def _handle_x_center(self, track_rect: QtCore.QRect) -> int:
        position = QtWidgets.QStyle.sliderPositionFromValue(
            self.minimum(),
            self.maximum(),
            self.value(),
            max(1, track_rect.width() - 1),
            self._upside_down(),
        )
        return track_rect.left() + position

    def _border_ink(self, x_center: int, track_rect: QtCore.QRect) -> QtGui.QColor:
        if self._track_buffer is None:
            return _DARK_INK
        buf = self._track_buffer
        # Map handle x in widget coords to a column in the cached track buffer.
        rel = (x_center - track_rect.left()) / max(1, track_rect.width() - 1)
        col = int(round(np.clip(rel, 0.0, 1.0) * (buf.shape[1] - 1)))
        row = buf.shape[0] // 2
        r, g, b = int(buf[row, col, 0]), int(buf[row, col, 1]), int(buf[row, col, 2])
        return _ink_for(r, g, b)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        track_rect = self._track_rect()
        if self._track_image is not None:
            painter.drawImage(track_rect, self._track_image)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 120), 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(track_rect)

        x = self._handle_x_center(track_rect)
        handle_rect = self._handle_rect()
        ink = self._border_ink(x, track_rect)
        if self._fallback_colour is not None:
            # Fill inside the border so OOG handles show a solid sample of the
            # clipped colour over the checker.
            inner = QtCore.QRectF(handle_rect).adjusted(
                _HANDLE_BORDER, _HANDLE_BORDER, -_HANDLE_BORDER, -_HANDLE_BORDER
            )
            if inner.width() > 0 and inner.height() > 0:
                painter.fillRect(inner, self._fallback_colour)
        pen = QtGui.QPen(ink, _HANDLE_BORDER)
        pen.setJoinStyle(QtCore.Qt.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        # Inset by half the pen width so the stroke stays inside handle_rect.
        inset = _HANDLE_BORDER / 2
        painter.drawRect(QtCore.QRectF(handle_rect).adjusted(inset, inset, -inset, -inset))
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return
        # Use the full widget height as the hit target, matching native
        # sliders while still mapping horizontally through the visible track.
        self.setSliderDown(True)
        self._pressed_handle = self._handle_rect().contains(event.pos())
        self._moved_since_press = False
        if not self._pressed_handle:
            self.setValue(self._value_at_x(event.x()))
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if not self.isSliderDown():
            super().mouseMoveEvent(event)
            return
        self._moved_since_press = True
        self.setValue(self._value_at_x(event.x()))
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.LeftButton or not self.isSliderDown():
            super().mouseReleaseEvent(event)
            return
        if not (self._pressed_handle and not self._moved_since_press):
            self.setValue(self._value_at_x(event.x()))
        self.setSliderDown(False)
        self._pressed_handle = False
        self._moved_since_press = False
        event.accept()

    def _value_at_x(self, x: int) -> int:
        track_rect = self._track_rect()
        return QtWidgets.QStyle.sliderValueFromPosition(
            self.minimum(),
            self.maximum(),
            x - track_rect.left(),
            max(1, track_rect.width() - 1),
            self._upside_down(),
        )

    def _upside_down(self) -> bool:
        right_to_left = self.layoutDirection() == QtCore.Qt.RightToLeft
        return right_to_left ^ self.invertedAppearance()

    def _handle_rect(self) -> QtCore.QRect:
        track_rect = self._track_rect()
        x = self._handle_x_center(track_rect)
        return QtCore.QRect(
            x - _HANDLE_WIDTH // 2,
            self.rect().top(),
            _HANDLE_WIDTH,
            self.rect().height() - 1,
        )


class _AxisRow(QtWidgets.QWidget):
    """One row: label, gradient slider, numeric spinbox."""

    valueChanged = QtCore.pyqtSignal(float, bool)  # (value, committed)
    editStarted = QtCore.pyqtSignal()
    editCancelled = QtCore.pyqtSignal()

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._step = float(step)
        self._decimals = int(decimals)
        self._edit_start_value: float | None = None
        self._edit_start_slider_value: int | None = None

        label_widget = QtWidgets.QLabel(label, self)
        label_widget.setFixedWidth(14)
        label_widget.setAlignment(QtCore.Qt.AlignCenter)

        self.slider = _GradientSlider(self)
        self.spin = QtWidgets.QDoubleSpinBox(self)
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(72)
        self.spin.setAlignment(QtCore.Qt.AlignRight)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.setAlignment(QtCore.Qt.AlignVCenter)
        layout.addWidget(label_widget, 0, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.slider, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.spin, 0, QtCore.Qt.AlignVCenter)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderPressed.connect(self._on_slider_pressed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.spin.installEventFilter(self)
        self.spin.editingFinished.connect(self._on_spin_committed)
        self.spin.valueChanged.connect(self._on_spin_value_changed)

    def set_value(self, value: float) -> None:
        clamped = float(np.clip(value, self._minimum, self._maximum))
        with QtCore.QSignalBlocker(self.spin), QtCore.QSignalBlocker(self.slider):
            self.spin.setValue(clamped)
            self.slider.setValue(self._value_to_slider(clamped))

    def value(self) -> float:
        return float(self.spin.value())

    def _value_to_slider(self, value: float) -> int:
        if self._maximum <= self._minimum:
            return 0
        fraction = (value - self._minimum) / (self._maximum - self._minimum)
        return int(round(fraction * self.slider.maximum()))

    def _slider_to_value(self, position: int) -> float:
        fraction = position / max(1, self.slider.maximum())
        return self._minimum + fraction * (self._maximum - self._minimum)

    def _on_slider_changed(self, position: int) -> None:
        self._begin_edit()
        value = self._slider_to_value(position)
        with QtCore.QSignalBlocker(self.spin):
            self.spin.setValue(value)
        committed = not self.slider.isSliderDown()
        self.valueChanged.emit(value, committed)

    def _on_slider_pressed(self) -> None:
        self._edit_start_slider_value = self.slider.value()
        self._begin_edit()

    def _on_slider_released(self) -> None:
        if self._edit_start_slider_value == self.slider.value():
            self._cancel_edit()
            return
        self.valueChanged.emit(self.value(), True)
        self._edit_start_value = None
        self._edit_start_slider_value = None

    def _on_spin_value_changed(self, value: float) -> None:
        self._begin_edit()
        with QtCore.QSignalBlocker(self.slider):
            self.slider.setValue(self._value_to_slider(value))
        self.valueChanged.emit(value, False)

    def _on_spin_committed(self) -> None:
        if self._edit_start_value is None:
            return
        self.spin.interpretText()
        value = self.value()
        if self._edit_start_value is not None and math.isclose(
            value, self._edit_start_value, abs_tol=10 ** -self._decimals
        ):
            self._cancel_edit()
            return
        self.valueChanged.emit(self.value(), True)
        self._edit_start_value = None

    def _begin_edit(self) -> None:
        if self._edit_start_value is None:
            self._edit_start_value = self.value()
            self.editStarted.emit()

    def _cancel_edit(self) -> None:
        self._edit_start_value = None
        self._edit_start_slider_value = None
        self.editCancelled.emit()

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self.spin:
            if event.type() == QtCore.QEvent.FocusIn:
                self._begin_edit()
            elif event.type() == QtCore.QEvent.KeyPress and event.key() == QtCore.Qt.Key_Escape:
                if self._edit_start_value is not None:
                    self.set_value(self._edit_start_value)
                self._cancel_edit()
                self.spin.clearFocus()
                return True
        return super().eventFilter(obj, event)


class _UnifiedSwatch(QtWidgets.QWidget):
    """Big colour swatch with overlaid hex text, corner revert button, and
    out-of-gamut indicator.

    The widget paints the fill itself; child widgets (hex line edit, revert
    button, OOG label) are positioned absolutely in :meth:`resizeEvent` so the
    overlay controls can sit *on top* of the colour fill without an intermediate
    layout.
    """

    hex_committed = QtCore.pyqtSignal(str)
    edit_started = QtCore.pyqtSignal()
    edit_cancelled = QtCore.pyqtSignal()
    revert_clicked = QtCore.pyqtSignal()
    _INK_STYLES: dict[str, tuple[str, str, str]] = {}

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_SWATCH_HEIGHT)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumWidth(48)
        self._colour = QtGui.QColor(0, 0, 0)
        self._hex_text = "#000000"
        self._oog_visible = False

        self._oog_label = QtWidgets.QLabel("⚠", self)
        oog_font = self._oog_label.font()
        oog_font.setBold(True)
        oog_font.setPointSizeF(oog_font.pointSizeF() + 1.0)
        self._oog_label.setFont(oog_font)
        self._oog_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._oog_label.setToolTip("Out of sRGB gamut")
        self._oog_label.setVisible(False)

        self._hex_edit = QtWidgets.QLineEdit(self)
        self._hex_edit.setMaxLength(7)
        self._hex_edit.setValidator(
            QtGui.QRegularExpressionValidator(
                QtCore.QRegularExpression(r"#?[0-9A-Fa-f]{6}"),
                self._hex_edit,
            )
        )
        self._hex_edit.setAlignment(QtCore.Qt.AlignCenter)
        hex_font = self._hex_edit.font()
        hex_font.setStyleHint(QtGui.QFont.Monospace)
        hex_font.setFamily("monospace")
        hex_font.setPointSizeF(hex_font.pointSizeF() + 2.0)
        hex_font.setBold(True)
        self._hex_edit.setFont(hex_font)
        self._hex_edit.setFrame(False)
        self._hex_edit.setStyleSheet("QLineEdit { background: transparent; border: none; }")
        self._hex_edit.setReadOnly(True)
        self._hex_edit.setCursor(QtCore.Qt.IBeamCursor)
        self._hex_edit.installEventFilter(self)
        self._hex_edit.editingFinished.connect(self._on_hex_finished)
        self._editing = False
        self._edit_start_hex = self._hex_text
        self._suppress_finish = False
        self._ink_name: str | None = None

        self._revert_button = QtWidgets.QToolButton(self)
        self._revert_button.setText("↶")
        self._revert_button.setFixedSize(_CORNER_BUTTON_SIZE, _CORNER_BUTTON_SIZE)
        self._revert_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._revert_button.setAutoRaise(True)
        self._revert_button.setEnabled(False)
        self._revert_button.setToolTip("No previous colour")
        self._revert_button.clicked.connect(self.revert_clicked.emit)

    # -- Public state --------------------------------------------------

    def set_srgb8(self, srgb8: tuple[int, int, int]) -> None:
        self._colour = _qcolor_from_srgb8(srgb8)
        self._hex_text = self._colour.name(QtGui.QColor.HexRgb)
        self._sync_hex_editor()
        self._apply_ink_styles()
        self.update()

    def _sync_hex_editor(self) -> None:
        if not self._editing and self._hex_edit.text().lower() != self._hex_text:
            self._suppress_finish = True
            try:
                self._hex_edit.setText(self._hex_text)
            finally:
                self._suppress_finish = False

    def set_oog_visible(self, visible: bool) -> None:
        self._oog_visible = bool(visible)
        self._oog_label.setVisible(self._oog_visible)
        self._apply_ink_styles()

    def set_revert_target(self, hex_text: str | None) -> None:
        if hex_text is None:
            self._revert_button.setEnabled(False)
            self._revert_button.setToolTip("No previous colour")
            return
        self._revert_button.setEnabled(True)
        tip = (
            f"Revert to <b>{hex_text}</b> "
            f"<span style='background:{hex_text};'>&nbsp;&nbsp;&nbsp;&nbsp;</span>"
        )
        self._revert_button.setToolTip(tip)

    @property
    def hex_text(self) -> str:
        return self._hex_text

    # -- Painting / interactions --------------------------------------

    def _apply_ink_styles(self) -> None:
        r, g, b = self._colour.red(), self._colour.green(), self._colour.blue()
        ink = _ink_for(r, g, b)
        ink_name = ink.name()
        if ink_name == self._ink_name:
            return
        self._ink_name = ink_name
        hex_style, oog_style, revert_style = self._styles_for_ink(ink_name)
        self._hex_edit.setStyleSheet(hex_style)
        # Match the OOG icon and the revert glyph to the ink colour.
        self._oog_label.setStyleSheet(oog_style)
        self._revert_button.setStyleSheet(revert_style)

    @classmethod
    def _styles_for_ink(cls, ink_name: str) -> tuple[str, str, str]:
        styles = cls._INK_STYLES.get(ink_name)
        if styles is not None:
            return styles
        styles = (
            f"QLineEdit {{ background: transparent; border: none; color: {ink_name}; }}",
            f"color: {ink_name}; background: transparent;",
            f"QToolButton {{ color: {ink_name}; background: transparent; border: none; }}"
            f"QToolButton:hover {{ background: rgba(127,127,127,80); border-radius: 3px; }}"
            f"QToolButton:disabled {{ color: rgba(127,127,127,160); }}",
        )
        cls._INK_STYLES[ink_name] = styles
        return styles

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(rect, self._colour)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 120), 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(rect)
        painter.end()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        margin = 4
        # OOG indicator top-left.
        self._oog_label.adjustSize()
        self._oog_label.move(margin, margin)
        # Revert button top-right.
        self._revert_button.move(
            self.width() - _CORNER_BUTTON_SIZE - margin, margin
        )
        # Hex edit centred horizontally, vertically centred in the swatch,
        # inset so it doesn't overlap the corner controls.
        edit_height = self._hex_edit.sizeHint().height()
        side_inset = _CORNER_BUTTON_SIZE + margin * 2
        self._hex_edit.setGeometry(
            side_inset,
            (self.height() - edit_height) // 2,
            max(40, self.width() - side_inset * 2),
            edit_height,
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.LeftButton and self._hex_edit.geometry().contains(event.pos()):
            self._enter_edit_mode()
            return
        super().mousePressEvent(event)

    def _enter_edit_mode(self) -> None:
        if self._editing:
            return
        self._editing = True
        self._edit_start_hex = self._hex_text
        self._hex_edit.setReadOnly(False)
        self._hex_edit.setFocus(QtCore.Qt.MouseFocusReason)
        self._hex_edit.selectAll()
        self.edit_started.emit()

    def _leave_edit_mode(self) -> None:
        if not self._editing:
            return
        self._editing = False
        self._hex_edit.setReadOnly(True)
        # Reset to the canonical text in case the user typed garbage.
        self._suppress_finish = True
        try:
            self._hex_edit.setText(self._hex_text)
        finally:
            self._suppress_finish = False
        self.edit_cancelled.emit()

    def _on_hex_finished(self) -> None:
        if self._suppress_finish or not self._editing:
            return
        text = self._hex_edit.text()
        self._editing = False
        self._hex_edit.setReadOnly(True)
        if text.strip().lower() == self._edit_start_hex.lower():
            self.edit_cancelled.emit()
            return
        self.hex_committed.emit(text)

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self._hex_edit and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Escape:
                self._leave_edit_mode()
                self.setFocus(QtCore.Qt.OtherFocusReason)
                return True
        return super().eventFilter(obj, event)


class ReadoutPanel(QtWidgets.QWidget):
    """Unified swatch + L/C/H gradient sliders.

    Emits :attr:`previewed` while a slider is being dragged or a spinbox/hex
    field is being edited; emits :attr:`committed` on slider release, spin
    Enter/blur, hex Enter, and revert-button click.
    """

    previewed = QtCore.pyqtSignal(object)
    committed = QtCore.pyqtSignal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._current: PresentedColour | None = None
        self._previous: PresentedColour | None = None
        self._state = _ReadoutState.IDLE
        self._latched_colour: _LatchedColour | None = None

        self._swatch = _UnifiedSwatch(self)
        self._swatch.edit_started.connect(self._begin_edit)
        self._swatch.edit_cancelled.connect(self._cancel_edit)
        self._swatch.hex_committed.connect(self._on_hex_committed)
        self._swatch.revert_clicked.connect(self._on_previous_clicked)

        self._row_l = _AxisRow("L", 0.0, 1.0, _STEP_L, 3, self)
        self._row_c = _AxisRow("C", 0.0, color_math.SRGB_MAX_CHROMA, _STEP_C, 3, self)
        self._row_h = _AxisRow("H", 0.0, 360.0, _STEP_H, 1, self)

        self._row_l.valueChanged.connect(self._on_l_changed)
        self._row_c.valueChanged.connect(self._on_c_changed)
        self._row_h.valueChanged.connect(self._on_h_changed)
        for row in (self._row_l, self._row_c, self._row_h):
            row.editStarted.connect(self._begin_edit)
            row.editCancelled.connect(self._cancel_edit)

        self._build_layout()
        initial_srgb8 = tuple(int(v) for v in color_math.oklab_to_srgb8(np.array([0.5, 0.0, 0.0], dtype=float)))
        self._row_l.set_value(0.5)
        self._row_c.set_value(0.0)
        self._row_h.set_value(0.0)
        self._swatch.set_srgb8(initial_srgb8)
        self._refresh_tracks(0.5, 0.0, 0.0)
        self._refresh_handle_fallback(initial_srgb8)
        self._previous = None
        self._swatch.set_revert_target(None)

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(self._swatch)
        layout.addWidget(self._row_l)
        layout.addWidget(self._row_c)
        layout.addWidget(self._row_h)

    # -- Public API ----------------------------------------------------

    @property
    def readout_state(self) -> str:
        return self._state.value

    @property
    def hue_intent(self) -> float:
        if self._state is _ReadoutState.EDITING:
            return math.radians(self._row_h.value()) % math.tau
        if self._current is not None and color_math.is_achromatic_chroma(self._current.intent.chroma):
            return math.radians(self._row_h.value()) % math.tau
        if self._current is None:
            return 0.0
        return self._current.intent.hue

    @property
    def _current_intent(self) -> ColourIntent | None:
        return None if self._current is None else self._current.intent

    @property
    def _previous_intent(self) -> ColourIntent | None:
        return None if self._previous is None else self._previous.intent

    @property
    def _current_oklab(self) -> np.ndarray | None:
        return None if self._current is None else self._current.paint_oklab

    @property
    def _previous_oklab(self) -> np.ndarray | None:
        return None if self._previous is None else self._previous.paint_oklab

    def show_colour(
        self,
        colour: PresentedColour | None,
        kind: object | None = None,
        *,
        model_factory: object | None = None,
    ) -> None:
        require_presented_colour(colour)
        # ``model_factory`` is part of the shared dock ``ColourView`` contract
        # but is selector-only; the readout has no slice model and ignores it.
        if kind is ChangeKind.INITIAL:
            # INITIAL is a seed, not a user commit: adopt it as current *and*
            # the revert baseline so a placeholder seed never becomes a bogus
            # previous colour.
            self._set_current_colour(colour, committed=False)
            self.set_previous_colour(colour)
            return
        self._set_current_colour(colour, committed=kind is not ChangeKind.PREVIEW)

    def _set_current_colour(
        self,
        colour: PresentedColour | None,
        *,
        committed: bool,
    ) -> None:
        if colour is None:
            return
        if self._state is _ReadoutState.EDITING:
            self._latched_colour = _LatchedColour(
                colour=colour,
                committed=committed,
            )
            return
        self._apply_colour(colour, committed=committed)

    def set_previous_colour(
        self,
        colour: PresentedColour | None,
    ) -> None:
        require_presented_colour(colour)
        if colour is None:
            self._previous = None
            self._swatch.set_revert_target(None)
            return
        self._previous = colour
        self._swatch.set_revert_target(self._hex_for_colour(self._previous))

    # -- Internal sync -------------------------------------------------

    def _apply_colour(self, colour: PresentedColour, *, committed: bool) -> None:
        if (
            committed
            and self._current is not None
            and colour.intent != self._current.intent
        ):
            self._previous = self._current
            self._swatch.set_revert_target(self._hex_for_colour(self._previous))
        self._current = colour
        self._sync_widgets_to_colour(colour)

    def _hex_for_colour(self, colour: PresentedColour) -> str:
        return _qcolor_from_srgb8(colour.srgb8).name(QtGui.QColor.HexRgb)

    def _sync_widgets_to_colour(self, colour: PresentedColour) -> None:
        intent = colour.intent
        l, c, h = intent.selector_lch
        self._row_l.set_value(float(l))
        self._row_c.set_value(float(c))
        self._row_h.set_value(math.degrees(h))
        self._sync_readout_presentation(colour, float(l), float(c), float(h))

    def _sync_readout_presentation(
        self,
        colour: PresentedColour,
        lightness: float,
        chroma: float,
        hue: float,
    ) -> None:
        self._swatch.set_srgb8(colour.srgb8)
        self._swatch.set_oog_visible(not colour.in_gamut)
        self._refresh_tracks(lightness, chroma, hue)
        self._refresh_handle_fallback(colour.srgb8)

    def _refresh_handle_fallback(self, srgb8: tuple[int, int, int]) -> None:
        colour = _qcolor_from_srgb8(srgb8)
        for row in (self._row_l, self._row_c, self._row_h):
            row.slider.set_fallback_colour(colour)

    def _refresh_tracks(self, lightness: float, chroma: float, hue: float) -> None:
        # Track widths can be 0 before the widget is laid out; skip then and
        # rely on the next sync (showEvent / resize) to populate.
        for axis, row, fixed in (
            (renderers.AXIS_L, self._row_l, (chroma, hue)),
            (renderers.AXIS_C, self._row_c, (lightness, hue)),
            (renderers.AXIS_H, self._row_h, (lightness, chroma)),
        ):
            slider = row.slider
            width = max(2, slider.width() - _HANDLE_WIDTH)
            height = max(2, slider.height() - 4)
            chroma_floor = _H_TRACK_CHROMA_FLOOR if axis == renderers.AXIS_H else 0.0
            key = (
                axis,
                round(fixed[0], 4),
                round(fixed[1], 4),
                width,
                height,
                chroma_floor,
            )
            if slider.cache_key() == key:
                continue
            rgba = renderers.render_axis_track(
                axis,
                fixed,
                color_math.SRGB_MAX_CHROMA,
                (width, height),
                hue_chroma_floor=chroma_floor,
            )
            slider.set_track(rgba)
            slider.set_cache_key(key)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._current is not None:
            l, c, h = self._current.selector_lch
            self._refresh_tracks(l, c, h)

    # -- Slot wiring ---------------------------------------------------

    def _current_lch(self) -> tuple[float, float, float]:
        if self._state is _ReadoutState.EDITING:
            return (
                self._row_l.value(),
                self._row_c.value(),
                math.radians(self._row_h.value()) % math.tau,
            )
        if self._current is None:
            return 0.5, 0.0, self.hue_intent
        return (
            self._row_l.value(),
            self._row_c.value(),
            math.radians(self._row_h.value()) % math.tau,
        )

    def _emit_from_lch(self, lightness: float, chroma: float, hue_rad: float, committed: bool) -> None:
        intent = ColourIntent.from_lch(lightness, chroma, hue_rad)
        if committed:
            self._finish_user_commit(intent)
            return
        self._begin_edit()
        self.previewed.emit(intent)

    def _on_l_changed(self, value: float, committed: bool) -> None:
        _, c, h = self._current_lch()
        self._emit_from_lch(value, c, h, committed)

    def _on_c_changed(self, value: float, committed: bool) -> None:
        l, _, h = self._current_lch()
        self._emit_from_lch(l, value, h, committed)

    def _on_h_changed(self, value_degrees: float, committed: bool) -> None:
        l, c, _ = self._current_lch()
        self._emit_from_lch(l, c, math.radians(value_degrees) % math.tau, committed)

    def _on_hex_committed(self, text: str) -> None:
        oklab = hex_to_oklab(text)
        if oklab is None:
            # Restore the swatch to the current colour on malformed input.
            if self._current is not None:
                self._swatch.set_srgb8(self._current.srgb8)
            self._finish_edit(_EditExit.CANCEL)
            return
        self._finish_user_commit(ColourIntent.from_oklab(oklab, achromatic_hue=self.hue_intent))

    def _on_previous_clicked(self) -> None:
        if self._previous is None:
            return
        self._finish_user_commit(self._previous.intent)

    def _begin_edit(self) -> None:
        if self._state is _ReadoutState.EDITING:
            return
        self._state = _ReadoutState.EDITING
        self._latched_colour = None

    def _cancel_edit(self) -> None:
        self._finish_edit(_EditExit.CANCEL)

    def _finish_user_commit(self, intent: ColourIntent) -> None:
        self._finish_edit(_EditExit.COMMIT)
        self.committed.emit(intent)

    def _finish_edit(self, exit_kind: _EditExit) -> None:
        if self._state is _ReadoutState.IDLE:
            return
        latched = self._latched_colour
        self._state = _ReadoutState.IDLE
        self._latched_colour = None
        if exit_kind is _EditExit.CANCEL and latched is not None:
            self._apply_colour(latched.colour, committed=latched.committed)
