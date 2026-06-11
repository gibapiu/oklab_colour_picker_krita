"""L/C/H readout axis controls."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from oklab_colour_picker.infrastructure.qt_facade import QtCore, QtGui, QtWidgets, event_pos

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.render import renderers
from oklab_colour_picker.ui.readout.style import (
    HANDLE_BORDER,
    HANDLE_WIDTH,
    ink_for,
    qcolor_from_srgb8,
)


STEP_L = 0.01
STEP_C = 0.005
STEP_H = 1.0

H_TRACK_CHROMA_FLOOR = 0.06


class GradientSlider(QtWidgets.QSlider):
    """Horizontal slider with a custom-painted gradient track and hollow handle."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.setMinimum(0)
        self.setMaximum(1000)
        self.setSingleStep(1)
        self.setPageStep(10)
        probe = QtWidgets.QSpinBox()
        self.setFixedHeight(max(20, probe.sizeHint().height()))
        self._track_image: QtGui.QImage | None = None
        self._track_buffer: np.ndarray | None = None
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
            QtGui.QImage.Format.Format_RGBA8888,
        )
        self.update()

    def set_fallback_colour(self, colour: QtGui.QColor | None) -> None:
        self._fallback_colour = None if colour is None else QtGui.QColor(colour)
        self.update()

    def track_rect(self) -> QtCore.QRect:
        pad = HANDLE_WIDTH // 2
        return self.rect().adjusted(pad, 2, -pad, -2)

    def handle_x_center(self, track_rect: QtCore.QRect) -> int:
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
            return QtGui.QColor("#1e1e1e")
        buf = self._track_buffer
        rel = (x_center - track_rect.left()) / max(1, track_rect.width() - 1)
        col = int(round(np.clip(rel, 0.0, 1.0) * (buf.shape[1] - 1)))
        row = buf.shape[0] // 2
        r, g, b = int(buf[row, col, 0]), int(buf[row, col, 1]), int(buf[row, col, 2])
        return ink_for(r, g, b)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        track_rect = self.track_rect()
        if self._track_image is not None:
            painter.drawImage(track_rect, self._track_image)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 120), 1))
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(track_rect)

        x = self.handle_x_center(track_rect)
        handle_rect = self._handle_rect()
        if self._fallback_colour is not None:
            inner = QtCore.QRectF(handle_rect).adjusted(
                HANDLE_BORDER, HANDLE_BORDER, -HANDLE_BORDER, -HANDLE_BORDER
            )
            if inner.width() > 0 and inner.height() > 0:
                painter.fillRect(inner, self._fallback_colour)
        pen = QtGui.QPen(self._border_ink(x, track_rect), HANDLE_BORDER)
        pen.setJoinStyle(QtCore.Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        inset = HANDLE_BORDER / 2
        painter.drawRect(QtCore.QRectF(handle_rect).adjusted(inset, inset, -inset, -inset))
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        point = event_pos(event)
        self.setSliderDown(True)
        self._pressed_handle = self._handle_rect().contains(point)
        self._moved_since_press = False
        if not self._pressed_handle:
            self.setValue(self._value_at_x(point.x()))
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if not self.isSliderDown():
            super().mouseMoveEvent(event)
            return
        self._moved_since_press = True
        self.setValue(self._value_at_x(event_pos(event).x()))
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.MouseButton.LeftButton or not self.isSliderDown():
            super().mouseReleaseEvent(event)
            return
        if not (self._pressed_handle and not self._moved_since_press):
            self.setValue(self._value_at_x(event_pos(event).x()))
        self.setSliderDown(False)
        self._pressed_handle = False
        self._moved_since_press = False
        event.accept()

    def _value_at_x(self, x: int) -> int:
        track_rect = self.track_rect()
        return QtWidgets.QStyle.sliderValueFromPosition(
            self.minimum(),
            self.maximum(),
            x - track_rect.left(),
            max(1, track_rect.width() - 1),
            self._upside_down(),
        )

    def _upside_down(self) -> bool:
        right_to_left = self.layoutDirection() == QtCore.Qt.LayoutDirection.RightToLeft
        return right_to_left ^ self.invertedAppearance()

    def _handle_rect(self) -> QtCore.QRect:
        track_rect = self.track_rect()
        x = self.handle_x_center(track_rect)
        return QtCore.QRect(
            x - HANDLE_WIDTH // 2,
            self.rect().top(),
            HANDLE_WIDTH,
            self.rect().height() - 1,
        )


class AxisRow(QtWidgets.QWidget):
    """One readout row: label, gradient slider, numeric spinbox."""

    valueChanged = QtCore.pyqtSignal(float, bool)
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
        self._decimals = int(decimals)
        self._edit_start_value: float | None = None
        self._edit_start_slider_value: int | None = None

        label_widget = QtWidgets.QLabel(label, self)
        label_widget.setFixedWidth(14)
        label_widget.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.slider = GradientSlider(self)
        self.spin = QtWidgets.QDoubleSpinBox(self)
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(72)
        self.spin.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(label_widget, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.slider, 1, QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.spin, 0, QtCore.Qt.AlignmentFlag.AlignVCenter)

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
        self.valueChanged.emit(value, not self.slider.isSliderDown())

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
        if math.isclose(value, self._edit_start_value, abs_tol=10 ** -self._decimals):
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
            if event.type() == QtCore.QEvent.Type.FocusIn:
                self._begin_edit()
            elif event.type() == QtCore.QEvent.Type.KeyPress and event.key() == QtCore.Qt.Key.Key_Escape:
                if self._edit_start_value is not None:
                    self.set_value(self._edit_start_value)
                self._cancel_edit()
                self.spin.clearFocus()
                return True
        return super().eventFilter(obj, event)


@dataclass(frozen=True)
class ReadoutAxisRows:
    lightness: AxisRow
    chroma: AxisRow
    hue: AxisRow

    @classmethod
    def create(cls, parent: QtWidgets.QWidget) -> "ReadoutAxisRows":
        return cls(
            AxisRow("L", 0.0, 1.0, STEP_L, 3, parent),
            AxisRow("C", 0.0, color_math.SRGB_MAX_CHROMA, STEP_C, 3, parent),
            AxisRow("H", 0.0, 360.0, STEP_H, 1, parent),
        )

    def as_tuple(self) -> tuple[AxisRow, AxisRow, AxisRow]:
        return (self.lightness, self.chroma, self.hue)

    def set_lch(self, lightness: float, chroma: float, hue_rad: float) -> None:
        self.lightness.set_value(lightness)
        self.chroma.set_value(chroma)
        self.hue.set_value(math.degrees(hue_rad))

    def current_lch(self) -> tuple[float, float, float]:
        return (
            self.lightness.value(),
            self.chroma.value(),
            math.radians(self.hue.value()) % math.tau,
        )

    def set_handle_fallback(self, srgb8: tuple[int, int, int]) -> None:
        colour = qcolor_from_srgb8(srgb8)
        for row in self.as_tuple():
            row.slider.set_fallback_colour(colour)


@dataclass(frozen=True)
class AxisTrackKey:
    axis: str
    fixed0: float
    fixed1: float
    width: int
    height: int
    chroma_floor: float


class AxisTrackPresenter:
    """Render and cache readout slider tracks."""

    def __init__(self) -> None:
        self._cache: dict[int, AxisTrackKey] = {}

    def refresh(
        self,
        rows: ReadoutAxisRows,
        lightness: float,
        chroma: float,
        hue: float,
    ) -> None:
        for axis, row, fixed in (
            (renderers.AXIS_L, rows.lightness, (chroma, hue)),
            (renderers.AXIS_C, rows.chroma, (lightness, hue)),
            (renderers.AXIS_H, rows.hue, (lightness, chroma)),
        ):
            slider = row.slider
            width = max(2, slider.width() - HANDLE_WIDTH)
            height = max(2, slider.height() - 4)
            chroma_floor = H_TRACK_CHROMA_FLOOR if axis == renderers.AXIS_H else 0.0
            cache_id = id(slider)
            key = AxisTrackKey(
                axis=axis,
                fixed0=round(fixed[0], 4),
                fixed1=round(fixed[1], 4),
                width=width,
                height=height,
                chroma_floor=chroma_floor,
            )
            if self._cache.get(cache_id) == key:
                continue
            rgba = renderers.render_axis_track(
                axis,
                fixed,
                color_math.SRGB_MAX_CHROMA,
                (width, height),
                hue_chroma_floor=chroma_floor,
            )
            slider.set_track(rgba)
            self._cache[cache_id] = key
