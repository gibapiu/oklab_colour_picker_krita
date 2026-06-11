"""Unified readout swatch widget."""

from __future__ import annotations

import numpy as np
from oklab_colour_picker.infrastructure.qt_facade import QtCore, QtGui, QtWidgets, event_pos

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.ui.readout.style import (
    CORNER_BUTTON_SIZE,
    SWATCH_HEIGHT,
    ink_for,
    qcolor_from_srgb8,
)


def hex_to_oklab(text: str) -> np.ndarray | None:
    """Parse ``#rrggbb`` / ``rrggbb`` to OKLab; ``None`` on malformed input."""

    candidate = (text or "").strip()
    if candidate.startswith("#"):
        candidate = candidate[1:]
    if len(candidate) != 6:
        return None
    try:
        rgb = np.array(
            [
                int(candidate[0:2], 16),
                int(candidate[2:4], 16),
                int(candidate[4:6], 16),
            ],
            dtype=float,
        )
    except ValueError:
        return None
    return color_math.srgb_to_oklab(rgb / 255.0)


class UnifiedSwatch(QtWidgets.QWidget):
    """Colour swatch with overlaid hex text, revert button, and OOG indicator."""

    hex_committed = QtCore.pyqtSignal(str)
    edit_started = QtCore.pyqtSignal()
    edit_cancelled = QtCore.pyqtSignal()
    revert_clicked = QtCore.pyqtSignal()
    _INK_STYLES: dict[str, tuple[str, str, str]] = {}

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(SWATCH_HEIGHT)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )
        self.setMinimumWidth(48)
        self._colour = QtGui.QColor(0, 0, 0)
        self._hex_text = "#000000"
        self._oog_visible = False

        self._oog_label = QtWidgets.QLabel("⚠", self)
        oog_font = self._oog_label.font()
        oog_font.setBold(True)
        oog_font.setPointSizeF(oog_font.pointSizeF() + 1.0)
        self._oog_label.setFont(oog_font)
        self._oog_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
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
        self._hex_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        hex_font = self._hex_edit.font()
        hex_font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        hex_font.setFamily("monospace")
        hex_font.setPointSizeF(hex_font.pointSizeF() + 2.0)
        hex_font.setBold(True)
        self._hex_edit.setFont(hex_font)
        self._hex_edit.setFrame(False)
        self._hex_edit.setStyleSheet("QLineEdit { background: transparent; border: none; }")
        self._hex_edit.setReadOnly(True)
        self._hex_edit.setCursor(QtCore.Qt.CursorShape.IBeamCursor)
        self._hex_edit.installEventFilter(self)
        self._hex_edit.editingFinished.connect(self._on_hex_finished)
        self._editing = False
        self._edit_start_hex = self._hex_text
        self._suppress_finish = False
        self._ink_name: str | None = None

        self._revert_button = QtWidgets.QToolButton(self)
        self._revert_button.setText("↶")
        self._revert_button.setFixedSize(CORNER_BUTTON_SIZE, CORNER_BUTTON_SIZE)
        self._revert_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self._revert_button.setAutoRaise(True)
        self._revert_button.setEnabled(False)
        self._revert_button.setToolTip("No previous colour")
        self._revert_button.clicked.connect(self.revert_clicked.emit)

    def set_srgb8(self, srgb8: tuple[int, int, int]) -> None:
        self._colour = qcolor_from_srgb8(srgb8)
        self._hex_text = self._colour.name(QtGui.QColor.NameFormat.HexRgb)
        self._sync_hex_editor()
        self._apply_ink_styles()
        self.update()

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

    def _sync_hex_editor(self) -> None:
        if not self._editing and self._hex_edit.text().lower() != self._hex_text:
            self._suppress_finish = True
            try:
                self._hex_edit.setText(self._hex_text)
            finally:
                self._suppress_finish = False

    def _apply_ink_styles(self) -> None:
        r, g, b = self._colour.red(), self._colour.green(), self._colour.blue()
        ink_name = ink_for(r, g, b).name()
        if ink_name == self._ink_name:
            return
        self._ink_name = ink_name
        hex_style, oog_style, revert_style = self._styles_for_ink(ink_name)
        self._hex_edit.setStyleSheet(hex_style)
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
        painter.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        painter.end()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        margin = 4
        self._oog_label.adjustSize()
        self._oog_label.move(margin, margin)
        self._revert_button.move(self.width() - CORNER_BUTTON_SIZE - margin, margin)
        edit_height = self._hex_edit.sizeHint().height()
        side_inset = CORNER_BUTTON_SIZE + margin * 2
        self._hex_edit.setGeometry(
            side_inset,
            (self.height() - edit_height) // 2,
            max(40, self.width() - side_inset * 2),
            edit_height,
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.MouseButton.LeftButton and self._hex_edit.geometry().contains(event_pos(event)):
            self._enter_edit_mode()
            return
        super().mousePressEvent(event)

    def _enter_edit_mode(self) -> None:
        if self._editing:
            return
        self._editing = True
        self._edit_start_hex = self._hex_text
        self._hex_edit.setReadOnly(False)
        self._hex_edit.setFocus(QtCore.Qt.FocusReason.MouseFocusReason)
        self._hex_edit.selectAll()
        self.edit_started.emit()

    def _leave_edit_mode(self) -> None:
        if not self._editing:
            return
        self._editing = False
        self._hex_edit.setReadOnly(True)
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
        if obj is self._hex_edit and event.type() == QtCore.QEvent.Type.KeyPress:
            if event.key() == QtCore.Qt.Key.Key_Escape:
                self._leave_edit_mode()
                self.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
                return True
        return super().eventFilter(obj, event)
