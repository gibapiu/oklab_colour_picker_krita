"""Qt selector widget backed by the pure selector interaction facade."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker import renderers, selector_interaction
from oklab_colour_picker.colour_presentation import PresentedColour
from oklab_colour_picker.colour_state import ColourIntent
from oklab_colour_picker.controller import normalize_oklab_for_krita
from oklab_colour_picker.models.base import positions_close
from oklab_colour_picker.selector_interaction import Indicator, Pick, PickResult, Ring
from oklab_colour_picker.selector_models import SelectorModel, SelectorSelection


@dataclass(frozen=True)
class _SelectedColour:
    intent: ColourIntent
    presentation: PresentedColour | None = None

    @classmethod
    def from_intent(cls, colour: ColourIntent | np.ndarray | Sequence[float]) -> "_SelectedColour":
        if isinstance(colour, ColourIntent):
            return cls(colour)
        return cls(ColourIntent.from_value(colour))

    @classmethod
    def from_presented(cls, colour: PresentedColour) -> "_SelectedColour":
        return cls(colour.intent, colour)

    @classmethod
    def from_selector_selection(
        cls,
        selection: SelectorSelection,
    ) -> "_SelectedColour":
        return cls.from_intent(ColourIntent.from_lch(*selection.lch))

    @property
    def paint_oklab(self) -> np.ndarray:
        return self.intent.paint_oklab

    @property
    def selector_lch(self) -> tuple[float, float, float]:
        return self.intent.selector_lch


class SelectorWidget(QtWidgets.QWidget):
    """Paint a selector model and translate Qt events into interaction commands."""

    previewed = QtCore.pyqtSignal(object)
    committed = QtCore.pyqtSignal(object)

    def __init__(
        self,
        model: SelectorModel,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._selection: _SelectedColour | None = None
        self._interaction = selector_interaction.SelectorInteraction()
        self._image_cache_key: tuple[SelectorModel, int, int] | None = None
        self._image_cache_buffer: np.ndarray | None = None
        self._image_cache: QtGui.QImage | None = None
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(32, 32)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    # -- State-machine surface ----------------------------------------

    @property
    def state(self) -> str:
        return self._interaction.state_name

    @property
    def anchor(self) -> tuple[float, float] | None:
        return self._interaction.anchor

    @property
    def transition_log(self) -> tuple[str, ...]:
        return self._interaction.transition_log

    def _dispatch(
        self, command: selector_interaction.SelectorCommand
    ) -> selector_interaction.InteractionResult:
        """Run one command through the machine and repaint."""

        result = self._interaction.dispatch(self, command)
        self.update()
        return result

    # -- Colour surface ------------------------------------------------

    @property
    def model(self) -> SelectorModel:
        return self._model

    @property
    def selected_colour(self) -> np.ndarray | None:
        return None if self._selection is None else self._selection.paint_oklab

    def set_model(self, model: SelectorModel) -> None:
        if self._model is model:
            return
        self._model = model
        self._clear_image_cache()
        self._dispatch(selector_interaction.Reframe())

    def show_colour(
        self,
        colour: PresentedColour | None,
        kind: object | None = None,
        *,
        model_factory: Callable[[], SelectorModel] | None = None,
    ) -> None:
        _require_presented_colour(colour)
        result = self._interaction.dispatch(
            self,
            selector_interaction.Broadcast(colour),
        )
        if result.absorbed_echo and colour is not None:
            self.set_colour(colour)
        if model_factory is not None and result.rendered_broadcast:
            self._apply_model(model_factory())
        elif model_factory is not None and result.absorbed_echo:
            model = model_factory()
            if self._model != model:
                self.set_model(model)
                self.set_colour(colour)
        self.update()

    def set_selected_colour(
        self,
        colour: PresentedColour | None,
        kind: object | None = None,
        *,
        model_factory: Callable[[], SelectorModel] | None = None,
    ) -> None:
        self.show_colour(colour, kind, model_factory=model_factory)

    def indicator_position(self) -> tuple[float, float] | None:
        if self._selection is None:
            return None
        return self._interaction.indicator_position(self)

    # -- Ctx port (used only by the state machine) ---------------------

    @property
    def colour(self) -> ColourIntent | None:
        return None if self._selection is None else self._selection.intent

    def set_colour(
        self,
        colour: PresentedColour | ColourIntent | np.ndarray | Sequence[float] | None,
    ) -> None:
        if colour is None:
            self._selection = None
            return
        if isinstance(colour, PresentedColour):
            self._selection = _SelectedColour.from_presented(colour)
            return
        self._selection = _SelectedColour.from_intent(colour)

    def preview(self, colour: ColourIntent | np.ndarray | None) -> None:
        self.previewed.emit(_emit_payload(colour))

    def commit(self, colour: ColourIntent | np.ndarray) -> None:
        self.committed.emit(_emit_payload(colour))

    def pick(self, point: tuple[float, float]) -> Pick:
        selection = self._model.selection_at_position(point, _widget_size(self))
        if selection is not None:
            return PickResult.exact(_SelectedColour.from_selector_selection(selection).intent)
        snapped = self._model.snapped_selector_selection_at_position(point, _widget_size(self))
        if snapped is not None:
            return PickResult.snapped(
                _SelectedColour.from_selector_selection(snapped).intent,
                snapped.position,
            )
        return PickResult.invalid()

    def intent_at(self, point: tuple[float, float]) -> ColourIntent | None:
        selection = self._model.selection_at_position((point[0], point[1]), _widget_size(self))
        if selection is None:
            return None
        return _SelectedColour.from_selector_selection(selection).intent

    @staticmethod
    def quantized_equal(
        a: ColourIntent | np.ndarray | None,
        b: ColourIntent | np.ndarray | None,
    ) -> bool:
        if a is None or b is None:
            return False
        return bool(
            np.array_equal(normalize_oklab_for_krita(_paint(a)), normalize_oklab_for_krita(_paint(b)))
        )

    def model_indicator(self) -> Indicator:
        if self._selection is None:
            return Indicator.nothing()
        spec = self._model.indicator_for_intent(self._selection.selector_lch, _widget_size(self))
        if spec is None:
            return Indicator.nothing()
        rings = [Ring(spec.desired, True)]
        fallback = None if self._selection.presentation is None else self._selection.presentation.fallback
        if fallback is None:
            return Indicator(tuple(rings))
        fallback_position = self._model.position_for_intent(
            fallback.fallback.selector_lch,
            _widget_size(self),
        )
        if (
            not fallback.in_gamut
            and fallback_position is not None
            and not positions_close(spec.desired, fallback_position)
        ):
            rings.append(Ring(fallback_position, False))
        return Indicator(tuple(rings))

    def model_position(self) -> tuple[float, float] | None:
        if self._selection is None:
            return None
        return self._model.position_for_intent(self._selection.selector_lch, _widget_size(self))

    # -- Qt event plumbing (input routing only) -----------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            event.ignore()
            return
        self.setFocus(QtCore.Qt.MouseFocusReason)
        self._dispatch(selector_interaction.PointerPress(_point(event)))
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        result = self._dispatch(selector_interaction.PointerMove(_point(event)))
        if not result.handled:
            event.ignore()
            return
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            event.ignore()
            return
        result = self._dispatch(selector_interaction.PointerRelease(_point(event)))
        if not result.handled:
            event.ignore()
            return
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._dispatch(selector_interaction.Reframe())
        super().resizeEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        self._dispatch(selector_interaction.FocusOut())
        super().focusOutEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        fallback = ((self.width() - 1.0) / 2.0, (self.height() - 1.0) / 2.0)
        position = self._interaction.navigation_origin(self, fallback)
        if position is None:
            event.ignore()
            return

        target = self._keyboard_target_position(position, event)
        if target is None:
            event.ignore()
            return
        intent = self.intent_at((target.x(), target.y()))
        if intent is None:
            event.ignore()
            return

        self._dispatch(
            selector_interaction.Navigation(
                (float(target.x()), float(target.y())), intent
            )
        )
        event.accept()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            event.accept()
            return
        if not self._is_keyboard_navigation_key(event.key()):
            event.ignore()
            return
        result = self._dispatch(selector_interaction.KeyRelease())
        if not result.handled:
            event.ignore()
            return
        event.accept()

    # -- Painting ------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setClipRect(event.rect())
        try:
            image = self._selector_image()
        except ValueError:
            painter.end()
            return
        painter.drawImage(0, 0, image)
        self._paint_indicator(painter)
        painter.end()

    def _paint_indicator(self, painter: QtGui.QPainter) -> None:
        indicator = self._interaction.indicator(self)
        if not indicator.rings:
            return
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setBrush(QtCore.Qt.NoBrush)
        for ring in indicator.rings:
            self._stroke_circle(painter, ring.position, solid=ring.solid)

    def _stroke_circle(self, painter: QtGui.QPainter, position: tuple[float, float], *, solid: bool) -> None:
        center = QtCore.QPointF(position[0], position[1])
        for colour, width in ((QtCore.Qt.black, 3.0), (QtCore.Qt.white, 1.5)):
            painter.setPen(_ring_pen(colour, width, solid=solid))
            painter.drawEllipse(center, 5.0, 5.0)

    def _selector_image(self) -> QtGui.QImage:
        key = (self._model, self.width(), self.height())
        if self._image_cache_key == key and self._image_cache is not None:
            return self._image_cache
        rgba = renderers.render_rgba(self._model, (self.width(), self.height()))
        bytes_per_line = int(rgba.strides[0])
        image = QtGui.QImage(
            rgba.data,
            self.width(),
            self.height(),
            bytes_per_line,
            QtGui.QImage.Format_RGBA8888,
        )
        self._image_cache_key = key
        self._image_cache_buffer = rgba
        self._image_cache = image
        return image

    def _clear_image_cache(self) -> None:
        self._image_cache_key = None
        self._image_cache_buffer = None
        self._image_cache = None

    def _apply_model(self, model: SelectorModel) -> None:
        if self._model == model:
            return
        self._model = model
        self._clear_image_cache()

    # -- Keyboard navigation maths ------------------------------------

    def _keyboard_target_position(
        self, position: tuple[float, float], event: QtGui.QKeyEvent
    ) -> QtCore.QPoint | None:
        x, y = position
        step = _keyboard_step(self.size(), event.modifiers())
        key = event.key()
        target_deltas = {
            QtCore.Qt.Key_Left: (-step, 0.0),
            QtCore.Qt.Key_Right: (step, 0.0),
            QtCore.Qt.Key_Up: (0.0, -step),
            QtCore.Qt.Key_Down: (0.0, step),
            QtCore.Qt.Key_Home: (-x, 0.0),
            QtCore.Qt.Key_End: (self.width() - 1.0 - x, 0.0),
            QtCore.Qt.Key_PageUp: (0.0, -y),
            QtCore.Qt.Key_PageDown: (0.0, self.height() - 1.0 - y),
        }
        if key in target_deltas:
            return self._nearest_valid_point(position, *target_deltas[key])
        return None

    def _nearest_valid_point(
        self, position: tuple[float, float], dx: float, dy: float
    ) -> QtCore.QPoint | None:
        start_x, start_y = position
        steps = max(1, int(max(abs(dx), abs(dy))))
        fractions = np.arange(steps, -1, -1, dtype=float) / steps
        x = np.rint(np.clip(start_x + dx * fractions, 0, self.width() - 1)).astype(float)
        y = np.rint(np.clip(start_y + dy * fractions, 0, self.height() - 1)).astype(float)
        _, valid = self._model.colors_at_positions(x, y, _widget_size(self))
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            index = int(valid_indices[0])
            return QtCore.QPoint(int(x[index]), int(y[index]))
        return None

    def _is_keyboard_navigation_key(self, key: int) -> bool:
        return key in {
            QtCore.Qt.Key_Left,
            QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Down,
            QtCore.Qt.Key_Home,
            QtCore.Qt.Key_End,
            QtCore.Qt.Key_PageUp,
            QtCore.Qt.Key_PageDown,
        }


def _widget_size(widget: QtWidgets.QWidget) -> tuple[int, int]:
    return widget.width(), widget.height()


def _point(event: QtGui.QMouseEvent) -> tuple[float, float]:
    return float(event.pos().x()), float(event.pos().y())


def _ring_pen(colour: QtCore.Qt.GlobalColor, width: float, *, solid: bool) -> QtGui.QPen:
    pen = QtGui.QPen(colour, width)
    if not solid:
        pen.setStyle(QtCore.Qt.DashLine)
        pen.setDashPattern([2.0, 2.0])
    return pen


def _paint(colour: PresentedColour | ColourIntent | np.ndarray | Sequence[float]) -> np.ndarray:
    if isinstance(colour, PresentedColour):
        return colour.paint_oklab
    if isinstance(colour, ColourIntent):
        return colour.paint_oklab
    return np.asarray(colour, dtype=float)


def _require_presented_colour(colour: PresentedColour | None) -> None:
    if colour is not None and not isinstance(colour, PresentedColour):
        raise TypeError("displayed selector colours must be PresentedColour")


def _emit_payload(
    colour: PresentedColour | ColourIntent | np.ndarray | Sequence[float] | None,
) -> object:
    if colour is None:
        return None
    if isinstance(colour, PresentedColour):
        return colour.intent
    if isinstance(colour, ColourIntent):
        return colour
    return np.asarray(colour, dtype=float).copy()


def _keyboard_step(size: QtCore.QSize, modifiers: QtCore.Qt.KeyboardModifiers) -> int:
    if modifiers & QtCore.Qt.ShiftModifier:
        return 1
    base = max(1, min(size.width(), size.height()) // 64)
    if modifiers & QtCore.Qt.ControlModifier:
        return max(1, base * 4)
    return base
