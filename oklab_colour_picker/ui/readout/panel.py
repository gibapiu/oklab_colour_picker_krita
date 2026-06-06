"""Readout panel: swatch plus L/C/H gradient sliders."""

from __future__ import annotations

import math

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.domain.colour_presentation import (
    PresentedColour,
    require_presented_colour,
)
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.app.controller import ChangeKind
from oklab_colour_picker.domain.readout_interaction import (
    EditExit,
    ReadoutAction,
    ReadoutResult,
    ReadoutSession,
    ReadoutState,
)
from oklab_colour_picker.ui.readout.axis import AxisTrackPresenter, ReadoutAxisRows
from oklab_colour_picker.ui.readout.swatch import UnifiedSwatch, hex_to_oklab
from oklab_colour_picker.ui.readout.style import qcolor_from_srgb8


class ReadoutPanel(QtWidgets.QWidget):
    """Unified swatch + L/C/H gradient sliders."""

    previewed = QtCore.pyqtSignal(object)
    committed = QtCore.pyqtSignal(object)

    def __init__(
        self,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._session = ReadoutSession()
        self._track_presenter = AxisTrackPresenter()

        self._swatch = UnifiedSwatch(self)
        self._swatch.edit_started.connect(self._begin_edit)
        self._swatch.edit_cancelled.connect(self._cancel_edit)
        self._swatch.hex_committed.connect(self._on_hex_committed)
        self._swatch.revert_clicked.connect(self._on_previous_clicked)

        self._rows = ReadoutAxisRows.create(self)
        self._row_l = self._rows.lightness
        self._row_c = self._rows.chroma
        self._row_h = self._rows.hue

        self._row_l.valueChanged.connect(self._on_l_changed)
        self._row_c.valueChanged.connect(self._on_c_changed)
        self._row_h.valueChanged.connect(self._on_h_changed)
        for row in self._rows.as_tuple():
            row.editStarted.connect(self._begin_edit)
            row.editCancelled.connect(self._cancel_edit)

        self._build_layout()
        initial_oklab = np.array([0.5, 0.0, 0.0], dtype=float)
        initial_srgb8 = tuple(int(v) for v in color_math.oklab_to_srgb8(initial_oklab))
        self._rows.set_lch(0.5, 0.0, 0.0)
        self._swatch.set_srgb8(initial_srgb8)
        self._refresh_tracks(0.5, 0.0, 0.0)
        self._rows.set_handle_fallback(initial_srgb8)
        self._swatch.set_revert_target(None)

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.addWidget(self._swatch)
        for row in self._rows.as_tuple():
            layout.addWidget(row)

    @property
    def readout_state(self) -> str:
        return self._session.state_name

    @property
    def hue_intent(self) -> float:
        if self._session.state is ReadoutState.EDITING:
            return math.radians(self._row_h.value()) % math.tau
        current = self._session.current
        if current is not None and color_math.is_achromatic_chroma(current.intent.chroma):
            return math.radians(self._row_h.value()) % math.tau
        if current is None:
            return 0.0
        return current.intent.hue

    @property
    def _current_oklab(self) -> np.ndarray | None:
        current = self._session.current
        return None if current is None else current.paint_oklab

    @property
    def _previous_oklab(self) -> np.ndarray | None:
        previous = self._session.previous
        return None if previous is None else previous.paint_oklab

    def show_colour(
        self,
        colour: PresentedColour | None,
        kind: object | None = None,
        *,
        model_factory: object | None = None,
    ) -> None:
        # Shared dock ColourView parameter. Selector widgets use it; readout has no model.
        _ = model_factory
        require_presented_colour(colour)
        if colour is None:
            return
        if kind is ChangeKind.INITIAL:
            result = self._session.seed_initial(colour)
        else:
            result = self._session.show_colour(
                colour,
                committed=kind is not ChangeKind.PREVIEW,
                preview=kind is ChangeKind.PREVIEW,
            )
        self._apply_session_result(result)
        self._sync_revert_target()

    def set_previous_colour(
        self,
        colour: PresentedColour | None,
    ) -> None:
        require_presented_colour(colour)
        self._session.set_previous_colour(colour)
        self._sync_revert_target()

    def _apply_session_result(self, result: ReadoutResult) -> None:
        if result.action is ReadoutAction.APPLY and result.colour is not None:
            self._sync_widgets_to_colour(result.colour)
        elif (
            result.action is ReadoutAction.SYNC_DRAFT_PRESENTATION
            and result.colour is not None
        ):
            l, c, h = result.colour.intent.selector_lch
            self._sync_readout_presentation(result.colour, float(l), float(c), float(h))

    def _sync_revert_target(self) -> None:
        previous = self._session.previous
        self._swatch.set_revert_target(None if previous is None else self._hex_for_colour(previous))

    def _hex_for_colour(self, colour: PresentedColour) -> str:
        return qcolor_from_srgb8(colour.srgb8).name(QtGui.QColor.HexRgb)

    def _sync_widgets_to_colour(self, colour: PresentedColour) -> None:
        l, c, h = colour.intent.selector_lch
        self._rows.set_lch(float(l), float(c), float(h))
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
        self._rows.set_handle_fallback(colour.srgb8)

    def _refresh_tracks(self, lightness: float, chroma: float, hue: float) -> None:
        self._track_presenter.refresh(self._rows, lightness, chroma, hue)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        current = self._session.current
        if current is not None:
            l, c, h = current.selector_lch
            self._refresh_tracks(l, c, h)

    def _current_lch(self) -> tuple[float, float, float]:
        if self._session.current is None:
            return 0.5, 0.0, self.hue_intent
        return self._rows.current_lch()

    def _emit_from_lch(
        self,
        lightness: float,
        chroma: float,
        hue_rad: float,
        committed: bool,
    ) -> None:
        intent = ColourIntent.from_lch(lightness, chroma, hue_rad)
        if committed:
            self._finish_user_commit(intent)
            return
        self._begin_edit()
        self._session.set_draft(intent)
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
            self._finish_edit(EditExit.CANCEL)
            return
        self._finish_user_commit(
            ColourIntent.from_oklab(oklab, achromatic_hue=self.hue_intent)
        )

    def _on_previous_clicked(self) -> None:
        previous = self._session.previous
        if previous is None:
            return
        self._finish_user_commit(previous.intent)

    def _begin_edit(self) -> None:
        self._session.begin_edit()

    def _cancel_edit(self) -> None:
        self._finish_edit(EditExit.CANCEL)

    def _finish_user_commit(self, intent: ColourIntent) -> None:
        self._finish_edit(EditExit.COMMIT)
        self.committed.emit(intent)

    def _finish_edit(self, exit_kind: EditExit) -> None:
        result = self._session.finish_edit(exit_kind)
        self._apply_session_result(result)
        self._sync_revert_target()
