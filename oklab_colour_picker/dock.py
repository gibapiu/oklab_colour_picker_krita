"""Qt dock content for the OKLab colour picker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np
from PyQt5 import QtCore, QtWidgets

from oklab_colour_picker import color_math
from oklab_colour_picker.colour_presentation import PresentedColour
from oklab_colour_picker.colour_state import ColourIntent
from oklab_colour_picker.controller import ChangeKind, ColourSnapshot
from oklab_colour_picker.selector_models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)
from oklab_colour_picker.widgets.readout_panel import ReadoutPanel
from oklab_colour_picker.widgets.selector import SelectorWidget


ColourListener = Callable[[ColourSnapshot], None]


class ColourView(Protocol):
    """The single inbound contract every dock view satisfies.

    The dock broadcasts uniformly through this one method; ``model_factory``
    is selector-only and ignored by views (e.g. the readout) that have no
    slice model. There is no per-view branching and no source tag.
    """

    def show_colour(
        self,
        colour: PresentedColour,
        kind: ChangeKind,
        *,
        model_factory: Callable[[], object] | None = None,
    ) -> None:
        ...


class DockController(Protocol):
    @property
    def selected_intent(self) -> ColourIntent | None:
        ...

    def set_preview_colour(self, oklab: ColourIntent | Sequence[float] | None) -> None:
        ...

    def request_foreground_commit(self, oklab: ColourIntent | Sequence[float] | None) -> None:
        ...

    def add_colour_listener(self, listener: ColourListener) -> None:
        ...

    def remove_colour_listener(self, listener: ColourListener) -> None:
        ...

    def sync_external_foreground(self, *, force: bool = False) -> bool:
        ...


class SelectorMode(str, Enum):
    LIGHTNESS_SLICE = "lightness_slice"
    HUE_LIGHTNESS_SLICE = "hue_lightness_slice"
    LIGHTNESS_CHROMA_SLICE = "lightness_chroma_slice"


ModelFactory = Callable[[float, float, float], object]
CoordinateFactory = Callable[[float, float, float], "SliceCoordinate"]
WidgetFactory = Callable[[object, QtWidgets.QWidget], SelectorWidget]

# OKLab -> OKLCh recovery can jitter by a few ulps for fixed hue/chroma slices.
# This epsilon is many orders below a visible slice step but large enough to
# make same-slice cache hits deterministic across normal float round-trips.
SLICE_COORDINATE_ROUNDTRIP_EPSILON = 1.0 / (255.0 ** 3)


@dataclass(frozen=True)
class ModeSpec:
    label: str
    object_name: str
    model_factory: ModelFactory
    coordinate_factory: CoordinateFactory
    widget_factory: WidgetFactory


@dataclass(frozen=True)
class SliceModelCacheEntry:
    """Cached selector model keyed by an equivalent fixed slice coordinate."""

    coordinate: "SliceCoordinate"
    model: object


class SliceCoordinate(Protocol):
    def equivalent_to(self, other: "SliceCoordinate") -> bool:
        ...


@dataclass(frozen=True)
class LinearSliceCoordinate:
    value: float

    def equivalent_to(self, other: SliceCoordinate) -> bool:
        return (
            isinstance(other, LinearSliceCoordinate)
            and math.isclose(
                self.value,
                other.value,
                rel_tol=0.0,
                abs_tol=SLICE_COORDINATE_ROUNDTRIP_EPSILON,
            )
        )


@dataclass(frozen=True)
class ChromaSliceCoordinate:
    value: float
    hue_when_achromatic: float

    def equivalent_to(self, other: SliceCoordinate) -> bool:
        if not isinstance(other, ChromaSliceCoordinate):
            return False
        if _both_achromatic_chroma(self.value, other.value):
            return (
                _circular_distance(self.hue_when_achromatic, other.hue_when_achromatic)
                <= SLICE_COORDINATE_ROUNDTRIP_EPSILON
            )
        if not math.isclose(
            self.value,
            other.value,
            rel_tol=0.0,
            abs_tol=SLICE_COORDINATE_ROUNDTRIP_EPSILON,
        ):
            return False
        return True


@dataclass(frozen=True)
class HueSliceCoordinate:
    radians: float

    def equivalent_to(self, other: SliceCoordinate) -> bool:
        return (
            isinstance(other, HueSliceCoordinate)
            and _circular_distance(self.radians, other.radians)
            <= SLICE_COORDINATE_ROUNDTRIP_EPSILON
        )


def _lightness_slice_model(lightness: float, _chroma: float, _hue: float) -> object:
    return LightnessSliceModel(lightness=lightness)


def _hue_lightness_slice_model(_lightness: float, chroma: float, hue: float) -> object:
    if color_math.is_achromatic_chroma(chroma):
        return HueLightnessSliceModel(chroma=chroma, achromatic_indicator_hue=hue)
    return HueLightnessSliceModel(chroma=chroma)


def _lightness_chroma_slice_model(_lightness: float, _chroma: float, hue: float) -> object:
    return LightnessChromaSliceModel(hue=hue)


def _lightness_coordinate(lightness: float, _chroma: float, _hue: float) -> SliceCoordinate:
    return LinearSliceCoordinate(lightness)


def _chroma_coordinate(_lightness: float, chroma: float, hue: float) -> SliceCoordinate:
    return ChromaSliceCoordinate(chroma, _hue_when_achromatic(chroma, hue))


def _hue_coordinate(_lightness: float, _chroma: float, hue: float) -> SliceCoordinate:
    return HueSliceCoordinate(hue % math.tau)


def _selector_widget(model: object, parent: QtWidgets.QWidget) -> SelectorWidget:
    return SelectorWidget(model, parent)


def _lightness_slice_widget(model: object, parent: QtWidgets.QWidget) -> SelectorWidget:
    from oklab_colour_picker.widgets.lightness_slice_disk import LightnessSliceDiskWidget

    return LightnessSliceDiskWidget(model, parent)


def _hue_lightness_slice_widget(model: object, parent: QtWidgets.QWidget) -> SelectorWidget:
    from oklab_colour_picker.widgets.hue_lightness_slice_disk import HueLightnessSliceDiskWidget

    return HueLightnessSliceDiskWidget(model, parent)


MODE_SPECS = {
    SelectorMode.LIGHTNESS_SLICE: ModeSpec(
        "Hue/Chroma",
        "lightness-slice-selector",
        _lightness_slice_model,
        _lightness_coordinate,
        _lightness_slice_widget,
    ),
    SelectorMode.HUE_LIGHTNESS_SLICE: ModeSpec(
        "Hue/Lightness",
        "hue-lightness-slice-selector",
        _hue_lightness_slice_model,
        _chroma_coordinate,
        _hue_lightness_slice_widget,
    ),
    SelectorMode.LIGHTNESS_CHROMA_SLICE: ModeSpec(
        "Lightness/Chroma",
        "lightness-chroma-slice-selector",
        _lightness_chroma_slice_model,
        _hue_coordinate,
        _selector_widget,
    ),
}

DEFAULT_COLOUR = np.array([0.5, 0.0, 0.0], dtype=float)


class ColourPickerDockPanel(QtWidgets.QWidget):
    """Build and synchronize the selector widgets shown inside the docker."""

    def __init__(
        self,
        controller: DockController,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller = controller
        self._view_seed_intent = _intent_or_default(controller.selected_intent)
        self._current_snapshot: ColourSnapshot | None = None
        self._selector_modes = tuple(MODE_SPECS)
        self._tabs = QtWidgets.QTabWidget(self)
        # SelectorWidget structurally satisfies ColourView; the readout does
        # too. The dock broadcasts to both through that one contract.
        self._selectors: dict[SelectorMode, SelectorWidget] = {}
        self._selector_model_cache: dict[SelectorMode, SliceModelCacheEntry] = {}
        self._readout_panel = ReadoutPanel(self)
        self._readout_panel.previewed.connect(self._preview_colour)
        self._readout_panel.committed.connect(self._commit_colour)
        self._build_selector_tabs()
        self._build_layout()
        self._tabs.currentChanged.connect(self._ensure_selector_for_tab)
        self._controller_subscription = ColourSubscription(controller, self._on_colour_changed)
        self.destroyed.connect(self._controller_subscription.disconnect)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._schedule_post_show_foreground_seed()

    def _schedule_post_show_foreground_seed(self) -> None:
        QtCore.QTimer.singleShot(0, self._seed_foreground_from_active_view)

    def _seed_foreground_from_active_view(self) -> None:
        self._controller.sync_external_foreground(force=True)

    @property
    def selector_widgets(self) -> tuple[SelectorWidget, ...]:
        return tuple(self.selector_for_mode(mode) for mode in SelectorMode)

    @property
    def mode(self) -> SelectorMode:
        index = self._tabs.currentIndex()
        if 0 <= index < len(self._selector_modes):
            return self._selector_modes[index]
        return self._selector_modes[0]

    @property
    def active_selector(self) -> SelectorWidget:
        return self.selector_for_mode(self.mode)

    def selector_for_mode(self, mode: SelectorMode | str) -> SelectorWidget:
        return self._ensure_selector(SelectorMode(mode))

    def set_mode(self, mode: SelectorMode | str) -> None:
        selector_mode = SelectorMode(mode)
        self._ensure_selector(selector_mode)
        self._tabs.setCurrentIndex(self._tab_index_for_mode(selector_mode))

    def set_selected_colour(
        self, oklab: Sequence[float] | None, *, committed: bool = True
    ) -> None:
        if committed:
            self._controller.request_foreground_commit(oklab)
        else:
            self._controller.set_preview_colour(oklab)

    def _on_colour_changed(self, snapshot: ColourSnapshot) -> None:
        self._show_on_views(snapshot)

    def _show_on_views(self, snapshot: ColourSnapshot) -> None:
        self._current_snapshot = snapshot
        colour = snapshot.colour
        self._view_seed_intent = colour.intent
        self._readout_panel.show_colour(colour, snapshot.kind)
        for mode, widget in self._selectors.items():
            widget.show_colour(
                colour,
                snapshot.kind,
                model_factory=self._selector_model_factory(mode, colour),
            )

    def _build_selector_tabs(self) -> None:
        for mode in self._selector_modes:
            if mode == self._selector_modes[0]:
                widget = self._ensure_selector(mode)
            else:
                widget = QtWidgets.QWidget(self)
                widget.setObjectName(f"{_mode_spec(mode).object_name}-placeholder")
            self._tabs.addTab(widget, _mode_spec(mode).label)

    def _ensure_selector_for_tab(self, index: int) -> None:
        if 0 <= index < len(self._selector_modes):
            self._ensure_selector(self._selector_modes[index])

    def _ensure_selector(self, mode: SelectorMode) -> SelectorWidget:
        existing = self._selectors.get(mode)
        if existing is not None:
            return existing

        seed = self._view_seed_intent
        widget = _build_selector_widget(
            mode,
            self._cached_model_for_colour(mode, seed),
            self,
        )
        widget.setObjectName(_mode_spec(mode).object_name)
        widget.previewed.connect(self._preview_colour)
        widget.committed.connect(self._commit_colour)
        self._selectors[mode] = widget
        if self._current_snapshot is not None:
            current_colour = self._current_snapshot.colour
            widget.show_colour(
                current_colour,
                self._current_snapshot.kind,
                model_factory=self._selector_model_factory(mode, current_colour),
            )

        index = self._tab_index_for_mode(mode)
        if index < self._tabs.count():
            current_index = self._tabs.currentIndex()
            placeholder = self._tabs.widget(index)
            self._tabs.removeTab(index)
            self._tabs.insertTab(index, widget, _mode_spec(mode).label)
            if placeholder is not None:
                placeholder.deleteLater()
            if current_index == index:
                self._tabs.setCurrentIndex(index)
        return widget

    def _tab_index_for_mode(self, mode: SelectorMode) -> int:
        return self._selector_modes.index(mode)

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self._tabs)
        layout.addWidget(self._readout_panel)

    def _selector_model_factory(
        self,
        mode: SelectorMode,
        colour: PresentedColour,
    ) -> Callable[[], object]:
        return lambda: self._cached_model_for_colour(mode, colour.intent)

    def _cached_model_for_colour(
        self,
        mode: SelectorMode,
        colour: ColourIntent | Sequence[float],
    ) -> object:
        intent = self._intent_from_value(colour)
        lch = intent.selector_lch
        coordinate = _fixed_slice_coordinate(mode, lch)
        cached = self._selector_model_cache.get(mode)
        if cached is not None and cached.coordinate.equivalent_to(coordinate):
            return cached.model
        model = _model_for_oklch(mode, lch)
        self._selector_model_cache[mode] = SliceModelCacheEntry(coordinate, model)
        return model

    def _preview_colour(self, oklab: ColourIntent | Sequence[float] | None) -> None:
        self._controller.set_preview_colour(oklab)

    def _commit_colour(self, oklab: ColourIntent | Sequence[float] | None) -> None:
        self._controller.request_foreground_commit(oklab)

    def _intent_from_value(self, value: ColourIntent | Sequence[float]) -> ColourIntent:
        return ColourIntent.from_value(value, achromatic_hue=self._view_seed_intent.hue)


class ColourSubscription:
    def __init__(self, controller: DockController, listener: ColourListener) -> None:
        self._controller = controller
        self._listener = listener
        self._connected = True
        self._controller.add_colour_listener(self._listener)

    def disconnect(self, *_args) -> None:
        if not self._connected:
            return
        try:
            self._controller.remove_colour_listener(self._listener)
        except (AttributeError, ValueError, RuntimeError):
            pass
        self._connected = False


def _build_selector_widget(
    mode: SelectorMode,
    model: object,
    parent: QtWidgets.QWidget,
) -> SelectorWidget | QtWidgets.QWidget:
    return _mode_spec(mode).widget_factory(model, parent)


def _model_for_oklch(mode: SelectorMode, oklch: tuple[float, float, float]) -> object:
    lightness, chroma, hue = oklch
    return _mode_spec(mode).model_factory(lightness, chroma, hue)


def _fixed_slice_coordinate(
    mode: SelectorMode,
    oklch: tuple[float, float, float],
) -> SliceCoordinate:
    return _mode_spec(mode).coordinate_factory(*oklch)


def _circular_distance(left: float, right: float) -> float:
    distance = abs((left - right) % math.tau)
    return min(distance, math.tau - distance)


def _both_achromatic_chroma(left: float, right: float) -> bool:
    return color_math.is_achromatic_chroma(max(left, right))


def _hue_when_achromatic(chroma: float, hue: float) -> float:
    if color_math.is_achromatic_chroma(chroma):
        return float(hue % math.tau)
    return 0.0


def _mode_spec(mode: SelectorMode) -> ModeSpec:
    return MODE_SPECS[mode]


def _intent_or_default(intent: ColourIntent | None) -> ColourIntent:
    if intent is None:
        return ColourIntent.from_oklab(DEFAULT_COLOUR)
    return intent
