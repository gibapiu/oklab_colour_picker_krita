"""Qt dock content for the OKLab colour picker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import numpy as np
from oklab_colour_picker.qt import QtCore, QtWidgets

from oklab_colour_picker.app.selector_model_cache import (
    SelectorMode,
    SelectorModelCache,
)
from oklab_colour_picker.domain.colour_presentation import PresentedColour
from oklab_colour_picker.domain.colour_state import ColourIntent
from oklab_colour_picker.app.controller import ChangeKind, ColourSnapshot
from oklab_colour_picker.models import SelectorModel
from oklab_colour_picker.ui.readout import ReadoutPanel
from oklab_colour_picker.ui.selectors.selector import SelectorWidget


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

    def set_fallback_strategy_provider(self, provider: object) -> None:
        ...

    def reproject(self) -> None:
        ...


WidgetFactory = Callable[[SelectorModel, QtWidgets.QWidget], SelectorWidget]


@dataclass(frozen=True)
class ModeSpec:
    label: str
    object_name: str
    widget_factory: WidgetFactory


def _selector_widget(model: SelectorModel, parent: QtWidgets.QWidget) -> SelectorWidget:
    return SelectorWidget(model, parent)


def _lightness_slice_widget(model: SelectorModel, parent: QtWidgets.QWidget) -> SelectorWidget:
    from oklab_colour_picker.ui.selectors.lightness_slice_disk import LightnessSliceDiskWidget

    return LightnessSliceDiskWidget(model, parent)


def _hue_lightness_slice_widget(
    model: SelectorModel,
    parent: QtWidgets.QWidget,
) -> SelectorWidget:
    from oklab_colour_picker.ui.selectors.hue_lightness_slice_disk import (
        HueLightnessSliceDiskWidget,
    )

    return HueLightnessSliceDiskWidget(model, parent)


MODE_SPECS = {
    SelectorMode.LIGHTNESS_SLICE: ModeSpec(
        "Hue/Chroma",
        "lightness-slice-selector",
        _lightness_slice_widget,
    ),
    SelectorMode.HUE_LIGHTNESS_SLICE: ModeSpec(
        "Hue/Lightness",
        "hue-lightness-slice-selector",
        _hue_lightness_slice_widget,
    ),
    SelectorMode.LIGHTNESS_CHROMA_SLICE: ModeSpec(
        "Lightness/Chroma",
        "lightness-chroma-slice-selector",
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
        self._selector_model_cache = SelectorModelCache()
        self._readout_panel = ReadoutPanel(self)
        self._readout_panel.previewed.connect(self._preview_colour)
        self._readout_panel.committed.connect(self._commit_colour)
        self._build_selector_tabs()
        self._build_layout()
        self._tabs.currentChanged.connect(self._on_tab_changed)
        self._controller.set_fallback_strategy_provider(self._active_fallback_strategy)
        self._controller_subscription = ColourSubscription(controller, self._on_colour_changed)
        self.destroyed.connect(self._release_controller)

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

    def _on_tab_changed(self, index: int) -> None:
        self._ensure_selector_for_tab(index)
        self._controller.reproject()

    def _ensure_selector_for_tab(self, index: int) -> None:
        if 0 <= index < len(self._selector_modes):
            self._ensure_selector(self._selector_modes[index])

    def _active_fallback_strategy(self, intent: ColourIntent) -> object:
        """Return the fallback strategy for the front tab's slice at ``intent``."""

        return self._selector_model_cache.fallback_strategy_for(self.mode, intent)

    def _release_controller(self, *_args: object) -> None:
        self._controller_subscription.disconnect()
        try:
            self._controller.set_fallback_strategy_provider(None)
        except (AttributeError, RuntimeError):
            pass

    def _ensure_selector(self, mode: SelectorMode) -> SelectorWidget:
        existing = self._selectors.get(mode)
        if existing is not None:
            return existing

        seed = self._view_seed_intent
        widget = _build_selector_widget(
            mode,
            self._selector_model_cache.model_for(mode, seed),
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
    ) -> Callable[[], SelectorModel]:
        return lambda: self._selector_model_cache.model_for(mode, colour.intent)

    def _preview_colour(self, oklab: ColourIntent | Sequence[float] | None) -> None:
        self._controller.set_preview_colour(oklab)

    def _commit_colour(self, oklab: ColourIntent | Sequence[float] | None) -> None:
        self._controller.request_foreground_commit(oklab)


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
    model: SelectorModel,
    parent: QtWidgets.QWidget,
) -> SelectorWidget:
    return _mode_spec(mode).widget_factory(model, parent)


def _mode_spec(mode: SelectorMode) -> ModeSpec:
    return MODE_SPECS[mode]


def _intent_or_default(intent: ColourIntent | None) -> ColourIntent:
    if intent is None:
        return ColourIntent.from_oklab(DEFAULT_COLOUR)
    return intent
