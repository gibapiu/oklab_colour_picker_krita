"""Krita and Qt boundary adapters for the colour picker controller."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from oklab_colour_picker.domain import color_math
from oklab_colour_picker.app.controller import normalize_oklab_for_krita


SRGB_COLOR_MODEL = "RGBA"
SRGB_COLOR_DEPTH = "U8"
SRGB_COLOR_PROFILE = "sRGB-elle-V2-srgbtrc.icc"


class QtSingleShotScheduler:
    """Coalesce work onto the next Qt event-loop turn."""

    def call_soon(self, callback: Callable[[], None]) -> None:
        from oklab_colour_picker.qt import QtCore

        QtCore.QTimer.singleShot(0, callback)


FOREGROUND_POLL_INTERVAL_MS = 250


class QtForegroundTimer:
    """QTimer wrapper that drives foreground polling at a fixed cadence."""

    def __init__(self) -> None:
        from oklab_colour_picker.qt import QtCore

        self._timer = QtCore.QTimer()
        self._timer.setInterval(FOREGROUND_POLL_INTERVAL_MS)
        self._connection = None

    def start(self, callback: Callable[[], None]) -> None:
        if self._connection is not None:
            self._timer.timeout.disconnect(self._connection)
        self._connection = self._timer.timeout.connect(callback)
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()


class KritaForegroundAdapter:
    """Read and write Krita's active foreground colour with null guards."""

    def __init__(self, krita_instance=None, *, managed_color_factory=None) -> None:
        self._krita = krita_instance if krita_instance is not None else _krita_instance()
        self._managed_color_factory = managed_color_factory

    def set_foreground(self, oklab: Sequence[float]) -> np.ndarray | None:
        view = self._active_view()
        if view is None:
            return None

        srgb = color_math.oklab_to_srgb8(oklab).astype(float) / 255.0
        managed = _managed_color_from_srgb(srgb, self._managed_color_factory)
        view.setForeGroundColor(managed)
        readback = self.get_foreground()
        return readback if readback is not None else normalize_oklab_for_krita(oklab)

    def get_foreground(self) -> np.ndarray | None:
        view = self._active_view()
        if view is None:
            return None

        foreground_color = view.foregroundColor()
        components = _srgb_components_from_managed_color(foreground_color)
        if components is None:
            return None
        return color_math.srgb_to_oklab(np.asarray(components[:3], dtype=float))

    def _active_view(self):
        if self._krita is None:
            return None

        active_window = getattr(self._krita, "activeWindow", lambda: None)()
        if active_window is None:
            return None
        return getattr(active_window, "activeView", lambda: None)()


def _krita_instance():
    from krita import Krita

    return Krita.instance()


def _managed_color_from_srgb(srgb: Sequence[float], managed_color_factory=None):
    if managed_color_factory is None:
        from krita import ManagedColor

        managed_color_factory = ManagedColor

    managed = managed_color_factory(SRGB_COLOR_MODEL, SRGB_COLOR_DEPTH, SRGB_COLOR_PROFILE)
    managed.setComponents(_krita_bgra_with_alpha_from_srgb(srgb))
    return managed


def _srgb_components_from_managed_color(managed_color) -> list[float] | None:
    if managed_color is None:
        return None

    srgb_color = _converted_to_srgb(managed_color)
    if srgb_color is None:
        return None

    bgra = list(srgb_color.components())
    if len(bgra) < 3:
        return None

    rgb = _srgb_from_krita_bgra(bgra)
    return [float(np.clip(component, 0.0, 1.0)) for component in rgb]


def _krita_bgra_with_alpha_from_srgb(srgb: Sequence[float]) -> list[float]:
    return [float(srgb[2]), float(srgb[1]), float(srgb[0]), 1.0]


def _srgb_from_krita_bgra(bgra: Sequence[float]) -> list[float]:
    return [float(bgra[2]), float(bgra[1]), float(bgra[0])]


def _converted_to_srgb(managed_color):
    if _is_target_srgb_space(managed_color):
        return managed_color
    try:
        result = managed_color.setColorSpace(SRGB_COLOR_MODEL, SRGB_COLOR_DEPTH, SRGB_COLOR_PROFILE)
    except (AttributeError, TypeError, RuntimeError, ValueError):
        return None
    if result is False:
        return None
    return managed_color


def _is_target_srgb_space(managed_color) -> bool:
    try:
        return (
            managed_color.colorModel() == SRGB_COLOR_MODEL
            and managed_color.colorDepth() == SRGB_COLOR_DEPTH
            and managed_color.colorProfile() == SRGB_COLOR_PROFILE
        )
    except (AttributeError, TypeError, RuntimeError):
        return False
