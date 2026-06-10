"""Qt binding seam: import QtCore/QtGui/QtWidgets from here, not from PyQt.

Resolves to PyQt6 when available, else PyQt5.
Set ``OKLAB_QT_API=PyQt5|PyQt6`` to force a binding.
"""

from __future__ import annotations

import importlib
import os
import sys


SUPPORTED_BINDINGS = ("PyQt6", "PyQt5")
_ENV_VAR = "OKLAB_QT_API"
_EXPORTS = ("QtCore", "QtGui", "QtWidgets", "QT_API")

_preferred_api: str | None = None
_loaded: dict[str, object] | None = None


def select_binding(qversion: str | None = None) -> None:
    """Pin the Qt binding from a ``qVersion()`` string, before any Qt access.

    A no-op when ``qversion`` gives no usable hint.
    Raises if a different binding has already loaded.
    """

    global _preferred_api
    chosen = _api_for_qversion(qversion)
    if chosen is None:
        return
    if _loaded is not None:
        if _loaded["QT_API"] != chosen:
            raise RuntimeError(
                f"Qt binding already loaded as {_loaded['QT_API']!r}."
                f"Cannot switch to {chosen!r}. Call select_binding() before any Qt access."
            )
        return
    _preferred_api = chosen


def event_pos(event) -> "QtCore.QPoint":
    """Local position of a mouse event as a ``QPoint``."""

    point = _event_local_position(event)
    to_point = getattr(point, "toPoint", None)
    if to_point is not None:
        return to_point()
    return _modules()["QtCore"].QPoint(point)


def event_xy(event) -> tuple[float, float]:
    """Local position of a mouse event as ``(x, y)`` floats."""

    point = _event_local_position(event)
    return float(point.x()), float(point.y())


def _event_local_position(event):
    position = getattr(event, "position", None)
    return position() if position is not None else event.pos()


def _api_for_qversion(qversion: str | None) -> str | None:
    if not qversion:
        return None
    major = str(qversion).split(".", 1)[0]
    if not major.isdigit():
        return None
    value = int(major)
    if value >= 6:
        return "PyQt6"
    if value == 5:
        return "PyQt5"
    return None


def _modules() -> dict[str, object]:
    global _loaded
    if _loaded is None:
        _loaded = _load_binding()
    return _loaded


def _load_binding() -> dict[str, object]:
    forced = _forced_api()
    if forced is not None:
        return _load(forced)
    return _load_first_available()


def _load_first_available() -> dict[str, object]:
    errors = []
    for candidate in SUPPORTED_BINDINGS:
        try:
            return _load(candidate)
        except ImportError as exc:
            if _partially_imported(candidate):
                raise
            errors.append(f"{candidate}: {exc}")
    raise ImportError("No usable Qt binding: " + "; ".join(errors))


def _partially_imported(api: str) -> bool:
    return api in sys.modules


def _forced_api() -> str | None:
    imported = _imported_binding()
    if _preferred_api is not None:
        _reject_conflict(_preferred_api, imported)
        return _preferred_api
    env = os.environ.get(_ENV_VAR)
    if env:
        if env not in SUPPORTED_BINDINGS:
            raise ValueError(
                f"{_ENV_VAR}={env!r} is not supported."
                f"Expected one of {SUPPORTED_BINDINGS}"
            )
        _reject_conflict(env, imported)
        return env
    return imported


def _imported_binding() -> str | None:
    present = [candidate for candidate in SUPPORTED_BINDINGS if candidate in sys.modules]
    if len(present) > 1:
        raise RuntimeError(
            f"Multiple Qt bindings already imported: {present}. "
            "The process must use exactly one."
        )
    return present[0] if present else None


def _reject_conflict(api: str, imported: str | None) -> None:
    if imported is not None and imported != api:
        raise RuntimeError(
            f"{imported!r} is already imported; cannot use {api!r}. "
            "Select the binding before any Qt module is imported."
        )


def _load(api: str) -> dict[str, object]:
    return {
        "QtCore": importlib.import_module(f"{api}.QtCore"),
        "QtGui": importlib.import_module(f"{api}.QtGui"),
        "QtWidgets": importlib.import_module(f"{api}.QtWidgets"),
        "QT_API": api,
    }


def __getattr__(name: str):
    if name in _EXPORTS:
        return _modules()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
