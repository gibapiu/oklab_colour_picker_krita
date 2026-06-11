"""Binding-neutral Qt event factories for widget tests.

Hide PyQt5/PyQt6 constructor and scoped-enum differences behind ``kind``
strings: mouse ``"press"|"move"|"release"``, key/focus per the maps below.
"""

from __future__ import annotations

from oklab_colour_picker.infrastructure.qt_facade import QtCore, QtGui


_NO_MODIFIER = QtCore.Qt.KeyboardModifier.NoModifier

_MOUSE_TYPES = {
    "press": QtCore.QEvent.Type.MouseButtonPress,
    "move": QtCore.QEvent.Type.MouseMove,
    "release": QtCore.QEvent.Type.MouseButtonRelease,
}

_MOUSE_BUTTONS = {
    "press": (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.LeftButton),
    "move": (QtCore.Qt.MouseButton.NoButton, QtCore.Qt.MouseButton.LeftButton),
    "release": (QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.NoButton),
}

_KEY_TYPES = {
    "press": QtCore.QEvent.Type.KeyPress,
    "release": QtCore.QEvent.Type.KeyRelease,
}

_FOCUS_TYPES = {
    "in": QtCore.QEvent.Type.FocusIn,
    "out": QtCore.QEvent.Type.FocusOut,
}


def mouse_event(kind: str, position, *, modifiers=_NO_MODIFIER) -> "QtGui.QMouseEvent":
    """Mouse event for gesture phase ``kind`` at ``position`` (a ``QPoint``)."""

    button, buttons = _MOUSE_BUTTONS[kind]
    return QtGui.QMouseEvent(
        _MOUSE_TYPES[kind], QtCore.QPointF(position), button, buttons, modifiers
    )


def key_event(kind: str, key: str, *, modifiers=_NO_MODIFIER) -> "QtGui.QKeyEvent":
    """Key ``"press"|"release"`` event for the named ``key`` (e.g. ``"Right"``)."""

    return QtGui.QKeyEvent(_KEY_TYPES[kind], getattr(QtCore.Qt.Key, f"Key_{key}"), modifiers)


def focus_event(
    kind: str, *, reason=QtCore.Qt.FocusReason.OtherFocusReason
) -> "QtGui.QFocusEvent":
    """Focus ``"in"|"out"`` event."""

    return QtGui.QFocusEvent(_FOCUS_TYPES[kind], reason)


def send_mouse(widget, kind: str, position) -> "QtGui.QMouseEvent":
    """Dispatch a mouse ``kind`` event to ``widget``; assert accepted; return it."""

    event = mouse_event(kind, position)
    QtCore.QCoreApplication.sendEvent(widget, event)
    assert event.isAccepted()
    return event


def send_focus(widget, kind: str) -> None:
    """Dispatch a focus ``"in"|"out"`` event to ``widget``."""

    QtCore.QCoreApplication.sendEvent(widget, focus_event(kind))
