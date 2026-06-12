import os


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


_OKLAB_TO_PYTEST_QT = {"PyQt5": "pyqt5", "PyQt6": "pyqt6"}
_PYTEST_TO_OKLAB_QT = {value: key for key, value in _OKLAB_TO_PYTEST_QT.items()}


def _align_qt_binding_env() -> None:
    oklab = os.environ.get("OKLAB_QT_API")
    pytest_qt = os.environ.get("PYTEST_QT_API")
    if oklab is not None and oklab not in _OKLAB_TO_PYTEST_QT:
        raise RuntimeError(
            f"OKLAB_QT_API={oklab!r} is not supported; expected one of "
            f"{sorted(_OKLAB_TO_PYTEST_QT)}"
        )
    if pytest_qt is not None and pytest_qt not in _PYTEST_TO_OKLAB_QT:
        raise RuntimeError(
            f"PYTEST_QT_API={pytest_qt!r} is not supported; expected one of "
            f"{sorted(_PYTEST_TO_OKLAB_QT)}"
        )
    if oklab is not None and pytest_qt is not None and _OKLAB_TO_PYTEST_QT[oklab] != pytest_qt:
        raise RuntimeError(
            f"OKLAB_QT_API={oklab!r} and PYTEST_QT_API={pytest_qt!r} select "
            "different Qt bindings"
        )
    if oklab is not None:
        os.environ["PYTEST_QT_API"] = _OKLAB_TO_PYTEST_QT[oklab]
    elif pytest_qt is not None:
        os.environ["OKLAB_QT_API"] = _PYTEST_TO_OKLAB_QT[pytest_qt]


_align_qt_binding_env()
