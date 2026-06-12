"""Krita plugin registration for the OKLab colour picker docker."""

from __future__ import annotations

from enum import Enum, auto
import logging
import sys
from typing import Callable

from oklab_colour_picker.infrastructure.dependency_bootstrap import install_numpy
from oklab_colour_picker.infrastructure import dependency_paths
from oklab_colour_picker.infrastructure.krita_facade import (
    KritaFacade,
    load_krita,
)


DOCK_FACTORY_ID = "oklab_colour_picker_dock"
DOCK_TITLE = "OKLab Colour Selector"


logger = logging.getLogger(__name__)


def register_plugin(*, krita_instance=None, krita_api: KritaFacade | None = None) -> bool:
    if krita_api is None:
        krita_api = load_krita()
    if krita_api is None:
        return False

    app = krita_instance if krita_instance is not None else krita_api.application()
    _seed_qt_binding(krita_api.qt_version(app))
    app_data_location = krita_api.app_data_location(app)
    _add_vendor_site_packages(app_data_location)
    dock_class = create_dock_widget_class(
        krita_api.dock_widget_base,
        app_data_location=app_data_location,
    )
    krita_api.register_dock_widget(app, DOCK_FACTORY_ID, dock_class)
    return True


def create_dock_widget_class(
    dock_widget_base: type,
    *,
    controller_factory: Callable | None = None,
    app_data_location: str | None = None,
    dependency_installer: Callable[[str], object] = install_numpy,
) -> type:
    class OKLabColourPickerDock(dock_widget_base):
        def __init__(self) -> None:
            super().__init__()
            self.setWindowTitle(DOCK_TITLE)
            self._controller = None
            self._panel = None

            try:
                from oklab_colour_picker.ui.dock import ColourPickerDockPanel
            except ImportError as exc:
                dependency_issue_kind = _runtime_dependency_issue_kind(exc)
                if dependency_issue_kind is None:
                    raise
                if dependency_issue_kind is _DependencyIssueKind.MISSING:
                    widget = _build_missing_dependency_widget(
                        vendor_path=dependency_paths.vendor_site_packages_path(app_data_location),
                        dependency_installer=dependency_installer,
                    )
                else:
                    logger.error("NumPy could not be loaded by Krita", exc_info=True)
                    widget = _build_dependency_load_failure_widget()
                self.setWidget(widget)
                return

            self._controller = _create_controller() if controller_factory is None else controller_factory()
            self._panel = ColourPickerDockPanel(self._controller, self)
            self.setWidget(self._panel)

        def canvasChanged(self, canvas) -> None:
            if self._controller is None:
                return
            self._controller.sync_external_foreground(force=True)

    OKLabColourPickerDock.__name__ = "OKLabColourPickerDock"
    return OKLabColourPickerDock


def _seed_qt_binding(qt_version: str | None) -> None:
    """Pin the Qt binding to Krita's runtime version before any UI import."""

    from oklab_colour_picker.infrastructure.qt_facade import select_binding

    select_binding(qt_version)


class _DependencyIssueKind(Enum):
    MISSING = auto()
    LOAD_FAILED = auto()


def _add_vendor_site_packages(app_data_location: str | None = None) -> None:
    dependency_path = dependency_paths.resolve_dependency_path(app_data_location)
    if dependency_path is not None and dependency_path not in sys.path:
        sys.path.insert(0, dependency_path)


def _runtime_dependency_issue_kind(error: ImportError) -> _DependencyIssueKind | None:
    name = getattr(error, "name", None) or ""
    root = name.split(".", 1)[0]
    message = str(error).lower()
    if root != "numpy" and "numpy" not in message:
        return None

    return (
        _DependencyIssueKind.MISSING
        if isinstance(error, ModuleNotFoundError) and name == "numpy"
        else _DependencyIssueKind.LOAD_FAILED
    )


def _build_missing_dependency_widget(
    *,
    vendor_path: str,
    dependency_installer: Callable[[str], object],
):
    from oklab_colour_picker.infrastructure.qt_facade import QtCore, QtWidgets

    widget = QtWidgets.QWidget()
    widget.setObjectName("oklab-missing-dependency")

    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(10)

    label = QtWidgets.QLabel(
        "OKLab Colour Selector could not start because NumPy is not installed.\n\n"
        "Krita does not always ship NumPy. You can install NumPy into Krita's app data, then restart Krita."
    )
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    layout.addWidget(label)

    button = QtWidgets.QPushButton("Install NumPy")
    button.setObjectName("oklab-install-numpy")
    layout.addWidget(button, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

    status = QtWidgets.QLabel("")
    status.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    status.setWordWrap(True)
    status.setObjectName("oklab-install-status")
    layout.addWidget(status)

    class InstallSignals(QtCore.QObject):
        finished = QtCore.pyqtSignal(bool, str)

    class InstallRunnable(QtCore.QRunnable):
        def __init__(self, signals: "InstallSignals") -> None:
            super().__init__()
            self._signals = signals

        def run(self) -> None:
            try:
                result = dependency_installer(vendor_path)
                self._signals.finished.emit(bool(result.success), str(result.message))
            except Exception as exc:
                self._signals.finished.emit(False, f"NumPy installation failed: {exc}")

    signals = InstallSignals(widget)

    def on_finished(success: bool, message: str) -> None:
        button.setEnabled(True)
        status.setText(message)
        if success:
            QtWidgets.QMessageBox.information(
                widget,
                "NumPy Installed",
                f"{message}\n\nRestart Krita to use the plugin.",
            )
        else:
            QtWidgets.QMessageBox.warning(widget, "NumPy Install Failed", message)

    signals.finished.connect(on_finished)

    def confirm_install() -> None:
        response = QtWidgets.QMessageBox.question(
            widget,
            "Install NumPy",
            "Install NumPy for OKLab Colour Picker?\n\n"
            "Krita will download NumPy from PyPI and install it into the plugin's private dependency folder.\n\n"
            "Restart Krita after installation.",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if response != QtWidgets.QMessageBox.StandardButton.Yes:
            return

        button.setEnabled(False)
        status.setText("Installing NumPy...")
        QtCore.QThreadPool.globalInstance().start(InstallRunnable(signals))

    button.clicked.connect(confirm_install)
    return widget


def _build_dependency_load_failure_widget():
    from oklab_colour_picker.infrastructure.qt_facade import QtCore, QtWidgets

    widget = QtWidgets.QWidget()
    widget.setObjectName("oklab-dependency-load-failure")

    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(10)

    label = QtWidgets.QLabel(
        "OKLab Colour Selector could not start because NumPy could not be loaded.\n\n"
        "The NumPy installation available to Krita is not working. "
        "Reinstall or update NumPy for Krita, then restart Krita.\n\n"
        "If the problem continues, see the plugin troubleshooting guide."
    )
    label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    label.setWordWrap(True)
    layout.addWidget(label)
    return widget


def _create_controller():
    from oklab_colour_picker.app.controller import ColourPickerController
    from oklab_colour_picker.infrastructure.krita_adapter import (
        KritaForegroundAdapter,
        QtForegroundTimer,
        QtSingleShotScheduler,
    )

    return ColourPickerController(
        KritaForegroundAdapter(),
        scheduler=QtSingleShotScheduler(),
        foreground_timer=QtForegroundTimer(),
    )
