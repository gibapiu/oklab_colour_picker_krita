"""Krita plugin registration for the OKLab colour picker docker."""

from __future__ import annotations

import os
import sys
from typing import Callable

from oklab_colour_picker.infrastructure.dependency_bootstrap import install_numpy
from oklab_colour_picker.infrastructure.krita_facade import (
    KritaFacade,
    load_krita,
)


DOCK_FACTORY_ID = "oklab_colour_picker_dock"
DOCK_TITLE = "OKLab Colour Selector"
VENDOR_ROOT_DIRECTORY_NAME = "oklab_colour_picker"
VENDOR_SITE_PACKAGES_DIRECTORY_NAME = "site-packages"


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
                if not _is_known_runtime_dependency(exc):
                    raise
                self.setWidget(
                    _build_missing_dependency_widget(
                        exc,
                        vendor_path=_vendor_site_packages_path(app_data_location),
                        dependency_installer=dependency_installer,
                    )
                )
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


_KNOWN_RUNTIME_DEPENDENCIES = frozenset({"numpy"})


def _add_vendor_site_packages(app_data_location: str | None = None) -> None:
    vendor_path = _vendor_site_packages_path(app_data_location)
    if os.path.isdir(vendor_path) and vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)


def _vendor_site_packages_path(app_data_location: str | None = None) -> str:
    if app_data_location:
        return os.path.join(
            app_data_location,
            VENDOR_ROOT_DIRECTORY_NAME,
            VENDOR_SITE_PACKAGES_DIRECTORY_NAME,
        )

    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        os.path.dirname(package_dir),
        VENDOR_ROOT_DIRECTORY_NAME,
        VENDOR_SITE_PACKAGES_DIRECTORY_NAME,
    )


def _is_known_runtime_dependency(error: ImportError) -> bool:
    name = getattr(error, "name", None) or ""
    root = name.split(".", 1)[0]
    if root in _KNOWN_RUNTIME_DEPENDENCIES:
        return True

    message = str(error).lower()
    return any(dependency in message for dependency in _KNOWN_RUNTIME_DEPENDENCIES)


def _build_missing_dependency_widget(
    error: ImportError,
    *,
    vendor_path: str,
    dependency_installer: Callable[[str], object],
):
    from oklab_colour_picker.infrastructure.qt_facade import QtCore, QtWidgets

    missing = error.name or str(error)
    widget = QtWidgets.QWidget()
    widget.setObjectName("oklab-missing-dependency")

    layout = QtWidgets.QVBoxLayout(widget)
    layout.setContentsMargins(16, 16, 16, 16)
    layout.setSpacing(10)

    label = QtWidgets.QLabel(
        f"OKLab Colour Selector could not start because Python dependency '{missing}' is missing.\n\n"
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
            "Install NumPy for OKLab Colour Selector?\n\n"
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
