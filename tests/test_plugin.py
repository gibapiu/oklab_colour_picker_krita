import sys
import types

import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets

import oklab_colour_picker
import oklab_colour_picker.plugin as plugin_module
from oklab_colour_picker.infrastructure.dependency_bootstrap import InstallResult
from oklab_colour_picker.plugin import (
    DOCK_FACTORY_ID,
    DOCK_TITLE,
    create_dock_widget_class,
    register_plugin,
)
from oklab_colour_picker.ui.dock import ColourPickerDockPanel


def test_registers_krita_dock_factory():
    app = FakeKritaApp()

    assert register_plugin(krita_instance=app, api=FakeKritaApi(app)) is True

    assert len(app.factories) == 1
    factory = app.factories[0]
    assert factory.identifier == DOCK_FACTORY_ID
    assert factory.area == FakeDockWidgetFactoryBase.DockRight


def test_vendor_site_packages_are_added_before_runtime_imports(tmp_path, monkeypatch):
    vendor_dir = _vendor_path(tmp_path)
    vendor_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))

    plugin_module._add_vendor_site_packages(str(tmp_path))

    assert sys.path[0] == str(vendor_dir)


def test_vendor_site_packages_fall_back_next_to_plugin_package(tmp_path, monkeypatch):
    package_dir = tmp_path / "pykrita" / "oklab_colour_picker"
    package_dir.mkdir(parents=True)
    expected = package_dir / plugin_module.VENDOR_SITE_PACKAGES_DIRECTORY_NAME
    monkeypatch.setattr(plugin_module, "__file__", str(package_dir / "plugin.py"))

    assert plugin_module._vendor_site_packages_path() == str(expected)


def test_created_dock_builds_panel(qtbot):
    controller = FakeController()
    dock_class = create_dock_widget_class(
        FakeDockWidget,
        controller_factory=lambda: controller,
    )
    dock = dock_class()
    qtbot.addWidget(dock)

    assert dock.windowTitle() == DOCK_TITLE
    assert isinstance(dock.widget(), ColourPickerDockPanel)


def test_created_dock_syncs_foreground_on_canvas_change(qtbot):
    controller = FakeController()
    dock_class = create_dock_widget_class(
        FakeDockWidget,
        controller_factory=lambda: controller,
    )
    dock = dock_class()
    qtbot.addWidget(dock)

    dock.canvasChanged(object())

    assert controller.sync_count == 1
    assert controller.last_force_sync is True


@pytest.mark.parametrize(
    "error",
    [
        ModuleNotFoundError("No module named 'numpy'", name="numpy"),
        ImportError("Importing the numpy C-extensions failed: incompatible binary"),
    ],
)
def test_dependency_failure_shows_numpy_installer(qtbot, monkeypatch, error):
    _replace_dock_module_with_error(monkeypatch, error)

    dock = create_dock_widget_class(FakeDockWidget)()
    qtbot.addWidget(dock)

    widget = dock.widget()
    assert widget.objectName() == "oklab-missing-dependency"
    assert "numpy" in widget.findChild(QtWidgets.QLabel).text().lower()
    assert _install_button(widget).text() == "Install NumPy"


def test_install_action_requires_confirmation(qtbot, monkeypatch, tmp_path):
    _replace_dock_module_with_missing_numpy(monkeypatch)
    messages = []
    installer_calls = []

    def reject_install(_parent, _title, message, *args, **kwargs):
        messages.append(message)
        return QtWidgets.QMessageBox.No

    monkeypatch.setattr(QtWidgets.QMessageBox, "question", reject_install)
    dock = create_dock_widget_class(
        FakeDockWidget,
        app_data_location=str(tmp_path),
        dependency_installer=lambda vendor_path: installer_calls.append(vendor_path),
    )()
    qtbot.addWidget(dock)

    qtbot.mouseClick(_install_button(dock.widget()), QtCore.Qt.LeftButton)

    assert installer_calls == []
    assert messages
    assert str(_vendor_path(tmp_path)) not in messages[0]
    assert "private dependency folder" in messages[0]


def test_confirmed_install_runs_installer(qtbot, monkeypatch, tmp_path):
    _replace_dock_module_with_missing_numpy(monkeypatch)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Yes,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Ok,
    )
    installer_calls = []

    def install(vendor_path):
        installer_calls.append(vendor_path)
        return InstallResult(True, "NumPy installed.")

    dock = create_dock_widget_class(
        FakeDockWidget,
        app_data_location=str(tmp_path),
        dependency_installer=install,
    )()
    qtbot.addWidget(dock)
    status = dock.widget().findChild(QtWidgets.QLabel, "oklab-install-status")
    button = _install_button(dock.widget())

    qtbot.mouseClick(button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(
        lambda: bool(installer_calls) and "installed" in status.text().lower(),
        timeout=5000,
    )
    assert installer_calls == [str(_vendor_path(tmp_path))]
    assert button.isEnabled()


def test_install_exception_is_reported(qtbot, monkeypatch, tmp_path):
    _replace_dock_module_with_missing_numpy(monkeypatch)
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Yes,
    )
    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda _parent, _title, message, *args, **kwargs: (
            warnings.append(message) or QtWidgets.QMessageBox.Ok
        ),
    )

    def fail_install(_vendor_path):
        raise RuntimeError("network is down")

    dock = create_dock_widget_class(
        FakeDockWidget,
        app_data_location=str(tmp_path),
        dependency_installer=fail_install,
    )()
    qtbot.addWidget(dock)
    button = _install_button(dock.widget())

    qtbot.mouseClick(button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(lambda: bool(warnings), timeout=5000)
    assert "network is down" in warnings[0]
    assert button.isEnabled()


def test_unexpected_import_error_is_propagated(monkeypatch):
    _replace_dock_module_with_error(
        monkeypatch,
        ModuleNotFoundError(
            "No module named 'something_else'",
            name="something_else",
        ),
    )

    with pytest.raises(ModuleNotFoundError):
        create_dock_widget_class(FakeDockWidget)()


def test_package_exports_register_plugin():
    assert oklab_colour_picker.__all__ == ["register_plugin"]
    assert oklab_colour_picker.register_plugin is register_plugin


def _replace_dock_module_with_missing_numpy(monkeypatch):
    _replace_dock_module_with_error(
        monkeypatch,
        ModuleNotFoundError("No module named 'numpy'", name="numpy"),
    )


def _replace_dock_module_with_error(monkeypatch, error):
    fake_dock = types.ModuleType("oklab_colour_picker.ui.dock")

    def raise_error(_name):
        raise error

    fake_dock.__getattr__ = raise_error
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.ui.dock", fake_dock)


def _vendor_path(root):
    return (
        root
        / plugin_module.VENDOR_ROOT_DIRECTORY_NAME
        / plugin_module.VENDOR_SITE_PACKAGES_DIRECTORY_NAME
    )


def _install_button(widget):
    return widget.findChild(QtWidgets.QPushButton, "oklab-install-numpy")


class FakeController:
    selected_intent = None

    def __init__(self):
        self.listeners = []
        self.sync_count = 0
        self.last_force_sync = False

    def set_preview_colour(self, _colour):
        pass

    def request_foreground_commit(self, _colour):
        pass

    def add_colour_listener(self, listener):
        self.listeners.append(listener)

    def remove_colour_listener(self, listener):
        self.listeners.remove(listener)

    def set_fallback_strategy_provider(self, _provider):
        pass

    def reproject(self):
        pass

    def sync_external_foreground(self, *, force=False):
        self.sync_count += 1
        self.last_force_sync = force
        return False


class FakeKritaApp:
    def __init__(self):
        self.factories = []

    def addDockWidgetFactory(self, factory):
        self.factories.append(factory)

    def getAppDataLocation(self):
        return "/tmp/fake-krita-app-data"


class FakeDockWidgetFactoryBase:
    DockRight = object()


class FakeDockWidgetFactory:
    def __init__(self, identifier, area, widget_class):
        self.identifier = identifier
        self.area = area
        self.widget_class = widget_class


class FakeKritaApi:
    def __init__(self, app):
        self.Krita = FakeKrita(app)
        self.DockWidget = FakeDockWidget
        self.DockWidgetFactory = FakeDockWidgetFactory
        self.DockWidgetFactoryBase = FakeDockWidgetFactoryBase


class FakeKrita:
    def __init__(self, app):
        self._app = app

    def instance(self):
        return self._app


class FakeDockWidget(QtWidgets.QDockWidget):
    def canvasChanged(self, canvas):
        pass
