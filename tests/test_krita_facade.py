import sys
import types

import pytest

from oklab_colour_picker.infrastructure.krita_facade import (
    KritaFacade,
    load_krita,
)


@pytest.mark.parametrize(
    ("dock_factory_base", "expected_area"),
    [
        (lambda area: types.SimpleNamespace(DockRight=area), "krita-5-area"),
        (
            lambda area: types.SimpleNamespace(
                DockPosition=types.SimpleNamespace(DockRight=area)
            ),
            "krita-6-area",
        ),
    ],
)
def test_registers_dock_widget_with_supported_enum_shapes(
    dock_factory_base,
    expected_area,
):
    application = FakeApplication()
    module = fake_krita_module(
        application,
        dock_factory_base=dock_factory_base(expected_area),
    )
    krita_api = KritaFacade.from_module(module)

    krita_api.register_dock_widget(application, "dock-id", FakeDockWidget)

    factory = application.factories[0]
    assert factory.identifier == "dock-id"
    assert factory.area == expected_area
    assert factory.widget_type is FakeDockWidget


def test_exposes_application_and_dock_widget_base():
    application = FakeApplication()
    module = fake_krita_module(application)
    krita_api = KritaFacade.from_module(module)

    assert krita_api.application() is application
    assert krita_api.dock_widget_base is FakeDockWidget


def test_reads_qt_version_from_first_available_runtime_source():
    application = FakeApplication()
    application.qVersion = lambda: "application-version"
    module = fake_krita_module(application)
    module.qVersion = lambda: "6.8.2"
    krita_api = KritaFacade.from_module(module)

    assert krita_api.qt_version(application) == "6.8.2"


def test_qt_version_ignores_broken_runtime_source():
    application = FakeApplication()
    module = fake_krita_module(application)

    def fail():
        raise RuntimeError("unavailable")

    module.qVersion = fail
    module.Krita.qVersion = lambda: "5.15.18"
    krita_api = KritaFacade.from_module(module)

    assert krita_api.qt_version(application) == "5.15.18"


def test_reads_app_data_location_from_application():
    application = FakeApplication(app_data_location="/application/path")
    krita_api = KritaFacade.from_module(fake_krita_module(application))

    assert krita_api.app_data_location(application) == "/application/path"


def test_reads_app_data_location_from_application_type():
    application = types.SimpleNamespace()
    module = fake_krita_module(application)
    module.Krita.getAppDataLocation = lambda: "/type/path"
    krita_api = KritaFacade.from_module(module)

    assert krita_api.app_data_location(application) == "/type/path"


def test_rejects_unsupported_dock_position_shape():
    module = fake_krita_module(
        FakeApplication(),
        dock_factory_base=types.SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="DockRight"):
        KritaFacade.from_module(module)


def test_load_returns_none_outside_krita(monkeypatch):
    monkeypatch.delitem(sys.modules, "krita", raising=False)

    import builtins

    real_import = builtins.__import__

    def import_without_krita(name, *args, **kwargs):
        if name == "krita":
            raise ModuleNotFoundError("not running in Krita", name="krita")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_krita)

    assert load_krita() is None


def test_load_builds_facade_from_krita_module(monkeypatch):
    application = FakeApplication()
    module = fake_krita_module(application)
    monkeypatch.setitem(sys.modules, "krita", module)

    krita_api = load_krita()

    assert krita_api is not None
    assert krita_api.application() is application


def test_load_propagates_krita_dependency_import_failures(monkeypatch):
    monkeypatch.delitem(sys.modules, "krita", raising=False)

    import builtins

    real_import = builtins.__import__

    def import_broken_krita(name, *args, **kwargs):
        if name == "krita":
            raise ModuleNotFoundError("missing binding", name="PyKrita")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_broken_krita)

    with pytest.raises(ModuleNotFoundError, match="missing binding"):
        load_krita()


def fake_krita_module(application, *, dock_factory_base=None):
    class FakeKrita:
        @staticmethod
        def instance():
            return application

    return types.SimpleNamespace(
        Krita=FakeKrita,
        DockWidget=FakeDockWidget,
        DockWidgetFactory=FakeDockWidgetFactory,
        DockWidgetFactoryBase=(
            dock_factory_base
            if dock_factory_base is not None
            else types.SimpleNamespace(DockRight="default-area")
        ),
    )


class FakeApplication:
    def __init__(self, *, app_data_location=None):
        self.app_data_location = app_data_location
        self.factories = []

    def addDockWidgetFactory(self, factory):
        self.factories.append(factory)

    def getAppDataLocation(self):
        return self.app_data_location


class FakeDockWidget:
    pass


class FakeDockWidgetFactory:
    def __init__(self, identifier, area, widget_type):
        self.identifier = identifier
        self.area = area
        self.widget_type = widget_type
