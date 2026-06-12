"""Unified plugin-bootstrap access to the Krita 5 and Krita 6 Python APIs."""

from __future__ import annotations

from dataclasses import dataclass


_DOCK_RIGHT_NAME = "DockRight"


@dataclass(frozen=True)
class KritaFacade:
    """Expose the subset of Krita used during plugin bootstrap."""

    _module: object
    _application_type: object
    dock_widget_base: type
    _dock_widget_factory_type: type
    _dock_right: object

    @classmethod
    def from_module(cls, module: object) -> "KritaFacade":
        application_type = _required_attribute(module, "Krita")
        dock_widget_base = _required_attribute(module, "DockWidget")
        dock_widget_factory_type = _required_attribute(module, "DockWidgetFactory")
        dock_factory_base = _required_attribute(module, "DockWidgetFactoryBase")

        return cls(
            _module=module,
            _application_type=application_type,
            dock_widget_base=dock_widget_base,
            _dock_widget_factory_type=dock_widget_factory_type,
            _dock_right=_dock_position(dock_factory_base, _DOCK_RIGHT_NAME),
        )

    def application(self):
        return self._application_type.instance()

    def qt_version(self, application) -> str | None:
        for source in (self._module, self._application_type, application):
            getter = getattr(source, "qVersion", None)
            if not callable(getter):
                continue
            try:
                version = getter()
            except Exception:
                continue
            if version:
                return str(version)
        return None

    def app_data_location(self, application) -> str | None:
        for source in (application, self._application_type):
            getter = getattr(source, "getAppDataLocation", None)
            if not callable(getter):
                continue
            location = getter()
            if location is not None:
                return str(location)
        return None

    def register_dock_widget(
        self,
        application,
        identifier: str,
        dock_widget_type: type,
    ) -> None:
        factory = self._dock_widget_factory_type(
            identifier,
            self._dock_right,
            dock_widget_type,
        )
        application.addDockWidgetFactory(factory)


def load_krita() -> KritaFacade | None:
    """Load Krita's Python API, or return ``None`` outside Krita."""

    try:
        import krita
    except ModuleNotFoundError as exc:
        if exc.name != "krita":
            raise
        return None
    return KritaFacade.from_module(krita)


def _dock_position(dock_factory_base: object, name: str) -> object:
    """Resolve unscoped Krita 5 or scoped Krita 6 dock-position enums."""

    dock_positions = getattr(dock_factory_base, "DockPosition", dock_factory_base)
    return _required_attribute(dock_positions, name)


def _required_attribute(source: object, name: str):
    try:
        return getattr(source, name)
    except AttributeError as exc:
        raise RuntimeError(f"Krita Python API does not provide {name!r}") from exc
