"""Dependency-free architecture policy shared by static and pytest checks."""

import ast
from pathlib import Path


KRITA_IMPORT_ALLOWED = {
    Path("oklab_colour_picker/infrastructure/krita_adapter.py"),
    Path("oklab_colour_picker/infrastructure/krita_facade.py"),
}

PACKAGE_LAYER_DIRECTORIES = (
    Path("oklab_colour_picker/app"),
    Path("oklab_colour_picker/domain"),
    Path("oklab_colour_picker/infrastructure"),
    Path("oklab_colour_picker/models"),
    Path("oklab_colour_picker/render"),
    Path("oklab_colour_picker/ui"),
)

PACKAGE_ROOT_MODULES = {
    Path("oklab_colour_picker/__init__.py"),
    Path("oklab_colour_picker/plugin.py"),
}

QT_FACADE_MODULE = "oklab_colour_picker.infrastructure.qt_facade"

QT_BINDING_MODULE_PREFIXES = (
    "PyQt5",
    "PyQt6",
    "PySide",
    "PySide2",
    "PySide6",
)

QT_BINDING_IMPORT_ALLOWED = {
    Path("oklab_colour_picker/infrastructure/qt_facade.py"),
}

PURE_LAYER_DIRECTORIES = (
    Path("oklab_colour_picker/domain"),
    Path("oklab_colour_picker/models"),
    Path("oklab_colour_picker/render"),
)

LOWER_LAYER_DIRECTORIES = (
    *PURE_LAYER_DIRECTORIES,
    Path("oklab_colour_picker/app"),
)

QT_OR_KRITA_MODULE_PREFIXES = (*QT_BINDING_MODULE_PREFIXES, "krita")

SET_FOREGROUND_ALLOWED = {
    Path("oklab_colour_picker/infrastructure/krita_adapter.py"),
}

# PresentedColour is a derived read model; only the presenter may build it so
# fallback policy keeps a single owner. Everyone else receives and reads it.
PRESENTED_COLOUR_CONSTRUCTION_ALLOWED = {
    Path("oklab_colour_picker/domain/colour_presentation.py"),
}

UI_LAYER_MODULE_PREFIXES = (
    "oklab_colour_picker.plugin",
    "oklab_colour_picker.ui",
)

LOWER_LAYER_FORBIDDEN_MODULE_PREFIXES = (
    *UI_LAYER_MODULE_PREFIXES,
    QT_FACADE_MODULE,
)


def is_pure_layer_file(path: Path) -> bool:
    return path.suffix == ".py" and is_within_any(path, PURE_LAYER_DIRECTORIES)


def is_lower_layer_file(path: Path) -> bool:
    return path.suffix == ".py" and is_within_any(path, LOWER_LAYER_DIRECTORIES)


def is_within_any(path: Path, directories: tuple[Path, ...]) -> bool:
    return any(directory == path or directory in path.parents for directory in directories)


def is_declared_package_module(path: Path) -> bool:
    if (
        path.suffix != ".py"
        or not path.parts
        or path.parts[0] != "oklab_colour_picker"
    ):
        return True
    return path in PACKAGE_ROOT_MODULES or is_within_any(path, PACKAGE_LAYER_DIRECTORIES)


def import_from_references(
    node: ast.ImportFrom,
    source_path: Path,
) -> tuple[str, ...]:
    module = node.module or ""
    if node.level:
        base = ".".join(_relative_import_base(source_path, node.level))
        resolved_module = ".".join(part for part in (base, module) if part)
        if module:
            return (
                resolved_module,
                *(f"{resolved_module}.{alias.name}" for alias in node.names),
            )
        return tuple(f"{base}.{alias.name}" for alias in node.names)
    if module != "oklab_colour_picker":
        return (module,)
    return tuple(f"{module}.{alias.name}" for alias in node.names)


def starts_with_any(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


def _relative_import_base(source_path: Path, level: int) -> tuple[str, ...]:
    module_parts = source_path.with_suffix("").parts
    package_parts = module_parts[:-1]
    keep = max(0, len(package_parts) - level + 1)
    return package_parts[:keep]
