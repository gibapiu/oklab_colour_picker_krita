import ast
from pathlib import Path

from scripts.checks.architecture_policy import (
    KRITA_IMPORT_ALLOWED,
    PRESENTED_COLOUR_CONSTRUCTION_ALLOWED,
    QT_OR_KRITA_MODULE_PREFIXES,
    SET_FOREGROUND_ALLOWED,
    UI_LAYER_MODULE_PREFIXES,
    import_from_references,
    is_declared_package_module,
    is_lower_layer_file,
    is_pure_layer_file,
    starts_with_any,
)


ROOT = Path(__file__).resolve().parents[1]


def test_krita_imports_are_limited_to_boundary_files():
    offenders = []
    for path, tree in _project_python_asts():
        for module in _imported_modules(tree):
            if _is_krita_module(module) and path not in KRITA_IMPORT_ALLOWED:
                offenders.append(f"{path}: {module}")

    assert offenders == []


def test_python_modules_live_in_declared_package_layers():
    offenders = [
        path.as_posix()
        for path, _tree in _project_python_asts()
        if not is_declared_package_module(path)
    ]

    assert offenders == []


def test_ui_layer_does_not_import_krita():
    offenders = []
    for full_path in _ui_python_files():
        path = full_path.relative_to(ROOT)
        tree = ast.parse(full_path.read_text(), filename=path.as_posix())
        for module in _imported_modules(tree):
            if _is_krita_module(module):
                offenders.append(f"{path}: {module}")

    assert offenders == []


def test_pure_layers_have_no_qt_or_krita_imports():
    offenders = []
    for path, tree in _project_python_asts():
        if not is_pure_layer_file(path):
            continue
        for module in _imported_modules(tree):
            if module.startswith(QT_OR_KRITA_MODULE_PREFIXES):
                offenders.append(f"{path}: {module}")

    assert offenders == []


def test_krita_foreground_writes_stay_behind_controller_boundary():
    offenders = []
    for path, tree in _project_python_asts():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setForeGroundColor"
                and path not in SET_FOREGROUND_ALLOWED
            ):
                offenders.append(path.as_posix())

    assert offenders == []


def test_presented_colour_is_built_only_by_the_presenter():
    offenders = []
    for path, tree in _project_python_asts():
        if path in PRESENTED_COLOUR_CONSTRUCTION_ALLOWED:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _constructs_presented_colour(node.func):
                offenders.append(path.as_posix())

    assert offenders == []


def _constructs_presented_colour(func):
    if isinstance(func, ast.Name):
        return func.id == "PresentedColour"
    return isinstance(func, ast.Attribute) and func.attr == "PresentedColour"


def test_selection_does_not_read_from_qimage_pixels():
    """Strict production tripwire for selector-by-rendered-pixel regressions."""

    offenders = []
    for path, tree in _project_python_asts():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pixelColor"
            ):
                offenders.append(path.as_posix())

    assert offenders == []


def test_lower_layers_do_not_import_ui_or_plugin_layers():
    offenders = []
    for path, tree in _project_python_asts():
        if not is_lower_layer_file(path):
            continue
        for module in _project_import_references(tree, path):
            if starts_with_any(module, UI_LAYER_MODULE_PREFIXES):
                offenders.append(f"{path}: {module}")

    assert offenders == []


def test_layer_policy_covers_future_nested_python_modules():
    assert is_pure_layer_file(Path("oklab_colour_picker/domain/future/policy.py"))
    assert is_pure_layer_file(Path("oklab_colour_picker/models/future/model.py"))
    assert is_pure_layer_file(Path("oklab_colour_picker/render/future/renderer.py"))
    assert is_lower_layer_file(Path("oklab_colour_picker/app/future/service.py"))
    assert not is_pure_layer_file(Path("oklab_colour_picker/app/future/service.py"))
    assert not is_lower_layer_file(Path("oklab_colour_picker/ui/future/widget.py"))
    assert not is_declared_package_module(Path("oklab_colour_picker/services/future.py"))


def test_relative_import_references_are_resolved_before_lower_layer_guard():
    tree = ast.parse("from ..ui.selectors import selector\nfrom ..ui import dock\n")
    imports = set(
        _project_import_references(
            tree,
            Path("oklab_colour_picker/domain/color_math.py"),
        )
    )
    assert "oklab_colour_picker.ui.selectors" in imports
    assert "oklab_colour_picker.ui.selectors.selector" in imports
    assert "oklab_colour_picker.ui.dock" in imports


def test_ui_layer_does_not_resolve_fallback_strategy_directly():
    offenders = []
    for full_path in _ui_python_files():
        path = full_path.relative_to(ROOT)
        tree = ast.parse(full_path.read_text(), filename=path.as_posix())
        imports = set(_project_import_references(tree, path))
        if "oklab_colour_picker.domain.gamut_fallback" in imports:
            offenders.append(f"{path}: imports gamut_fallback")
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "oklab_colour_picker.domain.colour_presentation"
            ):
                imported = {alias.name for alias in node.names}
                disallowed = imported - {"PresentedColour", "require_presented_colour"}
                if disallowed:
                    offenders.append(f"{path}: imports {sorted(disallowed)} from colour_presentation")

    assert offenders == []


def _project_python_asts():
    for full_path in sorted((ROOT / "oklab_colour_picker").rglob("*.py")):
        path = full_path.relative_to(ROOT)
        yield path, ast.parse(full_path.read_text(), filename=path.as_posix())


def _ui_python_files():
    ui_dir = ROOT / "oklab_colour_picker" / "ui"
    assert ui_dir.exists()
    yield from sorted(ui_dir.rglob("*.py"))


def _imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def _project_import_references(tree, source_path):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield from import_from_references(node, source_path)


def _is_krita_module(module):
    return module == "krita" or module.startswith("krita.")
