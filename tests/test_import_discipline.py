import ast
import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KRITA_IMPORT_ALLOWED = {
    Path("oklab_colour_picker/plugin.py"),
    Path("oklab_colour_picker/app/controller.py"),
    Path("oklab_colour_picker/infrastructure/krita_adapter.py"),
}
PURE_NO_QT_OR_KRITA = {
    Path("oklab_colour_picker/domain/__init__.py"),
    Path("oklab_colour_picker/domain/color_math.py"),
    Path("oklab_colour_picker/domain/colour_presentation.py"),
    Path("oklab_colour_picker/domain/colour_state.py"),
    Path("oklab_colour_picker/domain/gamut_fallback.py"),
    Path("oklab_colour_picker/domain/readout_interaction.py"),
    Path("oklab_colour_picker/domain/selector_interaction.py"),
    Path("oklab_colour_picker/models/__init__.py"),
    Path("oklab_colour_picker/models/base.py"),
    Path("oklab_colour_picker/models/geometry.py"),
    Path("oklab_colour_picker/models/hue_lightness_slice.py"),
    Path("oklab_colour_picker/models/lightness_chroma_slice.py"),
    Path("oklab_colour_picker/models/lightness_slice.py"),
    Path("oklab_colour_picker/render/__init__.py"),
    Path("oklab_colour_picker/render/renderers.py"),
}
SET_FOREGROUND_ALLOWED = {
    Path("oklab_colour_picker/app/controller.py"),
    Path("oklab_colour_picker/infrastructure/krita_adapter.py"),
}
LOWER_LAYER_FILES = {
    Path("oklab_colour_picker/domain/__init__.py"),
    Path("oklab_colour_picker/domain/color_math.py"),
    Path("oklab_colour_picker/domain/colour_presentation.py"),
    Path("oklab_colour_picker/domain/colour_state.py"),
    Path("oklab_colour_picker/domain/gamut_fallback.py"),
    Path("oklab_colour_picker/domain/readout_interaction.py"),
    Path("oklab_colour_picker/domain/selector_interaction.py"),
    Path("oklab_colour_picker/models/__init__.py"),
    Path("oklab_colour_picker/models/base.py"),
    Path("oklab_colour_picker/models/geometry.py"),
    Path("oklab_colour_picker/models/hue_lightness_slice.py"),
    Path("oklab_colour_picker/models/lightness_chroma_slice.py"),
    Path("oklab_colour_picker/models/lightness_slice.py"),
    Path("oklab_colour_picker/render/__init__.py"),
    Path("oklab_colour_picker/render/renderers.py"),
    Path("oklab_colour_picker/app/controller.py"),
}
UI_LAYER_MODULE_PREFIXES = (
    "oklab_colour_picker.plugin",
    "oklab_colour_picker.ui",
)


def test_krita_imports_are_limited_to_boundary_files():
    offenders = []
    for path, tree in _project_python_asts():
        for module in _imported_modules(tree):
            if _is_krita_module(module) and path not in KRITA_IMPORT_ALLOWED:
                offenders.append(f"{path}: {module}")

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


def test_pure_color_math_has_no_qt_or_krita_imports():
    offenders = []
    for path in sorted(PURE_NO_QT_OR_KRITA):
        full_path = ROOT / path
        if not full_path.exists():
            continue
        tree = ast.parse(full_path.read_text(), filename=path.as_posix())
        for module in _imported_modules(tree):
            if module.startswith(("PyQt5", "PySide", "krita")):
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
        if path not in LOWER_LAYER_FILES:
            continue
        for module in _project_import_references(tree, path):
            if _starts_with_any(module, UI_LAYER_MODULE_PREFIXES):
                offenders.append(f"{path}: {module}")

    assert offenders == []


def test_lower_layer_guard_constants_match_dev_check_runner():
    dev_checks = _load_dev_checks_module()

    assert LOWER_LAYER_FILES == dev_checks.LOWER_LAYER_FILES
    assert UI_LAYER_MODULE_PREFIXES == dev_checks.UI_LAYER_MODULE_PREFIXES


def test_relative_import_references_are_resolved_before_lower_layer_guard():
    tree = ast.parse("from ..ui.selectors import selector\nfrom ..ui import dock\n")
    import_from_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    dev_checks = _load_dev_checks_module()

    imports = set(
        _project_import_references(
            tree,
            Path("oklab_colour_picker/domain/color_math.py"),
        )
    )
    runner_imports = {
        imported
        for node in import_from_nodes
        for imported in dev_checks.project_import_references(
            node,
            Path("oklab_colour_picker/domain/color_math.py"),
        )
    }

    assert "oklab_colour_picker.ui.selectors" in imports
    assert "oklab_colour_picker.ui.selectors.selector" in imports
    assert "oklab_colour_picker.ui.dock" in imports
    assert imports == runner_imports


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
            module = node.module or ""
            if node.level:
                base = ".".join(_relative_import_base(source_path, node.level))
                resolved_module = ".".join(part for part in (base, module) if part)
                if module:
                    yield resolved_module
                    for alias in node.names:
                        yield f"{resolved_module}.{alias.name}"
                    continue
                for alias in node.names:
                    yield f"{base}.{alias.name}"
                continue
            if module != "oklab_colour_picker":
                yield module
                continue
            for alias in node.names:
                yield f"{module}.{alias.name}"


def _is_krita_module(module):
    return module == "krita" or module.startswith("krita.")


def _starts_with_any(module, prefixes):
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


def _relative_import_base(source_path, level):
    module_parts = source_path.with_suffix("").parts
    package_parts = module_parts[:-1]
    keep = max(0, len(package_parts) - level + 1)
    return package_parts[:keep]


def _load_dev_checks_module():
    path = ROOT / "scripts" / "checks" / "dev_checks.py"
    spec = importlib.util.spec_from_file_location("dev_checks_for_import_discipline", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
