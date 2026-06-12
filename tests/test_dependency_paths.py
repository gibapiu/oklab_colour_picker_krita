import types

import pytest

from oklab_colour_picker.infrastructure import dependency_paths


@pytest.mark.parametrize(
    ("soabi", "expected"),
    [
        ("cpython-314-x86_64-linux-gnu", "cpython-314-x86_64-linux-gnu"),
        ("cpython-314/test", "cpython-314_test"),
    ],
)
def test_runtime_abi_tag_uses_sanitized_soabi(monkeypatch, soabi, expected):
    monkeypatch.setattr(
        dependency_paths.sysconfig,
        "get_config_var",
        lambda name: soabi if name == "SOABI" else None,
    )

    assert dependency_paths.runtime_abi_tag() == expected


@pytest.mark.parametrize(
    ("soabi", "platform_tag", "expected"),
    [
        (None, "win-amd64", "cpython-314-win-amd64"),
        ("///", "macosx-15-arm64", "cpython-314-macosx-15-arm64"),
    ],
)
def test_runtime_abi_tag_falls_back_to_interpreter_and_platform(
    monkeypatch,
    soabi,
    platform_tag,
    expected,
):
    monkeypatch.setattr(
        dependency_paths.sysconfig,
        "get_config_var",
        lambda name: soabi if name == "SOABI" else None,
    )
    monkeypatch.setattr(dependency_paths.sysconfig, "get_platform", lambda: platform_tag)
    monkeypatch.setattr(
        dependency_paths.sys,
        "implementation",
        types.SimpleNamespace(cache_tag="cpython-314"),
    )

    assert dependency_paths.runtime_abi_tag() == expected


@pytest.mark.parametrize(
    ("extension_name", "expected"),
    [
        ("_multiarray_umath.cpython-314-x86_64-linux-gnu.so", True),
        ("_multiarray_umath.cpython-310-x86_64-linux-gnu.so", False),
        ("__init__.py", False),
    ],
)
def test_legacy_numpy_compatibility(tmp_path, monkeypatch, extension_name, expected):
    numpy_path = tmp_path / "numpy" / "_core"
    numpy_path.mkdir(parents=True)
    (numpy_path / extension_name).touch()
    (numpy_path / "__init__.cpython-314.pyc").touch()
    monkeypatch.setattr(
        dependency_paths.sysconfig,
        "get_config_var",
        lambda name: ".cpython-314-x86_64-linux-gnu.so" if name == "EXT_SUFFIX" else None,
    )

    assert dependency_paths._legacy_numpy_is_compatible(str(tmp_path)) is expected


@pytest.mark.parametrize(
    "abi_tag",
    [
        "cpython-310-x86_64-linux-gnu",
        "cpython-314-x86_64-linux-gnu",
    ],
)
def test_vendor_path_is_isolated_by_runtime_abi(tmp_path, monkeypatch, abi_tag):
    monkeypatch.setattr(dependency_paths, "runtime_abi_tag", lambda: abi_tag)

    path = dependency_paths.vendor_site_packages_path(str(tmp_path))

    assert path == str(
        tmp_path
        / dependency_paths.VENDOR_ROOT_DIRECTORY_NAME
        / dependency_paths.VENDOR_SITE_PACKAGES_DIRECTORY_NAME
        / abi_tag
    )


def test_vendor_path_falls_back_next_to_plugin_package(tmp_path, monkeypatch):
    module_path = tmp_path / "oklab_colour_picker" / "infrastructure" / "dependency_paths.py"
    monkeypatch.setattr(dependency_paths, "__file__", str(module_path))
    monkeypatch.setattr(dependency_paths, "runtime_abi_tag", lambda: "cpython-314-win-amd64")

    path = dependency_paths.vendor_site_packages_path(None)

    assert path == str(
        tmp_path
        / "oklab_colour_picker"
        / dependency_paths.VENDOR_SITE_PACKAGES_DIRECTORY_NAME
        / "cpython-314-win-amd64"
    )


def test_resolver_prefers_runtime_specific_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_paths, "runtime_abi_tag", lambda: "cpython-314")
    vendor_path = tmp_path / "oklab_colour_picker" / "site-packages" / "cpython-314"
    vendor_path.mkdir(parents=True)
    monkeypatch.setattr(
        dependency_paths,
        "_legacy_numpy_is_compatible",
        lambda _path: pytest.fail("legacy path should not be inspected"),
    )

    assert dependency_paths.resolve_dependency_path(str(tmp_path)) == str(vendor_path)


@pytest.mark.parametrize(
    ("legacy_is_compatible", "expected_legacy_path"),
    [(True, True), (False, False)],
)
def test_resolver_reuses_only_compatible_legacy_directory(
    tmp_path,
    monkeypatch,
    legacy_is_compatible,
    expected_legacy_path,
):
    legacy_path = tmp_path / "oklab_colour_picker" / "site-packages"
    legacy_path.mkdir(parents=True)
    monkeypatch.setattr(
        dependency_paths,
        "_legacy_numpy_is_compatible",
        lambda path: path == str(legacy_path) and legacy_is_compatible,
    )

    resolved = dependency_paths.resolve_dependency_path(str(tmp_path))

    assert resolved == (str(legacy_path) if expected_legacy_path else None)
