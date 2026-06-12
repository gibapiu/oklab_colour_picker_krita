"""Runtime-specific locations for private binary dependencies."""

from __future__ import annotations

from pathlib import Path
import platform
import re
import struct
import sys
import sysconfig


VENDOR_ROOT_DIRECTORY_NAME = "oklab_colour_picker"
VENDOR_SITE_PACKAGES_DIRECTORY_NAME = "site-packages"


def runtime_abi_tag() -> str:
    """Return a filesystem-safe tag for the active Python extension ABI."""

    safe_tag = _sanitize_tag(sysconfig.get_config_var("SOABI"))
    if safe_tag:
        return safe_tag
    return _fallback_abi_tag()


def resolve_dependency_path(app_data_location: str | None) -> str | None:
    """Return the private dependency directory that is safe to import."""

    vendor_path = vendor_site_packages_path(app_data_location)
    if Path(vendor_path).is_dir():
        return vendor_path

    legacy_path = vendor_site_packages_root(app_data_location)
    if _legacy_numpy_is_compatible(legacy_path):
        return legacy_path
    return None


def _legacy_numpy_is_compatible(vendor_root: str) -> bool:
    """Return whether a legacy NumPy tree matches the active extension ABI."""

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not extension_suffix:
        return False

    numpy_path = Path(vendor_root) / "numpy"
    if not numpy_path.is_dir():
        return False

    abi_extensions = [
        path
        for path in numpy_path.rglob("*")
        if path.is_file() and _has_python_abi_marker(path.name)
    ]
    return bool(abi_extensions) and all(
        path.name.endswith(extension_suffix) for path in abi_extensions
    )


def vendor_site_packages_path(app_data_location: str | None) -> str:
    """Return the private dependency path for the active Python ABI."""

    return str(Path(vendor_site_packages_root(app_data_location)) / runtime_abi_tag())


def vendor_site_packages_root(app_data_location: str | None) -> str:
    """Return the legacy unversioned root that contains ABI-specific paths."""

    if app_data_location:
        return str(
            Path(app_data_location)
            / VENDOR_ROOT_DIRECTORY_NAME
            / VENDOR_SITE_PACKAGES_DIRECTORY_NAME
        )

    package_dir = Path(__file__).resolve().parents[1]
    return str(package_dir / VENDOR_SITE_PACKAGES_DIRECTORY_NAME)


def _fallback_abi_tag() -> str:
    interpreter_tag = sys.implementation.cache_tag or (
        f"python-{sys.version_info.major}.{sys.version_info.minor}"
    )
    platform_tag = sysconfig.get_platform() or (
        f"{platform.system()}-{platform.machine()}-{struct.calcsize('P') * 8}bit"
    )
    safe_tag = _sanitize_tag(f"{interpreter_tag}-{platform_tag}")
    if safe_tag:
        return safe_tag
    return f"python-{sys.version_info.major}.{sys.version_info.minor}-{struct.calcsize('P') * 8}bit"


def _sanitize_tag(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")


def _has_python_abi_marker(filename: str) -> bool:
    lowered = filename.lower()
    return lowered.endswith((".so", ".pyd")) and (
        ".cpython-" in lowered or re.search(r"\.cp\d{2,}", lowered) is not None
    )
