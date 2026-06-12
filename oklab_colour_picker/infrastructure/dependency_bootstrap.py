"""Opt-in runtime dependency installation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import platform
from pathlib import Path
import runpy
import shutil
import ssl
import subprocess
import sys
import sysconfig
import tempfile
import threading
from urllib import request


NUMPY_REQUIREMENT = "numpy>=1.26,<3"

PIP_WHEEL_DIRECTORY_NAME = "pip-wheel"
PIP_DOWNLOAD_TIMEOUT_SECONDS = 120
PIP_INSTALL_TIMEOUT_SECONDS = 600
PYTHON_PROBE_TIMEOUT_SECONDS = 30

_PIP_MODULE_PREFIX = "pip."
_PIP_RUN_LOCK = threading.RLock()
_NUMPY_TARGET_NAMES = frozenset({"numpy", "numpy.libs"})

_PIP_WHEEL_INSTALL_SCRIPT = """
import runpy
import sys

pip_wheel = sys.argv[1]
pip_args = sys.argv[2:]
sys.path.insert(0, pip_wheel)
sys.argv = ["pip", *pip_args]
runpy.run_module("pip", run_name="__main__")
"""


@dataclass(frozen=True)
class InstallResult:
    success: bool
    message: str


def install_numpy(vendor_path: str, *, requirement: str = NUMPY_REQUIREMENT) -> InstallResult:
    Path(vendor_path).mkdir(parents=True, exist_ok=True)

    python = find_krita_python()
    if python is not None:
        subprocess_result = _install_via_subprocess(python, vendor_path, requirement)
        if subprocess_result is not None:
            if subprocess_result.success or not _could_be_network_failure(subprocess_result.message):
                return subprocess_result

            offline_result = _install_from_host_wheelhouse(python, vendor_path, requirement)
            if offline_result is not None:
                return offline_result
            return subprocess_result
        # The interpreter cannot be exercised; let the in-process path try.

    in_process_result = _install_in_process(vendor_path, requirement)
    if in_process_result.success:
        return in_process_result

    offline_result = _install_from_host_wheelhouse(None, vendor_path, requirement)
    if offline_result is not None:
        return offline_result
    return in_process_result


def _find_host_python() -> str | None:
    """Return a system Python on PATH other than Krita's bundled interpreter."""
    krita_prefix = os.path.realpath(sys.prefix) + os.sep
    for name in ("python3", "python"):
        executable = shutil.which(name)
        if executable is None:
            continue
        resolved = os.path.realpath(executable)
        if resolved.startswith(krita_prefix):
            continue
        if resolved == os.path.realpath(sys.executable) and _python_matches_krita_runtime(resolved):
            continue
        return executable
    return None


def _host_pip_download_args(wheelhouse: str, requirement: str) -> list[str]:
    major, minor = sys.version_info[:2]
    args = [
        "-m", "pip", "download",
        "--no-input",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--dest", wheelhouse,
        "--only-binary=:all:",
        "--python-version", f"{major}.{minor}",
        "--abi", f"cp{major}{minor}",
        "--implementation", "cp",
    ]
    for platform_tag in _target_platforms():
        args.extend(("--platform", platform_tag))
    args.append(requirement)
    return args


def _target_platforms() -> list[str]:
    """Return explicit pip platform tags when host defaults can pick the wrong ABI."""
    platform_name = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    if not platform_name.startswith("linux_"):
        return []

    arch = platform_name.removeprefix("linux_")
    libc_name, libc_version = platform.libc_ver()
    manylinux = _manylinux_platforms(arch, libc_version) if libc_name == "glibc" else []
    return [*manylinux, platform_name]


def _manylinux_platforms(arch: str, glibc_version: str) -> list[str]:
    version_parts = glibc_version.split(".")
    if len(version_parts) < 2:
        return []
    try:
        major = int(version_parts[0])
        minor = int(version_parts[1])
    except ValueError:
        return []
    if major != 2 or minor < 17:
        return []
    return [
        *(f"manylinux_2_{version}_{arch}" for version in range(minor, 16, -1)),
        f"manylinux2014_{arch}",
    ]


def find_krita_python() -> str | None:
    """Locate a Python executable that matches Krita's runtime.

    On Linux ``sys.executable`` may be a host Python, so it is used only after proving that its major/minor version matches Krita's runtime.
    On Windows ``sys.executable`` is ``krita.exe`` and the bundled interpreter sits next to it.
    On macOS the bundle ships ``krita_python`` alongside ``krita`` inside ``Contents/MacOS``.

    Some Krita builds ship without a Python interpreter at all. In that case this returns ``None`` and the caller must fall back to in-process pip.
    """
    executable = sys.executable
    if executable and _looks_like_python(Path(executable).name) and _python_matches_krita_runtime(executable):
        return executable

    if not executable:
        return None

    here = Path(executable).parent
    candidates = [
        here / "python.exe",
        here / "python3.exe",
        here / "python3",
        here / "python",
        here / "krita_python",
    ]
    for candidate in candidates:
        if candidate.is_file() and _python_matches_krita_runtime(str(candidate)):
            return str(candidate)
    return None


def _python_matches_krita_runtime(python: str) -> bool:
    """Check that an external Python executable uses Krita's Python ABI."""
    try:
        completed = subprocess.run(
            [
                python,
                "-I",
                "-c",
                "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=PYTHON_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    expected = f"{sys.version_info[0]}.{sys.version_info[1]}"
    return completed.returncode == 0 and completed.stdout.strip() == expected


def _install_via_subprocess(
    python: str,
    vendor_path: str,
    requirement: str,
    *,
    wheelhouse: str | None = None,
) -> InstallResult | None:
    """Run an explicit pip wheel via the discovered interpreter.

    Runs in isolated mode (``-I``) so the interpreter ignores ``PYTHONPATH`` and user-site,
    so the install script's ``sys.path[0]`` wheel wins over any leaked host pip.

    Returns ``None`` only when the interpreter cannot be exercised, so the caller can fall back to the in-process path.
    """
    try:
        pip_wheel = _get_or_download_pip_wheel(vendor_path)
        _clear_numpy_target(vendor_path)
    except PipBootstrapError as exc:
        return InstallResult(False, str(exc))

    try:
        completed = subprocess.run(
            [
                python,
                "-I",
                "-c",
                _PIP_WHEEL_INSTALL_SCRIPT,
                str(pip_wheel),
                *_pip_install_args(vendor_path, requirement, wheelhouse=wheelhouse),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=PIP_INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return InstallResult(False, "pip install timed out. Check your network connection and retry.")
    except OSError:
        return None

    if completed.returncode == 0:
        return InstallResult(True, "NumPy installed. Restart Krita to load the colour selector.")
    return InstallResult(False, _format_process_failure(completed))


def _install_in_process(
    vendor_path: str,
    requirement: str,
    *,
    wheelhouse: str | None = None,
) -> InstallResult:
    """Run an explicit pip wheel inside Krita's interpreter.

    Used on Krita builds that bundle a Python runtime without exposing a standalone python executable.
    """
    try:
        pip_wheel = _get_or_download_pip_wheel(vendor_path)
        _clear_numpy_target(vendor_path)
    except PipBootstrapError as exc:
        return InstallResult(False, str(exc))

    try:
        exit_code = _run_pip_wheel_in_process(
            pip_wheel,
            _pip_install_args(vendor_path, requirement, wheelhouse=wheelhouse),
        )
    except Exception as exc:
        return InstallResult(False, f"NumPy installation failed: {exc}")

    if exit_code == 0:
        return InstallResult(True, "NumPy installed. Restart Krita to load the colour selector.")
    return InstallResult(False, f"pip exited with status {exit_code}.")


def _pip_install_args(vendor_path: str, requirement: str, *, wheelhouse: str | None = None) -> list[str]:
    args = [
        "--isolated",
        "--no-input",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "install",
        "--upgrade",
        "--force-reinstall",
        "--only-binary=:all:",
    ]
    if wheelhouse is not None:
        args.extend(("--no-index", "--find-links", wheelhouse))
    args.extend(
        [
            "--target",
            vendor_path,
            requirement,
        ]
    )
    return args


class PipBootstrapError(Exception):
    pass


@dataclass(frozen=True)
class _PipWheel:
    url: str
    filename: str
    sha256: str


_PINNED_PIP_WHEEL = _PipWheel(
    url=(
        "https://files.pythonhosted.org/packages/29/a2/"
        "d40fb2460e883eca5199c62cfc2463fd261f760556ae6290f88488c362c0/"
        "pip-25.1.1-py3-none-any.whl"
    ),
    filename="pip-25.1.1-py3-none-any.whl",
    sha256="2913a38a2abf4ea6b64ab507bd9e967f3b53dc1ede74b01b0931e1ce548751af",
)


def _get_or_download_pip_wheel(vendor_path: str) -> Path:
    wheel_dir = Path(vendor_path).parent / PIP_WHEEL_DIRECTORY_NAME
    wheel_dir.mkdir(parents=True, exist_ok=True)

    wheel = _PINNED_PIP_WHEEL
    wheel_path = wheel_dir / wheel.filename
    if wheel_path.exists() and _path_matches_sha256(wheel_path, wheel.sha256):
        return wheel_path

    try:
        wheel_bytes = _download(wheel.url)
    except PipBootstrapError as direct_error:
        host_wheel = _download_pinned_pip_with_host_python(wheel_dir, wheel)
        if isinstance(host_wheel, Path):
            return host_wheel
        if host_wheel is not None:
            raise PipBootstrapError(f"{direct_error} Host Python fallback also failed: {host_wheel}") from direct_error
        raise

    _verify_pip_wheel_bytes(wheel_bytes, wheel.sha256, "Downloaded pip wheel")

    temporary_path = wheel_path.with_suffix(f"{wheel_path.suffix}.tmp")
    temporary_path.write_bytes(wheel_bytes)
    temporary_path.replace(wheel_path)
    return wheel_path


def _download(url: str) -> bytes:
    try:
        with request.urlopen(url, timeout=PIP_DOWNLOAD_TIMEOUT_SECONDS, context=_pypi_ssl_context()) as response:
            return response.read()
    except OSError as exc:
        raise PipBootstrapError(f"Could not download pip from PyPI: {exc}") from exc


# Krita's bundled OpenSSL may resolve a CA path that is absent on the host distro,
# so verification falls back to whichever common store can be located.
_CA_BUNDLE_CANDIDATES = (
    "/etc/ssl/certs/ca-certificates.crt",
    "/etc/pki/tls/certs/ca-bundle.crt",
    "/etc/ssl/cert.pem",
)


def _pypi_ssl_context() -> ssl.SSLContext | None:
    cafile = os.environ.get("SSL_CERT_FILE")
    if not (cafile and os.path.isfile(cafile)):
        cafile = next((path for path in _CA_BUNDLE_CANDIDATES if os.path.isfile(path)), None)
    if cafile is None:
        return None
    return ssl.create_default_context(cafile=cafile)


def _path_matches_sha256(path: Path, expected_sha256: str) -> bool:
    return hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256


def _verify_pip_wheel_bytes(payload: bytes, expected_sha256: str, label: str) -> None:
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise PipBootstrapError(f"{label} failed its PyPI sha256 check.")


def _clear_numpy_target(vendor_path: str) -> None:
    """Remove stale NumPy files that pip --target may leave behind."""
    root = Path(vendor_path)
    if not root.exists():
        return
    for child in root.iterdir():
        if not _is_numpy_target_artifact(child.name):
            continue
        try:
            if child.is_symlink() or not child.is_dir():
                child.unlink()
            else:
                shutil.rmtree(child)
        except OSError as exc:
            raise PipBootstrapError(f"Could not remove stale NumPy files from {child}: {exc}") from exc


def _is_numpy_target_artifact(name: str) -> bool:
    lowered = name.lower()
    return (
        lowered in _NUMPY_TARGET_NAMES
        or lowered.startswith("numpy-") and lowered.endswith((".dist-info", ".egg-info"))
    )


def _download_pinned_pip_with_host_python(wheel_dir: Path, wheel: _PipWheel) -> Path | str | None:
    host_python = _find_host_python()
    if host_python is None:
        return None

    try:
        completed = subprocess.run(
            [
                host_python,
                "-m", "pip", "download",
                "--no-input",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "--no-deps",
                "--only-binary=:all:",
                "--dest", str(wheel_dir),
                f"pip=={_pip_version_from_wheel_filename(wheel.filename)}",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=PIP_DOWNLOAD_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return "host pip download timed out."
    except OSError as exc:
        return str(exc)

    if completed.returncode != 0:
        return _format_process_failure(completed)

    wheel_path = wheel_dir / wheel.filename
    if not wheel_path.exists():
        return f"host pip did not download {wheel.filename}."
    if not _path_matches_sha256(wheel_path, wheel.sha256):
        try:
            wheel_path.unlink()
        except OSError:
            pass
        return "host-downloaded pip wheel failed its pinned sha256 check."
    return wheel_path


def _pip_version_from_wheel_filename(filename: str) -> str:
    prefix = "pip-"
    suffix = "-py3-none-any.whl"
    return filename.removeprefix(prefix).removesuffix(suffix)


def _install_from_host_wheelhouse(
    python: str | None,
    vendor_path: str,
    requirement: str,
) -> InstallResult | None:
    host_python = _find_host_python()
    if host_python is None or sys.implementation.name != "cpython":
        return None

    with tempfile.TemporaryDirectory(prefix="oklab-wheelhouse-", dir=str(Path(vendor_path).parent)) as wheelhouse:
        if not _download_wheelhouse_with_host_python(host_python, wheelhouse, requirement):
            return None
        if python is not None:
            return _install_via_subprocess(python, vendor_path, requirement, wheelhouse=wheelhouse)
        return _install_in_process(vendor_path, requirement, wheelhouse=wheelhouse)


def _download_wheelhouse_with_host_python(host_python: str, wheelhouse: str, requirement: str) -> bool:
    try:
        completed = subprocess.run(
            [host_python, *_host_pip_download_args(wheelhouse, requirement)],
            check=False,
            capture_output=True,
            text=True,
            timeout=PIP_INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return False
    except OSError:
        return False

    return completed.returncode == 0 and any(Path(wheelhouse).iterdir())


def _could_be_network_failure(message: str) -> bool:
    needles = (
        "certificate_verify_failed",
        "could not fetch url",
        "could not install packages due to an oserror",
        "failed to establish a new connection",
        "network connection",
        "proxy",
        "ssl",
        "temporary failure in name resolution",
        "timed out",
        "timeout",
        "urlopen error",
    )
    lowered = message.lower()
    return any(needle in lowered for needle in needles)


def _run_pip_wheel_in_process(pip_wheel: Path, argv: list[str]) -> int:
    with _PIP_RUN_LOCK:
        original_argv = sys.argv
        original_exit = sys.exit
        original_path = sys.path[:]
        original_pip_modules = {
            module_name: module
            for module_name, module in sys.modules.items()
            if module_name == "pip" or module_name.startswith(_PIP_MODULE_PREFIX)
        }

        def _exit(code=0):
            raise SystemExit(code)

        try:
            _clear_pip_modules()
            sys.path.insert(0, str(pip_wheel))
            sys.argv = ["pip", *argv]
            sys.exit = _exit
            try:
                runpy.run_module("pip", run_name="__main__")
            except SystemExit as exc:
                if exc.code is None:
                    return 0
                if isinstance(exc.code, int):
                    return exc.code
                return 1
            return 0
        finally:
            # Drop the wheel's pip modules and restore any that pre-existed.
            _clear_pip_modules()
            sys.modules.update(original_pip_modules)
            sys.path[:] = original_path
            sys.argv = original_argv
            sys.exit = original_exit


def _clear_pip_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "pip" or module_name.startswith(_PIP_MODULE_PREFIX):
            sys.modules.pop(module_name, None)


def _looks_like_python(executable_name: str) -> bool:
    name = executable_name.lower()
    return name.startswith("python") or name == "krita_python"


def _format_process_failure(completed: subprocess.CompletedProcess) -> str:
    output = (completed.stderr or completed.stdout or "").strip()
    if output:
        return output
    return f"pip exited with status {completed.returncode}."
