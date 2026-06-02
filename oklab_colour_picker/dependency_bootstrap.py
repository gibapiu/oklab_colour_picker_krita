"""Opt-in runtime dependency installation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import runpy
import subprocess
import sys
import threading
from urllib import request
from urllib.error import URLError
from urllib.parse import urlparse


NUMPY_REQUIREMENT = "numpy>=1.26,<3"
PIP_PROJECT_METADATA_URL = "https://pypi.org/pypi/pip/json"
PIP_WHEEL_DIRECTORY_NAME = "pip-wheel"
PIP_DOWNLOAD_TIMEOUT_SECONDS = 120
PIP_INSTALL_TIMEOUT_SECONDS = 600
_PIP_MODULE_PREFIX = "pip."
_PIP_RUN_LOCK = threading.RLock()
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
            return subprocess_result
        # The interpreter or pip bootstrap is unusable; let the in-process
        # path try. We only fall through on infrastructure failures, never on
        # genuine pip install failures (e.g. "no matching wheel"), since
        # retrying those in-process would just repeat the same error.
    return _install_in_process(vendor_path, requirement)


def find_krita_python() -> str | None:
    """Locate a Python executable that matches Krita's runtime.

    On Linux Krita usually runs under system Python, so ``sys.executable`` is
    already python. On Windows ``sys.executable`` is ``krita.exe`` and the
    bundled interpreter sits next to it. On macOS the bundle ships
    ``krita_python`` alongside ``krita`` inside ``Contents/MacOS``. Some Krita
    builds ship without a Python interpreter at all; in that case this returns
    ``None`` and the caller must fall back to in-process pip.
    """
    executable = sys.executable
    if executable and _looks_like_python(Path(executable).name):
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
        if candidate.is_file():
            return str(candidate)
    return None


def _install_via_subprocess(python: str, vendor_path: str, requirement: str) -> InstallResult | None:
    """Run an explicit pip wheel via the discovered interpreter.

    The ``-I`` flag keeps Krita's Python from importing host Python packages
    such as Arch's system pip. Returns ``None`` only when the interpreter cannot
    be exercised, so the caller can fall back to the in-process path.
    """
    try:
        pip_wheel = _get_or_download_pip_wheel(vendor_path)
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
                *_pip_install_args(vendor_path, requirement),
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


def _install_in_process(vendor_path: str, requirement: str) -> InstallResult:
    """Run an explicit pip wheel inside Krita's interpreter.

    Used on Krita builds that bundle a Python runtime without exposing a
    standalone python executable. This path mutates process-global Python state,
    so it scopes the changes tightly and clears only pip modules before/after
    the run to avoid mixing a leaked host pip with the selected wheel.
    """
    try:
        pip_wheel = _get_or_download_pip_wheel(vendor_path)
    except PipBootstrapError as exc:
        return InstallResult(False, str(exc))

    try:
        exit_code = _run_pip_wheel_in_process(
            pip_wheel,
            _pip_install_args(vendor_path, requirement),
        )
    except Exception as exc:
        return InstallResult(False, f"NumPy installation failed: {exc}")

    if exit_code == 0:
        return InstallResult(True, "NumPy installed. Restart Krita to load the colour selector.")
    return InstallResult(False, f"pip exited with status {exit_code}.")


def _pip_install_args(vendor_path: str, requirement: str) -> list[str]:
    return [
        "--isolated",
        "--no-input",
        "--disable-pip-version-check",
        "install",
        "--upgrade",
        "--only-binary=:all:",
        "--target",
        vendor_path,
        requirement,
    ]


class PipBootstrapError(Exception):
    pass


def _get_or_download_pip_wheel(vendor_path: str) -> Path:
    wheel_dir = Path(vendor_path).parent / PIP_WHEEL_DIRECTORY_NAME
    wheel_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        wheel_dir.glob("pip-*.whl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if existing:
        return existing[0]

    wheel_url = _latest_pip_wheel_url()
    wheel_name = Path(urlparse(wheel_url).path).name
    if not wheel_name:
        raise PipBootstrapError("Could not determine the pip wheel filename from PyPI.")

    try:
        with request.urlopen(wheel_url, timeout=PIP_DOWNLOAD_TIMEOUT_SECONDS) as response:
            wheel_bytes = response.read()
    except (OSError, URLError) as exc:
        raise PipBootstrapError(f"Could not download pip from PyPI: {exc}") from exc

    wheel_path = wheel_dir / wheel_name
    temporary_wheel_path = wheel_path.with_suffix(f"{wheel_path.suffix}.tmp")
    temporary_wheel_path.write_bytes(wheel_bytes)
    temporary_wheel_path.replace(wheel_path)
    return wheel_path


def _latest_pip_wheel_url() -> str:
    try:
        with request.urlopen(PIP_PROJECT_METADATA_URL, timeout=PIP_DOWNLOAD_TIMEOUT_SECONDS) as response:
            metadata = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        raise PipBootstrapError(f"Could not fetch pip metadata from PyPI: {exc}") from exc

    for package in metadata.get("urls", []):
        if package.get("packagetype") == "bdist_wheel":
            url = package.get("url")
            if isinstance(url, str) and url:
                return url

    raise PipBootstrapError("PyPI did not return a pip wheel download URL.")


def _run_pip_wheel_in_process(pip_wheel: Path, argv: list[str]) -> int:
    with _PIP_RUN_LOCK:
        original_argv = sys.argv
        original_exit = sys.exit
        original_path = sys.path[:]

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
            _clear_pip_modules()
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
