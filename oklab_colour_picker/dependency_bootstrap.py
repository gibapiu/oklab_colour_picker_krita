"""Opt-in runtime dependency installation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import runpy
import subprocess
import sys


NUMPY_REQUIREMENT = "numpy>=1.26,<3"
ENSUREPIP_TIMEOUT_SECONDS = 120
PIP_INSTALL_TIMEOUT_SECONDS = 600


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
    """Run pip via the discovered interpreter.

    Returns ``None`` when the interpreter or pip bootstrap cannot be exercised
    (interpreter not runnable, ensurepip fails) so the caller can fall back to
    in-process pip. Returns an ``InstallResult`` for any actual pip-install
    outcome — success, timeout, or a normal pip failure such as
    "no matching wheel" — since retrying those in-process would recur.
    """
    if not _ensure_pip_available_subprocess(python):
        return None

    try:
        completed = subprocess.run(
            [python, "-m", "pip", *_pip_install_args(vendor_path, requirement)],
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
    """Run pip inside Krita's own interpreter when no python executable is reachable.

    Used on Krita builds that bundle a Python runtime without exposing a
    standalone python.exe. We mutate sys.argv/sys.exit while pip's CLI runs and
    restore them afterwards.
    """
    if not _ensure_pip_available_in_process():
        return InstallResult(
            False,
            "pip is unavailable in Krita's bundled Python and `ensurepip` did not bootstrap it.",
        )

    try:
        exit_code = _run_pip_in_process(["pip", *_pip_install_args(vendor_path, requirement)])
    except Exception as exc:
        return InstallResult(False, f"NumPy installation failed: {exc}")

    if exit_code == 0:
        return InstallResult(True, "NumPy installed. Restart Krita to load the colour selector.")
    return InstallResult(False, f"pip exited with status {exit_code}.")


def _pip_install_args(vendor_path: str, requirement: str) -> list[str]:
    return [
        "install",
        "--upgrade",
        "--only-binary=:all:",
        "--target",
        vendor_path,
        requirement,
    ]


def _ensure_pip_available_subprocess(python: str) -> bool:
    if _python_can_import(python, "pip"):
        return True

    try:
        subprocess.run(
            [python, "-m", "ensurepip", "--upgrade"],
            check=False,
            capture_output=True,
            text=True,
            timeout=ENSUREPIP_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False

    return _python_can_import(python, "pip")


def _ensure_pip_available_in_process() -> bool:
    if _can_import("pip"):
        return True

    try:
        import ensurepip

        ensurepip.bootstrap(upgrade=True)
    except Exception:
        return False

    return _can_import("pip")


def _can_import(module: str) -> bool:
    import importlib.util

    importlib.invalidate_caches()
    return importlib.util.find_spec(module) is not None


def _python_can_import(python: str, module: str) -> bool:
    try:
        completed = subprocess.run(
            [python, "-c", f"import {module}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=ENSUREPIP_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


def _run_pip_in_process(argv: list[str]) -> int:
    original_argv = sys.argv
    original_exit = sys.exit

    def _exit(code=0):
        raise SystemExit(code)

    try:
        sys.argv = argv
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
        sys.argv = original_argv
        sys.exit = original_exit


def _looks_like_python(executable_name: str) -> bool:
    name = executable_name.lower()
    return name.startswith("python") or name == "krita_python"


def _format_process_failure(completed: subprocess.CompletedProcess) -> str:
    output = (completed.stderr or completed.stdout or "").strip()
    if output:
        return output
    return f"pip exited with status {completed.returncode}."
