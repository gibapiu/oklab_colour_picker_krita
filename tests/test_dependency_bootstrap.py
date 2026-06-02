import os
import subprocess
from pathlib import Path

import pytest

from oklab_colour_picker import dependency_bootstrap


KRITA_PYTHON = "/fake/krita/bin/python.exe"
PIP_WHEEL = Path("/fake/pip-25.1.1-py3-none-any.whl")


def _completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


def _pip_args(tmp_path):
    return [
        "--isolated",
        "--no-input",
        "--disable-pip-version-check",
        "install",
        "--upgrade",
        "--only-binary=:all:",
        "--target",
        str(tmp_path),
        dependency_bootstrap.NUMPY_REQUIREMENT,
    ]


def test_install_numpy_invokes_krita_python_with_isolated_pip_wheel(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _completed(args, returncode=0, stdout="ok")

    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor_path: PIP_WHEEL)
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert calls == [
        (
            [
                KRITA_PYTHON,
                "-I",
                "-c",
                dependency_bootstrap._PIP_WHEEL_INSTALL_SCRIPT,
                str(PIP_WHEEL),
                *_pip_args(tmp_path),
            ],
            {
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": dependency_bootstrap.PIP_INSTALL_TIMEOUT_SECONDS,
            },
        )
    ]


def test_install_numpy_falls_back_to_in_process_when_no_python_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: None)
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor_path: PIP_WHEEL)

    captured = []
    monkeypatch.setattr(
        dependency_bootstrap,
        "_run_pip_wheel_in_process",
        lambda pip_wheel, argv: captured.append((pip_wheel, argv)) or 0,
    )

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert captured == [(PIP_WHEEL, _pip_args(tmp_path))]


def test_install_numpy_in_process_reports_pip_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: None)
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor_path: PIP_WHEEL)
    monkeypatch.setattr(dependency_bootstrap, "_run_pip_wheel_in_process", lambda _pip_wheel, _argv: 2)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "status 2" in result.message


def test_install_numpy_reports_pip_bootstrap_error(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: None)
    monkeypatch.setattr(
        dependency_bootstrap,
        "_get_or_download_pip_wheel",
        lambda _vendor_path: (_ for _ in ()).throw(dependency_bootstrap.PipBootstrapError("could not fetch pip")),
    )

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "could not fetch pip" in result.message


def test_install_numpy_surfaces_pip_stderr_on_failure(tmp_path, monkeypatch):
    def fake_run(args, **kwargs):
        return _completed(args, returncode=1, stderr="ERROR: no matching wheel")

    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor_path: PIP_WHEEL)
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "no matching wheel" in result.message


def test_install_numpy_reports_timeout(tmp_path, monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 0))

    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor_path: PIP_WHEEL)
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "timed out" in result.message.lower()


def test_install_numpy_falls_through_when_subprocess_interpreter_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor_path: PIP_WHEEL)
    monkeypatch.setattr(
        dependency_bootstrap.subprocess,
        "run",
        lambda _args, **_kwargs: (_ for _ in ()).throw(OSError("interpreter failed")),
    )

    captured = []
    monkeypatch.setattr(
        dependency_bootstrap,
        "_run_pip_wheel_in_process",
        lambda pip_wheel, argv: captured.append((pip_wheel, argv)) or 0,
    )

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert captured == [(PIP_WHEEL, _pip_args(tmp_path))]


def test_install_numpy_does_not_retry_in_process_after_pip_install_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor_path: PIP_WHEEL)
    monkeypatch.setattr(
        dependency_bootstrap.subprocess,
        "run",
        lambda args, **kwargs: _completed(args, returncode=1, stderr="ERROR: no matching wheel"),
    )

    in_process_called = []
    monkeypatch.setattr(
        dependency_bootstrap,
        "_install_in_process",
        lambda *args, **kwargs: in_process_called.append(args) or dependency_bootstrap.InstallResult(True, "unexpected"),
    )

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "no matching wheel" in result.message
    assert in_process_called == []


def _stub_urlopen(monkeypatch, payload):
    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return payload

    monkeypatch.setattr(dependency_bootstrap.request, "urlopen", lambda *a, **k: _Response())


def _metadata(filename, sha256):
    return {
        "urls": [
            {
                "packagetype": "bdist_wheel",
                "filename": filename,
                "url": f"https://pypi.org/x/{filename}",
                "digests": {"sha256": sha256},
            }
        ],
    }


def test_get_or_download_pip_wheel_reuses_cached_wheel(tmp_path):
    vendor_path = tmp_path / "site-packages"
    wheel_dir = tmp_path / dependency_bootstrap.PIP_WHEEL_DIRECTORY_NAME
    wheel_dir.mkdir()
    wheel = wheel_dir / "pip-25.1.1-py3-none-any.whl"
    wheel.write_bytes(b"wheel")

    assert dependency_bootstrap._get_or_download_pip_wheel(str(vendor_path)) == wheel


def test_get_or_download_pip_wheel_reuses_most_recent_cached_wheel(tmp_path):
    vendor_path = tmp_path / "site-packages"
    wheel_dir = tmp_path / dependency_bootstrap.PIP_WHEEL_DIRECTORY_NAME
    wheel_dir.mkdir()
    older_wheel = wheel_dir / "pip-24.0-py3-none-any.whl"
    newer_wheel = wheel_dir / "pip-25.1.1-py3-none-any.whl"
    older_wheel.write_bytes(b"older")
    newer_wheel.write_bytes(b"newer")
    os.utime(older_wheel, (100, 100))
    os.utime(newer_wheel, (200, 200))

    assert dependency_bootstrap._get_or_download_pip_wheel(str(vendor_path)) == newer_wheel


def test_get_or_download_pip_wheel_downloads_and_verifies(tmp_path, monkeypatch):
    import hashlib

    payload = b"verified wheel bytes"
    metadata = _metadata("pip-26.0-py3-none-any.whl", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(dependency_bootstrap, "_fetch_pip_metadata", lambda: metadata)
    _stub_urlopen(monkeypatch, payload)

    result = dependency_bootstrap._get_or_download_pip_wheel(str(tmp_path / "site-packages"))

    assert result.name == "pip-26.0-py3-none-any.whl"
    assert result.read_bytes() == payload


def test_get_or_download_pip_wheel_rejects_bad_digest(tmp_path, monkeypatch):
    metadata = _metadata("pip-26.0-py3-none-any.whl", "0" * 64)
    monkeypatch.setattr(dependency_bootstrap, "_fetch_pip_metadata", lambda: metadata)
    _stub_urlopen(monkeypatch, b"tampered")

    with pytest.raises(dependency_bootstrap.PipBootstrapError, match="sha256"):
        dependency_bootstrap._get_or_download_pip_wheel(str(tmp_path / "site-packages"))


def test_run_pip_wheel_in_process_front_loads_wheel_and_restores_global_state(monkeypatch):
    original_argv = dependency_bootstrap.sys.argv
    original_exit = dependency_bootstrap.sys.exit
    original_path = dependency_bootstrap.sys.path[:]
    saved_pip_modules = {
        module_name: module
        for module_name, module in dependency_bootstrap.sys.modules.items()
        if module_name == "pip" or module_name.startswith("pip.")
    }
    # Stand in for a host pip already imported into the process; it must be
    # cleared while the wheel runs and then restored afterwards.
    host_pip = object()
    host_pip_vendor = object()
    dependency_bootstrap.sys.modules["pip"] = host_pip
    dependency_bootstrap.sys.modules["pip._vendor"] = host_pip_vendor
    captured = []

    def fake_run_module(module, run_name):
        captured.append((module, run_name, dependency_bootstrap.sys.path[0], dependency_bootstrap.sys.argv[:]))
        assert "pip" not in dependency_bootstrap.sys.modules
        assert "pip._vendor" not in dependency_bootstrap.sys.modules
        raise SystemExit(0)

    monkeypatch.setattr(dependency_bootstrap.runpy, "run_module", fake_run_module)

    try:
        result = dependency_bootstrap._run_pip_wheel_in_process(PIP_WHEEL, ["install", "numpy"])

        assert result == 0
        assert captured == [("pip", "__main__", str(PIP_WHEEL), ["pip", "install", "numpy"])]
        # The pre-existing host pip modules are put back exactly as they were.
        assert dependency_bootstrap.sys.modules["pip"] is host_pip
        assert dependency_bootstrap.sys.modules["pip._vendor"] is host_pip_vendor
        assert dependency_bootstrap.sys.argv is original_argv
        assert dependency_bootstrap.sys.exit is original_exit
        assert dependency_bootstrap.sys.path == original_path
    finally:
        for module_name in list(dependency_bootstrap.sys.modules):
            if module_name == "pip" or module_name.startswith("pip."):
                dependency_bootstrap.sys.modules.pop(module_name, None)
        dependency_bootstrap.sys.modules.update(saved_pip_modules)


def test_find_krita_python_uses_sys_executable_when_already_python(monkeypatch):
    monkeypatch.setattr(dependency_bootstrap.sys, "executable", "/usr/bin/python3.10")

    assert dependency_bootstrap.find_krita_python() == "/usr/bin/python3.10"


def test_find_krita_python_locates_sibling_python_exe(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    krita = bin_dir / "krita.exe"
    krita.write_text("")
    python = bin_dir / "python.exe"
    python.write_text("")

    monkeypatch.setattr(dependency_bootstrap.sys, "executable", str(krita))

    assert dependency_bootstrap.find_krita_python() == str(python)


def test_find_krita_python_locates_macos_krita_python(tmp_path, monkeypatch):
    macos_dir = tmp_path / "Krita.app" / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True)
    krita = macos_dir / "krita"
    krita.write_text("")
    krita_python = macos_dir / "krita_python"
    krita_python.write_text("")

    monkeypatch.setattr(dependency_bootstrap.sys, "executable", str(krita))

    assert dependency_bootstrap.find_krita_python() == str(krita_python)


def test_find_krita_python_returns_none_when_no_candidate(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    krita = bin_dir / "krita.exe"
    krita.write_text("")

    monkeypatch.setattr(dependency_bootstrap.sys, "executable", str(krita))

    assert dependency_bootstrap.find_krita_python() is None
