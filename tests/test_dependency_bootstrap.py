import subprocess
from pathlib import Path

import pytest

from oklab_colour_picker import dependency_bootstrap


KRITA_PYTHON = "/fake/krita/bin/python.exe"


def _completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


def test_install_numpy_invokes_krita_python_with_target_and_upgrade(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _completed(args, returncode=0, stdout="ok")

    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_subprocess", lambda _python: True)
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert calls == [
        (
            [
                KRITA_PYTHON,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--only-binary=:all:",
                "--target",
                str(tmp_path),
                dependency_bootstrap.NUMPY_REQUIREMENT,
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
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_in_process", lambda: True)

    captured = []
    monkeypatch.setattr(
        dependency_bootstrap,
        "_run_pip_in_process",
        lambda argv: captured.append(argv) or 0,
    )

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert captured == [
        [
            "pip",
            "install",
            "--upgrade",
            "--only-binary=:all:",
            "--target",
            str(tmp_path),
            dependency_bootstrap.NUMPY_REQUIREMENT,
        ]
    ]


def test_install_numpy_in_process_reports_pip_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: None)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_in_process", lambda: True)
    monkeypatch.setattr(dependency_bootstrap, "_run_pip_in_process", lambda _argv: 2)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "status 2" in result.message


def test_install_numpy_in_process_reports_missing_pip(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: None)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_in_process", lambda: False)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "ensurepip" in result.message


def test_install_numpy_surfaces_pip_stderr_on_failure(tmp_path, monkeypatch):
    def fake_run(args, **kwargs):
        return _completed(args, returncode=1, stderr="ERROR: no matching wheel")

    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_subprocess", lambda _python: True)
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "no matching wheel" in result.message


def test_install_numpy_reports_timeout(tmp_path, monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 0))

    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_subprocess", lambda _python: True)
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "timed out" in result.message.lower()


def test_install_numpy_falls_through_when_subprocess_pip_bootstrap_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_subprocess", lambda _python: False)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_in_process", lambda: True)

    captured = []
    monkeypatch.setattr(
        dependency_bootstrap,
        "_run_pip_in_process",
        lambda argv: captured.append(argv) or 0,
    )

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert captured and captured[0][0] == "pip"


def test_install_numpy_does_not_retry_in_process_after_pip_install_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_subprocess", lambda _python: True)
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


def test_ensure_pip_available_subprocess_short_circuits_when_pip_imports(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return _completed(args, returncode=0)

    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    assert dependency_bootstrap._ensure_pip_available_subprocess(KRITA_PYTHON) is True
    assert len(calls) == 1
    assert calls[0][1:] == ["-c", "import pip"]


def test_ensure_pip_available_subprocess_runs_ensurepip_when_pip_missing(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if "ensurepip" in args:
            return _completed(args, returncode=0)
        if calls.count(args) == 1:
            return _completed(args, returncode=1)
        return _completed(args, returncode=0)

    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    assert dependency_bootstrap._ensure_pip_available_subprocess(KRITA_PYTHON) is True
    assert any("ensurepip" in args for args in calls)


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
