import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from oklab_colour_picker.infrastructure import dependency_bootstrap


KRITA_PYTHON = "/fake/krita/bin/python.exe"
HOST_PYTHON = "/usr/bin/python3"
PIP_WHEEL = Path("/fake/pip-25.1.1-py3-none-any.whl")

# The autouse fixture stubs _find_host_python; keep the real one for its own test.
_REAL_FIND_HOST_PYTHON = dependency_bootstrap._find_host_python


@pytest.fixture(autouse=True)
def _no_host_python(monkeypatch):
    # Default every test to "no separate host Python" so the in-Krita paths are
    # exercised; host-pip tests opt back in by overriding _find_host_python.
    monkeypatch.setattr(dependency_bootstrap, "_find_host_python", lambda: None)


def _completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


def _pip_args(tmp_path, *, wheelhouse=None):
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
        args.extend(("--no-index", "--find-links", str(wheelhouse)))
    args.extend(
        [
            "--target",
            str(tmp_path),
            dependency_bootstrap.NUMPY_REQUIREMENT,
        ]
    )
    return args


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


def test_install_via_subprocess_runs_explicit_pip_wheel_script(tmp_path, monkeypatch):
    vendor_path = tmp_path / "site-packages"
    fake_wheel = tmp_path / "pip-25.1.1-py3-none-any.whl"
    fake_pip = fake_wheel / "pip"
    fake_pip.mkdir(parents=True)
    (fake_pip / "__init__.py").write_text("", encoding="utf-8")
    fake_pip_main = fake_pip / "__main__.py"
    fake_pip_main.write_text(
        """
import json
import pathlib
import sys

target_path = pathlib.Path(sys.argv[sys.argv.index("--target") + 1])
target_path.mkdir(parents=True, exist_ok=True)
(target_path.parent / "pip-subprocess-observed.json").write_text(
    json.dumps({"argv": sys.argv, "path0": sys.path[0]}),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor_path: fake_wheel)

    result = dependency_bootstrap._install_via_subprocess(
        sys.executable,
        str(vendor_path),
        dependency_bootstrap.NUMPY_REQUIREMENT,
    )

    observed = json.loads((tmp_path / "pip-subprocess-observed.json").read_text(encoding="utf-8"))
    assert result == dependency_bootstrap.InstallResult(
        True,
        "NumPy installed. Restart Krita to load the colour selector.",
    )
    assert observed == {
        "argv": ["pip", *_pip_args(vendor_path)],
        "path0": str(fake_wheel),
    }


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


def test_install_numpy_falls_back_to_in_process_when_sys_executable_version_mismatches(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor_path: PIP_WHEEL)

    def fake_run(args, **kwargs):
        assert args[:3] == ["/usr/bin/python3", "-I", "-c"]
        return _completed(args, returncode=0, stdout="3.14\n")

    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)
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


def _pinned_wheel(filename, payload, *, url=None):
    return dependency_bootstrap._PipWheel(
        url=url or f"https://pypi.org/x/{filename}",
        filename=filename,
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_get_or_download_pip_wheel_reuses_verified_cached_wheel(tmp_path, monkeypatch):
    vendor_path = tmp_path / "site-packages"
    wheel_dir = tmp_path / dependency_bootstrap.PIP_WHEEL_DIRECTORY_NAME
    wheel_dir.mkdir()
    wheel = wheel_dir / "pip-25.1.1-py3-none-any.whl"
    payload = b"verified wheel"
    wheel.write_bytes(payload)
    monkeypatch.setattr(dependency_bootstrap, "_PINNED_PIP_WHEEL", _pinned_wheel(wheel.name, payload))
    monkeypatch.setattr(
        dependency_bootstrap,
        "_download",
        lambda _url: (_ for _ in ()).throw(AssertionError("unexpected download")),
    )

    assert dependency_bootstrap._get_or_download_pip_wheel(str(vendor_path)) == wheel


def test_get_or_download_pip_wheel_replaces_bad_cached_wheel(tmp_path, monkeypatch):
    vendor_path = tmp_path / "site-packages"
    wheel_dir = tmp_path / dependency_bootstrap.PIP_WHEEL_DIRECTORY_NAME
    wheel_dir.mkdir()
    wheel = wheel_dir / "pip-25.1.1-py3-none-any.whl"
    wheel.write_bytes(b"planted wheel")
    payload = b"verified wheel"
    monkeypatch.setattr(dependency_bootstrap, "_PINNED_PIP_WHEEL", _pinned_wheel(wheel.name, payload))
    monkeypatch.setattr(dependency_bootstrap, "_download", lambda _url: payload)

    assert dependency_bootstrap._get_or_download_pip_wheel(str(vendor_path)) == wheel
    assert wheel.read_bytes() == payload


def test_get_or_download_pip_wheel_ignores_other_cached_wheels(tmp_path, monkeypatch):
    vendor_path = tmp_path / "site-packages"
    wheel_dir = tmp_path / dependency_bootstrap.PIP_WHEEL_DIRECTORY_NAME
    wheel_dir.mkdir()
    planted_wheel = wheel_dir / "pip-99.0-py3-none-any.whl"
    planted_wheel.write_bytes(b"planted wheel")
    payload = b"verified wheel"
    monkeypatch.setattr(
        dependency_bootstrap,
        "_PINNED_PIP_WHEEL",
        _pinned_wheel("pip-25.1.1-py3-none-any.whl", payload),
    )
    monkeypatch.setattr(dependency_bootstrap, "_download", lambda _url: payload)

    result = dependency_bootstrap._get_or_download_pip_wheel(str(vendor_path))

    assert result == wheel_dir / "pip-25.1.1-py3-none-any.whl"
    assert result.read_bytes() == payload


def test_get_or_download_pip_wheel_downloads_and_verifies(tmp_path, monkeypatch):
    payload = b"verified wheel bytes"
    monkeypatch.setattr(
        dependency_bootstrap,
        "_PINNED_PIP_WHEEL",
        _pinned_wheel("pip-26.0-py3-none-any.whl", payload),
    )
    _stub_urlopen(monkeypatch, payload)

    result = dependency_bootstrap._get_or_download_pip_wheel(str(tmp_path / "site-packages"))

    assert result.name == "pip-26.0-py3-none-any.whl"
    assert result.read_bytes() == payload


def test_get_or_download_pip_wheel_rejects_bad_digest(tmp_path, monkeypatch):
    monkeypatch.setattr(
        dependency_bootstrap,
        "_PINNED_PIP_WHEEL",
        dependency_bootstrap._PipWheel(
            url="https://pypi.org/x/pip-26.0-py3-none-any.whl",
            filename="pip-26.0-py3-none-any.whl",
            sha256="0" * 64,
        ),
    )
    _stub_urlopen(monkeypatch, b"tampered")

    with pytest.raises(dependency_bootstrap.PipBootstrapError, match="sha256"):
        dependency_bootstrap._get_or_download_pip_wheel(str(tmp_path / "site-packages"))


def test_get_or_download_pip_wheel_falls_back_to_host_python_download(tmp_path, monkeypatch):
    payload = b"verified host wheel"
    wheel = _pinned_wheel("pip-25.1.1-py3-none-any.whl", payload)
    calls = []
    monkeypatch.setattr(
        dependency_bootstrap,
        "_PINNED_PIP_WHEEL",
        wheel,
    )
    monkeypatch.setattr(dependency_bootstrap, "_download", lambda _url: (_ for _ in ()).throw(
        dependency_bootstrap.PipBootstrapError("Krita SSL failed")
    ))
    monkeypatch.setattr(dependency_bootstrap, "_find_host_python", lambda: HOST_PYTHON)

    def fake_run(args, **kwargs):
        calls.append(args)
        wheel_dir = Path(args[args.index("--dest") + 1])
        wheel_dir.mkdir(parents=True, exist_ok=True)
        (wheel_dir / wheel.filename).write_bytes(payload)
        return _completed(args, returncode=0)

    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap._get_or_download_pip_wheel(str(tmp_path / "site-packages"))

    assert result == tmp_path / dependency_bootstrap.PIP_WHEEL_DIRECTORY_NAME / wheel.filename
    assert calls == [
        [
            HOST_PYTHON,
            "-m", "pip", "download",
            "--no-input",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--no-deps",
            "--only-binary=:all:",
            "--dest", str(tmp_path / dependency_bootstrap.PIP_WHEEL_DIRECTORY_NAME),
            "pip==25.1.1",
        ]
    ]


def test_get_or_download_pip_wheel_rejects_bad_host_download(tmp_path, monkeypatch):
    wheel = dependency_bootstrap._PipWheel(
        url="https://pypi.org/x/pip-25.1.1-py3-none-any.whl",
        filename="pip-25.1.1-py3-none-any.whl",
        sha256="0" * 64,
    )
    monkeypatch.setattr(dependency_bootstrap, "_PINNED_PIP_WHEEL", wheel)
    monkeypatch.setattr(dependency_bootstrap, "_download", lambda _url: (_ for _ in ()).throw(
        dependency_bootstrap.PipBootstrapError("Krita SSL failed")
    ))
    monkeypatch.setattr(dependency_bootstrap, "_find_host_python", lambda: HOST_PYTHON)

    def fake_run(args, **kwargs):
        wheel_dir = Path(args[args.index("--dest") + 1])
        wheel_dir.mkdir(parents=True, exist_ok=True)
        (wheel_dir / wheel.filename).write_bytes(b"tampered")
        return _completed(args, returncode=0)

    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    with pytest.raises(dependency_bootstrap.PipBootstrapError, match="Host Python fallback"):
        dependency_bootstrap._get_or_download_pip_wheel(str(tmp_path / "site-packages"))


def test_clear_numpy_target_removes_stale_numpy_artifacts(tmp_path):
    vendor_path = tmp_path / "site-packages"
    vendor_path.mkdir()
    stale_extension = vendor_path / "numpy" / "_core" / "_multiarray_umath.cpython-314-x86_64-linux-gnu.so"
    stale_extension.parent.mkdir(parents=True)
    stale_extension.write_text("")
    stale_lib = vendor_path / "numpy.libs"
    stale_lib.mkdir()
    stale_metadata = vendor_path / "numpy-2.4.6.dist-info"
    stale_metadata.mkdir()
    unrelated = vendor_path / "scipy"
    unrelated.mkdir()

    dependency_bootstrap._clear_numpy_target(str(vendor_path))

    assert not (vendor_path / "numpy").exists()
    assert not stale_lib.exists()
    assert not stale_metadata.exists()
    assert unrelated.exists()


def test_clear_numpy_target_unlinks_symlink_without_following_it(tmp_path):
    vendor_path = tmp_path / "site-packages"
    vendor_path.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "kept.txt").write_text("")
    (vendor_path / "numpy").symlink_to(outside, target_is_directory=True)

    dependency_bootstrap._clear_numpy_target(str(vendor_path))

    assert not (vendor_path / "numpy").exists()
    assert (outside / "kept.txt").exists()


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


def test_install_numpy_uses_krita_pip_even_when_host_python_exists(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dependency_bootstrap, "_find_host_python", lambda: HOST_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor: PIP_WHEEL)
    monkeypatch.setattr(
        dependency_bootstrap.subprocess,
        "run",
        lambda args, **kwargs: calls.append(args) or _completed(args, returncode=0),
    )

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert calls == [
        [
            KRITA_PYTHON,
            "-I",
            "-c",
            dependency_bootstrap._PIP_WHEEL_INSTALL_SCRIPT,
            str(PIP_WHEEL),
            *_pip_args(tmp_path),
        ]
    ]


def test_install_numpy_retries_krita_pip_offline_from_host_wheelhouse(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dependency_bootstrap, "_find_host_python", lambda: HOST_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor: PIP_WHEEL)
    monkeypatch.setattr(dependency_bootstrap, "_target_platforms", lambda: ["manylinux_2_28_x86_64", "linux_x86_64"])

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[0] == HOST_PYTHON:
            wheelhouse = Path(args[args.index("--dest") + 1])
            wheelhouse.mkdir(parents=True, exist_ok=True)
            (wheelhouse / "numpy.whl").write_text("")
            return _completed(args, returncode=0)
        if "--no-index" in args:
            return _completed(args, returncode=0)
        return _completed(args, returncode=1, stderr="CERTIFICATE_VERIFY_FAILED")

    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    host_call = calls[1]
    wheelhouse = host_call[host_call.index("--dest") + 1]
    assert calls == [
        [
            KRITA_PYTHON,
            "-I",
            "-c",
            dependency_bootstrap._PIP_WHEEL_INSTALL_SCRIPT,
            str(PIP_WHEEL),
            *_pip_args(tmp_path),
        ],
        [
            HOST_PYTHON,
            "-m", "pip", "download",
            "--no-input",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--dest", wheelhouse,
            "--only-binary=:all:",
            "--python-version", f"{sys.version_info[0]}.{sys.version_info[1]}",
            "--abi", f"cp{sys.version_info[0]}{sys.version_info[1]}",
            "--implementation", "cp",
            "--platform", "manylinux_2_28_x86_64",
            "--platform", "linux_x86_64",
            dependency_bootstrap.NUMPY_REQUIREMENT,
        ],
        [
            KRITA_PYTHON,
            "-I",
            "-c",
            dependency_bootstrap._PIP_WHEEL_INSTALL_SCRIPT,
            str(PIP_WHEEL),
            *_pip_args(tmp_path, wheelhouse=wheelhouse),
        ],
    ]


def test_install_numpy_ignores_empty_host_wheelhouse(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dependency_bootstrap, "_find_host_python", lambda: HOST_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_get_or_download_pip_wheel", lambda _vendor: PIP_WHEEL)

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[0] == HOST_PYTHON:
            return _completed(args, returncode=0)
        return _completed(args, returncode=1, stderr="CERTIFICATE_VERIFY_FAILED")

    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert result.message == "CERTIFICATE_VERIFY_FAILED"
    assert len(calls) == 2
    assert calls[0][0] == KRITA_PYTHON
    assert calls[1][0] == HOST_PYTHON


def test_host_pip_download_args_constrain_platform_when_detected(monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "_target_platforms", lambda: ["manylinux_2_28_x86_64", "linux_x86_64"])

    args = dependency_bootstrap._host_pip_download_args("/tmp/wheelhouse", "numpy")

    assert args[:7] == [
        "-m",
        "pip",
        "download",
        "--no-input",
        "--disable-pip-version-check",
        "--no-cache-dir",
        "--dest",
    ]
    assert "--target" not in args
    assert args[-5:] == ["--platform", "manylinux_2_28_x86_64", "--platform", "linux_x86_64", "numpy"]


def test_target_platforms_include_manylinux_compatibility_tags(monkeypatch):
    monkeypatch.setattr(dependency_bootstrap.sysconfig, "get_platform", lambda: "linux-x86_64")
    monkeypatch.setattr(dependency_bootstrap.platform, "libc_ver", lambda: ("glibc", "2.28"))

    assert dependency_bootstrap._target_platforms() == [
        "manylinux_2_28_x86_64",
        "manylinux_2_27_x86_64",
        "manylinux_2_26_x86_64",
        "manylinux_2_25_x86_64",
        "manylinux_2_24_x86_64",
        "manylinux_2_23_x86_64",
        "manylinux_2_22_x86_64",
        "manylinux_2_21_x86_64",
        "manylinux_2_20_x86_64",
        "manylinux_2_19_x86_64",
        "manylinux_2_18_x86_64",
        "manylinux_2_17_x86_64",
        "manylinux2014_x86_64",
        "linux_x86_64",
    ]


def test_target_platforms_accept_glibc_patch_version(monkeypatch):
    monkeypatch.setattr(dependency_bootstrap.sysconfig, "get_platform", lambda: "linux-x86_64")
    monkeypatch.setattr(dependency_bootstrap.platform, "libc_ver", lambda: ("glibc", "2.28.1"))

    assert dependency_bootstrap._target_platforms()[0] == "manylinux_2_28_x86_64"


def test_target_platforms_leave_non_linux_as_pip_default(monkeypatch):
    monkeypatch.setattr(dependency_bootstrap.sysconfig, "get_platform", lambda: "macosx-14.0-arm64")

    assert dependency_bootstrap._target_platforms() == []


def test_find_host_python_skips_krita_bundled_interpreter(tmp_path, monkeypatch):
    bundled = tmp_path / "krita" / "usr"
    bundled_python = bundled / "bin" / "python3"
    host_python = tmp_path / "host" / "python3"
    for path in (bundled_python, host_python):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    monkeypatch.setattr(dependency_bootstrap, "_find_host_python", _REAL_FIND_HOST_PYTHON)
    monkeypatch.setattr(dependency_bootstrap.sys, "prefix", str(bundled))
    monkeypatch.setattr(dependency_bootstrap.sys, "executable", str(bundled_python))
    which = {"python3": str(bundled_python), "python": str(host_python)}
    monkeypatch.setattr(dependency_bootstrap.shutil, "which", lambda name: which.get(name))

    assert dependency_bootstrap._find_host_python() == str(host_python)


def test_find_host_python_allows_sys_executable_when_it_is_not_krita_runtime(monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "_find_host_python", _REAL_FIND_HOST_PYTHON)
    monkeypatch.setattr(dependency_bootstrap.sys, "prefix", "/tmp/krita/usr")
    monkeypatch.setattr(dependency_bootstrap.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(dependency_bootstrap.shutil, "which", lambda name: "/usr/bin/python3" if name == "python3" else None)
    monkeypatch.setattr(dependency_bootstrap, "_python_matches_krita_runtime", lambda _python: False)

    assert dependency_bootstrap._find_host_python() == "/usr/bin/python3"


def test_pypi_ssl_context_uses_first_available_ca_bundle(tmp_path, monkeypatch):
    bundle = tmp_path / "ca-certificates.crt"
    bundle.write_text("")
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(dependency_bootstrap, "_CA_BUNDLE_CANDIDATES", (str(bundle),))

    captured = {}
    monkeypatch.setattr(
        dependency_bootstrap.ssl,
        "create_default_context",
        lambda cafile=None: captured.setdefault("cafile", cafile),
    )

    dependency_bootstrap._pypi_ssl_context()

    assert captured["cafile"] == str(bundle)


def test_pypi_ssl_context_returns_none_without_any_ca_store(monkeypatch):
    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.setattr(dependency_bootstrap, "_CA_BUNDLE_CANDIDATES", ("/nonexistent/ca.crt",))

    assert dependency_bootstrap._pypi_ssl_context() is None


def test_find_krita_python_uses_sys_executable_when_already_python(monkeypatch):
    monkeypatch.setattr(dependency_bootstrap.sys, "executable", "/usr/bin/python3.10")
    monkeypatch.setattr(dependency_bootstrap, "_python_matches_krita_runtime", lambda _python: True)

    assert dependency_bootstrap.find_krita_python() == "/usr/bin/python3.10"


def test_find_krita_python_locates_sibling_python_exe(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    krita = bin_dir / "krita.exe"
    krita.write_text("")
    python = bin_dir / "python.exe"
    python.write_text("")

    monkeypatch.setattr(dependency_bootstrap.sys, "executable", str(krita))
    monkeypatch.setattr(dependency_bootstrap, "_python_matches_krita_runtime", lambda _python: True)

    assert dependency_bootstrap.find_krita_python() == str(python)


def test_find_krita_python_locates_macos_krita_python(tmp_path, monkeypatch):
    macos_dir = tmp_path / "Krita.app" / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True)
    krita = macos_dir / "krita"
    krita.write_text("")
    krita_python = macos_dir / "krita_python"
    krita_python.write_text("")

    monkeypatch.setattr(dependency_bootstrap.sys, "executable", str(krita))
    monkeypatch.setattr(dependency_bootstrap, "_python_matches_krita_runtime", lambda _python: True)

    assert dependency_bootstrap.find_krita_python() == str(krita_python)


def test_find_krita_python_returns_none_when_no_candidate(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    krita = bin_dir / "krita.exe"
    krita.write_text("")

    monkeypatch.setattr(dependency_bootstrap.sys, "executable", str(krita))

    assert dependency_bootstrap.find_krita_python() is None


def test_find_krita_python_rejects_python_executable_with_wrong_runtime(monkeypatch):
    monkeypatch.setattr(dependency_bootstrap.sys, "executable", "/usr/bin/python3")
    monkeypatch.setattr(dependency_bootstrap, "_python_matches_krita_runtime", lambda _python: False)

    assert dependency_bootstrap.find_krita_python() is None
