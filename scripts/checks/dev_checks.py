#!/usr/bin/env python3
"""Deterministic development-loop checks for the rewrite.

The checks intentionally avoid third-party dependencies so they can run from
Git hooks before the project has a full Python package/test setup.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from scripts.checks.architecture_policy import (
        KRITA_IMPORT_ALLOWED,
        LOWER_LAYER_FORBIDDEN_MODULE_PREFIXES,
        QT_BINDING_IMPORT_ALLOWED,
        QT_BINDING_MODULE_PREFIXES,
        QT_OR_KRITA_MODULE_PREFIXES,
        SET_FOREGROUND_ALLOWED,
        import_from_references,
        is_declared_package_module,
        is_lower_layer_file,
        is_pure_layer_file,
        starts_with_any,
    )
except ModuleNotFoundError:
    from architecture_policy import (  # type: ignore[no-redef]
        KRITA_IMPORT_ALLOWED,
        LOWER_LAYER_FORBIDDEN_MODULE_PREFIXES,
        QT_BINDING_IMPORT_ALLOWED,
        QT_BINDING_MODULE_PREFIXES,
        QT_OR_KRITA_MODULE_PREFIXES,
        SET_FOREGROUND_ALLOWED,
        import_from_references,
        is_declared_package_module,
        is_lower_layer_file,
        is_pure_layer_file,
        starts_with_any,
    )


ROOT = Path(__file__).resolve().parents[2]
LEGACY_PREFIX = "legacy-plugin/"

# Whitespace fixes/checks apply only to regular files known to be text.
# Symlinks (120000) and gitlinks (160000) carry no rewriteable text and excluded from check.
REGULAR_FILE_MODES = frozenset({"100644", "100755"})
TEXT_SUFFIXES = frozenset({
    ".py", ".pyi", ".md", ".rst", ".txt", ".ini", ".cfg", ".toml",
    ".yml", ".yaml", ".json", ".html", ".css", ".js", ".sh", ".desktop",
})
TEXT_FILENAMES = frozenset({
    "LICENSE", ".gitignore", ".gitattributes", ".editorconfig",
    "pre-commit", "pre-push",
})


@dataclass(frozen=True)
class SourceFile:
    """A file snapshot from either the working tree or the staged index."""

    path: Path
    data: bytes
    mode: str

    @property
    def suffix(self) -> str:
        return self.path.suffix

    @property
    def posix(self) -> str:
        return self.path.as_posix()

    @property
    def is_legacy(self) -> bool:
        return self.posix.startswith(LEGACY_PREFIX)

    @property
    def is_test(self) -> bool:
        return bool(self.path.parts) and self.path.parts[0] == "tests"

    @property
    def is_binary(self) -> bool:
        return b"\0" in self.data

    @property
    def is_regular(self) -> bool:
        return self.mode in REGULAR_FILE_MODES

    @property
    def is_text(self) -> bool:
        """True only for regular files whose name and contents are safe to rewrite as text."""
        if not self.is_regular or self.is_binary:
            return False
        if self.suffix not in TEXT_SUFFIXES and self.path.name not in TEXT_FILENAMES:
            return False
        try:
            self.data.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def git_blob(path: Path) -> bytes:
    return subprocess.check_output(["git", "show", f":{path.as_posix()}"], cwd=ROOT)


def tracked_paths() -> list[Path]:
    return [Path(line) for line in run_git(["ls-files"]).splitlines() if line]


def staged_paths() -> list[Path]:
    lines = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"]).splitlines()
    return [Path(line) for line in lines if line]


def staged_modes() -> dict[str, str]:
    modes = {}
    for line in run_git(["ls-files", "-s"]).splitlines():
        meta, _, path = line.partition("\t")
        if path:
            modes[path] = meta.split()[0]
    return modes


def tracked_mode(full_path: Path) -> str:
    st = full_path.lstat()
    if stat.S_ISLNK(st.st_mode):
        return "120000"
    return "100755" if st.st_mode & 0o111 else "100644"


def source_files(scope: str) -> list[SourceFile]:
    sources = []
    if scope == "staged":
        modes = staged_modes()
        for path in staged_paths():
            sources.append(
                SourceFile(path=path, data=git_blob(path), mode=modes.get(path.as_posix(), "100644"))
            )
    else:
        for path in tracked_paths():
            full_path = ROOT / path
            if not full_path.is_file():
                continue
            sources.append(
                SourceFile(path=path, data=full_path.read_bytes(), mode=tracked_mode(full_path))
            )
    return sources


def python_sources(sources: list[SourceFile]) -> list[SourceFile]:
    return [source for source in sources if source.suffix == ".py" and not source.is_legacy]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)


def check_no_legacy(sources: list[SourceFile]) -> int:
    bad = [source.posix for source in sources if source.is_legacy]
    if not bad:
        return 0
    fail("legacy-plugin files must remain untracked:")
    for path in bad:
        print(f"  {path}", file=sys.stderr)
    return 1


def check_python_rules(sources: list[SourceFile]) -> int:
    errors = 0
    for source in python_sources(sources):
        rp = source.path
        if not is_declared_package_module(rp):
            fail(f"{rp}: Python modules must live in a declared package layer")
            errors += 1
        try:
            tree = ast.parse(source.data, filename=source.posix)
            compile(tree, source.posix, "exec")
        except SyntaxError as exc:
            fail(f"Cannot parse {rp}: {exc}")
            errors += 1
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                if any(module == "krita" or module.startswith("krita.") for module in modules) and rp not in KRITA_IMPORT_ALLOWED:
                    fail(f"{rp}: Krita imports are only allowed in plugin/controller adapter files")
                    errors += 1
                if any(module.startswith("legacy_plugin") for module in modules):
                    fail(f"{rp}: imports from legacy plugin are forbidden")
                    errors += 1
                if is_pure_layer_file(rp) and any(
                    module.startswith(QT_OR_KRITA_MODULE_PREFIXES) for module in modules
                ):
                    fail(f"{rp}: pure domain/model/render modules must not import Qt or Krita")
                    errors += 1
                if (
                    any(starts_with_any(module, QT_BINDING_MODULE_PREFIXES) for module in modules)
                    and rp not in QT_BINDING_IMPORT_ALLOWED
                ):
                    fail(f"{rp}: Qt bindings may only be imported in the oklab_colour_picker.qt shim")
                    errors += 1
                if is_lower_layer_file(rp):
                    for module in modules:
                        if starts_with_any(module, LOWER_LAYER_FORBIDDEN_MODULE_PREFIXES):
                            fail(f"{rp}: lower layers must not import UI, plugin, or the Qt shim")
                            errors += 1

            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if (module == "krita" or module.startswith("krita.")) and rp not in KRITA_IMPORT_ALLOWED:
                    fail(f"{rp}: Krita imports are only allowed in plugin/controller adapter files")
                    errors += 1
                if module.startswith("legacy_plugin"):
                    fail(f"{rp}: imports from legacy plugin are forbidden")
                    errors += 1
                if is_pure_layer_file(rp) and module.startswith(QT_OR_KRITA_MODULE_PREFIXES):
                    fail(f"{rp}: pure domain/model/render modules must not import Qt or Krita")
                    errors += 1
                if (
                    starts_with_any(module, QT_BINDING_MODULE_PREFIXES)
                    and rp not in QT_BINDING_IMPORT_ALLOWED
                ):
                    fail(f"{rp}: Qt bindings may only be imported in the oklab_colour_picker.qt shim")
                    errors += 1
                if is_lower_layer_file(rp):
                    for imported_module in import_from_references(node, rp):
                        if starts_with_any(imported_module, LOWER_LAYER_FORBIDDEN_MODULE_PREFIXES):
                            fail(f"{rp}: lower layers must not import UI, plugin, or the Qt shim")
                            errors += 1

            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Name-based AST guardrail by design: it catches direct calls in
            # production code, but it is not a type-aware semantic analysis.
            if isinstance(func, ast.Attribute) and func.attr == "pixelColor":
                fail(f"{rp}: selection must not read colours from QImage.pixelColor")
                errors += 1
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "setForeGroundColor"
                and rp not in SET_FOREGROUND_ALLOWED
                and not source.is_test
            ):
                fail(f"{rp}: setForeGroundColor is only allowed behind the controller/Krita adapter boundary")
                errors += 1
    return errors


def check_formatting(sources: list[SourceFile]) -> int:
    errors = 0
    for source in sources:
        if not source.is_text:
            continue
        data = source.data
        rp = source.path
        if b"\r\n" in data:
            fail(f"{rp}: CRLF line endings are not allowed")
            errors += 1
        if data and not data.endswith(b"\n"):
            fail(f"{rp}: file must end with a newline")
            errors += 1
        for line_no, line in enumerate(data.splitlines(), start=1):
            if line.rstrip(b" \t") != line:
                fail(f"{rp}:{line_no}: trailing whitespace is not allowed")
                errors += 1
    return errors


def normalize_bytes(data: bytes) -> bytes:
    """Return data with CRLF normalized, trailing whitespace stripped, and a single final newline."""
    text = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    lines = [line.rstrip(b" \t") for line in text.split(b"\n")]
    fixed = b"\n".join(lines).rstrip(b"\n")
    return fixed + b"\n" if fixed else fixed


def restage_blob(path: Path, data: bytes, mode: str) -> None:
    proc = subprocess.run(
        ["git", "hash-object", "-w", "--path", path.as_posix(), "--stdin"],
        input=data,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        check=True,
    )
    sha = proc.stdout.decode().strip()
    subprocess.check_call(
        ["git", "update-index", "--cacheinfo", f"{mode},{sha},{path.as_posix()}"],
        cwd=ROOT,
    )


def autofix_formatting(sources: list[SourceFile], scope: str) -> list[str]:
    """Apply whitespace/newline fixes to regular text files; return the paths that changed.
    Only files reported as text are touched, so symlinks and binary blobs are left intact
    """
    fixed = []
    for source in sources:
        if not source.is_text:
            continue
        new = normalize_bytes(source.data)
        if new == source.data:
            continue
        full_path = ROOT / source.path
        if scope == "staged":
            restage_blob(source.path, new, source.mode)
            if full_path.is_file() and full_path.read_bytes() == source.data:
                full_path.write_bytes(new)
        else:
            full_path.write_bytes(new)
        fixed.append(source.posix)
    return fixed


def tests_command() -> list[str]:
    """Return the lean test command, preferring the tox matrix for full CI parity."""
    lean = ["-q", "--no-header", "--tb=short", "--disable-warnings"]
    if importlib.util.find_spec("tox") is not None:
        return [sys.executable, "-m", "tox", "-q", "-e", "pyqt5,pyqt6", "--", *lean]
    if shutil.which("tox") is not None:
        return ["tox", "-q", "-e", "pyqt5,pyqt6", "--", *lean]
    print("WARNING: tox not found; running single-binding pytest only (perf included). "
          "CI also checks the other Qt binding -- install requirements-dev.txt for full parity.",
          file=sys.stderr)
    return [sys.executable, "-m", "pytest", *lean]


def run_tests() -> int:
    tests_dir = ROOT / "tests"
    if not tests_dir.exists():
        print("No tests/ directory yet; skipping tests.")
        return 0
    return subprocess.call(tests_command(), cwd=ROOT)


def install_hooks() -> int:
    subprocess.check_call(["git", "config", "core.hooksPath", ".githooks"], cwd=ROOT)
    print("Configured Git to use .githooks for this repository.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("tracked", "staged"), default="tracked")
    parser.add_argument("--fix", action="store_true", help="Auto-fix whitespace/newline issues before checking.")
    parser.add_argument("--pytest", action="store_true", help="Run the lean test suite after static checks.")
    parser.add_argument("--install-hooks", action="store_true", help="Set core.hooksPath=.githooks.")
    args = parser.parse_args()

    if args.install_hooks:
        return install_hooks()

    os.chdir(ROOT)
    sources = source_files(args.scope)
    fixed = []
    if args.fix:
        fixed = autofix_formatting(sources, args.scope)
        if fixed:
            print(f"auto-fixed formatting in {len(fixed)} file(s):")
            for path in fixed:
                print(f"  {path}")
            sources = source_files(args.scope)
    errors = 0
    errors += check_no_legacy(sources)
    errors += check_formatting(sources)
    errors += check_python_rules(sources)
    if args.pytest:
        errors += run_tests()
    if fixed:
        # Re-staging the fixes changed the index, so abort and require an explicit review + re-commit.
        fail("auto-fixes were applied and staged; review them (git diff --cached) and re-run the commit.")
        return 1
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
