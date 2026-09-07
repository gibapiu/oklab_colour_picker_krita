#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

REQUIRED_COMMIT_IDENTITY = "gibapiu <279996898+gibapiu@users.noreply.github.com>"

WEB_MERGE_COMMITTER = "GitHub <noreply@github.com>"

RETIRED_MAINTAINER_IDENTITIES = frozenset({
    "gibapiu <gibapiu>",
    "gibapiu <workshophatemachine@gmail.com>",
})

ALLOWED_IDENTITIES = (
    RETIRED_MAINTAINER_IDENTITIES
    | {REQUIRED_COMMIT_IDENTITY, WEB_MERGE_COMMITTER}
)

ALLOWED_TRAILER_KEYS: frozenset[str] = frozenset()


def git(*args: str) -> str:
    """Return the stdout of a git invocation, raising on a non-zero exit."""
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True,
        stdout=subprocess.PIPE, text=True,
    ).stdout


def commits(rev_range: str) -> list[str]:
    """Return the commit hashes in rev_range, oldest last."""
    return git("rev-list", rev_range).split()


def identities(sha: str) -> tuple[str, str]:
    """Return the (author, committer) identity strings of a commit."""
    author = git("show", "-s", "--format=%an <%ae>", sha).strip()
    committer = git("show", "-s", "--format=%cn <%ce>", sha).strip()
    return author, committer


def trailers(sha: str) -> list[tuple[str, str]]:
    """Return the (key, value) trailers of a commit message."""
    message = git("show", "-s", "--format=%B", sha)
    parsed = subprocess.run(
        ["git", "interpret-trailers", "--parse"], cwd=ROOT, check=True,
        input=message, stdout=subprocess.PIPE, text=True,
    ).stdout
    found = []
    for line in parsed.splitlines():
        key, _, value = line.partition(":")
        if value:
            found.append((key.strip(), value.strip()))
    return found


def configured_identity(role: str) -> str:
    """Return the identity git would stamp on a new commit for role author/committer."""
    ident = git("var", f"GIT_{role.upper()}_IDENT").strip()
    return ident.rsplit(">", 1)[0] + ">"


def check_identity() -> list[str]:
    """Return the deviations of the configured commit identity from the required one."""
    problems = []
    for role in ("author", "committer"):
        if configured_identity(role) != REQUIRED_COMMIT_IDENTITY:
            problems.append(f"configured {role} identity is not the maintainer account")
    return problems


def remediation() -> str:
    """Return the operator instructions for restoring the maintainer identity."""
    name, _, email = REQUIRED_COMMIT_IDENTITY.partition(" <")
    return "\n".join((
        "Set this repository to the maintainer account before committing:",
        f"  git config --local user.name {name}",
        f"  git config --local user.email {email.rstrip('>')}",
        f"  gh auth switch -u {name}",
        "Also clear any GIT_AUTHOR_* / GIT_COMMITTER_* variables in your environment.",
    ))


def check(sha: str, verbose: bool) -> list[str]:
    """Return the policy violations of a single commit, empty when it conforms."""
    problems = []
    author, committer = identities(sha)
    for role, identity in (("author", author), ("committer", committer)):
        if identity not in ALLOWED_IDENTITIES:
            detail = f": {identity}" if verbose else ""
            problems.append(f"{sha[:12]}: unrecognised {role} identity{detail}")
    for key, value in trailers(sha):
        if key.lower() not in ALLOWED_TRAILER_KEYS:
            detail = f": {key}: {value}" if verbose else ""
            problems.append(f"{sha[:12]}: trailer key not permitted{detail}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rev_range", nargs="?", default="HEAD")
    parser.add_argument("--identity", action="store_true",
                        help="Check the configured commit identity instead of a range.")
    parser.add_argument("--verbose", action="store_true",
                        help="Include the offending values in the report.")
    args = parser.parse_args()

    if args.identity:
        problems = check_identity()
    else:
        problems = []
        for sha in commits(args.rev_range):
            problems.extend(check(sha, args.verbose))

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{len(problems)} contributor policy violation(s).", file=sys.stderr)
        print(remediation(), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
