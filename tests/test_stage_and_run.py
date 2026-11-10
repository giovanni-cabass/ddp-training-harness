"""Tests for stage_and_run.sh's safety guard against staging into / cleaning
up a non-scratch path. These invoke the real script as a subprocess -- the
guard is checked before anything else happens, so this never touches disk
beyond the guard's own early exit.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "stage_and_run.sh"


@pytest.mark.parametrize(
    "bad_scratch_base",
    [
        os.path.expanduser("~"),
        os.path.join(os.path.expanduser("~"), "some_subdir"),
        "/",
    ],
)
def test_refuses_non_scratch_scratch_base(bad_scratch_base):
    env = dict(os.environ)
    env["SCRATCH_BASE"] = bad_scratch_base
    result = subprocess.run(
        [str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "refusing to use SCRATCH_BASE" in result.stderr


@pytest.mark.parametrize("relative_scratch_base", [".", "some_subdir", "../elsewhere"])
def test_refuses_relative_scratch_base(relative_scratch_base):
    env = dict(os.environ)
    env["SCRATCH_BASE"] = relative_scratch_base
    result = subprocess.run(
        [str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "must be an absolute path" in result.stderr


def test_refuses_dotdot_path_that_resolves_into_home():
    # Textually this doesn't match "$HOME" or "$HOME/*" at all, but it
    # resolves to exactly $HOME -- the guard must canonicalize before
    # comparing, not just pattern-match the literal string.
    home = os.path.expanduser("~")
    parent, base = os.path.dirname(home), os.path.basename(home)
    env = dict(os.environ)
    env["SCRATCH_BASE"] = os.path.join(parent, "some_other_dir", "..", base)
    result = subprocess.run(
        [str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "doesn't look like disposable scratch" in result.stderr


def test_fails_loudly_when_src_data_dir_does_not_exist(tmp_path):
    env = dict(os.environ)
    env["SCRATCH_BASE"] = str(tmp_path / "scratch")
    env["SRC_DATA_DIR"] = str(tmp_path / "does_not_exist")
    result = subprocess.run(
        [str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "does not exist or is not a directory" in result.stderr


def test_fails_loudly_when_space_check_cannot_determine_sizes(tmp_path):
    # The guard protects against du/df *succeeding* (exit 0) but producing
    # no usable line for awk to parse (e.g. an unexpected output format) --
    # not against du/df failing outright, which `set -e`/pipefail already
    # catches on its own, earlier, with a different error. To exercise this
    # guard specifically, fake binaries on PATH that exit 0 with no output.
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    fake_tool_body = '#!/bin/bash\nif [[ "$1" == "--version" ]]; then echo "(GNU coreutils) 9.0"; fi\nexit 0\n'
    for name in ("gdu", "gdf", "du", "df"):
        tool_path = fake_bin / name
        tool_path.write_text(fake_tool_body)
        tool_path.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env.get('PATH', '')}"
    env["SCRATCH_BASE"] = str(tmp_path / "scratch")
    env["SRC_DATA_DIR"] = str(tmp_path / "source")
    (tmp_path / "source").mkdir()

    result = subprocess.run(
        [str(SCRIPT)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "unable to determine source size or available scratch space" in result.stderr
