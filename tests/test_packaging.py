"""Tests for the installable packaging of TicketSense.

These tests read ``pyproject.toml`` and ``requirements.txt`` from
the project tree, so they run without any install. The two tests
that check the installed distribution skip when TicketSense is
not installed in the current environment — they run for real in
a fresh venv after ``pip install -e .``.
"""

import tomllib
from importlib import metadata
from pathlib import Path

import pytest

import ticketsense
from ticketsense import app, predict, train

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SCRIPTS = {
    "ticketsense-train": "ticketsense.train:main",
    "ticketsense-predict": "ticketsense.predict:main",
    "ticketsense-demo": "ticketsense.app:main",
}


def _load_pyproject() -> dict:
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as handle:
        return tomllib.load(handle)


def _pinned_versions(path: Path) -> dict[str, str]:
    """Read ``name==version`` pins from a requirements-style file."""
    versions: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        versions[name.strip().lower()] = version.strip()
    return versions


def test_pyproject_names_the_package_and_takes_its_version() -> None:
    data = _load_pyproject()
    assert data["project"]["name"] == "ticketsense"
    # The version lives in the package, not in the file.
    assert data["project"]["dynamic"] == ["version"]
    assert data["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "ticketsense.__version__"
    }
    assert ticketsense.__version__ == "0.1.0"


def test_pyproject_declares_the_src_layout() -> None:
    data = _load_pyproject()
    assert data["tool"]["setuptools"]["packages"]["find"]["where"] == ["src"]
    assert data["project"]["requires-python"] == ">=3.10"


def test_pyproject_declares_the_console_scripts() -> None:
    data = _load_pyproject()
    assert data["project"]["scripts"] == EXPECTED_SCRIPTS


def test_console_script_entry_points_are_callable() -> None:
    # The scripts must call the existing module entry points, not
    # duplicate their logic.
    assert callable(train.main)
    assert callable(predict.main)
    assert callable(app.main)


def test_runtime_versions_match_requirements() -> None:
    """pyproject.toml and requirements.txt must not disagree."""
    data = _load_pyproject()
    declared = _pinned_versions(PROJECT_ROOT / "requirements.txt")
    for requirement in data["project"]["dependencies"]:
        name, _, version = requirement.partition("==")
        assert version, f"{requirement} is not pinned"
        assert declared[name.lower()] == version, (
            f"{name} is {version} in pyproject.toml but "
            f"{declared.get(name.lower())} in requirements.txt"
        )
    # pytest is a dev-only extra in requirements.txt, not a runtime dep.
    assert "pytest" not in {
        req.partition("==")[0].lower() for req in data["project"]["dependencies"]
    }


def test_installed_version_matches_the_package() -> None:
    try:
        installed = metadata.version("ticketsense")
    except metadata.PackageNotFoundError:
        pytest.skip("TicketSense is not installed in this environment")
    assert installed == ticketsense.__version__


def test_installed_console_scripts_are_registered() -> None:
    try:
        distribution = metadata.distribution("ticketsense")
    except metadata.PackageNotFoundError:
        pytest.skip("TicketSense is not installed in this environment")
    scripts = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    for name, target in EXPECTED_SCRIPTS.items():
        assert scripts[name] == target
