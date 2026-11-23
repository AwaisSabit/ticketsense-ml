"""Release hygiene checks for TicketSense.

These tests enforce the findings of the final security, config
and leakage review before the 1.0.0 release: no secret-shaped
text anywhere in the repo, no model binary or environment file
tracked by git, no real contact details in the synthetic demo
data, and every dependency pinned to an exact version. They run
on the checked-out tree, so a future change that breaks one of
these rules fails the suite instead of slipping through.
"""

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Directories and file types that are never scanned: the virtual
# environment, git metadata, caches and build output, and binary
# artefacts (the models and charts are checked separately below).
SKIP_DIRS = {".venv", ".git", "__pycache__", ".pytest_cache", "ticketsense.egg-info"}
SKIP_SUFFIXES = {".png", ".joblib", ".pkl", ".pyc"}

SECRET_PATTERNS = {
    "api key assignment": re.compile(
        r"(?i)(api[_-]?key)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
    "secret or password assignment": re.compile(
        r"(?i)(secret|passwd|password)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
    "token assignment": re.compile(
        r"(?i)(token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9]{16,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "GitHub token": re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    "private key block": re.compile(
        r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"
    ),
    "Slack token": re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
}

EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_PATTERN = re.compile(r"\+?\d[\d\s()-]{9,}\d")


def _scannable_files() -> list[Path]:
    """Text files in the tree, minus venvs, caches and binaries."""
    files = []
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix in SKIP_SUFFIXES:
            continue
        if path.name == "PKG-INFO":
            continue
        files.append(path)
    return files


def _tracked_files() -> list[str]:
    """Paths git tracks, relative to the project root."""
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def test_no_secret_shaped_text_in_the_repo() -> None:
    hits = []
    for path in _scannable_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                hits.append(f"{path.relative_to(PROJECT_ROOT)}: {name}")
    assert hits == [], f"secret-shaped text found: {hits}"


def test_no_model_binary_or_env_file_is_tracked() -> None:
    tracked = _tracked_files()
    bad = [
        path
        for path in tracked
        if path.endswith((".joblib", ".pkl", ".env"))
        or "__pycache__" in path
        or path.endswith(".egg-info/PKG-INFO")
        or "/.egg-info/" in path
    ]
    assert bad == [], f"binaries or env files tracked: {bad}"


def test_report_charts_are_not_tracked_except_the_two_balance_charts() -> None:
    tracked = _tracked_files()
    allowed = {
        "reports/category_balance.png",
        "reports/priority_balance.png",
    }
    bad = [
        path
        for path in tracked
        if path.startswith("reports/") and path.endswith(".png")
        and path not in allowed
    ]
    assert bad == [], f"generated charts tracked: {bad}"


def test_demo_data_has_no_real_contact_details() -> None:
    text = (PROJECT_ROOT / "data" / "demo_tickets.csv").read_text(
        encoding="utf-8"
    )
    assert EMAIL_PATTERN.search(text) is None, "email address in demo data"
    assert PHONE_PATTERN.search(text) is None, "phone number in demo data"


def test_runtime_dependencies_are_pinned() -> None:
    def pinned_versions(path: Path) -> dict[str, str]:
        versions: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            name, sep, version = line.partition("==")
            assert sep, f"unpinned dependency in {path.name}: {line!r}"
            versions[name.strip().lower()] = version.strip()
        return versions

    requirements = pinned_versions(PROJECT_ROOT / "requirements.txt")
    assert "pytest" in requirements, "pytest pin missing from requirements"
    assert len(requirements) >= 9, "requirements.txt lost entries"
