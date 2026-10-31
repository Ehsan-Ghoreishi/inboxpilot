"""Static checks on the container build files: they can't drift from the
project's Python version, leak secrets, or run as root."""

import re
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text()
DOCKERIGNORE = (REPO_ROOT / ".dockerignore").read_text().splitlines()


def _instructions(keyword: str) -> list[str]:
    return [
        line.split(maxsplit=1)[1]
        for line in DOCKERFILE.splitlines()
        if line.startswith(f"{keyword} ")
    ]


def test_base_image_matches_requires_python():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    requires = pyproject["project"]["requires-python"]
    (base,) = [image for image in _instructions("FROM") if image.startswith("python:")]
    version = re.fullmatch(r"python:(\d+\.\d+)-slim", base).group(1)

    assert f">={version}" in requires
    major, minor = version.split(".")
    assert f"<{major}.{int(minor) + 1}" in requires


def test_dependencies_install_from_the_lockfile_without_dev_group():
    assert "uv sync --frozen --no-dev" in _instructions("RUN")


def test_copies_only_runtime_code():
    copied = {line.split()[0] for line in _instructions("COPY") if "--from" not in line}

    assert copied == {
        "pyproject.toml",
        "app/",
        "alembic/",
        "alembic.ini",
        "scripts/",
    }


def test_runs_as_non_root_user():
    (user,) = _instructions("USER")

    assert user not in ("root", "0")


def test_serves_on_cloud_runs_port():
    (cmd,) = _instructions("CMD")

    assert "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}" in cmd


def test_dockerignore_excludes_secrets_and_dev_files():
    for entry in [".env*", ".git", ".venv", "__pycache__", "eval/results", "tests", "requests/"]:
        assert entry in DOCKERIGNORE


def test_env_files_other_than_the_example_are_git_ignored():
    gitignore = (REPO_ROOT / ".gitignore").read_text().splitlines()

    assert ".env*" in gitignore
    assert "!.env.example" in gitignore


def test_raw_event_file_copy_is_off_in_the_image():
    assert "STORE_RAW_EVENT_FILES=false" in DOCKERFILE
