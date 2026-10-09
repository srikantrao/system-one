"""Capture the experiment's code, installed environment, and visible hardware."""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from system_one.config import digest_json


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_snapshot(project_root: Path) -> dict:
    files = [project_root / "pyproject.toml", project_root / "uv.lock"]
    for directory in ("src", "scripts"):
        files.extend(sorted((project_root / directory).rglob("*.py")))
    hashes = {
        str(path.relative_to(project_root)): file_sha256(path) for path in files if path.is_file()
    }
    git = {"commit": None, "dirty": None}
    if shutil.which("git"):
        try:
            command = ["git", "-C", str(project_root)]
            status = subprocess.run(
                [*command, "status", "--porcelain"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            # An initialized repository can have changes before its first commit.
            commit = subprocess.run(
                [*command, "rev-parse", "--verify", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            git = {
                "commit": commit.stdout.strip() if commit.returncode == 0 else None,
                "dirty": bool(status.stdout.strip()),
            }
        except (subprocess.SubprocessError, OSError):
            pass
    return {"sha256": digest_json(hashes), "files": hashes, "git": git}


def environment_snapshot() -> dict:
    packages = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata["Name"]
    }
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "packages": dict(sorted(packages.items())),
    }


def hardware_snapshot() -> dict:
    result = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "nvidia": {"status": "not_detected", "devices": []},
    }
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            completed = subprocess.run(
                [
                    executable,
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            result["nvidia"] = {
                "status": "reported",
                "devices": completed.stdout.strip().splitlines(),
            }
        except (subprocess.SubprocessError, OSError):
            result["nvidia"]["status"] = "unavailable"
    return result
