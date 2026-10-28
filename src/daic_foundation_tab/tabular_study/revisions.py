from __future__ import annotations

import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

from daic_foundation_tab.data.participants import fingerprint

ROOT = Path(__file__).resolve().parents[3]
FILENAME = "execution_revisions.json"


def _identity(files: dict) -> str:
    names = sorted(name for name in files if name.startswith("src/") and name.endswith(".py"))
    if set(files) != {*names, "pyproject.toml", "uv.lock"}:
        raise ValueError("Software snapshot contains unexpected paths")
    return fingerprint([(name, files[name]) for name in [*names, "pyproject.toml", "uv.lock"]])


def _git_files(commit: str) -> dict:
    archive = subprocess.run(
        ["git", "archive", commit, "src", "pyproject.toml", "uv.lock"],
        cwd=ROOT, check=True, capture_output=True,
    ).stdout
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        return {
            member.name: hashlib.sha256(source.extractfile(member).read()).hexdigest()
            for member in source.getmembers()
            if member.isfile() and (member.name.endswith(".py") or member.name in {"pyproject.toml", "uv.lock"})
        }


def _working_files() -> dict:
    paths = sorted((ROOT / "src").rglob("*.py")) + [ROOT / "pyproject.toml", ROOT / "uv.lock"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def load_revisions(directory: Path, study: dict) -> dict:
    path = directory / FILENAME
    if not path.exists():
        return {}
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1 or manifest.get("study_id") != study["study_id"]:
        raise ValueError("Execution revision manifest belongs to a different study")
    revisions = {}
    for record in manifest["revisions"]:
        payload = {key: value for key, value in record.items() if key != "revision_id"}
        before, after = record["baseline_files"], record["executed_files"]
        changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
        if (
            record.get("revision_id") != fingerprint(payload)
            or _identity(before) != study["software_identity"]
            or _identity(after) != record["software_identity"]
            or not changed
            or not changed <= set(record["allowed_files"])
            or any(not name.startswith("src/") or not name.endswith(".py") or ".." in Path(name).parts for name in record["allowed_files"])
            or not isinstance(record.get("reason"), str) or not record["reason"].strip()
            or not isinstance(record.get("git_commit"), str) or len(record["git_commit"]) != 40
            or record["software_identity"] in revisions
        ):
            raise ValueError("Execution revision is invalid or exceeds its declared source changes")
        revisions[record["software_identity"]] = record
    return revisions


def register_revision(directory: str | Path, baseline_commit: str, allowed_files: list[str], reason: str) -> dict:
    from daic_foundation_tab.tabular_study.bundle import _json, verify_bundle

    directory = Path(directory).resolve()
    study = verify_bundle(directory, verify_software=False)
    revisions = load_revisions(directory, study)
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.strip()
    baseline = _git_files(baseline_commit)
    current = _working_files()
    if _git_files(current_commit) != current:
        raise ValueError("Commit all executed source changes before registering a revision")
    if _identity(baseline) != study["software_identity"]:
        raise ValueError("Baseline commit does not match the prepared study source")
    identity = _identity(current)
    if identity in revisions:
        return revisions[identity]
    payload = {
        "software_identity": identity,
        "git_commit": current_commit,
        "baseline_commit": baseline_commit,
        "baseline_files": baseline,
        "executed_files": current,
        "allowed_files": sorted(set(allowed_files)),
        "reason": reason,
    }
    record = {**payload, "revision_id": fingerprint(payload)}
    manifest = {"version": 1, "study_id": study["study_id"], "revisions": [*revisions.values(), record]}
    pending = directory / "execution_revisions.pending"
    _json(pending, manifest)
    original = directory / FILENAME
    try:
        changed = {name for name in baseline.keys() | current.keys() if baseline.get(name) != current.get(name)}
        if not changed or not changed <= set(allowed_files) or any(
            not name.startswith("src/") or not name.endswith(".py") or ".." in Path(name).parts
            for name in allowed_files
        ) or not reason.strip():
            raise ValueError("Execution revision exceeds its declared source changes")
        pending.replace(original)
    finally:
        pending.unlink(missing_ok=True)
    load_revisions(directory, study)
    return record
