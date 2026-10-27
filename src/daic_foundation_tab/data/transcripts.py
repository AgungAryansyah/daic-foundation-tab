from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from daic_foundation_tab.models.text_base import TextExample

from .participants import fingerprint, load_manifest, prepare_manifest, source_matches


class TranscriptError(ValueError):
    pass


@dataclass(frozen=True)
class TranscriptDataset:
    texts: dict[str, str]
    manifest: dict
    cache_key: str

    def examples(self, role: str, *, labeled: bool = False) -> list[TextExample]:
        column = "target_binary" if self.manifest["task"] == "classification" else "target_phq8"
        if role == "test" and labeled:
            raise TranscriptError("Test labels cannot enter model examples")
        return [
            TextExample(
                str(row["participant_id"]),
                self.texts[str(row["participant_id"])],
                float(row[column]) if labeled else None,
            )
            for row in self.manifest["participants"]
            if row["role"] == role
        ]


def read_transcript(path: Path, settings: dict) -> str:
    if any(part.startswith("._") for part in path.parts):
        raise TranscriptError("Filesystem metadata is not a transcript")
    columns = settings.get("columns", {})
    speaker_column = columns.get("speaker", "speaker")
    text_column = columns.get("text", "value")
    try:
        frame = pd.read_csv(
            path,
            sep=settings.get("delimiter", "\t"),
            encoding=settings.get("encoding", "utf-8-sig"),
            dtype=str,
            keep_default_na=False,
        )
    except (ValueError, UnicodeError, pd.errors.ParserError) as error:
        raise TranscriptError(f"Cannot parse transcript {path.name}") from error
    if not {speaker_column, text_column} <= set(frame.columns):
        raise TranscriptError(f"Missing speaker/text columns in {path.name}")
    policy = settings.get("speaker_policy", "participant_only")
    if policy not in {"participant_only", "dialogue"}:
        raise TranscriptError("Unsupported transcript speaker policy")
    participant = {
        str(x).strip().lower() for x in settings.get("participant_speakers", ["Participant"])
    }
    interviewer = {
        str(x).strip().lower()
        for x in settings.get("interviewer_speakers", ["Ellie", "Interviewer"])
    }
    if participant & interviewer:
        raise TranscriptError("Participant and interviewer speaker labels overlap")
    utterances = []
    for speaker, text in frame[[speaker_column, text_column]].itertuples(index=False, name=None):
        speaker = speaker.strip().lower()
        text = " ".join(text.split())
        if speaker not in participant | interviewer:
            raise TranscriptError(f"Unrecognized speaker in {path.name}")
        if not text or (policy == "participant_only" and speaker not in participant):
            continue
        label = "Participant" if speaker in participant else "Interviewer"
        utterances.append(f"{label}: {text}" if policy == "dialogue" else text)
    if not utterances:
        raise TranscriptError(f"Empty selected transcript in {path.name}")
    return "\n".join(utterances)


def load_transcripts(
    config: dict, progress: Callable[[int, int], None] | None = None
) -> TranscriptDataset:
    manifest_path = config["experiment"].get("partition_manifest")
    manifest = load_manifest(manifest_path, config) if manifest_path else prepare_manifest(config)
    settings = config["data"]["transcripts"]
    root = Path(config["data"]["root"]).expanduser().resolve()
    selected = manifest["participants"]
    cache_root = Path(config["project"]["cache_root"]).expanduser() / "transcripts"
    cache_root.mkdir(parents=True, exist_ok=True)
    texts = {}
    keys = []
    if progress:
        progress(0, len(selected))
    for completed, row in enumerate(selected, start=1):
        participant_id = str(row["participant_id"])
        matches = source_matches(root, settings["pattern"], participant_id)
        if len(matches) != 1:
            raise TranscriptError(
                f"Expected exactly one transcript for participant {participant_id}"
            )
        digest = hashlib.sha256(matches[0].read_bytes()).hexdigest()
        key = fingerprint({"version": 1, "source": digest, "settings": settings})
        cache = cache_root / f"{key}.json"
        if cache.is_file():
            text = json.loads(cache.read_text(encoding="utf-8"))["text"]
        else:
            text = read_transcript(matches[0], settings)
            temporary = cache.with_suffix(".tmp")
            temporary.write_text(json.dumps({"text": text}), encoding="utf-8")
            temporary.replace(cache)
        texts[participant_id] = text
        keys.append((participant_id, key))
        if progress:
            progress(completed, len(selected))
    return TranscriptDataset(
        texts, manifest, fingerprint({"sources": keys, "manifest": manifest["manifest_id"]})
    )
