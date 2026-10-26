import pandas as pd
import pytest

from daic_foundation_tab.data.transcripts import TranscriptError, load_transcripts, read_transcript
from tests.test_participants import manifest_config


def transcript_config(tmp_path):
    config = manifest_config(tmp_path)
    config["input"] = {"kind": "transcript"}
    config["data"]["transcripts"] = {"pattern": "data/{participant_id}_TRANSCRIPT.csv"}
    for participant_id in ["300", "301", "302", "303", "304", "305", "400", "401", "500", "501"]:
        pd.DataFrame(
            {
                "speaker": ["Ellie", "Participant", "Participant"],
                "value": ["How are you?", "I feel tired.", "Sleep is difficult."],
            }
        ).to_csv(tmp_path / "data/data" / f"{participant_id}_TRANSCRIPT.csv", sep="\t", index=False)
    return config


def test_speaker_policies_preserve_order_and_do_not_include_identifiers(tmp_path):
    config = transcript_config(tmp_path)
    dataset = load_transcripts(config)
    example = dataset.examples("fit", labeled=True)[0]
    assert example.text == "I feel tired.\nSleep is difficult."
    assert example.participant_id not in example.text
    assert dataset.examples("dev")[0].target is None
    with pytest.raises(TranscriptError, match="Test labels"):
        dataset.examples("test", labeled=True)
    config["data"]["transcripts"]["speaker_policy"] = "dialogue"
    dialogue = load_transcripts(config)
    assert dialogue.texts[example.participant_id].startswith(
        "Interviewer: How are you?\nParticipant:"
    )
    assert dialogue.cache_key != dataset.cache_key


def test_source_changes_invalidate_cleaned_cache(tmp_path):
    config = transcript_config(tmp_path)
    before = load_transcripts(config)
    path = tmp_path / "data/data/300_TRANSCRIPT.csv"
    path.write_text("speaker\tvalue\nParticipant\tA different answer.\n")
    after = load_transcripts(config)
    assert after.cache_key != before.cache_key
    assert after.texts["300"] == "A different answer."


@pytest.mark.parametrize(
    "content",
    [
        "wrong,columns\na,b\n",
        "speaker\tvalue\nEllie\tQuestion\n",
        "speaker\tvalue\nUnknown\tAnswer\n",
    ],
)
def test_rejects_invalid_or_empty_transcripts(tmp_path, content):
    path = tmp_path / "transcript.tsv"
    path.write_text(content)
    with pytest.raises(TranscriptError):
        read_transcript(path, {})


def test_custom_delimiter_and_column_mapping(tmp_path):
    path = tmp_path / "transcript.csv"
    path.write_text("who,utterance\nP,Hello\n")
    assert (
        read_transcript(
            path,
            {
                "delimiter": ",",
                "columns": {"speaker": "who", "text": "utterance"},
                "participant_speakers": ["P"],
            },
        )
        == "Hello"
    )
