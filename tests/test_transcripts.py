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


def test_participant_speech_preserves_order_and_does_not_include_identifiers(tmp_path):
    config = transcript_config(tmp_path)
    dataset = load_transcripts(config)
    example = dataset.examples("fit", labeled=True)[0]
    assert example.text == "I feel tired.\nSleep is difficult."
    assert example.participant_id not in example.text
    assert dataset.examples("dev")[0].target is None
    with pytest.raises(TranscriptError, match="Test labels"):
        dataset.examples("test", labeled=True)
    config["data"]["transcripts"]["speaker_policy"] = "dialogue"
    with pytest.raises(TranscriptError, match="participant_only"):
        load_transcripts(config)


def test_reader_rejects_dialogue_input(tmp_path):
    path = tmp_path / "transcript.tsv"
    path.write_text("speaker\tvalue\nEllie\tQuestion\nParticipant\tAnswer\n")
    with pytest.raises(TranscriptError, match="participant_only"):
        read_transcript(path, {"speaker_policy": "dialogue"})


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


def test_native_edaic_reads_text_without_speaker_labels(tmp_path):
    path = tmp_path / "300_Transcript.csv"
    path.write_text(
        'Start_Time,End_Time,Text,Confidence\n'
        '10,12,"I feel tired, most days.",0.9\n'
        '15,18,"  Sleep   is difficult.  ",0.1\n'
        '20,21,   ,0.8\n'
    )
    assert read_transcript(path, {"format": "edaic"}) == (
        "I feel tired, most days.\nSleep is difficult."
    )
    with pytest.raises(TranscriptError, match="Missing speaker/text"):
        read_transcript(path, {"delimiter": ","})


def test_native_edaic_requires_text_column(tmp_path):
    path = tmp_path / "300_Transcript.csv"
    path.write_text("Start_Time,End_Time,Confidence\n1,2,0.9\n")
    with pytest.raises(TranscriptError, match="Missing text"):
        read_transcript(path, {"format": "edaic"})


@pytest.mark.parametrize(
    "content",
    ["Start_Time,End_Time,Text,Confidence\n1,2,   ,0.8\n", "Start_Time,End_Time,Text,Confidence\n"],
)
def test_native_edaic_rejects_empty_text(tmp_path, content):
    path = tmp_path / "300_Transcript.csv"
    path.write_text(content)
    with pytest.raises(TranscriptError, match="Empty selected"):
        read_transcript(path, {"format": "edaic"})


def test_native_edaic_loading_and_source_cache_invalidation(tmp_path):
    config = transcript_config(tmp_path)
    config["data"]["dataset_id"] = "edaic"
    config["data"]["transcripts"] = {
        "format": "edaic", "pattern": "data/{participant_id}_P/{participant_id}_Transcript.csv"
    }
    paths = []
    for source in (tmp_path / "data/data").glob("*_TRANSCRIPT.csv"):
        participant = source.name.split("_")[0]
        path = source.parent / f"{participant}_P" / f"{participant}_Transcript.csv"
        path.parent.mkdir()
        path.write_text("Start_Time,End_Time,Text,Confidence\n1,2,Participant speech,0.9\n")
        paths.append(path)
    before = load_transcripts(config)
    assert set(before.texts.values()) == {"Participant speech"}
    paths[0].write_text("Start_Time,End_Time,Text,Confidence\n1,2,Changed speech,0.9\n")
    after = load_transcripts(config)
    assert after.cache_key != before.cache_key
    assert "Changed speech" in after.texts.values()


def test_reader_rejects_unknown_format(tmp_path):
    path = tmp_path / "transcript.tsv"
    path.write_text("speaker\tvalue\nParticipant\tAnswer\n")
    with pytest.raises(TranscriptError, match="format"):
        read_transcript(path, {"format": "automatic"})
