import pytest

from daic_foundation_tab.models.text_base import TextExample
from daic_foundation_tab.models.text_prompt import PromptEncoder, parse_score, select_demonstrations


class TinyTokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return [
            3 + ord(character) % 100 for message in messages for character in message["content"]
        ]

    def encode(self, text, **kwargs):
        return [3 + ord(character) % 100 for character in text]


def test_answer_only_loss_and_no_query_label_or_identifier_in_prompt(tmp_path):
    example = TextExample("secret_id", "I feel tired", 12)
    encoder = PromptEncoder(
        TinyTokenizer(), "regression", {"max_length": 4096}, {"revision": "one"}, tmp_path
    )
    prompt = str(encoder.messages(example))
    assert "secret_id" not in prompt and "12" not in prompt
    record = encoder.training_record(example)
    count = len(encoder.prompt(example))
    assert record["labels"][:count] == [-100] * count
    assert record["labels"][count:] == encoder.completion("12")
    assert encoder.prompt(example) == encoder.prompt(TextExample("different", example.text))
    encoder.identity = {"revision": "two"}
    encoder.prompt(example)
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_context_overflow_never_silently_drops_demonstrations():
    encoder = PromptEncoder(TinyTokenizer(), "classification", {"max_length": 10}, {})
    encoder.demonstrations = [TextExample("train", "Example", 1)]
    with pytest.raises(ValueError, match="no truncation"):
        encoder.prompt(TextExample("test", "Transcript"))
    assert len(encoder.demonstrations) == 1


def test_demonstrations_are_stable_and_reject_oversampling():
    pool = [TextExample(str(index), "text", index % 2) for index in range(10)]
    assert select_demonstrations(pool, 3, 42) == select_demonstrations(pool[::-1], 3, 42)
    assert select_demonstrations(pool, 0, 42) == []
    with pytest.raises(ValueError):
        select_demonstrations(pool, 11, 42)


@pytest.mark.parametrize("text", ["NaN", "25", "-1", "Score: 12", "12 because", "", "1e1"])
def test_invalid_scores_are_rejected(text):
    with pytest.raises(ValueError):
        parse_score(text)


def test_decimal_scores_are_preserved():
    assert parse_score(" 12.5\n") == 12.5
