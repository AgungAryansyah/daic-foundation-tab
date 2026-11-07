import json

import pytest
import yaml

from daic_foundation_tab.transcript_study.audit import require_audit
from daic_foundation_tab.transcript_study.bundle import prepare_bundle, read_bundle
from tests.test_text_prompt import TinyTokenizer
from tests.test_transcript_study_bundle import transcript_input


def prepared_input(tmp_path, family="qwen35_2b", *, adapter_method=None):
    config, parent, _ = transcript_input(tmp_path, family=family)
    if adapter_method is not None:
        raw = yaml.safe_load(config.read_text())
        raw.setdefault("model", {}).setdefault("parameters", {})["adapter_method"] = adapter_method
        config.write_text(yaml.safe_dump(raw))
    bundle = prepare_bundle(config, parent, tmp_path / "transcript", tokenizer=TinyTokenizer())
    return bundle


def test_audit_needs_no_model_weights_and_preserves_demonstration_roles(tmp_path, monkeypatch):
    from daic_foundation_tab.models.qwen import QwenTextModel

    monkeypatch.setattr(QwenTextModel, "_load", lambda self: pytest.fail("Model weights loaded"))
    bundle = prepared_input(tmp_path)
    study, _, _, folds = read_bundle(bundle)
    report = require_audit(bundle, study)
    assert len(report["conditions"]) == 24
    for condition in study["conditions"]["cv"]:
        record = report["conditions"][condition["condition_id"]]
        partition = folds[condition["task"]]["folds"][condition["fold"] - 1]
        assert set(record["demonstrations"]) <= set(partition["fit"])
        assert len(record["demonstrations"]) == (2 if condition["mode"] == "frozen_icl" else 0)
        assert record["statistics"]["count"] > 0
        assert record["statistics"]["max"] < 32768
    report["conditions"].pop(next(iter(report["conditions"])))
    (bundle / "audit.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="incomplete, or changed"):
        require_audit(bundle, study)


def test_overflow_writes_a_failed_audit_and_never_creates_a_ready_campaign(tmp_path):
    config, parent, _ = transcript_input(tmp_path)
    raw = yaml.safe_load(config.read_text())
    raw["model"] = {"parameters": {"max_length": 150}}
    config.write_text(yaml.safe_dump(raw))
    bundle = tmp_path / "transcript"
    with pytest.raises(ValueError, match="No truncation"):
        prepare_bundle(config, parent, bundle, tokenizer=TinyTokenizer())
    study = read_bundle(bundle)[0]
    report = json.loads((bundle / "audit.json").read_text())
    assert report["status"] == "failed"
    assert any(record["errors"] for record in report["conditions"].values())
    with pytest.raises(ValueError, match="failed"):
        require_audit(bundle, study)
