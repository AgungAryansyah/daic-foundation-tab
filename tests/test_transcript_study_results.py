import json

import pytest

from daic_foundation_tab.tabular_study.bundle import file_hash
from daic_foundation_tab.transcript_study import groups
from daic_foundation_tab.transcript_study.bundle import read_bundle
from daic_foundation_tab.transcript_study.results import verify_result
from tests.test_transcript_study_audit import prepared_input
from tests.test_transcript_study_groups import host_input, install_model


@pytest.mark.parametrize("problem", ["unfrozen_base", "missing_weights"])
def test_result_rejects_inconsistent_adapter_provenance_even_with_updated_file_hashes(tmp_path, monkeypatch, problem):
    bundle, host = prepared_input(tmp_path), host_input(tmp_path)
    install_model(monkeypatch)
    context = read_bundle(bundle)
    condition = next(c for c in context[0]["conditions"]["cv"] if c["mode"] == "fine_tuning")
    output = groups.condition_worker(bundle, host, condition["condition_id"], "cv")
    result = json.loads((output / "result.json").read_text())
    if problem == "unfrozen_base":
        path = output / "model_metadata.json"
        metadata = json.loads(path.read_text())
        metadata["base_parameters_frozen"] = False
        path.write_text(json.dumps(metadata))
        result["files"][path.name] = file_hash(path)
    else:
        result["checkpoints"] = {name: digest for name, digest in result["checkpoints"].items()
                                 if not name.endswith("adapter_model.safetensors")}
    (output / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="adaptation metadata|adapter weights"):
        verify_result(context, output)
