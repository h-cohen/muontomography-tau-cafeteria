import json

from cafetomo.provenance import write_meta


def test_meta_records_commit_and_config_hash(tmp_path):
    p = write_meta(tmp_path, config_path="configs/cafeteria.yaml", stage="test", extra={"n": 1})
    doc = json.loads(p.read_text())
    assert doc["stage"] == "test" and doc["n"] == 1
    assert len(doc["config_sha256"]) == 16
    assert len(doc["git_commit"].split("+")[0]) == 40
