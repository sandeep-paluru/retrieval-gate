import pytest

from retrieval_gate.dataset import load_dataset
from retrieval_gate.errors import DatasetError


def test_jsonl_and_yaml_match(tmp_path):
    jsonl = tmp_path / "cases.jsonl"
    jsonl.write_text(
        '{"id": "q1", "query": " cats ", "relevant": ["d1", "d2"]}\n'
        '{"id": "q2", "query": "dogs", "relevant": {"d3": 2, "d1": 1}}\n',
        encoding="utf-8",
    )
    yaml_path = tmp_path / "cases.yaml"
    yaml_path.write_text(
        "- id: q1\n"
        "  query: cats\n"
        "  relevant: [d1, d2]\n"
        "- id: q2\n"
        "  query: dogs\n"
        "  relevant:\n"
        "    d3: 2\n"
        "    d1: 1\n",
        encoding="utf-8",
    )
    from_json = [case.model_dump() for case in load_dataset(jsonl)]
    from_yaml = [case.model_dump() for case in load_dataset(yaml_path)]
    assert from_json == from_yaml
    assert from_json[0]["query"] == "cats"
    assert from_json[0]["relevant"] == {"d1": 1, "d2": 1}
    assert from_json[1]["relevant"] == {"d3": 2, "d1": 1}


def test_jsonl_error_includes_line_and_case_id(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "q1", "query": "ok", "relevant": ["d1"]}\n'
        "\n"
        '{"id": "q2", "query": "bad", "relevant": []}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match=r"cases\.jsonl:3: case 'q2':") as caught:
        load_dataset(path)
    assert "at least one document" in str(caught.value)


def test_invalid_json_names_the_line(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "q1", "query": "ok", "relevant": ["d1"]}\n{nope\n', encoding="utf-8")
    with pytest.raises(DatasetError, match=r"cases\.jsonl:2: invalid JSON"):
        load_dataset(path)


def test_duplicate_case_id(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id": "q1", "query": "a", "relevant": ["d1"]}\n'
        '{"id": "q1", "query": "b", "relevant": ["d2"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError, match="duplicate case id"):
        load_dataset(path)


def test_duplicate_relevant_id(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "q1", "query": "a", "relevant": ["d1", "d1"]}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate doc id 'd1'"):
        load_dataset(path)


def test_missing_field(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "q1", "relevant": ["d1"]}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="missing field 'query'"):
        load_dataset(path)


def test_grade_must_be_positive(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "q1", "query": "a", "relevant": {"d1": 0}}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="positive integer"):
        load_dataset(path)


def test_bool_grade_is_rejected(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text('{"id": "q1", "query": "a", "relevant": {"d1": true}}\n', encoding="utf-8")
    with pytest.raises(DatasetError, match="case 'q1'"):
        load_dataset(path)


def test_yaml_must_be_a_list(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text("id: q1\nquery: a\nrelevant: [d1]\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="must be a list"):
        load_dataset(path)


def test_invalid_yaml(tmp_path):
    path = tmp_path / "cases.yaml"
    path.write_text("[\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="invalid YAML"):
        load_dataset(path)


def test_empty_file_is_rejected(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="no cases"):
        load_dataset(path)


def test_bad_suffix_and_missing_file(tmp_path):
    path = tmp_path / "cases.txt"
    path.write_text("hi\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="expected a .jsonl"):
        load_dataset(path)
    with pytest.raises(DatasetError, match="dataset not found"):
        load_dataset(tmp_path / "missing.jsonl")


def test_non_object_case(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="case must be an object"):
        load_dataset(path)
