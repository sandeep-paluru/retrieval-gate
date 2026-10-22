"""Load labeled cases from JSONL or YAML."""

import json
from pathlib import Path

import yaml
from pydantic import ValidationError

from retrieval_gate.errors import DatasetError
from retrieval_gate.models import Case


def load_dataset(path: Path) -> list[Case]:
    """Load cases from a ``.jsonl``, ``.yaml``, or ``.yml`` file.

    JSONL is one JSON object per line. Blank lines are ignored. YAML is
    a list of the same objects. An empty relevant set is rejected because
    recall and nDCG are undefined for it. ``tags``, when present, is a
    list of non-empty strings. A missing field means the case has no tags.
    """

    if not path.is_file():
        raise DatasetError(f"dataset not found: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError(f"could not read {path}: {exc}") from exc

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        cases = _load_jsonl(path, text)
    elif suffix in {".yaml", ".yml"}:
        cases = _load_yaml(path, text)
    else:
        raise DatasetError(f"{path}: expected a .jsonl, .yaml, or .yml file")
    if not cases:
        raise DatasetError(f"{path}: dataset has no cases")
    return cases


def _load_jsonl(path: Path, text: str) -> list[Case]:
    cases: list[Case] = []
    seen: dict[str, int] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{line_no}: invalid JSON ({exc.msg})") from exc
        case = _parse_case(raw, f"{path}:{line_no}")
        _remember(seen, case.id, f"{path}:{line_no}")
        cases.append(case)
    return cases


def _load_yaml(path: Path, text: str) -> list[Case]:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        if mark is not None:
            detail = getattr(exc, "problem", None) or exc
            raise DatasetError(f"{path}:{mark.line + 1}: invalid YAML ({detail})") from exc
        raise DatasetError(f"{path}: invalid YAML ({exc})") from exc
    if not isinstance(raw, list):
        raise DatasetError(f"{path}: YAML dataset must be a list of cases")
    cases: list[Case] = []
    seen: dict[str, str] = {}
    for index, item in enumerate(raw, start=1):
        where = f"{path}: item {index}"
        case = _parse_case(item, where)
        if case.id in seen:
            raise DatasetError(
                f"{where}: case {case.id!r}: duplicate case id (first at {seen[case.id]})"
            )
        seen[case.id] = where
        cases.append(case)
    return cases


def _parse_case(raw: object, where: str) -> Case:
    label = where
    if isinstance(raw, dict) and isinstance(raw.get("id"), str) and raw["id"].strip():
        label = f"{where}: case {raw['id'].strip()!r}"
    try:
        return Case.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(f"{label}: {format_validation_error(exc)}") from exc


def _remember(seen: dict[str, int], case_id: str, where: str) -> None:
    if case_id in seen:
        raise DatasetError(
            f"{where}: case {case_id!r}: duplicate case id (first at line {seen[case_id]})"
        )
    # ``where`` ends with the line number for JSONL.
    line_no = int(where.rsplit(":", 1)[-1])
    seen[case_id] = line_no


def format_validation_error(exc: ValidationError) -> str:
    parts: list[str] = []
    for err in exc.errors():
        loc = [str(item) for item in err["loc"]]
        msg = err["msg"]
        prefix = "Value error, "
        if msg.startswith(prefix):
            msg = msg[len(prefix) :]
        if err["type"] == "missing" and loc:
            parts.append(f"missing field '{loc[-1]}'")
            continue
        if loc:
            parts.append(f"{'.'.join(loc)}: {msg}")
        else:
            parts.append(msg)
    return "; ".join(parts)
