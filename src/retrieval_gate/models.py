"""Labeled query cases."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Case(BaseModel):
    """One labeled query.

    ``relevant`` is a mapping of document id to a positive integer grade.
    A list of ids in the file is stored as grade 1 for each id.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    relevant: dict[str, int]

    @model_validator(mode="before")
    @classmethod
    def normalize_relevant(cls, data: object) -> object:
        if not isinstance(data, dict):
            raise ValueError("case must be an object")
        if "relevant" not in data:
            return data
        relevant = data["relevant"]
        if isinstance(relevant, dict):
            return data
        if not isinstance(relevant, list):
            raise ValueError(
                "relevant must be a list of doc ids or a mapping of doc id to grade"
            )
        grades: dict[str, int] = {}
        for doc_id in relevant:
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("relevant doc ids must be non-empty strings")
            key = doc_id.strip()
            if key in grades:
                raise ValueError(f"duplicate doc id {key!r} in relevant")
            grades[key] = 1
        return {**data, "relevant": grades}

    @field_validator("id", "query", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip()
        return value

    @field_validator("relevant")
    @classmethod
    def grades_must_be_positive(cls, value: dict[str, int]) -> dict[str, int]:
        if not value:
            raise ValueError("relevant must contain at least one document")
        cleaned: dict[str, int] = {}
        for doc_id, grade in value.items():
            if not isinstance(doc_id, str) or not doc_id.strip():
                raise ValueError("relevant doc ids must be non-empty strings")
            key = doc_id.strip()
            if key in cleaned:
                raise ValueError(f"duplicate doc id {key!r} in relevant")
            if isinstance(grade, bool) or not isinstance(grade, int) or grade < 1:
                raise ValueError(f"grade for {key!r} must be a positive integer")
            cleaned[key] = grade
        return cleaned
