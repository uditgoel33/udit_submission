"""Stable question and question-country keys built from original text only."""

import hashlib
import json

import pandas as pd

QUESTION_FIELDS = ("question_text", "qualifying_options", "disqualifying_options")
RAW_FIELDS = ("category_name", "geography_name", *QUESTION_FIELDS)


def text_hash(values: list[str]) -> str:
    """Hash a structured text tuple without ambiguous separator collisions."""
    encoded = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def add_question_keys(data: pd.DataFrame) -> pd.DataFrame:
    """Attach one wording key and one wording-country key to every original row."""
    missing = sorted(set(RAW_FIELDS) - set(data.columns))
    if missing:
        raise ValueError(f"Raw question input is missing columns: {missing}")
    for column in RAW_FIELDS:
        if data[column].isna().any() or not data[column].map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Raw question field must contain text without missing values: {column}")
    result = data.copy()
    result["question_id"] = [text_hash(list(values)) for values in result[list(QUESTION_FIELDS)].itertuples(index=False, name=None)]
    result["question_geo_id"] = [text_hash([question, country]) for question, country in
                                 result[["question_id", "geography_name"]].itertuples(index=False, name=None)]
    return result


def build_question_tables(training: pd.DataFrame, prediction: pd.DataFrame) -> tuple:
    """Return keyed datasets and unique target-free tables shared by independent feature sets."""
    mapped_training = add_question_keys(training)
    mapped_prediction = add_question_keys(prediction)
    columns = ["question_id", "question_geo_id", *RAW_FIELDS]
    combined = pd.concat([mapped_training[columns], mapped_prediction[columns]], ignore_index=True)
    questions = combined[["question_id", *QUESTION_FIELDS]].drop_duplicates().reset_index(drop=True)
    question_geographies = combined[["question_geo_id", "question_id", "geography_name", *QUESTION_FIELDS]].drop_duplicates().reset_index(drop=True)
    if not questions["question_id"].is_unique or not question_geographies["question_geo_id"].is_unique:
        raise ValueError("Question keys must map to exactly one raw-text definition")
    for mapped in (mapped_training, mapped_prediction):
        if not mapped["question_id"].isin(questions["question_id"]).all():
            raise ValueError("Every row must map to one question")
        if not mapped["question_geo_id"].isin(question_geographies["question_geo_id"]).all():
            raise ValueError("Every row must map to one question-country pair")
    return mapped_training, mapped_prediction, questions, question_geographies