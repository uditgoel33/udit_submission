"""Feature set C: five-class population knowledge estimates for question-country pairs."""

from collections import Counter
import json
import re
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from incidence.llm.client import PROMPT_DIR, ask_many
from incidence.llm.questions import QUESTION_FIELDS, text_hash

PopulationClass = Literal["very_low", "low", "medium", "high", "very_high"]
CLASS_ORDER = ("very_low", "low", "medium", "high", "very_high")
ESTIMATE_LABEL_FIELDS = (
    "est_eligible_class", "est_active_of_eligible_class",
    "est_open_of_rest_class", "est_direct_total_class",
)
ESTIMATE_AGREEMENT_FIELDS = tuple(f"{field}_agreement" for field in ESTIMATE_LABEL_FIELDS)
ESTIMATE_TIE_FIELDS = tuple(f"{field}_tie" for field in ESTIMATE_LABEL_FIELDS)
ESTIMATE_OUTPUT_FIELDS = (*ESTIMATE_LABEL_FIELDS, *ESTIMATE_AGREEMENT_FIELDS,
                          *ESTIMATE_TIE_FIELDS, "est_vote_counts", "est_basis")
GENDER_SUFFIX = re.compile(r"\((female|male) only\)\s*$", flags=re.IGNORECASE)
PROMPT_RAW_FIELDS = (*QUESTION_FIELDS, "geography_name")


class PopulationEstimate(BaseModel):
    """Allow only anchored five-band labels; basis is audit text, never a predictor."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    eligible_class: PopulationClass
    active_of_eligible_class: PopulationClass
    open_of_rest_class: PopulationClass
    direct_total_class: PopulationClass
    basis: str = Field(min_length=1)


def _gender_line(category_names: list[str]) -> str:
    """Derive a restricted-population line only when category labels consistently specify one gender."""
    restrictions = []
    for category_name in category_names:
        match = GENDER_SUFFIX.search(category_name.strip())
        restrictions.append(match.group(1).lower() if match else None)
    specified = {restriction for restriction in restrictions if restriction is not None}
    if not specified:
        return ""
    if len(specified) != 1 or any(restriction is None for restriction in restrictions):
        raise ValueError("Conflicting gender restrictions share one question_geo_id")
    gender = next(iter(specified))
    return f"Only {gender} respondents are asked."


def build_estimate_questions(training: pd.DataFrame,
                             prediction: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build one target-free row per wording-country pair and derive any gender line by rule."""
    required = {"category_name", "geography_name", *QUESTION_FIELDS}
    prediction = pd.DataFrame(columns=list(required)) if prediction is None else prediction
    for label, data in (("training", training), ("prediction", prediction)):
        missing = sorted(required - set(data.columns))
        if missing:
            raise ValueError(f"{label} estimate input is missing columns: {missing}")
    combined = pd.concat([training[list(required)], prediction[list(required)]], ignore_index=True)
    for column in required:
        if combined[column].isna().any():
            raise ValueError(f"Estimate input contains missing raw field: {column}")
        combined[column] = combined[column].astype(str)
    question_ids = [text_hash(list(values)) for values in
                    combined[list(QUESTION_FIELDS)].itertuples(index=False, name=None)]
    combined["question_id"] = question_ids
    combined["question_geo_id"] = [text_hash([question_id, country]) for question_id, country in
                                   combined[["question_id", "geography_name"]].itertuples(index=False, name=None)]
    records = []
    for question_geo_id, group in combined.groupby("question_geo_id", sort=True):
        for field in (*QUESTION_FIELDS, "geography_name"):
            if group[field].nunique() != 1:
                raise ValueError(f"Question-country key maps to conflicting raw field: {field}")
        gender_lines = group["category_name"].map(
            lambda category: _gender_line([category])
        )
        scoped_group = group.assign(gender_line=gender_lines)
        for gender_line, population_group in scoped_group.groupby("gender_line", sort=True):
            first = population_group.iloc[0]
            records.append({
                "question_geo_id": question_geo_id,
                "estimate_id": text_hash([question_geo_id, gender_line]),
                **{field: first[field] for field in PROMPT_RAW_FIELDS},
                "gender_line": gender_line,
            })
    return pd.DataFrame(records)


def build_estimate_prompt(question: dict) -> str:
    """Fill the independent estimate prompt from raw wording, country and rule-derived gender line."""
    missing = sorted(set((*PROMPT_RAW_FIELDS, "gender_line")) - set(question))
    if missing:
        raise ValueError(f"Population estimate is missing prompt fields: {missing}")
    values = {field: question[field] for field in (*PROMPT_RAW_FIELDS, "gender_line")}
    values["country"] = values["geography_name"]
    if any(not isinstance(value, str) for value in values.values()):
        raise ValueError("Population estimate prompt values must be text")
    if any(not value.strip() for field, value in values.items() if field != "gender_line"):
        raise ValueError("Population estimate raw fields must not be empty")
    if values["gender_line"] not in {"", "Only female respondents are asked.", "Only male respondents are asked."}:
        raise ValueError("Gender line must be derived from a single-gender category label")
    template = (PROMPT_DIR / "estimate.txt").read_text(encoding="utf-8")
    return template.format(**values)


def summarize_samples(question_geo_id: str, samples: list[PopulationEstimate]) -> dict:
    """Aggregate five class votes by deterministic mode without inventing numeric midpoints."""
    if len(samples) != 5:
        raise ValueError("Each question-country estimate requires exactly five samples")
    result = {"estimate_id": question_geo_id}
    vote_counts = {}
    for field in ESTIMATE_LABEL_FIELDS:
        labels = [getattr(sample, field.removeprefix("est_")) for sample in samples]
        counts = Counter(labels)
        winning_count = max(counts.values())
        result[field] = next(label for label in CLASS_ORDER if counts[label] == winning_count)
        result[f"{field}_agreement"] = winning_count / len(samples)
        result[f"{field}_tie"] = sum(count == winning_count for count in counts.values()) > 1
        vote_counts[field] = {label: counts.get(label, 0) for label in CLASS_ORDER}
    result["est_vote_counts"] = json.dumps(vote_counts, sort_keys=True, separators=(",", ":"))
    result["est_basis"] = samples[0].basis
    return result


def build_estimate_features(question_geographies: pd.DataFrame,
                            limit: int | None = None) -> pd.DataFrame:
    """Request five cached readings per question-country pair and return categorical summaries."""
    required = {"question_geo_id", "estimate_id", *PROMPT_RAW_FIELDS, "gender_line"}
    missing = sorted(required - set(question_geographies.columns))
    if missing or question_geographies["estimate_id"].duplicated().any():
        raise ValueError(f"Estimate question table is missing fields or has duplicate keys: {missing}")
    selected = question_geographies if limit is None else question_geographies.head(limit)
    records = selected.to_dict("records")
    prompts = [build_estimate_prompt(record) for record in records]
    sample_sets = []
    for sample_index in range(5):
        sample_sets.append(ask_many("estimate", prompts, PopulationEstimate,
                                    temperature=0.7, sample_index=sample_index))
    results = []
    for row_index, record in enumerate(records):
        results.append(summarize_samples(record["estimate_id"],
                                         [samples[row_index] for samples in sample_sets]))
    return pd.DataFrame(results)


def merge_estimate_features(training_data: pd.DataFrame,
                            estimates: pd.DataFrame) -> pd.DataFrame:
    """Append estimates to every matching row without exposing question_geo_id."""
    missing_raw = sorted({"geography_name", *QUESTION_FIELDS} - set(training_data.columns))
    required_estimates = {"estimate_id", *ESTIMATE_OUTPUT_FIELDS}
    missing_estimates = sorted(required_estimates - set(estimates.columns))
    if missing_raw or missing_estimates:
        raise ValueError(f"Cannot merge estimates; missing raw={missing_raw}, estimates={missing_estimates}")
    if estimates["estimate_id"].isna().any() or estimates["estimate_id"].duplicated().any():
        raise ValueError("Estimate outputs must have one row per population-scope estimate key")
    result = training_data.drop(columns=[*ESTIMATE_OUTPUT_FIELDS, "question_geo_id", "estimate_id"], errors="ignore").copy()
    question_ids = [text_hash(list(values)) for values in
                    result[list(QUESTION_FIELDS)].itertuples(index=False, name=None)]
    question_geo_ids = [text_hash([question_id, country]) for question_id, country in
                        zip(question_ids, result["geography_name"])]
    gender_lines = result["category_name"].map(lambda category: _gender_line([category]))
    estimate_ids = [text_hash([question_geo_id, gender_line]) for question_geo_id, gender_line in
                    zip(question_geo_ids, gender_lines)]
    lookup = estimates.set_index("estimate_id")
    missing_ids = sorted(set(estimate_ids) - set(lookup.index))
    if missing_ids:
        raise ValueError(f"No population estimate for {len(missing_ids)} training question-country-scope row(s)")
    for column in ESTIMATE_OUTPUT_FIELDS:
        result[column] = [lookup.at[estimate_id, column] for estimate_id in estimate_ids]
    if len(result) != len(training_data) or {"question_geo_id", "estimate_id"}.intersection(result.columns):
        raise ValueError("Estimate merge changed row count or exposed internal population keys")
    return result