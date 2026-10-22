"""Feature set B: independently classify screener wording into anchored profile categories."""

from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from incidence.llm.client import PROMPT_DIR, ask_many
from incidence.deterministic import parse_recall_months
from incidence.llm.questions import QUESTION_FIELDS, text_hash

ProfileAction = Literal["purchase", "use", "visit", "donate", "status", "intent"]
ProfileGate = Literal[
    "none", "pet_owner", "parent_or_expecting", "vehicle_owner", "homeowner_or_property",
    "alcohol_drinker", "employed_or_business", "single_gender", "specific_location",
    "specific_condition", "other",
]
FrequencyBucket = Literal[
    "weekly_or_more", "monthly", "every_few_months", "yearly", "every_few_years",
    "rarely", "not_repeating",
]
PriceBand = Literal["free_or_na", "under_20", "20_to_200", "200_to_2000", "over_2000"]
AnswerFormat = Literal["yes_no", "pick_from_list", "statements"]

PROFILE_FIELDS = (*QUESTION_FIELDS,)
PROFILE_LABEL_FIELDS = (
    "prof_action", "prof_gate", "prof_frequency", "prof_price_band",
    "prof_answer_format", "notes",
)
TYPICAL_FREQUENCY_MONTHS = {
    "weekly_or_more": 0.25,
    "monthly": 1.0,
    "every_few_months": 3.0,
    "yearly": 12.0,
    "every_few_years": 36.0,
    "rarely": 120.0,
    "not_repeating": np.nan,
}
PROFILE_OUTPUT_FIELDS = (*PROFILE_LABEL_FIELDS, "freq_months", "typical_frequency_months")


class QuestionProfile(BaseModel):
    """Fixed-definition profile labels; notes are retained only as audit text."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    action: ProfileAction
    eligibility_gate: ProfileGate
    frequency_bucket: FrequencyBucket
    price_band: PriceBand
    answer_format: AnswerFormat
    notes: str = Field(min_length=1)


def build_profile_prompt(question: dict) -> str:
    """Fill the profile prompt from only its three raw wording fields."""
    missing = sorted(set(QUESTION_FIELDS) - set(question))
    if missing:
        raise ValueError(f"Question profile is missing raw fields: {missing}")
    values = {field: question[field] for field in QUESTION_FIELDS}
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise ValueError("Question profile inputs must be nonempty raw text")
    template = (PROMPT_DIR / "profile.txt").read_text(encoding="utf-8")
    return template.format(**values)


def _profile_record(question_id: str, question_text: str, profile: QuestionProfile) -> dict:
    """Separate the question's recall window from typical behaviour frequency."""
    return {
        "question_id": question_id,
        "prof_action": profile.action,
        "prof_gate": profile.eligibility_gate,
        "prof_frequency": profile.frequency_bucket,
        "prof_price_band": profile.price_band,
        "prof_answer_format": profile.answer_format,
        "notes": profile.notes,
        "freq_months": parse_recall_months(question_text),
        "typical_frequency_months": TYPICAL_FREQUENCY_MONTHS[profile.frequency_bucket],
    }


def build_profile_features(questions: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    """Label each unique question once and persist the feature set separately by question ID."""
    required = {"question_id", *QUESTION_FIELDS}
    missing = sorted(required - set(questions.columns))
    if missing:
        raise ValueError(f"Question profile table is missing columns: {missing}")
    if questions["question_id"].isna().any() or questions["question_id"].duplicated().any():
        raise ValueError("Question profile input must contain unique nonempty question IDs")
    selected = questions if limit is None else questions.head(limit)
    prompt_records = selected[["question_id", *QUESTION_FIELDS]].to_dict("records")
    prompts = [build_profile_prompt(record) for record in prompt_records]
    profiles = ask_many("profile", prompts, QuestionProfile, temperature=0.0, sample_index=0)
    return pd.DataFrame([_profile_record(record["question_id"], record["question_text"], profile)
                         for record, profile in zip(prompt_records, profiles)])


def profile_reliability(questions: pd.DataFrame, baseline_profiles: pd.DataFrame,
                        sample_size: int = 60, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Repeat a deterministic random question sample and report exact field-label agreement."""
    if sample_size < 1 or not questions["question_id"].is_unique:
        raise ValueError("Reliability requires unique questions and a positive sample size")
    required_profiles = {"question_id", "prof_action", "prof_gate", "prof_frequency",
                         "prof_price_band", "prof_answer_format"}
    if required_profiles - set(baseline_profiles.columns) or baseline_profiles["question_id"].duplicated().any():
        raise ValueError("Baseline profile output must contain one validated profile per question_id")
    sample = questions.sample(n=min(sample_size, len(questions)), random_state=seed)
    prompts = [build_profile_prompt(record)
               for record in sample[["question_id", *QUESTION_FIELDS]].to_dict("records")]
    repeats = ask_many("profile", prompts, QuestionProfile, temperature=0.7, sample_index=1)
    reference = pd.DataFrame([_profile_record(record["question_id"], record["question_text"], profile)
                              for record, profile in zip(sample.to_dict("records"), repeats)])
    sample_ids = set(sample["question_id"])
    first = baseline_profiles.loc[baseline_profiles["question_id"].isin(sample_ids)].set_index("question_id")
    repeated = reference.set_index("question_id")
    fields = ["prof_action", "prof_gate", "prof_frequency", "prof_price_band", "prof_answer_format"]
    agreement = pd.DataFrame([{
        "field": field,
        "questions": len(sample),
        "agreement": float(first.loc[repeated.index, field].eq(repeated[field]).mean()),
        "below_80pct": bool(first.loc[repeated.index, field].eq(repeated[field]).mean() < 0.8),
    } for field in fields])
    detail = pd.DataFrame({"question_id": repeated.index})
    for field in fields:
        detail[f"{field}_first"] = first.loc[repeated.index, field].to_numpy()
        detail[f"{field}_repeat"] = repeated[field].to_numpy()
    return agreement, detail


def refresh_profile_time_columns(profiles: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """Repair derived time columns from saved labels and the original question wording, without API calls."""
    required = {"question_id", *PROFILE_LABEL_FIELDS}
    missing = sorted(required - set(profiles.columns))
    if missing or not {"question_id", "question_text"}.issubset(questions.columns):
        raise ValueError(f"Cannot recompute time columns; missing profile fields: {missing}")
    if profiles["question_id"].isna().any() or profiles["question_id"].duplicated().any():
        raise ValueError("Saved profiles must contain one row per question_id")
    question_texts = questions.set_index("question_id")["question_text"]
    missing_ids = sorted(set(profiles["question_id"]) - set(question_texts.index))
    if missing_ids:
        raise ValueError(f"No raw question text for {len(missing_ids)} saved profile(s)")
    result = profiles.drop(columns=["freq_months", "typical_frequency_months"], errors="ignore").copy()
    result["freq_months"] = [parse_recall_months(question_texts.at[question_id])
                              for question_id in result["question_id"]]
    result["typical_frequency_months"] = result["prof_frequency"].map(TYPICAL_FREQUENCY_MONTHS)
    return result


def merge_profile_features(taxonomy_data: pd.DataFrame,
                           profiles: pd.DataFrame) -> pd.DataFrame:
    """Append question profiles to each taxonomy row using an internal hash, not an ID column."""
    missing_raw = sorted(set(QUESTION_FIELDS) - set(taxonomy_data.columns))
    missing_profiles = sorted(set(PROFILE_LABEL_FIELDS) - set(profiles.columns))
    if missing_raw or missing_profiles or "question_id" not in profiles:
        raise ValueError(f"Cannot merge profiles; missing raw fields={missing_raw}, profile fields={missing_profiles}")
    if profiles["question_id"].isna().any() or profiles["question_id"].duplicated().any():
        raise ValueError("Profile output must contain one label record per question_id")
    existing_features = [column for column in PROFILE_OUTPUT_FIELDS if column in taxonomy_data]
    if existing_features:
        taxonomy_data = taxonomy_data.drop(columns=existing_features)
    question_ids = [text_hash(list(values)) for values in
                    taxonomy_data[list(QUESTION_FIELDS)].itertuples(index=False, name=None)]
    lookup = profiles.set_index("question_id")
    missing_ids = sorted(set(question_ids) - set(lookup.index))
    if missing_ids:
        raise ValueError(f"No saved profile labels for {len(missing_ids)} taxonomy row(s)")
    result = taxonomy_data.copy()
    for column in PROFILE_LABEL_FIELDS:
        result[column] = [lookup.at[question_id, column] for question_id in question_ids]
    result["freq_months"] = result["question_text"].map(parse_recall_months)
    result["typical_frequency_months"] = result["prof_frequency"].map(TYPICAL_FREQUENCY_MONTHS)
    if len(result) != len(taxonomy_data) or "question_id" in result.columns:
        raise ValueError("Profile merge must preserve row count and keep question_id internal")
    return result