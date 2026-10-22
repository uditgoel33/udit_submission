"""Generate final-model prediction features using the training prompt/schema definitions."""

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from incidence.eda_model import prepare_data
from incidence.llm.client import usage_summary
from incidence.llm.fs_estimate import (
    build_estimate_features, build_estimate_prompt, build_estimate_questions, merge_estimate_features,
)
from incidence.llm.fs_profile import (
    build_profile_features, build_profile_prompt, merge_profile_features,
)
from incidence.llm.fs_taxonomy import (
    ASSIGNMENT_FIELDS, assign_fixed_taxonomy, assignment_prompt, load_fixed_taxonomy,
)
from incidence.llm.questions import QUESTION_FIELDS, text_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "prediction_set.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "outputs/prediction_set_with_taxonomy.csv")
    parser.add_argument("--prepare-only", action="store_true")
    arguments = parser.parse_args()
    raw = pd.read_csv(arguments.input)
    required = {"prediction_id", "geography_name", *ASSIGNMENT_FIELDS}
    if required - set(raw.columns) or raw.empty:
        parser.error("Prediction data must contain IDs, country and all raw screener fields")
    if raw.prediction_id.isna().any() or raw.prediction_id.duplicated().any() or raw.prediction_id.astype(str).str.strip().eq("").any():
        parser.error("Prediction IDs must be nonempty and unique")
    if raw[list(required)].isna().any().any():
        parser.error("Prediction raw fields must not be missing")
    taxonomy = load_fixed_taxonomy(PROJECT_ROOT / "data/taxonomy.csv")
    profiles = pd.read_csv(PROJECT_ROOT / "outputs/llm/profile_features.csv")
    questions = raw[list(QUESTION_FIELDS)].copy()
    questions["question_id"] = [text_hash(list(values)) for values in questions.itertuples(index=False, name=None)]
    questions = questions.drop_duplicates("question_id")
    missing_profiles = questions.loc[~questions.question_id.isin(profiles.question_id)]
    estimate_questions = build_estimate_questions(raw)
    assignment_inputs = raw[list(ASSIGNMENT_FIELDS)].drop_duplicates()
    directory = PROJECT_ROOT / "outputs/llm/prediction"
    directory.mkdir(parents=True, exist_ok=True)
    summary = {"status": "prepared_not_generated", "prediction_rows": len(raw),
               "taxonomy_requests": len(assignment_inputs), "missing_profile_requests": len(missing_profiles),
               "estimate_scopes": len(estimate_questions), "samples_per_estimate": 5,
               "output": str(arguments.output), "model": os.environ.get("OPENAI_MODEL", "")}
    (directory / "taxonomy_prompts.json").write_text(json.dumps([
        assignment_prompt(row, taxonomy) for row in assignment_inputs.to_dict("records")
    ], indent=2) + "\n")
    (directory / "estimate_prompts.json").write_text(json.dumps([
        build_estimate_prompt(row) for row in estimate_questions.to_dict("records")
    ], indent=2) + "\n")
    (directory / "profile_prompts.json").write_text(json.dumps([
        build_profile_prompt(row) for row in missing_profiles.to_dict("records")
    ], indent=2) + "\n")
    summary_path = directory / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if arguments.prepare_only:
        print(json.dumps(summary, indent=2))
        print("Prompts prepared only. No API calls or completed feature/prediction file created.")
        return
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("OPENAI_MODEL"):
        parser.error("Configure OPENAI_API_KEY and the training OPENAI_MODEL in your terminal; never put secrets in chat")
    usage_summary(reset=True)
    augmented, _ = assign_fixed_taxonomy(raw, taxonomy)
    if not missing_profiles.empty:
        profiles = pd.concat([profiles, build_profile_features(missing_profiles)], ignore_index=True)
    augmented = merge_profile_features(augmented, profiles)
    estimates = build_estimate_features(estimate_questions)
    augmented = merge_estimate_features(augmented, estimates)
    prepared = prepare_data(augmented, require_target=False)
    pd.testing.assert_frame_equal(prepared[raw.columns], raw.reset_index(drop=True))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    prepared.to_csv(arguments.output, index=False)
    summary.update(status="complete", usage=usage_summary())
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Validated final-format features for {len(prepared)} prediction rows: {arguments.output}")


if __name__ == "__main__":
    main()