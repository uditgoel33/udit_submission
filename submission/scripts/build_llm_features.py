"""Append taxonomy labels selected exclusively from the user's fixed CSV."""

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd

from incidence.config import DATA_DIR, OUTPUT_DIR, PREDICTION_DATA_PATH, TRAINING_DATA_PATH
from incidence.llm.client import prompt_hash, usage_summary
from incidence.llm.fs_taxonomy import (
    ASSIGNMENT_FIELDS, assign_fixed_taxonomy, assignment_prompt, load_fixed_taxonomy,
)
from incidence.llm.fs_profile import (
    PROFILE_OUTPUT_FIELDS, build_profile_features, build_profile_prompt,
    merge_profile_features, profile_reliability, refresh_profile_time_columns,
)
from incidence.llm.fs_estimate import (
    ESTIMATE_OUTPUT_FIELDS, build_estimate_features, build_estimate_prompt,
    build_estimate_questions, merge_estimate_features,
)
from incidence.llm.questions import RAW_FIELDS, build_question_tables, text_hash


def prepare_inputs(training: pd.DataFrame, taxonomy: pd.DataFrame, directory: Path, limit=None):
    """Save exactly the allowed prompt inputs for review, with no targets or old labels."""
    raw = training[list(ASSIGNMENT_FIELDS)].copy()
    raw["taxonomy_input_id"] = [text_hash(list(values)) for values in
                                 raw[list(ASSIGNMENT_FIELDS)].itertuples(index=False, name=None)]
    unique = raw.drop_duplicates("taxonomy_input_id", keep="first")
    selected = unique if limit is None else unique.head(limit)
    directory.mkdir(parents=True, exist_ok=True)
    selected.to_csv(directory / "taxonomy_assignment_inputs.csv", index=False)
    prompts = [assignment_prompt(row, taxonomy) for row in selected.to_dict("records")]
    (directory / "taxonomy_assignment_prompts.json").write_text(json.dumps(prompts, indent=2) + "\n", encoding="utf-8")
    counts = {"training_rows": len(training), "unique_assignment_inputs": len(unique),
              "selected_assignment_inputs": len(selected), "allowed_pairs": len(taxonomy),
              "allowed_sectors": taxonomy["sector"].nunique()}
    print(json.dumps(counts, indent=2))
    return counts


def write_assignments(training, taxonomy, output_dir, limit):
    """Write a complete augmented training file or a clearly named limited preview."""
    augmented, unique = assign_fixed_taxonomy(training, taxonomy, limit)
    if limit is None and len(augmented) != len(training):
        raise ValueError("Full taxonomy output must preserve every original training row")
    pd.testing.assert_frame_equal(augmented[training.columns], training.loc[augmented.index])
    filename = "incidence_dataset_with_taxonomy.csv" if limit is None else "incidence_dataset_with_taxonomy_preview.csv"
    path = output_dir / filename
    augmented.to_csv(path, index=False)
    sizes = augmented.groupby(["tax_sector", "tax_subsector"], sort=True)["category_name"].nunique()
    sizes.rename("unique_categories").to_csv(output_dir / "llm" / "taxonomy_node_sizes.csv")
    print(sizes.to_string())
    print(f"Appended tax_sector and tax_subsector: {len(augmented)} rows from {len(unique)} unique inputs.")
    print(f"Output: {path}. Original training file unchanged; fixed labels were not merged or renamed.")
    return path


def append_existing_profiles(taxonomy_path: Path, profile_path: Path) -> Path:
    """Append cached profile columns to taxonomy output without rewriting the source CSV."""
    taxonomy_data = pd.read_csv(taxonomy_path)
    profiles = pd.read_csv(profile_path)
    questions_path = profile_path.parent / "questions.csv"
    if questions_path.exists():
        profiles = refresh_profile_time_columns(profiles, pd.read_csv(questions_path))
        profiles.to_csv(profile_path, index=False)
    merged = merge_profile_features(taxonomy_data, profiles)
    preserved_columns = [column for column in taxonomy_data.columns if column not in PROFILE_OUTPUT_FIELDS]
    if len(merged) != len(taxonomy_data) or not merged[preserved_columns].equals(taxonomy_data[preserved_columns]):
        raise ValueError("Profile attachment must preserve all existing taxonomy data exactly")
    temporary_path = taxonomy_path.with_suffix(".tmp")
    merged.to_csv(temporary_path, index=False)
    temporary_path.replace(taxonomy_path)
    print(f"Added profile columns: {list(PROFILE_OUTPUT_FIELDS)}")
    print(f"Updated {taxonomy_path}: {len(merged)} rows; question_id remains internal.")
    return taxonomy_path


def run_profile(training_path: Path, prediction_path: Path, output_dir: Path,
                limit: int | None, prepare_only: bool) -> dict:
    """Build independent question profiles and a seeded repeatability report, never using targets."""
    training = pd.read_csv(training_path, usecols=list(RAW_FIELDS))
    prediction = pd.read_csv(prediction_path, usecols=list(RAW_FIELDS))
    mapped_train, mapped_pred, questions, question_geographies = build_question_tables(training, prediction)
    directory = output_dir / "llm"
    directory.mkdir(parents=True, exist_ok=True)
    questions.to_csv(directory / "questions.csv", index=False)
    question_geographies.to_csv(directory / "question_geographies.csv", index=False)
    for label, mapped in (("train", mapped_train), ("pred", mapped_pred)):
        mapping = mapped[["question_id", "question_geo_id"]].copy()
        mapping.insert(0, "source_row_id", range(len(mapping)))
        mapping.to_csv(directory / f"question_map_{label}.csv", index=False)
    selected = questions if limit is None else questions.head(limit)
    prompts = [build_profile_prompt(record) for record in selected.to_dict("records")]
    pd.DataFrame({"question_id": selected["question_id"], "prompt": prompts}).to_json(
        directory / "profile_prompts_preview.jsonl", orient="records", lines=True, force_ascii=False,
    )
    counts = {"train_rows": len(training), "prediction_rows": len(prediction),
              "unique_combined_questions": len(questions), "selected_profile_questions": len(selected),
              "unique_combined_question_geographies": len(question_geographies)}
    print(json.dumps(counts, indent=2))
    if prepare_only:
        print("Profile prompts prepared only; no API calls made.")
        return counts
    profiles = build_profile_features(selected.reset_index(drop=True))
    profiles_path = directory / ("profile_features.csv" if limit is None else "profile_features_preview.csv")
    profiles.to_csv(profiles_path, index=False)
    print(f"Saved independent question profile features: {profiles_path}")
    if limit is None:
        agreement, details = profile_reliability(questions, profiles, sample_size=60, seed=2026)
        agreement.to_csv(directory / "profile_reliability.csv", index=False)
        details.to_csv(directory / "profile_reliability_details.csv", index=False)
        print("Reliability agreement (below 80% is flagged):")
        print(agreement.to_string(index=False))
        taxonomy_path = output_dir / "incidence_dataset_with_taxonomy.csv"
        if taxonomy_path.exists():
            append_existing_profiles(taxonomy_path, profiles_path)
        else:
            print("Taxonomy-augmented training data not found; profile labels remain separately available.")
    else:
        print("Limited preview: reliability reruns are deferred until the full profile run.")
    return counts


def run_estimate(output_dir: Path, limit: int | None, prepare_only: bool) -> dict:
    """Add independent five-class population estimates to the existing taxonomy training file."""
    taxonomy_path = output_dir / "incidence_dataset_with_taxonomy.csv"
    if not taxonomy_path.exists():
        raise ValueError(f"Run taxonomy assignment first; augmented training file not found: {taxonomy_path}")
    training = pd.read_csv(taxonomy_path)
    question_geographies = build_estimate_questions(training)
    selected = question_geographies if limit is None else question_geographies.head(limit)
    directory = output_dir / "llm"
    directory.mkdir(parents=True, exist_ok=True)
    prompt_records = selected.to_dict("records")
    prompts = [build_estimate_prompt(record) for record in prompt_records]
    prompt_preview = pd.DataFrame({"estimate_id": selected["estimate_id"], "prompt": prompts})
    if prepare_only:
        prompt_preview.to_json(directory / "estimate_prompts_preview.jsonl", orient="records", lines=True, force_ascii=False)
    counts = {"training_rows": len(training),
              "unique_question_country_pairs": int(question_geographies["question_geo_id"].nunique()),
              "population_scopes": len(question_geographies),
              "selected_population_scopes": len(selected), "samples_per_scope": 5}
    print(json.dumps(counts, indent=2))
    if prepare_only:
        print("Estimate prompts prepared only; no API calls made.")
        return counts
    estimates = build_estimate_features(selected.reset_index(drop=True))
    if limit is not None:
        estimates.to_csv(directory / "population_estimates_preview.csv", index=False)
        print(f"Preview only: {len(estimates)} question-country estimates saved under {directory}; training CSV unchanged.")
        return counts
    augmented = merge_estimate_features(training, estimates)
    if len(augmented) != len(training):
        raise ValueError("Population-estimate merge changed the training row count")
    preserved_columns = [column for column in training.columns if column not in ESTIMATE_OUTPUT_FIELDS]
    if not augmented[preserved_columns].equals(training[preserved_columns]):
        raise ValueError("Population estimates changed existing taxonomy or profile data")
    temporary = taxonomy_path.with_suffix(".tmp")
    augmented.to_csv(temporary, index=False)
    temporary.replace(taxonomy_path)
    print(f"Appended population-class fields {list(ESTIMATE_OUTPUT_FIELDS)} to {taxonomy_path}.")
    print("No question_geo_id column is included; question-country keys remain internal.")
    return counts


def main() -> None:
    """Label the original training data from a user-approved list and stop after taxonomy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["taxonomy", "profile", "estimate"], default="taxonomy",
                        help="Run one feature set only: fixed taxonomy, question profile or population classes.")
    parser.add_argument("--prepare-only", action="store_true", help="Save allowed-list prompts with zero API calls.")
    parser.add_argument("--merge-existing-profile", action="store_true",
                        help="Attach cached profile labels to taxonomy data without API calls.")
    parser.add_argument("--limit", type=int, help="Label only the first N unique inputs and write a clearly marked preview.")
    parser.add_argument("--train", type=Path, default=TRAINING_DATA_PATH)
    parser.add_argument("--taxonomy", type=Path, default=DATA_DIR / "taxonomy.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be positive")
    usage_summary(reset=True)
    counts = {}
    try:
        if arguments.only == "profile":
            if arguments.merge_existing_profile:
                append_existing_profiles(arguments.output_dir / "incidence_dataset_with_taxonomy.csv",
                                         arguments.output_dir / "llm" / "profile_features.csv")
                return
            counts = run_profile(arguments.train, PREDICTION_DATA_PATH, arguments.output_dir,
                                 arguments.limit, arguments.prepare_only)
        elif arguments.only == "estimate":
            counts = run_estimate(arguments.output_dir, arguments.limit, arguments.prepare_only)
        elif arguments.merge_existing_profile:
            parser.error("--merge-existing-profile requires --only profile")
        else:
            training = pd.read_csv(arguments.train)
            taxonomy = load_fixed_taxonomy(arguments.taxonomy)
            directory = arguments.output_dir / "llm"
            counts = prepare_inputs(training, taxonomy, directory, arguments.limit)
            if arguments.prepare_only:
                print("Prepared only. No API calls or labels generated. The supplied taxonomy is fixed.")
            else:
                write_assignments(training, taxonomy, arguments.output_dir, arguments.limit)
    except (ValueError, RuntimeError) as error:
        parser.exit(2, f"LLM feature build stopped: {error}\n")
    finally:
        summary = {"stage": arguments.only, "prepare_only": arguments.prepare_only,
               "input": str(arguments.train), "taxonomy": str(arguments.taxonomy),
               "prompt_hash": prompt_hash({"taxonomy": "taxonomy_assign", "profile": "profile",
                                            "estimate": "estimate"}[arguments.only]),
               "counts": counts, "usage": usage_summary()}
        if arguments.output_dir.exists():
            summary_directory = arguments.output_dir / "llm"
            summary_directory.mkdir(exist_ok=True)
            (summary_directory / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print("Run usage: " + json.dumps(summary["usage"]))


if __name__ == "__main__":
    main()