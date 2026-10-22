"""Prepare fold-specific neighbor-incidence features without fitting a prediction model."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from incidence.data import normalize_category_name
from incidence.config import CV_FOLDS, CV_REPEATS
from incidence.embeddings import CLEANING_VERSION, build_text_table
from incidence.retrieval import MATCH_FIELDS, MIN_SIMILARITY, candidate_table, choose_neighbor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true",
                        help="Freeze pools and, when embeddings exist, export top-five candidates without LLM calls.")
    parser.add_argument("--views", nargs="+", choices=["cv", "evaluation", "deployment"],
                        default=["cv", "evaluation", "deployment"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cv-repeats", type=int, choices=range(1, CV_REPEATS + 1), default=1,
                        help="Saved CV repeats to prepare; default 1 gives five folds rather than fifteen.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/retrieval_features")
    parser.add_argument("--min-similarity", type=float, default=MIN_SIMILARITY,
                        help="Optional prefilter; omitted by default so the LLM sees the top five and may return null.")
    arguments = parser.parse_args()
    if arguments.workers < 1:
        parser.error("Workers must be positive")
    if arguments.min_similarity is not None and (not np.isfinite(arguments.min_similarity) or not -1 <= arguments.min_similarity <= 1):
        parser.error("Minimum similarity must be finite and within [-1, 1]")
        run = (PROJECT_ROOT / "evaluation" if (PROJECT_ROOT / "evaluation").exists()
            else PROJECT_ROOT / "outputs/all_estimates_cross_model")
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(PROJECT_ROOT / "incidence_dataset.csv")
    prediction = pd.read_csv(PROJECT_ROOT / "prediction_set.csv")
    training["source_row_id"] = np.arange(len(training))
    training["category_key"] = training.category_name.map(normalize_category_name)
    split = pd.read_csv(run / "split_assignments.csv")
    if not np.array_equal(split.row_id, training.source_row_id):
        parser.error("Saved split source IDs do not align to training rows")
    pd.testing.assert_frame_equal(split[["category_name", "geography_name"]], training[["category_name", "geography_name"]])
    development_ids = split.loc[split.partition.eq("development"), "row_id"].tolist()
    development_set = set(development_ids)
    queries_train = training[["source_row_id", *MATCH_FIELDS]].copy()
    queries_train["query_id"] = "train:" + queries_train.source_row_id.astype(str)
    queries_train["source"] = "training"
    queries_prediction = prediction[["prediction_id", *MATCH_FIELDS]].copy()
    if queries_prediction.prediction_id.isna().any() or not queries_prediction.prediction_id.is_unique:
        parser.error("Prediction IDs must be nonempty and unique")
    queries_prediction["source_row_id"] = np.arange(len(prediction))
    queries_prediction["query_id"] = "prediction:" + queries_prediction.prediction_id.astype(str)
    queries_prediction["source"] = "prediction"
    queries = pd.concat([queries_train, queries_prediction], ignore_index=True)
    queries.to_csv(output / "query_rows.csv", index=False)
    training[["source_row_id", *MATCH_FIELDS, "percentage_3"]].to_csv(output / "reference_rows.csv", index=False)
    contexts = []
    if "cv" in arguments.views:
        selection = json.loads((run / "metrics.json").read_text())["selected"]["model"]
        oof = pd.read_csv(run / "oof_predictions.csv", usecols=["model", "split", "row_id"])
        selected_folds = oof.loc[oof.model.eq(selection) & oof.split.lt(CV_FOLDS * arguments.cv_repeats)]
        if selected_folds["split"].nunique() != CV_FOLDS * arguments.cv_repeats:
            parser.error("Requested repeats exceed the frozen submission folds; use --cv-repeats 1")
        for split_number, validation in selected_folds.groupby("split", sort=True):
            validation_ids = set(validation.row_id)
            if not validation_ids.issubset(development_set):
                parser.error("Saved CV validation contains rows outside development")
            reference_ids = [row_id for row_id in development_ids if row_id not in validation_ids]
            train_groups = set(training.iloc[reference_ids].category_key)
            validation_groups = set(training.iloc[list(validation_ids)].category_key)
            if train_groups & validation_groups:
                parser.error("Saved CV categories overlap")
            contexts.append({"context": f"cv_{int(split_number):02d}", "view": "cv",
                             "reference_row_ids": reference_ids, "query_positions": development_ids,
                             "query_roles": ["validation" if row_id in validation_ids else "training"
                                             for row_id in development_ids]})
    if "evaluation" in arguments.views:
        contexts.append({"context": "evaluation", "view": "evaluation",
                         "reference_row_ids": development_ids, "query_positions": list(range(len(training))),
                         "query_roles": ["training" if row_id in development_set else "holdout"
                                         for row_id in range(len(training))]})
    if "deployment" in arguments.views:
        contexts.append({"context": "deployment", "view": "deployment",
                         "reference_row_ids": list(range(len(training))),
                         "query_positions": list(range(len(queries))),
                         "query_roles": ["training"] * len(training) + ["prediction"] * len(prediction)})
    (output / "reference_pools.json").write_text(json.dumps(contexts, indent=2) + "\n")
    plan = pd.DataFrame([{"context": context["context"], "view": context["view"],
                          "reference_rows": len(context["reference_row_ids"]),
                          "query_rows": len(context["query_positions"]),
                          "training_queries": context["query_roles"].count("training"),
                          "validation_queries": context["query_roles"].count("validation"),
                          "holdout_queries": context["query_roles"].count("holdout"),
                          "prediction_queries": context["query_roles"].count("prediction")}
                         for context in contexts])
    plan.to_csv(output / "preparation_plan.csv", index=False)
    manifest = {"status": "reference_pools_prepared_embeddings_pending", "top_k": 5,
                "cv_repeats": arguments.cv_repeats, "cv_folds_per_repeat": CV_FOLDS,
                "min_similarity": arguments.min_similarity, "no_match_value": None,
                "views": arguments.views, "contexts": len(contexts),
                "maximum_llm_choices_before_cache": int(plan.query_rows.sum()),
                "same_category_excluded": True, "targets_shown_to_llm": False,
                "llm_input_fields": ["question_text", "qualifying_options", "disqualifying_options"],
                "llm_candidate_ids": "Opaque local IDs only; no source row IDs, metadata, ranks or scores",
                "llm_can_abstain": True,
                "training_model_fitted": False, "feature_name": "retrieved_incidence",
                "warning": "CV tables are context-specific; never use a single global target-derived column across folds"}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(plan.to_string(index=False))
    embedding_directory = PROJECT_ROOT / "outputs/text_embeddings"
    embedding_manifest_path = embedding_directory / "manifest.json"
    if not embedding_manifest_path.exists():
        embedding_directory = PROJECT_ROOT / "features"
        embedding_manifest_path = embedding_directory / "embedding_manifest.json"
    embedding_manifest = json.loads(embedding_manifest_path.read_text())
    if embedding_manifest.get("status") != "complete":
        print("Reference pools and raw audit tables prepared. Real embeddings are pending; no neighbors or incidence features invented.")
        if not arguments.prepare_only:
            parser.exit(2, "Run scripts/build_text_embeddings.py first.\n")
        return
    if embedding_manifest["cleaning_version"] != CLEANING_VERSION:
        parser.error("Embedding cleaning version is stale; regenerate embeddings")
    matrices = []
    for label, raw in (("training", training), ("prediction", prediction)):
        expected = build_text_table(raw, label)
        with np.load(embedding_directory / f"{label}_embeddings.npz", allow_pickle=False) as stored:
            np.testing.assert_array_equal(stored["source_row_id"], expected.source_row_id)
            np.testing.assert_array_equal(stored["embedding_text_id"], expected.embedding_text_id)
            if stored["embeddings"].shape != (len(raw), embedding_manifest["dimensions"]):
                parser.error("Embedding matrix dimensions do not match the current manifest")
            matrices.append(stored["embeddings"].copy())
    vectors = np.vstack(matrices)
    compact_candidates, selected_features, query_status_tables = [], [], []
    for context in contexts:
        positions = context["query_positions"]
        reference_ids = context["reference_row_ids"]
        query_rows = queries.iloc[positions]
        references = training.iloc[reference_ids]
        candidates = candidate_table(query_rows, references, vectors[positions], vectors[reference_ids],
                         min_similarity=arguments.min_similarity)
        roles = dict(zip(query_rows.query_id, context["query_roles"]))
        candidates["context"] = context["context"]
        candidates["query_role"] = candidates.query_id.map(roles)
        compact = candidates[["context", "query_id", "query_role", "rank", "candidate_row_id", "cosine_similarity"]]
        compact.to_csv(output / f"{context['context']}_candidates.csv", index=False)
        compact_candidates.append(compact)
        counts = candidates.groupby("query_id").size()
        query_status = pd.DataFrame({"context": context["context"], "query_id": query_rows.query_id.to_numpy(),
                                     "query_role": context["query_roles"],
                                     "eligible_candidates": query_rows.query_id.map(counts).fillna(0).to_numpy(dtype=int),
                                     "retrieved_incidence": np.nan})
        query_status["retrieval_status"] = np.where(query_status.eligible_candidates.gt(0),
                                                    "llm_pending", "no_candidates")
        if not arguments.prepare_only:
            candidate_groups = dict(tuple(candidates.groupby("query_id", sort=False)))
            def select(query):
                eligible = candidate_groups.get(query["query_id"], candidates.iloc[:0])
                return choose_neighbor(query, eligible, references, min_similarity=arguments.min_similarity)
            with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
                selected = list(executor.map(select, query_rows.to_dict("records")))
            features = pd.DataFrame(selected)
            features["context"] = context["context"]
            features["query_role"] = features.query_id.map(roles)
            features.to_csv(output / f"{context['context']}_features.csv", index=False)
            selected_features.append(features)
            lookup = features.set_index("query_id")
            query_status["retrieval_status"] = query_status.query_id.map(lookup.retrieval_status)
            query_status["retrieved_incidence"] = query_status.query_id.map(lookup.retrieved_incidence)
        query_status_tables.append(query_status)
        print(f"Prepared {context['context']}: {len(query_rows)} queries, {len(reference_ids)} authorized reference rows.")
    pd.concat(compact_candidates, ignore_index=True).to_csv(output / "top5_candidates.csv", index=False)
    all_status = pd.concat(query_status_tables, ignore_index=True)
    all_status.to_csv(output / "query_match_status.csv", index=False)
    if selected_features:
        pd.concat(selected_features, ignore_index=True).to_csv(output / "retrieved_incidence_features.csv", index=False)
    manifest.update(status="candidates_prepared_llm_pending" if arguments.prepare_only else "complete",
                    embedding_model=embedding_manifest["embedding_model"],
                    embedding_cleaning_version=embedding_manifest["cleaning_version"],
                    eligible_queries=int(all_status.eligible_candidates.gt(0).sum()),
                    no_match_queries=int(all_status.eligible_candidates.eq(0).sum()),
                    llm_abstentions=int(all_status.retrieval_status.eq("llm_no_relevant_match").sum()))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Data preparation finished: {manifest['status']}. No prediction model trained.")


if __name__ == "__main__":
    main()