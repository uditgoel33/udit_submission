"""Train ridge comparisons with context-specific, category-isolated retrieved targets."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from incidence.eda_model import (
    MODEL_FEATURES, SUBSECTOR_TOTAL_INTERACTION, build_ridge, calibrate_oof, prepare_data, score,
)
from incidence.calibration import interval_radius, prediction_intervals
from incidence.models import logit, probability

BASE_FEATURES = (*MODEL_FEATURES, SUBSECTOR_TOTAL_INTERACTION)
RETRIEVAL_NUMERIC = ("retrieved_incidence", "retrieved_incidence_missing")
RETRIEVAL_FEATURES = (*BASE_FEATURES, *RETRIEVAL_NUMERIC)


def attach_retrieval(data: pd.DataFrame, rows: pd.DataFrame, query_ids: list[str],
                     references: pd.DataFrame, allowed_ids: set[int]) -> pd.DataFrame:
    """Validate feature lineage before attaching the correct reference-context values."""
    if len(query_ids) != len(data) or len(set(query_ids)) != len(query_ids):
        raise ValueError("Query identities must align one-to-one with model rows")
    if rows.query_id.duplicated().any() or not set(query_ids).issubset(rows.query_id):
        raise ValueError("Retrieval context is missing queries or contains duplicates")
    aligned = rows.set_index("query_id").loc[query_ids]
    values = aligned.retrieved_incidence.to_numpy(dtype=float)
    selected = aligned.selected_reference_row_id.to_numpy(dtype=float)
    if not np.array_equal(np.isnan(values), np.isnan(selected)):
        raise ValueError("Null retrieval features must have null reference IDs")
    present = ~np.isnan(values)
    if not np.isfinite(values[present]).all() or not ((values[present] >= 0) & (values[present] <= 1)).all():
        raise ValueError("Retrieved incidence must be missing or a proportion")
    if not np.isfinite(selected[present]).all() or not np.equal(selected[present], np.floor(selected[present])).all():
        raise ValueError("Selected reference IDs must be integers")
    selected_ids = selected[present].astype(int)
    if not set(selected_ids).issubset(allowed_ids):
        raise ValueError("Retrieved reference is outside the authorized training pool")
    chosen = references.loc[selected_ids]
    if not np.allclose(values[present], chosen.percentage_3.to_numpy(), atol=1e-12, rtol=0):
        raise ValueError("Copied incidence does not match its reference target")
    if np.any(data.category_key.to_numpy()[present] == chosen.category_key.to_numpy()):
        raise ValueError("Retrieved reference shares the query category")
    result = data.copy()
    result["retrieved_incidence"] = values
    result["retrieved_incidence_missing"] = np.isnan(values).astype(float)
    return result


def build_retrieval_ridge(alpha: float):
    """Use the established fold-local ridge preprocessing with two extra numeric inputs."""
    model = build_ridge(alpha, BASE_FEATURES)
    transforms = model.named_steps["features"].transformers
    model.set_params(features__transformers=[
        (name, transformer, [*columns, *RETRIEVAL_NUMERIC] if name == "numeric" else columns)
        for name, transformer, columns in transforms
    ])
    return model


def train_retrieval_model(project_root: Path, output_dir: Path) -> dict:
    """Tune within frozen CV pools, evaluate once, then separately refit for deployment."""
    packaged_features = project_root / "features"
    retrieval_dir = project_root / "outputs/retrieval_features"
    if not (retrieval_dir / "retrieved_incidence_features.csv").exists():
        retrieval_dir = packaged_features
    packaged = retrieval_dir == packaged_features
    previous_run = (project_root / "evaluation" if (project_root / "evaluation").exists()
                    else project_root / "outputs/all_estimates_cross_model")
    manifest = json.loads((retrieval_dir / ("retrieval_manifest.json" if packaged else "manifest.json")).read_text())
    if manifest["status"] != "complete" or not manifest["same_category_excluded"]:
        raise ValueError("Completed category-disjoint retrieval features are required")
    training_path = project_root / "outputs/incidence_dataset_with_taxonomy.csv"
    if not training_path.exists():
        training_path = packaged_features / "training_features.csv"
    data = prepare_data(pd.read_csv(training_path))
    split = pd.read_csv(previous_run / "split_assignments.csv")
    np.testing.assert_array_equal(split.row_id, data.index)
    pd.testing.assert_frame_equal(split[["category_key", "geography_name"]], data[["category_key", "geography_name"]])
    development_ids = split.loc[split.partition.eq("development"), "row_id"].tolist()
    holdout_ids = split.loc[split.partition.eq("holdout"), "row_id"].tolist()
    if set(data.loc[development_ids].category_key) & set(data.loc[holdout_ids].category_key):
        raise ValueError("Development and holdout categories overlap")
    feature_rows = pd.read_csv(retrieval_dir / ("retrieval_features.csv" if packaged else "retrieved_incidence_features.csv"))
    if feature_rows.duplicated(["context", "query_id"]).any():
        raise ValueError("Retrieval context/query keys must be unique")
    contexts = json.loads((retrieval_dir / "reference_pools.json").read_text())
    pools = {context["context"]: context for context in contexts}
    cv_contexts = [context for context in contexts if context["view"] == "cv"]
    if not cv_contexts:
        raise ValueError("No completed CV retrieval contexts found")
    folds = []
    for context in cv_contexts:
        reference_ids = set(context["reference_row_ids"])
        if not reference_ids.issubset(development_ids):
            raise ValueError("CV retrieval pool includes holdout rows")
        training_ids = [row_id for row_id in development_ids if row_id in reference_ids]
        validation_ids = [row_id for row_id in development_ids if row_id not in reference_ids]
        if not training_ids or not validation_ids:
            raise ValueError("Each CV context needs training and validation rows")
        if set(data.loc[training_ids].category_key) & set(data.loc[validation_ids].category_key):
            raise ValueError("CV categories overlap")
        attached = attach_retrieval(data.loc[development_ids],
                                    feature_rows.loc[feature_rows.context.eq(context["context"])],
                                    [f"train:{row_id}" for row_id in development_ids], data, reference_ids)
        folds.append((attached.loc[training_ids], attached.loc[validation_ids]))
    validation_counts = pd.Series([row_id for _, validation in folds for row_id in validation.index]).value_counts()
    if set(validation_counts.index) != set(development_ids) or validation_counts.nunique() != 1:
        raise ValueError("CV must validate every development row equally often")
    output_dir.mkdir(parents=True, exist_ok=True)
    records, oof_lookup, diagnostics = [], {}, []
    for variant in ("baseline", "retrieval"):
        features = BASE_FEATURES if variant == "baseline" else RETRIEVAL_FEATURES
        for alpha in (.1, 1., 10., 100.):
            frames, train_errors = [], []
            name = f"{variant}_alpha_{alpha:g}"
            for split_number, (training, validation) in enumerate(folds):
                model = build_ridge(alpha, BASE_FEATURES) if variant == "baseline" else build_retrieval_ridge(alpha)
                model.fit(training[list(features)], logit(training.percentage_3))
                train_error = np.abs(probability(model.predict(training[list(features)])) - training.percentage_3).mean()
                train_errors.append(train_error)
                frame = validation[["category_key", "geography_name", "percentage_3"]].copy()
                frame["row_id"], frame["split"] = validation.index, split_number
                frame["predicted_logit"] = model.predict(validation[list(features)])
                frames.append(frame)
                diagnostics.append({"model": name, "split": split_number, "train_mae": train_error,
                                    "validation_mae": np.abs(probability(frame.predicted_logit) - frame.percentage_3).mean()})
            predictions = calibrate_oof(pd.concat(frames, ignore_index=True))
            oof_lookup[name] = predictions
            record = {"model": name, "variant": variant, "alpha": alpha,
                      "train_mae": float(np.mean(train_errors)), **score(predictions)}
            records.append(record)
            print(f"{name}: CV MAE={100 * record['mae']:.2f} pp; interval score={record['interval_score']:.4f}", flush=True)
    results = pd.DataFrame(records).sort_values(["interval_score", "mae"])
    results.to_csv(output_dir / "cv_results.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "fold_diagnostics.csv", index=False)
    pd.concat([frame.assign(model=name) for name, frame in oof_lookup.items()]).to_csv(output_dir / "oof_predictions.csv", index=False)
    evaluation_pool = set(pools["evaluation"]["reference_row_ids"])
    if evaluation_pool != set(development_ids):
        raise ValueError("Evaluation retrieval pool must be development only")
    evaluation_data = attach_retrieval(data, feature_rows.loc[feature_rows.context.eq("evaluation")],
                                      [f"train:{row_id}" for row_id in data.index], data, evaluation_pool)
    comparisons = []
    selected = None
    radius = None
    for variant in ("baseline", "retrieval"):
        choice = results.loc[results.variant.eq(variant)].iloc[0].to_dict()
        features = BASE_FEATURES if variant == "baseline" else RETRIEVAL_FEATURES
        model = build_ridge(choice["alpha"], BASE_FEATURES) if variant == "baseline" else build_retrieval_ridge(choice["alpha"])
        model.fit(evaluation_data.loc[development_ids, list(features)], logit(data.loc[development_ids, "percentage_3"]))
        calibration = oof_lookup[choice["model"]].groupby("row_id").agg(
            actual=("percentage_3", "first"), predicted_logit=("predicted_logit", "mean"), category_key=("category_key", "first"))
        candidate_radius = interval_radius(calibration.actual, calibration.predicted_logit, calibration.category_key)
        logits = model.predict(evaluation_data.loc[holdout_ids, list(features)])
        lower, upper = prediction_intervals(logits, candidate_radius)
        frame = data.loc[holdout_ids, ["category_key", "geography_name", "percentage_3"]].copy()
        frame.insert(0, "row_id", holdout_ids)
        frame["point_estimate"], frame["lower_80"], frame["upper_80"] = probability(logits), lower, upper
        errors = np.abs(frame.point_estimate - frame.percentage_3)
        metrics = {"variant": variant, "alpha": choice["alpha"], "cv_mae": choice["mae"],
                   "cv_interval_score": choice["interval_score"], **score(frame),
                   "median_absolute_error": float(errors.median()), "p25_absolute_error": float(errors.quantile(.25)),
                   "p75_absolute_error": float(errors.quantile(.75))}
        comparisons.append(metrics)
        frame.to_csv(output_dir / ("holdout_predictions.csv" if variant == "retrieval" else "baseline_holdout_predictions.csv"), index=False)
        if variant == "retrieval":
            selected, radius = choice, candidate_radius
            artifact = {"model": model, "features": list(features), "fit_rows": len(development_ids),
                        "fit_categories": int(data.loc[development_ids].category_key.nunique()),
                        "training_partition": "development", "selection": {**choice, "model_type": "retrieval_ridge"},
                        "logit_interval_radius": radius, "retrieval_context": "evaluation"}
            joblib.dump(artifact, output_dir / "fitted_model.joblib")
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(output_dir / "holdout_comparison.csv", index=False)
    split.to_csv(output_dir / "split_assignments.csv", index=False)
    deployment_pool = set(pools["deployment"]["reference_row_ids"])
    if deployment_pool != set(data.index):
        raise ValueError("Deployment retrieval pool must include all labeled rows")
    deployment_rows = feature_rows.loc[feature_rows.context.eq("deployment")]
    deployment_data = attach_retrieval(data, deployment_rows, [f"train:{row_id}" for row_id in data.index], data, deployment_pool)
    deployment_model = build_retrieval_ridge(selected["alpha"]).fit(
        deployment_data[list(RETRIEVAL_FEATURES)], logit(deployment_data.percentage_3))
    deployment_artifact = {**artifact, "model": deployment_model, "fit_rows": len(data),
                           "fit_categories": int(data.category_key.nunique()),
                           "training_partition": "full_data_deployment", "retrieval_context": "deployment"}
    joblib.dump(deployment_artifact, output_dir / "deployment_model.joblib")
    prediction_path = project_root / "outputs/prediction_set_with_taxonomy.csv"
    if not prediction_path.exists():
        prediction_path = packaged_features / "prediction_features.csv"
    prediction_data = prepare_data(pd.read_csv(prediction_path), require_target=False)
    prediction_data = attach_retrieval(prediction_data, deployment_rows,
                                      [f"prediction:{value}" for value in prediction_data.prediction_id], data, deployment_pool)
    prediction_data.to_csv(output_dir / "prediction_features.csv", index=False)
    from incidence.predict import make_predictions
    predictions = make_predictions(prediction_data, deployment_artifact)
    predictions.to_csv(output_dir / "predictions.csv", index=False, float_format="%.8f")
    prediction_data.merge(predictions, on="prediction_id", validate="one_to_one", sort=False).to_csv(
        output_dir / "predictions_detailed.csv", index=False)
    metrics = {"selected": selected, "cv_folds": len(folds), "features": list(RETRIEVAL_FEATURES),
               "development_rows": len(development_ids), "holdout_rows": len(holdout_ids),
               "holdout_rows_used_for_evaluation_model_fit": False,
               "holdout_comparison": comparisons, "logit_interval_radius": float(radius),
               "evaluation_retrieval_missing_rate": float(evaluation_data.retrieved_incidence.isna().mean()),
               "prediction_retrieval_missing_rate": float(prediction_data.retrieved_incidence.isna().mean()),
               "warning": "Previously examined holdout; approximate grouped-OOF calibration. Deployment uses all labeled rows; its coverage is not independently evaluated."}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    lines = ["# Retrieval-Feature Ridge", "",
             "Five-fold grouped development CV compares identical ridge grids with and without retrieved incidence.",
             "Every retrieval value is checked against its authorized reference pool and excludes the query category.",
             "Null matches remain missing in exported features; fold-local median imputation and a missingness indicator enter the model.",
             "No raw embeddings, similarity scores, LLM reasons or reference IDs are predictors.", "",
             "| Model | CV MAE (pp) | Holdout MAE (pp) | Median Error (pp) | Coverage | Width (pp) | Interval Score |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['variant']} | {100*row['cv_mae']:.2f} | {100*row['mae']:.2f} | {100*row['median_absolute_error']:.2f} | {100*row['coverage_80']:.1f}% | {100*row['mean_width']:.2f} | {row['interval_score']:.4f} |")
    lines += ["", "Selection uses development-CV interval score, not holdout results.",
              "Evaluation artifact fits development only; deployment artifact fits all 833 labeled rows with the same alpha and development radius.",
              "The holdout was previously examined; these results are diagnostic. Intervals use approximate category-max OOF calibration, not exact independent split-conformal inference.",
              "The final selected model and the median-versus-mean error trade-off are documented in the root README."]
    (output_dir / "results.md").write_text("\n".join(lines) + "\n")
    print(comparison.to_string(index=False), flush=True)
    print(f"Saved separate evaluation/deployment models and {len(predictions)} predictions under {output_dir}", flush=True)
    return metrics