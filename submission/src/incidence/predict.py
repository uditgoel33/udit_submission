"""Predict incidence and calibrated intervals from supplied features and a saved model."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from incidence.calibration import prediction_intervals
from incidence.config import OUTPUT_DIR, PROJECT_ROOT
from incidence.models import probability
from incidence.eda_model import prepare_data


def make_predictions(data: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Apply saved preprocessing and interval calibration without fitting or reading outcomes."""
    features = prepare_data(data, require_target=False)
    identifiers = features["prediction_id"]
    if identifiers.isna().any() or identifiers.duplicated().any() or identifiers.astype(str).str.strip().eq("").any():
        raise ValueError("Prediction IDs must be nonempty and unique")
    if features.empty:
        raise ValueError("Prediction input must contain at least one row")
    radius = float(artifact["logit_interval_radius"])
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("Saved interval radius must be finite and nonnegative")
    logits = artifact["model"].predict(features[artifact["features"]])
    lower, upper = prediction_intervals(logits, radius)
    result = pd.DataFrame({"prediction_id": identifiers, "point_estimate": probability(logits),
                           "lower_80": lower, "upper_80": upper})
    values = result[["lower_80", "point_estimate", "upper_80"]].to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Predictions must be finite and within [0, 1]")
    if not (result["lower_80"].le(result["point_estimate"]) & result["point_estimate"].le(result["upper_80"])).all():
        raise ValueError("Prediction intervals must contain their point estimates")
    return result


def run_prediction(input_path: Path, model_path: Path, output_path: Path) -> None:
    """Write one prediction per supplied row using the trusted local training artifact."""
    data = pd.read_csv(input_path)
    artifact = joblib.load(model_path)
    predictions = make_predictions(data, artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False, float_format="%.8f")
    print(f"Model: {artifact['selection']['model']}")
    print(f"Validated predictions: {len(predictions)} rows with unique IDs and ordered intervals.")
    print(f"Output: {output_path}")
    print("No retraining or LLM calls. Intervals use approximate development-OOF calibration.")


def main() -> None:
    """Expose target-free prediction for the supplied enriched prediction set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "features/prediction_features.csv")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models/deployment_model.joblib")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "predictions.csv")
    arguments = parser.parse_args()
    run_prediction(arguments.input, arguments.model, arguments.output)


if __name__ == "__main__":
    main()