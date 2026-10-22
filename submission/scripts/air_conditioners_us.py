"""Score Air Conditioners with the saved model, changing only the country from AU to US."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import joblib
import pandas as pd

from incidence.predict import make_predictions


def compare_countries(features: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Score the same screener in two countries, explicitly retaining AU estimate labels."""
    australian = features.loc[features["category_name"].eq("Air Conditioners")
                              & features["geography_name"].eq("Australia")].copy()
    if len(australian) != 1:
        raise ValueError("Country comparison expects one Australian Air Conditioners row")
    scenarios = pd.concat([
        australian.assign(prediction_id="AC_AU"),
        australian.assign(prediction_id="AC_US", geography_name="United States"),
    ], ignore_index=True)
    predictions = make_predictions(scenarios, artifact)
    predictions.insert(1, "geography_name", scenarios["geography_name"])
    return predictions


def main() -> None:
    """Report model scores, country-only difference and observed AU incidence for context."""
    features = pd.read_csv(PROJECT_ROOT / "features/training_features.csv")
    directory = PROJECT_ROOT / "evaluation"
    artifact = joblib.load(PROJECT_ROOT / "models/deployment_model.joblib")
    predictions = compare_countries(features, artifact)
    print(f"Saved model: {artifact['selection']['model']}")
    print("Country-only scenario: original recall/frequency and AU population labels are retained; country interactions are rebuilt.")
    print("The same text-derived neighbor incidence is retained in both scenarios; no US-specific retrieval or labels are generated.")
    print(predictions.to_string(index=False))
    difference = predictions.iloc[1]["point_estimate"] - predictions.iloc[0]["point_estimate"]
    observed = features.loc[features["category_name"].eq("Air Conditioners")
                            & features["geography_name"].eq("Australia"), "percentage_3"].iloc[0]
    print(f"Country-only US minus AU model difference: {100*difference:+.2f} percentage points.")
    print(f"Observed AU incidence (context only, not a prediction input): {100*observed:.2f}%.")
    print("Use the same two-year purchase/future-intent screener; these are not AC ownership rates.")
    print("Low-to-moderate trust: population labels still describe the Australian screener; no US AC outcome validates this transfer scenario.")
    print("Nominal 80% intervals use the existing approximate OOF calibration, not US-AC-specific calibration.")
    print("No retraining, observed-AU anchoring, submission changes or LLM calls.")
    predictions.to_csv(directory / "air_conditioners_country_scenario.csv", index=False)


if __name__ == "__main__":
    main()