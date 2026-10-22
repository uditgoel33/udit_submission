"""Verify this standalone folder and reproduce its frozen prediction outputs."""

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd

from incidence.predict import make_predictions


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text())
    changed = [name for name, checksum in manifest["sha256"].items()
               if not (ROOT / name).is_file()
               or hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != checksum]
    if changed:
        raise ValueError(f"Packaged files changed or are missing: {changed}")
    raw = pd.read_csv(ROOT / "prediction_set.csv")
    features = pd.read_csv(ROOT / "features/prediction_features.csv")
    predictions = pd.read_csv(ROOT / "predictions.csv")
    model = joblib.load(ROOT / "models/deployment_model.joblib")
    if model["fit_rows"] != 833 or len(model["features"]) != 19:
        raise ValueError("Unexpected selected deployment model")
    pd.testing.assert_frame_equal(features[raw.columns], raw)
    if len(predictions) != 110 or not predictions.prediction_id.is_unique:
        raise ValueError("Submission must contain 110 unique prediction IDs")
    if predictions.columns.tolist() != ["prediction_id", "point_estimate", "lower_80", "upper_80"]:
        raise ValueError("Unexpected submission schema")
    if predictions.prediction_id.tolist() != raw.prediction_id.tolist():
        raise ValueError("Prediction IDs or row order changed")
    recomputed = make_predictions(features, model)
    np.testing.assert_allclose(predictions.iloc[:, 1:], recomputed.iloc[:, 1:], atol=6e-9, rtol=0)
    print(f"Verified {len(manifest['sha256'])} frozen files, all 19 model inputs and all 110 predictions.")
    print("No external project folders, API calls or model fitting required.")


if __name__ == "__main__":
    main()