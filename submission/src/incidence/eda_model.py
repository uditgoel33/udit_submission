"""Frozen final-model features, preprocessing and OOF calibration helpers."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from incidence.config import TARGET_COLUMN
from incidence.data import normalize_category_name
from incidence.models import logit, probability
from incidence.calibration import interval_radius, prediction_intervals, regression_metrics

BASE_NUMERIC = ("freq_months", "typical_frequency_months")


BASE_CATEGORICAL = (
    "geography_name", "tax_sector", "tax_subsector", "prof_action", "prof_gate",
    "prof_frequency", "prof_price_band", "est_eligible_class",
    "est_active_of_eligible_class", "est_open_of_rest_class", "est_direct_total_class",
)


INTERACTION_FEATURES = ("geo_x_sector", "price_x_frequency", "gate_x_eligible_class")


SUBSECTOR_TOTAL_INTERACTION = "subsector_x_direct_total_class"


MODEL_CATEGORICAL = (*BASE_CATEGORICAL, *INTERACTION_FEATURES)


MODEL_FEATURES = (*BASE_NUMERIC, *MODEL_CATEGORICAL)


def prepare_data(data: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Validate the model allowlist and derive interpretable categorical interactions."""
    required = {"category_name", *BASE_NUMERIC, *BASE_CATEGORICAL}
    if require_target:
        required.add(TARGET_COLUMN)
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Augmented data is missing EDA-selected model features: {missing}")
    result = data.copy().reset_index(drop=True)
    if require_target and (result[TARGET_COLUMN].isna().any() or not result[TARGET_COLUMN].between(0, 1).all()):
        raise ValueError("Incidence target must be complete and between zero and one")
    result["category_key"] = result["category_name"].map(normalize_category_name)
    result["sector"] = result["tax_sector"]
    result["geo_x_sector"] = result["geography_name"] + " | " + result["tax_sector"]
    result["price_x_frequency"] = result["prof_price_band"] + " | " + result["prof_frequency"]
    result["gate_x_eligible_class"] = result["prof_gate"] + " | " + result["est_eligible_class"]
    result[SUBSECTOR_TOTAL_INTERACTION] = result["tax_subsector"] + " | " + result["est_direct_total_class"]
    if result[list(BASE_CATEGORICAL)].isna().any().any():
        raise ValueError("Categorical model features must not be missing")
    return result


def build_ridge(alpha: float, features: tuple = MODEL_FEATURES) -> Pipeline:
    """Fit standardized numeric windows and one-hot class labels with rare levels pooled per fold."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=3)),
    ])
    preprocessing = ColumnTransformer([
        ("numeric", numeric, [feature for feature in BASE_NUMERIC if feature in features]),
        ("categorical", categorical, [feature for feature in (*MODEL_CATEGORICAL, SUBSECTOR_TOTAL_INTERACTION)
                           if feature in features]),
    ])
    return Pipeline([("features", preprocessing), ("regression", Ridge(alpha=alpha))])


def calibrate_oof(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calibrate each fold from other categories' OOF errors and add 80% intervals."""
    result = predictions.copy()
    for _, validation in result.groupby("split"):
        other = result.loc[~result["category_key"].isin(validation["category_key"])]
        calibration = other.groupby("row_id", sort=False).agg(
            actual=(TARGET_COLUMN, "first"), predicted_logit=("predicted_logit", "mean"),
            category_key=("category_key", "first"),
        )
        radius = interval_radius(calibration["actual"], calibration["predicted_logit"], calibration["category_key"])
        lower, upper = prediction_intervals(validation["predicted_logit"], radius)
        result.loc[validation.index, "lower_80"] = lower
        result.loc[validation.index, "upper_80"] = upper
    result["point_estimate"] = probability(result["predicted_logit"])
    return result


def score(predictions: pd.DataFrame) -> dict[str, float]:
    """Score probability error and calibrated-interval quality."""
    return regression_metrics(predictions[TARGET_COLUMN], predictions["point_estimate"],
                              predictions["lower_80"], predictions["upper_80"])
