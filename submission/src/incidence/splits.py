"""Leakage-safe grouped holdout and cross-validation splits."""

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from sklearn.model_selection import StratifiedGroupKFold

from incidence.config import (
    CV_FOLDS,
    CV_REPEATS,
    GEOGRAPHY_COLUMN,
    GROUP_COLUMN,
    HOLDOUT_FOLDS,
    RANDOM_SEED,
    TARGET_COLUMN,
)


def _make_stratification_labels(data: pd.DataFrame) -> pd.Series:
    """Build category-level bins so folds have similar targets and geography mix."""
    required = {GROUP_COLUMN, GEOGRAPHY_COLUMN, TARGET_COLUMN}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Data is missing split columns: {missing}")
    if data[GROUP_COLUMN].isna().any():
        raise ValueError("Category groups must not be missing")

    grouped = data.groupby(GROUP_COLUMN, sort=False)
    category_summary = grouped[TARGET_COLUMN].mean().to_frame("mean_incidence")
    category_summary["geo_profile"] = grouped[GEOGRAPHY_COLUMN].agg(
        lambda geographies: "multi" if geographies.nunique() > 1 else "single"
    )
    if "sector" in data.columns:
        category_summary["sector"] = grouped["sector"].agg(
            lambda values: values.mode().iloc[0] if not values.mode().empty else "unknown"
        )
    else:
        category_summary["sector"] = "unassigned"

    clipped_rate = category_summary["mean_incidence"].clip(0.001, 0.999)
    logit_rate = np.log(clipped_rate / (1 - clipped_rate))
    category_summary["target_bin"] = pd.qcut(
        logit_rate.rank(method="first"), q=4, labels=False, duplicates="drop"
    ).astype(str)
    category_summary["stratum"] = (
        category_summary["target_bin"]
        + "|" + category_summary["sector"].astype(str)
        + "|" + category_summary["geo_profile"]
    )
    return data[GROUP_COLUMN].map(category_summary["stratum"])


def make_holdout_split(
    data: pd.DataFrame,
    n_splits: int = HOLDOUT_FOLDS,
    seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one stratified category-group holdout and its development rows.

    Grouping prevents duplicate and cross-geography category rows from leaking
    between development and holdout data.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    labels = _make_stratification_labels(data)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    development_indices, holdout_indices = next(
        splitter.split(data, labels, groups=data[GROUP_COLUMN])
    )
    print(
        f"Grouped holdout: {len(holdout_indices)} rows; "
        f"development: {len(development_indices)} rows"
    )
    return development_indices, holdout_indices


def _total_variation(full: pd.Series, subset: pd.Series) -> float:
    """Compare categorical shares with total-variation distance, treating absent levels as zero."""
    full_share = full.value_counts(normalize=True)
    subset_share = subset.value_counts(normalize=True)
    levels = full_share.index.union(subset_share.index)
    return float((full_share.reindex(levels, fill_value=0) -
                  subset_share.reindex(levels, fill_value=0)).abs().sum() / 2)


def _category_profile(data: pd.DataFrame) -> pd.DataFrame:
    """Give each normalized category one stable modal sector/subsector for balance summaries."""
    def stable_mode(values: pd.Series) -> str:
        """Resolve label ties alphabetically without consulting target values."""
        modes = values.astype(str).mode()
        return sorted(modes.tolist())[0]

    return data.groupby(GROUP_COLUMN, sort=False).agg(
        tax_sector=("tax_sector", stable_mode),
        tax_subsector=("tax_subsector", stable_mode),
    ).reset_index()


def make_representative_holdout_split(
    data: pd.DataFrame,
    holdout_fraction: float = 0.15,
    seed: int = RANDOM_SEED,
    candidates: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Optimize a category-disjoint holdout for representative country/taxonomy shares without targets."""
    required = {GROUP_COLUMN, GEOGRAPHY_COLUMN, "tax_sector", "tax_subsector"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Representative split is missing non-target balance columns: {missing}")
    if not 0 < holdout_fraction < 0.5:
        raise ValueError("Holdout fraction must be in (0, 0.5)")
    if data[list(required)].isna().any().any():
        raise ValueError("Representative split columns must not be missing")
    group_data = _category_profile(data).set_index(GROUP_COLUMN)
    group_keys = group_data.index.tolist()
    group_count = len(group_keys)
    group_index = {key: index for index, key in enumerate(group_keys)}
    group_rows = data.groupby(GROUP_COLUMN, sort=False).size().reindex(group_keys).to_numpy(dtype=float)
    countries = sorted(data[GEOGRAPHY_COLUMN].unique())
    country_row_matrix = pd.crosstab(data[GEOGRAPHY_COLUMN], data[GROUP_COLUMN]).reindex(
        index=countries, columns=group_keys, fill_value=0,
    ).to_numpy(dtype=float)
    country_membership_matrix = (country_row_matrix > 0).astype(float)
    sectors = sorted(group_data["tax_sector"].unique())
    subsectors = sorted(group_data["tax_subsector"].unique())
    sector_matrix = np.vstack([group_data["tax_sector"].eq(value).to_numpy(float) for value in sectors])
    subsector_matrix = np.vstack([group_data["tax_subsector"].eq(value).to_numpy(float) for value in subsectors])
    supported_subsector_count = int((subsector_matrix.sum(axis=1) >= 2).sum())

    # Each balance row gets positive/negative deviation variables around its target share.
    balance_vectors = [np.ones(group_count), group_rows, *country_row_matrix, *sector_matrix, *subsector_matrix]
    balance_targets = [holdout_fraction * group_count, holdout_fraction * len(data)]
    balance_targets += [holdout_fraction * data[GEOGRAPHY_COLUMN].eq(country).sum() for country in countries]
    balance_targets += [holdout_fraction * row.sum() for row in sector_matrix]
    balance_targets += [holdout_fraction * row.sum() for row in subsector_matrix]
    weights = [1.0 / group_count, 5.0 / len(data)]
    weights += [12.0 / max(data[GEOGRAPHY_COLUMN].eq(country).sum(), 1) for country in countries]
    weights += [5.0 / max(row.sum(), 1) for row in sector_matrix]
    weights += [2.0 / max(row.sum(), 1) for row in subsector_matrix]
    balance_count = len(balance_vectors)
    variable_count = group_count + 2 * balance_count
    objective = np.zeros(variable_count)
    for index, weight in enumerate(weights):
        objective[group_count + index] = weight
        objective[group_count + balance_count + index] = weight
    objective[:group_count] = np.random.default_rng(seed).uniform(0, 1e-9, group_count)

    # Coverage constraints ensure every geography and every repeatable taxonomy node appears on both sides.
    coverage_vectors, coverage_lower, coverage_upper = [], [], []
    for matrix in (country_membership_matrix, sector_matrix, subsector_matrix):
        for vector in matrix:
            number_of_groups = int(vector.sum())
            if number_of_groups >= 2:
                coverage_vectors.append(vector)
                coverage_lower.append(1)
                coverage_upper.append(number_of_groups - 1)
    constraint_count = balance_count + 1 + len(coverage_vectors)
    constraint_matrix = lil_matrix((constraint_count, variable_count), dtype=float)
    lower = np.full(constraint_count, -np.inf)
    upper = np.full(constraint_count, np.inf)
    for index, (vector, target) in enumerate(zip(balance_vectors, balance_targets)):
        constraint_matrix[index, :group_count] = vector
        constraint_matrix[index, group_count + index] = -1
        constraint_matrix[index, group_count + balance_count + index] = 1
        lower[index] = upper[index] = target
    group_target = holdout_fraction * group_count
    constraint_matrix[balance_count, :group_count] = 1
    lower[balance_count] = np.floor(group_target)
    upper[balance_count] = np.ceil(group_target)
    for row_index, (vector, minimum, maximum) in enumerate(
            zip(coverage_vectors, coverage_lower, coverage_upper), start=balance_count + 1):
        constraint_matrix[row_index, :group_count] = vector
        lower[row_index], upper[row_index] = minimum, maximum
    integrality = np.zeros(variable_count, dtype=int)
    integrality[:group_count] = 1
    bounds = Bounds(np.zeros(variable_count),
                    np.concatenate([np.ones(group_count), np.full(2 * balance_count, np.inf)]))
    solution = milp(objective, integrality=integrality, bounds=bounds,
                    constraints=LinearConstraint(constraint_matrix.tocsr(), lower, upper),
                    options={"time_limit": 120, "mip_rel_gap": 0.001})
    if not solution.success or solution.x is None:
        raise RuntimeError(f"Representative holdout optimization failed: {solution.message}")
    selected_groups = {key for index, key in enumerate(group_keys) if solution.x[index] > 0.5}
    is_holdout = data[GROUP_COLUMN].isin(selected_groups).to_numpy()
    holdout_indices, development_indices = np.flatnonzero(is_holdout), np.flatnonzero(~is_holdout)
    development = data.iloc[development_indices]
    holdout = data.iloc[holdout_indices]
    if set(development[GROUP_COLUMN]) & set(holdout[GROUP_COLUMN]):
        raise ValueError("A category group appears in both representative partitions")
    all_pairs = set(zip(data[GROUP_COLUMN], data[GEOGRAPHY_COLUMN]))
    partition_pairs = set(zip(development[GROUP_COLUMN], development[GEOGRAPHY_COLUMN])) | set(
        zip(holdout[GROUP_COLUMN], holdout[GEOGRAPHY_COLUMN])
    )
    if all_pairs != partition_pairs:
        raise ValueError("Every observed category-country pair must be assigned exactly to a partition")
    summary = {"seed": seed, "optimizer": "scipy.optimize.milp",
               "objective_value": float(solution.fun),
               "optimality_gap": float(getattr(solution, "mip_gap", np.nan)),
               "target_used_for_split": False, "holdout_rows": len(holdout),
               "development_rows": len(development), "holdout_groups": int(holdout[GROUP_COLUMN].nunique()),
               "development_groups": int(development[GROUP_COLUMN].nunique()),
               "supported_subsectors": supported_subsector_count,
               "countries_covered_in_both": sorted(set(holdout[GEOGRAPHY_COLUMN]) & set(development[GEOGRAPHY_COLUMN]))}
    if set(summary["countries_covered_in_both"]) != set(data[GEOGRAPHY_COLUMN]):
        raise RuntimeError("Could not represent every geography in both partitions")
    print(f"Representative grouped holdout: {len(holdout)} rows/{summary['holdout_groups']} categories; "
          f"development: {len(development)} rows/{summary['development_groups']} categories; "
            f"balance objective={summary['objective_value']:.4f}")
    return development_indices, holdout_indices, summary


def representative_balance_table(data: pd.DataFrame, development_indices: np.ndarray,
                                 holdout_indices: np.ndarray) -> pd.DataFrame:
    """Report overall, country, sector and subsector shares for the selected partition."""
    development, holdout = data.iloc[development_indices], data.iloc[holdout_indices]
    records = []
    for feature, partition in ((GEOGRAPHY_COLUMN, "rows"), ("tax_sector", "categories"),
                               ("tax_subsector", "categories")):
        full_values = data[feature] if partition == "rows" else _category_profile(data)[feature]
        development_values = development[feature] if partition == "rows" else _category_profile(development)[feature]
        holdout_values = holdout[feature] if partition == "rows" else _category_profile(holdout)[feature]
        full_shares = full_values.value_counts(normalize=True)
        development_shares = development_values.value_counts(normalize=True)
        holdout_shares = holdout_values.value_counts(normalize=True)
        for value in full_shares.index:
            records.append({"dimension": feature, "value": str(value), "unit": partition,
                            "all_share": float(full_shares[value]),
                            "development_share": float(development_shares.get(value, 0)),
                            "holdout_share": float(holdout_shares.get(value, 0)),
                            "holdout_minus_all_pp": 100 * (holdout_shares.get(value, 0) - full_shares[value])})
    return pd.DataFrame(records)


def make_group_folds(
    data: pd.DataFrame,
    n_splits: int = CV_FOLDS,
    repeats: int = CV_REPEATS,
    seed: int = RANDOM_SEED,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create repeated stratified folds while keeping every category intact."""
    if n_splits < 2 or repeats < 1:
        raise ValueError("n_splits must be at least 2 and repeats must be positive")
    labels = _make_stratification_labels(data)
    groups = data[GROUP_COLUMN]
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for repeat in range(repeats):
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat
        )
        folds.extend(splitter.split(data, labels, groups=groups))
    print(f"Grouped CV folds: {len(folds)} ({n_splits} folds × {repeats} repeats)")
    return folds
