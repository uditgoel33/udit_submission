"""Tests for category-grouped holdout and cross-validation."""

import numpy as np
import pandas as pd
import unittest

from incidence.splits import (
    make_group_folds, make_holdout_split, make_representative_holdout_split,
    representative_balance_table,
)


def make_sample_data(category_count: int = 40) -> pd.DataFrame:
    """Create repeated category rows with varied targets and geography profiles."""
    rows = []
    for category_index in range(category_count):
        category_key = f"category-{category_index}"
        incidence = 0.15 + 0.7 * category_index / (category_count - 1)
        geographies = ["Australia", "United States"] if category_index % 2 == 0 else ["Australia"]
        for geography in geographies:
            rows.append(
                {
                    "category_key": category_key,
                    "geography_name": geography,
                    "percentage_3": incidence,
                    "sector": f"sector-{category_index % 2}",
                }
            )
    return pd.DataFrame(rows)


def assert_categories_are_disjoint(
    data: pd.DataFrame, train_indices: np.ndarray, test_indices: np.ndarray
) -> None:
    """Ensure a category never crosses a train/validation boundary."""
    train_groups = set(data.iloc[train_indices]["category_key"])
    test_groups = set(data.iloc[test_indices]["category_key"])
    assert train_groups.isdisjoint(test_groups)


class SplitTests(unittest.TestCase):
    """Check that grouping prevents category leakage in every split."""

    def test_holdout_keeps_category_rows_together(self) -> None:
        """The held-out category set must be absent from development rows."""
        data = make_sample_data()
        development, holdout = make_holdout_split(data, n_splits=5, seed=17)
        assert_categories_are_disjoint(data, development, holdout)
        self.assertEqual(len(development) + len(holdout), len(data))

    def test_repeated_folds_assign_every_row_once_per_repeat(self) -> None:
        """Each repetition should validate every row once without group leakage."""
        data = make_sample_data()
        folds = make_group_folds(data, n_splits=5, repeats=2, seed=21)
        validation_indices = [validation for _, validation in folds]
        counts = np.bincount(np.concatenate(validation_indices), minlength=len(data))
        self.assertTrue(np.all(counts == 2))
        for train_indices, test_indices in folds:
            assert_categories_are_disjoint(data, train_indices, test_indices)

    def test_representative_holdout_balances_countries_and_never_uses_targets(self):
        """The representative split is deterministic from country/taxonomy only and keeps groups intact."""
        rows = []
        countries = ["Australia", "United States", "New Zealand", "United Kingdom", "Canada"]
        sectors = ["Food", "Home", "Services"]
        for category_index in range(45):
            country = countries[category_index % len(countries)]
            sector = sectors[category_index % len(sectors)]
            rows.append({"category_key": f"cat-{category_index}", "category_name": f"cat-{category_index}",
                         "geography_name": country, "percentage_3": (category_index % 9) / 10,
                         "tax_sector": sector, "tax_subsector": f"sub-{category_index % 6}"})
        data = pd.DataFrame(rows)
        development, holdout, summary = make_representative_holdout_split(
            data, holdout_fraction=0.2, seed=17,
        )
        changed_targets = data.copy()
        changed_targets["percentage_3"] = np.linspace(0, 1, len(data))
        second_development, second_holdout, _ = make_representative_holdout_split(
            changed_targets, holdout_fraction=0.2, seed=17,
        )
        np.testing.assert_array_equal(holdout, second_holdout)
        np.testing.assert_array_equal(development, second_development)
        assert_categories_are_disjoint(data, development, holdout)
        self.assertEqual(len(development) + len(holdout), len(data))
        self.assertEqual(set(summary["countries_covered_in_both"]), set(countries))
        balance = representative_balance_table(data, development, holdout)
        country_balance = balance.loc[balance["dimension"].eq("geography_name")]
        self.assertTrue((country_balance["holdout_share"] > 0).all())
        self.assertTrue((country_balance["development_share"] > 0).all())
