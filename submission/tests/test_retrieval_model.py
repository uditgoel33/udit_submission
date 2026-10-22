"""Checks for context-specific target-derived feature attachment."""

import unittest

import numpy as np
import pandas as pd

from incidence.retrieval_model import attach_retrieval


class RetrievalModelTests(unittest.TestCase):
    def make_data(self):
        references = pd.DataFrame({"category_key": ["a", "b", "c"], "percentage_3": [.2, .7, .4]})
        rows = pd.DataFrame({"query_id": ["train:0", "train:1"],
                             "retrieved_incidence": [.4, np.nan],
                             "selected_reference_row_id": [2., np.nan]})
        return references, rows

    def test_attach_preserves_nulls_and_uses_query_id_not_row_order(self):
        references, rows = self.make_data()
        result = attach_retrieval(references.iloc[:2], rows.iloc[::-1], ["train:0", "train:1"], references, {2})
        self.assertEqual(result.loc[0, "retrieved_incidence"], .4)
        self.assertTrue(pd.isna(result.loc[1, "retrieved_incidence"]))
        self.assertEqual(result.retrieved_incidence_missing.tolist(), [0., 1.])

    def test_rejects_reference_outside_fold_and_incorrect_target(self):
        references, rows = self.make_data()
        with self.assertRaisesRegex(ValueError, "authorized"):
            attach_retrieval(references.iloc[:2], rows, ["train:0", "train:1"], references, {1})
        with self.assertRaisesRegex(ValueError, "does not match"):
            attach_retrieval(references.iloc[:2], rows.assign(retrieved_incidence=[.9, np.nan]),
                             ["train:0", "train:1"], references, {2})

    def test_rejects_same_category_even_with_different_row_id(self):
        references, rows = self.make_data()
        references.loc[2, "category_key"] = "a"
        with self.assertRaisesRegex(ValueError, "query category"):
            attach_retrieval(references.iloc[:2], rows, ["train:0", "train:1"], references, {2})


if __name__ == "__main__":
    unittest.main()