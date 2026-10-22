"""Checks for category leakage, target-blind prompts and neighbor feature lineage."""

import unittest

import numpy as np
import pandas as pd

from incidence.retrieval import NeighborChoice, candidate_table, choose_neighbor, match_prompt


class RetrievalTests(unittest.TestCase):
    def make_rows(self):
        return pd.DataFrame([{
            "source_row_id": index, "category_name": f"Category {index}",
            "geography_name": "Australia", "question_text": f"Have you purchased item {index}?",
            "qualifying_options": "Yes", "disqualifying_options": "No",
            "percentage_3": .123456789 + index / 10,
        } for index in range(7)])

    def test_top_five_exclude_self_category_and_respect_reference_pool(self):
        data = self.make_rows()
        query = data.iloc[:1].assign(query_id="train:0")
        vectors = np.column_stack([np.ones(7), .1 * np.arange(7)])
        candidates = candidate_table(query, data.iloc[:6], vectors[:1], vectors[:6])
        self.assertEqual(len(candidates), 5)
        self.assertNotIn(0, candidates.candidate_row_id.tolist())
        self.assertNotIn(6, candidates.candidate_row_id.tolist())
        changed = query.assign(percentage_3=.999999)
        pd.testing.assert_frame_equal(candidates, candidate_table(changed, data.iloc[:6], vectors[:1], vectors[:6]))

    def test_prompt_excludes_targets_and_choice_attaches_only_selected_target(self):
        data = self.make_rows()
        query = data.iloc[0].to_dict() | {"query_id": "train:0"}
        candidates = candidate_table(pd.DataFrame([query]), data, np.array([[1., 0.]]), np.ones((7, 2)))
        prompt = match_prompt(query, candidates)
        self.assertNotIn("percentage_3", prompt)
        self.assertNotIn("0.123456789", prompt)
        for field in ("category_name", "geography_name", "cosine_similarity", "candidate_row_id", "percentage_3", "rank"):
            self.assertNotIn(f'"{field}"', prompt)
        changed_candidates = candidates.assign(percentage_3=.999999, weight_3=99999,
                                                category_name="Hidden category", geography_name="Hidden country",
                                                cosine_similarity=-.99, rank=99)
        changed_query = query | {"percentage_3": .999999, "category_name": "Hidden category",
                                 "geography_name": "Hidden country", "base_weight_3": 99999}
        self.assertEqual(prompt, match_prompt(changed_query, changed_candidates))
        def chooser(*args, **kwargs):
            return NeighborChoice(candidate_id=3, reason="Comparable wording")
        result = choose_neighbor(query, candidates, data, chooser)
        selected_id = int(candidates.iloc[2].candidate_row_id)
        self.assertEqual(result["retrieved_incidence"], data.loc[selected_id, "percentage_3"])
        self.assertEqual(result["selected_reference_row_id"], selected_id)

    def test_threshold_is_inclusive_and_fewer_than_five_is_valid(self):
        data = self.make_rows()
        query = data.iloc[:1].assign(query_id="train:0")
        vectors = np.array([[.7, np.sqrt(1 - .7 ** 2)], [.69, np.sqrt(1 - .69 ** 2)]])
        candidates = candidate_table(query, data.iloc[1:3], np.array([[1., 0.]]), vectors, min_similarity=.7)
        self.assertEqual(candidates.candidate_row_id.tolist(), [1])
        self.assertTrue(candidates.cosine_similarity.ge(.7).all())
        self.assertIn("candidate_id", match_prompt(query.iloc[0].to_dict(), candidates))

    def test_below_threshold_returns_null_and_never_calls_llm(self):
        from unittest.mock import Mock
        data = self.make_rows()
        query = data.iloc[:1].assign(query_id="train:0")
        candidates = candidate_table(query, data.iloc[1:], np.array([[1., 0.]]), np.tile([0., 1.], (6, 1)), min_similarity=.7)
        self.assertTrue(candidates.empty)
        chooser = Mock(side_effect=AssertionError("No eligible candidates must skip LLM"))
        result = choose_neighbor(query.iloc[0].to_dict(), candidates, data, chooser)
        self.assertIsNone(result["retrieved_incidence"])
        self.assertIsNone(result["selected_reference_row_id"])
        self.assertEqual(result["retrieval_status"], "no_candidates")
        chooser.assert_not_called()

    def test_invalid_llm_selection_is_rejected(self):
        data = self.make_rows()
        query = data.iloc[0].to_dict() | {"query_id": "train:0"}
        candidates = candidate_table(pd.DataFrame([query]), data, np.array([[1., 0.]]), np.ones((7, 2)))
        def chooser(*args, **kwargs):
            return NeighborChoice(candidate_id=999, reason="Invalid")
        with self.assertRaises(ValueError):
            choose_neighbor(query, candidates, data, chooser)

    def test_no_default_cutoff_and_llm_can_abstain_without_accessing_targets(self):
        data = self.make_rows()
        query = data.iloc[:1].assign(query_id="train:0")
        candidates = candidate_table(query, data.iloc[1:], np.array([[1., 0.]]), np.tile([0., 1.], (6, 1)))
        self.assertEqual(len(candidates), 5)
        self.assertTrue(candidates.cosine_similarity.lt(.7).all())
        def chooser(*args, **kwargs):
            return NeighborChoice(candidate_id=None, reason="None describes comparable behavior")
        result = choose_neighbor(query.iloc[0].to_dict(), candidates, data.drop(columns="percentage_3"), chooser)
        self.assertIsNone(result["retrieved_incidence"])
        self.assertIsNone(result["selected_reference_row_id"])
        self.assertEqual(result["retrieval_status"], "llm_no_relevant_match")


if __name__ == "__main__":
    unittest.main()