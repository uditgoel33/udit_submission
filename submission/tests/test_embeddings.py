"""Tests for semantic-safe cleaning and embedding cache alignment."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from incidence.embeddings import build_text_table, clean_record, embed_texts


class EmbeddingTests(unittest.TestCase):
    def test_cleaning_preserves_action_notes_and_time_without_category_prefix(self):
        record = clean_record("Premium Toilet Paper", "In the past 3 months have you purchased premium toilet paper? (Note: eco-friendly only)", "Yes | No, but I am open or likely to in the near future")
        self.assertTrue(record["cleaned_question"].startswith("purchased premium"))
        self.assertIn("eco-friendly only", record["embedding_text"])
        self.assertIn("3 months", record["recall_context"])
        self.assertNotIn("Category:", record["embedding_text"])
        self.assertNotIn("note:", record["embedding_text"].lower())
        self.assertNotIn("No, but I am open or likely", record["embedding_text"])
        self.assertNotIn("Qualifying answers:", record["embedding_text"])
        self.assertEqual(record["cleaned_qualifying_options"], "")
        self.assertIn("No, but I am open or likely", record["removed_standard_options"])

    def test_category_changes_do_not_change_embedding_and_empty_input_is_flagged(self):
        first = clean_record("Category Alpha", "Have you purchased this product?", "Yes")
        second = clean_record("Category Beta", "Have you purchased this product?", "Yes")
        self.assertEqual(first["embedding_text_id"], second["embedding_text_id"])
        self.assertNotIn("Category Alpha", first["embedding_text"])
        empty = clean_record("Category Alpha", "Have you/are you...", "Yes")
        self.assertTrue(empty["empty_after_cleaning"])
        self.assertTrue(empty["embedding_text"])
        self.assertNotIn("Category Alpha", empty["embedding_text"])

    def test_generic_stem_removed_without_losing_qualifying_semantics(self):
        record = clean_record("Peanut Butter", "Which of the following best describes you?", "I do some grocery shopping but have not purchased peanut butter | I intend to buy peanut butter")
        self.assertEqual(record["cleaned_question"], "")
        self.assertIn("have not purchased", record["embedding_text"])
        self.assertIn("intend to buy", record["embedding_text"])

    def test_single_year_and_repeated_auxiliary_stems_are_cleaned(self):
        record = clean_record("Electronics", "In the past year have you bought electronics?", "Yes")
        self.assertEqual(record["cleaned_question"], "bought electronics")
        self.assertEqual(record["recall_context"], "past year")
        self.assertEqual(clean_record("Furniture", "Have you/are you...", "Yes")["cleaned_question"], "")
        self.assertEqual(clean_record("A", "Have you not bought A?", "Yes")["cleaned_question"], "not bought A")

    def test_option_order_does_not_change_the_embedding_input(self):
        first = clean_record("A", "Have you bought A?", "Yes | No, but I would consider it")
        second = clean_record("A", "Have you bought A?", "No, but I would consider it | Yes")
        self.assertEqual(first["embedding_text_id"], second["embedding_text_id"])

    def test_only_complete_generic_answers_are_removed(self):
        record = clean_record("Electric Cars", "Have you purchased an electric car?",
                              "Yes | No, but I would consider purchasing in the near future | No, but I would consider buying an electric car")
        self.assertEqual(record["cleaned_qualifying_options"], "No, but I would consider buying an electric car")
        self.assertNotIn("No, but I would consider purchasing in the near future", record["embedding_text"])
        generic = clean_record("A", "Have you bought A?", "Yes | No, but would be open to these products in the future")
        self.assertEqual(generic["cleaned_qualifying_options"], "")

    def test_target_free_tables_keep_row_ids_and_reject_missing_text(self):
        data = pd.DataFrame({"category_name": ["A", "A"], "question_text": ["Have you bought A?", "Have you bought A?"],
                             "qualifying_options": ["Yes", "Yes"], "percentage_3": [.1, .9]})
        table = build_text_table(data, "train")
        self.assertEqual(table.source_row_id.tolist(), [0, 1])
        self.assertEqual(table.embedding_text_id.nunique(), 1)
        self.assertNotIn("percentage_3", table)
        with self.assertRaises(ValueError):
            clean_record("A", None, "Yes")

    def test_embeddings_deduplicate_preserve_order_and_reuse_cache(self):
        client = Mock()
        client.embeddings.create.return_value = SimpleNamespace(
            data=[SimpleNamespace(index=1, embedding=[0., 1.]), SimpleNamespace(index=0, embedding=[1., 0.])],
            usage=SimpleNamespace(total_tokens=8),
        )
        with TemporaryDirectory() as directory:
            matrix, summary = embed_texts(["A", "B", "A"], "test-model", Path(directory), client)
            np.testing.assert_array_equal(matrix, [[1, 0], [0, 1], [1, 0]])
            self.assertEqual(summary["api_calls"], 1)
            repeated, summary = embed_texts(["B", "A"], "test-model", Path(directory), client)
            np.testing.assert_array_equal(repeated, [[0, 1], [1, 0]])
            self.assertEqual(summary["cache_hits"], 2)
            self.assertEqual(client.embeddings.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()