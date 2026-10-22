"""Independent question-profile labels and repeatability reporting tests."""

import unittest
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
from pydantic import ValidationError

from incidence.llm.fs_profile import (
    PROFILE_OUTPUT_FIELDS, TYPICAL_FREQUENCY_MONTHS, QuestionProfile,
    build_profile_features, build_profile_prompt, profile_reliability,
    refresh_profile_time_columns,
)
from incidence.llm.questions import text_hash


def question(question_id, text="Question text"):
    """Make a raw question keyed only by its exact screening wording."""
    return {"question_id": question_id, "question_text": text,
            "qualifying_options": "Yes | No, but open to it", "disqualifying_options": "No"}


def profile(frequency="yearly", action="purchase"):
    """Return a complete schema-valid classification."""
    return QuestionProfile(action=action, eligibility_gate="none", frequency_bucket=frequency,
                           price_band="20_to_200", answer_format="yes_no",
                           notes="Recent purchase is the main qualifying behaviour.")


class ProfileFeatureTests(unittest.TestCase):
    """Fixed labels, exact prompt inputs and country-invariant question caching."""

    def test_profile_schema_has_only_anchored_categories(self):
        """No free numeric rating can enter any profile field."""
        schema = QuestionProfile.model_json_schema()
        self.assertEqual(schema["properties"]["action"]["enum"],
                         ["purchase", "use", "visit", "donate", "status", "intent"])
        for field in ("action", "eligibility_gate", "frequency_bucket", "price_band", "answer_format"):
            self.assertIn("enum", schema["properties"][field])
        with self.assertRaises(ValidationError):
            QuestionProfile(action=0.7, eligibility_gate="none", frequency_bucket="monthly",
                            price_band="20_to_200", answer_format="yes_no", notes="A note.")

    def test_prompt_uses_three_raw_fields_only(self):
        """Question content is text-only; no geography, target or taxonomy is interpolated."""
        fields = question("question-1")
        prompt = build_profile_prompt(fields)
        self.assertIn(fields["question_text"], prompt)
        self.assertIn(fields["qualifying_options"], prompt)
        self.assertIn(fields["disqualifying_options"], prompt)
        self.assertIn("monthly            = roughly every 3-6 weeks", prompt)
        self.assertNotIn("question-1", prompt)
        self.assertNotIn("percentage_3", prompt)
        with self.assertRaises(TypeError):
            build_profile_prompt({**fields, "geography_name": "Australia"}, llm_sector="old label")

    def test_frequency_anchors_match_requested_midpoints(self):
        """The numeric month feature is derived in code from the fixed bucket mapping."""
        self.assertEqual(TYPICAL_FREQUENCY_MONTHS["weekly_or_more"], 0.25)
        self.assertEqual(TYPICAL_FREQUENCY_MONTHS["monthly"], 1)
        self.assertEqual(TYPICAL_FREQUENCY_MONTHS["every_few_months"], 3)
        self.assertEqual(TYPICAL_FREQUENCY_MONTHS["yearly"], 12)
        self.assertEqual(TYPICAL_FREQUENCY_MONTHS["every_few_years"], 36)
        self.assertEqual(TYPICAL_FREQUENCY_MONTHS["rarely"], 120)
        self.assertTrue(np.isnan(TYPICAL_FREQUENCY_MONTHS["not_repeating"]))

    def test_builder_labels_once_per_unique_question_and_writes_expected_fields(self):
        """Identical screener text shares one profile even if it appears in multiple countries upstream."""
        questions = pd.DataFrame([
            question("question-1", "In the past 3 months, have you bought this?"),
            question("question-2", "In the past 2 years, have you done this?"),
        ])
        answers = [profile("yearly"), profile("not_repeating", "status")]
        with patch("incidence.llm.fs_profile.ask_many", return_value=answers) as ask:
            result = build_profile_features(questions)
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(ask.call_args.args[0], "profile")
        self.assertEqual(ask.call_args.kwargs["temperature"], 0.0)
        self.assertEqual(ask.call_args.kwargs["sample_index"], 0)
        self.assertEqual(result.columns.tolist(), ["question_id", *PROFILE_OUTPUT_FIELDS])
        self.assertEqual(result["prof_gate"].tolist(), ["none", "none"])
        self.assertEqual(result["freq_months"].tolist(), [3, 24])
        self.assertEqual(result["typical_frequency_months"].iloc[0], 12)
        self.assertTrue(np.isnan(result["typical_frequency_months"].iloc[1]))

    def test_reliability_uses_seeded_sample_and_compares_against_first_pass(self):
        """The 60-question repeat requests sample 1 at 0.7 and flags agreement below 80%."""
        questions = pd.DataFrame([question(f"question-{index}") for index in range(80)])
        first_pass = pd.DataFrame([{"question_id": row["question_id"], "prof_action": "purchase",
                                    "prof_gate": "none", "prof_frequency": "yearly",
                                    "prof_price_band": "20_to_200", "prof_answer_format": "yes_no"}
                                   for row in questions.to_dict("records")])
        repeats = [profile("yearly", "purchase") for _ in range(60)]
        repeats[0] = profile("yearly", "use")
        with patch("incidence.llm.fs_profile.ask_many", return_value=repeats) as ask:
            agreement, details = profile_reliability(questions, first_pass, seed=2026)
        self.assertEqual(ask.call_count, 1)
        self.assertEqual(ask.call_args.kwargs["temperature"], 0.7)
        self.assertEqual(ask.call_args.kwargs["sample_index"], 1)
        self.assertTrue((ask.call_args.args[1] == profile_prompt_values(ask)).all())
        action = agreement.loc[agreement["field"].eq("prof_action")].iloc[0]
        self.assertAlmostEqual(action["agreement"], 59 / 60)
        self.assertFalse(action["below_80pct"])
        self.assertEqual(len(details), 60)
        self.assertTrue({"prof_action_first", "prof_action_repeat"}.issubset(details.columns))

    def test_profile_merge_appends_to_taxonomy_data_without_question_id(self):
        """Labels fan out to repeated country rows without changing the source data."""
        from incidence.llm.fs_profile import _profile_record
        script = Path(__file__).resolve().parents[1] / "scripts" / "build_llm_features.py"
        append_existing_profiles = runpy.run_path(str(script))["append_existing_profiles"]

        base = pd.DataFrame([{
            "category_name": "Example", "geography_name": country, "wave_date": "2024-09-01",
            "question_text": "In the past 2 years, have you bought this?", "qualifying_options": "Yes", "disqualifying_options": "No",
            "percentage_3": 0.5, "tax_sector": "Home", "tax_subsector": "Appliances",
        } for country in ("Australia", "United States")])
        question_id = text_hash(["In the past 2 years, have you bought this?", "Yes", "No"])
        saved_profiles = pd.DataFrame([_profile_record(question_id, "In the past 2 years, have you bought this?", profile("rarely"))])
        with TemporaryDirectory() as directory:
            root = Path(directory)
            taxonomy_path, profiles_path = root / "taxonomy.csv", root / "profiles.csv"
            original_path = root / "original.csv"
            base.to_csv(taxonomy_path, index=False)
            base.to_csv(original_path, index=False)
            original_bytes = original_path.read_bytes()
            saved_profiles.to_csv(profiles_path, index=False)
            pd.DataFrame({"question_id": [question_id],
                          "question_text": ["In the past 2 years, have you bought this?"]}).to_csv(
                              root / "questions.csv", index=False)
            append_existing_profiles(taxonomy_path, profiles_path)
            refreshed_profiles = pd.read_csv(profiles_path)
            merged = pd.read_csv(taxonomy_path)
            self.assertEqual(len(merged), len(base))
            self.assertEqual(merged["prof_action"].tolist(), ["purchase", "purchase"])
            self.assertEqual(merged["freq_months"].tolist(), [24, 24])
            self.assertEqual(merged["typical_frequency_months"].tolist(), [120, 120])
            self.assertEqual(refreshed_profiles["freq_months"].tolist(), [24])
            self.assertEqual(refreshed_profiles["typical_frequency_months"].tolist(), [120])
            self.assertEqual(merged["tax_sector"].tolist(), ["Home", "Home"])
            self.assertNotIn("question_id", merged.columns)
            pd.testing.assert_frame_equal(merged[base.columns], base)
            self.assertEqual(original_path.read_bytes(), original_bytes)


def profile_prompt_values(mocked_ask):
    """Recover prompt order to assert the random sample is deterministic across runs."""
    return pd.Series(mocked_ask.call_args.args[1])


if __name__ == "__main__":
    unittest.main()