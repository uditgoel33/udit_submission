"""Five-band population estimates use fixed classes and explicit denominators."""

import unittest
from unittest.mock import patch

import pandas as pd
from pydantic import ValidationError

from incidence.llm.fs_estimate import (
    CLASS_ORDER, ESTIMATE_OUTPUT_FIELDS, PopulationEstimate,
    build_estimate_features, build_estimate_prompt, build_estimate_questions,
    merge_estimate_features, summarize_samples,
)


def sample(eligible="medium", active="high", opened="very_low", total="medium", basis="Market context."):
    """Build one valid class-only response for unit tests."""
    return PopulationEstimate(eligible_class=eligible, active_of_eligible_class=active,
                             open_of_rest_class=opened, direct_total_class=total, basis=basis)


def category(name, country, text="In the past 2 years have you purchased this?"):
    """Build raw category/question data without any targets or precomputed labels."""
    return {"category_name": name, "geography_name": country, "question_text": text,
            "qualifying_options": "Yes | No, but open or likely", "disqualifying_options": "No"}


class EstimateFeatureTests(unittest.TestCase):
    """Check the distinct denominators, exact allowed classes, five-sample mode and row merge."""

    def test_schema_only_allows_the_five_classes(self):
        """Population fields reject numeric estimates and other labels."""
        schema = PopulationEstimate.model_json_schema()
        for field in ("eligible_class", "active_of_eligible_class", "open_of_rest_class", "direct_total_class"):
            self.assertEqual(schema["properties"][field]["enum"], list(CLASS_ORDER))
        with self.assertRaises(ValidationError):
            PopulationEstimate(eligible_class=0.4, active_of_eligible_class="high",
                               open_of_rest_class="very_low", direct_total_class="low", basis="Basis.")
        with self.assertRaises(ValidationError):
            PopulationEstimate(eligible_class="very_high", active_of_eligible_class="high",
                               open_of_rest_class="very_low", direct_total_class="low",
                               basis="Basis.", estimate_pct=50)

    def test_prompt_defines_distinct_population_denominators(self):
        """Eligibility uses all adults; active is conditional on eligibility; future openness is the rest."""
        fields = {"question_text": "Question text", "qualifying_options": "Yes",
                  "disqualifying_options": "No", "geography_name": "Australia",
                  "gender_line": ""}
        prompt = build_estimate_prompt(fields)
        self.assertIn("% of ALL adults", prompt)
        self.assertIn("among ONLY the eligible adults", prompt)
        self.assertIn("ALL adults NOT already counted", prompt)
        self.assertIn("very_low = 0% to less than 20%", prompt)
        for forbidden in ("percentage_3", "weight_3", "tax_sector", "prof_action"):
            self.assertNotIn(forbidden, prompt)
        self.assertIn("Only female respondents are asked.", build_estimate_prompt({**fields, "gender_line": "Only female respondents are asked."}))
        with self.assertRaisesRegex(ValueError, "Gender line"):
            build_estimate_prompt({**fields, "gender_line": "Old label"})

    def test_gender_restrictions_create_distinct_population_keys(self):
        """Same wording and country uses separate estimate IDs for all-adult and female-only denominators."""
        data = pd.DataFrame([category("Skincare (Both Genders)", "United States"),
                             category("Skincare (Female Only)", "United States")])
        questions = build_estimate_questions(data)
        self.assertEqual(len(questions), 2)
        self.assertEqual(questions["question_geo_id"].nunique(), 1)
        self.assertEqual(questions["estimate_id"].nunique(), 2)
        self.assertEqual(set(questions["gender_line"]), {"", "Only female respondents are asked."})

    def test_five_sample_mode_and_ties_use_fixed_class_order(self):
        """The selected class is the modal vote and agreement is a deterministic vote share."""
        results = summarize_samples("internal-estimate-id", [
            sample("high"), sample("medium"), sample("high"), sample("medium"), sample("high"),
        ])
        self.assertEqual(results["est_eligible_class"], "high")
        self.assertEqual(results["est_eligible_class_agreement"], 0.6)
        self.assertFalse(results["est_eligible_class_tie"])
        tied = summarize_samples("internal-estimate-id", [
            sample("low"), sample("medium"), sample("low"), sample("medium"), sample("high"),
        ])
        self.assertEqual(tied["est_eligible_class"], "low")
        self.assertTrue(tied["est_eligible_class_tie"])
        self.assertNotIn("est_structured_total", results)

    def test_builder_requests_five_independent_sample_indices(self):
        """Each unique question-country prompt gets exactly five cached structured readings."""
        row = {"question_geo_id": "same-wording-au", "estimate_id": "estimate-1",
               **category("Example", "Australia"), "gender_line": ""}
        responses = [[sample()]] * 5
        with patch("incidence.llm.fs_estimate.ask_many", side_effect=responses) as ask:
            result = build_estimate_features(pd.DataFrame([row]))
        self.assertEqual(ask.call_count, 5)
        self.assertEqual([call.kwargs["sample_index"] for call in ask.call_args_list], list(range(5)))
        self.assertTrue(all(call.kwargs["temperature"] == 0.7 for call in ask.call_args_list))
        self.assertEqual(len(result), 1)
        self.assertEqual(result.loc[0, "est_eligible_class"], "medium")
        self.assertNotIn("question_geo_id", ESTIMATE_OUTPUT_FIELDS)

    def test_estimates_merge_into_same_file_without_internal_ids(self):
        """Class labels fan out per population scope while source columns and rows stay intact."""
        data = pd.DataFrame([category("Skincare (Both Genders)", "United States"),
                             category("Skincare (Female Only)", "United States")])
        data["tax_sector"], data["tax_subsector"] = "Health & beauty", "Personal care"
        questions = build_estimate_questions(data)
        estimates = pd.DataFrame([summarize_samples(row.estimate_id, [sample("low"), sample("low"), sample("low"), sample("low"), sample("low")])
                                  for row in questions.itertuples()])
        merged = merge_estimate_features(data, estimates)
        self.assertEqual(len(merged), len(data))
        self.assertEqual(merged["est_eligible_class"].tolist(), ["low", "low"])
        self.assertNotIn("question_geo_id", merged)
        self.assertNotIn("estimate_id", merged)
        pd.testing.assert_frame_equal(merged[data.columns], data)


if __name__ == "__main__":
    unittest.main()