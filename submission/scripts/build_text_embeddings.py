"""Prepare semantic survey text and embed it without changing the finalized model."""

import argparse
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from incidence.config import OPENAI_EMBEDDING_MODEL
from incidence.embeddings import (
    CLEANING_VERSION, STANDARD_OPTION_PATTERN, STEM_PATTERNS, TIME_PATTERN, build_text_table, embed_texts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Write cleaned text and audit metadata without API calls.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/text_embeddings")
    arguments = parser.parse_args()
    training = pd.read_csv(PROJECT_ROOT / "incidence_dataset.csv")
    prediction = pd.read_csv(PROJECT_ROOT / "prediction_set.csv")
    split_path = PROJECT_ROOT / "evaluation/split_assignments.csv"
    if not split_path.exists():
        split_path = PROJECT_ROOT / "outputs/all_estimates_cross_model/split_assignments.csv"
    split = pd.read_csv(split_path)
    if not np.array_equal(split.row_id.to_numpy(), np.arange(len(training))):
        parser.error("Saved split must map every training row in source order")
    pd.testing.assert_frame_equal(split[["category_name", "geography_name"]], training[["category_name", "geography_name"]])
    if prediction.prediction_id.isna().any() or not prediction.prediction_id.is_unique:
        parser.error("Prediction IDs must be nonempty and unique")
    train_text = build_text_table(training, "training")
    train_text["partition"] = split.partition
    prediction_text = build_text_table(prediction, "prediction")
    prediction_text["partition"] = "prediction"
    output = arguments.output_dir
    output.mkdir(parents=True, exist_ok=True)
    train_text.to_csv(output / "training_text.csv", index=False)
    prediction_text.to_csv(output / "prediction_text.csv", index=False)
    combined = pd.concat([train_text, prediction_text], ignore_index=True)
    unique = combined[["embedding_text_id", "embedding_text"]].drop_duplicates("embedding_text_id")
    unique.to_csv(output / "unique_embedding_text.csv", index=False)
    manifest = {
        "status": "text_prepared_embeddings_pending", "cleaning_version": CLEANING_VERSION,
        "embedding_model": arguments.model, "training_rows": len(train_text),
        "prediction_rows": len(prediction_text), "unique_texts": len(unique),
        "input_fields": ["question_text", "qualifying_options"],
        "metadata_only_fields": ["category_name", "geography_name"],
        "category_in_embedding": False,
        "empty_after_cleaning_rows": int(combined.empty_after_cleaning.sum()),
        "rule_selection": "Reviewed scaffolding informed by unique development-only 3-7-gram frequencies",
        "stem_patterns": list(STEM_PATTERNS), "time_pattern": TIME_PATTERN.pattern,
        "standard_option_pattern": STANDARD_OPTION_PATTERN.pattern,
        "standard_options_handling": "Omit complete generic answers from embedding text; retain originals and removed options for audit",
        "time_handling": "Remove generic recall phrase from question but preserve its text in Timeframe",
        "preserved": ["action", "specific-answer negation/future intent", "eligibility", "note content without label", "time context"],
        "targets_used": False, "final_model_modified": False,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Cleaned text: {len(train_text)} training + {len(prediction_text)} prediction rows; {len(unique)} unique inputs.")
    print(f"Model: {arguments.model}; outputs: {output}")
    if arguments.prepare_only:
        print("Prepared only: no API calls or embedding vectors generated.")
        return
    vectors, usage = embed_texts(unique.embedding_text.tolist(), arguments.model, PROJECT_ROOT / "cache/embeddings")
    vector_lookup = dict(zip(unique.embedding_text_id, vectors))
    for label, frame in (("training", train_text), ("prediction", prediction_text)):
        values = np.stack([vector_lookup[text_id] for text_id in frame.embedding_text_id])
        np.savez_compressed(output / f"{label}_embeddings.npz", embeddings=values,
                            embedding_text_id=frame.embedding_text_id.to_numpy(dtype=str),
                            source_row_id=frame.source_row_id.to_numpy())
    manifest.update(status="complete", dimensions=vectors.shape[1], usage=usage)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated vectors saved: dimensions={vectors.shape[1]}; API batches={usage['api_calls']}; cache hits={usage['cache_hits']}.")


if __name__ == "__main__":
    main()