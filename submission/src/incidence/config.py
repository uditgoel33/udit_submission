"""Shared paths and reproducibility settings for the incidence project."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache" / "llm"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"

# The supplied CSVs currently sit at the repository root; prefer data/ when populated.
TRAINING_DATA_PATH = (
    DATA_DIR / "incidence_dataset.csv"
    if (DATA_DIR / "incidence_dataset.csv").exists()
    else PROJECT_ROOT / "incidence_dataset.csv"
)
PREDICTION_DATA_PATH = (
    DATA_DIR / "prediction_set.csv"
    if (DATA_DIR / "prediction_set.csv").exists()
    else PROJECT_ROOT / "prediction_set.csv"
)

RANDOM_SEED = 2026
HOLDOUT_FRACTION = 0.15
HOLDOUT_FOLDS = 7
CV_FOLDS = 5
CV_REPEATS = 3
TARGET_COLUMN = "percentage_3"
GROUP_COLUMN = "category_key"
GEOGRAPHY_COLUMN = "geography_name"
LLM_PROVIDER = "copilot"
COPILOT_LABELS_PATH = PROJECT_ROOT / "cache" / "copilot" / "question_features.json"
OLLAMA_MODEL = "qwen3.5:9b-mlx"
OLLAMA_HOST = "http://localhost:11434"
LLM_TEMPERATURE = 0.7
LLM_SAMPLES = 1
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "")
OPENAI_TEMPERATURE_MODE = os.environ.get("OPENAI_TEMPERATURE_MODE", "auto")
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_WORKERS = 4
OPENAI_INPUT_USD_PER_MILLION = os.environ.get("OPENAI_INPUT_USD_PER_MILLION")
OPENAI_OUTPUT_USD_PER_MILLION = os.environ.get("OPENAI_OUTPUT_USD_PER_MILLION")

TRAINING_COLUMNS = (
    "category_name",
    "geography_name",
    "wave_date",
    "question_text",
    "qualifying_options",
    "disqualifying_options",
    "weight_3",
    "base_weight_3",
    "percentage_3",
)
PREDICTION_COLUMNS = (
    "prediction_id",
    "category_name",
    "geography_name",
    "wave_date",
    "question_text",
    "qualifying_options",
    "disqualifying_options",
)
