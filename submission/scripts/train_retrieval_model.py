"""Compare the retrieval-incidence feature using its five context-specific CV folds."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from incidence.retrieval_model import train_retrieval_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/retrieval_model")
    arguments = parser.parse_args()
    train_retrieval_model(PROJECT_ROOT, arguments.output_dir)