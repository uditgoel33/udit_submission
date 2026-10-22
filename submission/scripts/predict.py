"""Generate predictions using the full-data deployment refit and development calibration."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from incidence.predict import main


if __name__ == "__main__":
    main()