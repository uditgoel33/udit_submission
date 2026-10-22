"""Rule-based features parsed directly from survey wording."""

import re

import numpy as np

_NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "twelve": 12, "a": 1, "an": 1}


def parse_recall_months(question: str) -> float:
    """Parse a stated past/last period into months; return NaN when no recall window is present."""
    text = str(question).lower()
    match = re.search(
        r"\b(?:past|last)\s+(\d+(?:\.\d+)?|one|two|three|four|five|six|twelve|a|an)"
        r"\s+(weeks?|months?|years?)\b", text,
    )
    if match is None:
        match = re.search(r"\b(?:past|last)\s+(week|month|year)\b", text)
        if match is None:
            return np.nan
        return {"week": 12 / 52, "month": 1, "year": 12}[match.group(1)]
    amount = _NUMBER_WORDS.get(match.group(1))
    amount = float(match.group(1)) if amount is None else amount
    unit = match.group(2)
    multiplier = 12 if unit.startswith("year") else 12 / 52 if unit.startswith("week") else 1
    return amount * multiplier