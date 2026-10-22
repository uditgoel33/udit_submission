"""Category normalization used for leakage isolation."""
import re

def normalize_category_name(category_name: str) -> str:
    """Normalize common category-name variants to a stable group key.

    Suffix cleanup groups gender variants and retailer naming variants without
    merging unrelated category descriptions.
    """
    normalized = category_name.lower().strip()
    normalized = re.sub(r"\s*\((female|male|both genders) only\)\s*$", "", normalized)
    # This known singular/plural spelling variant should share one leakage group.
    normalized = re.sub(r"\belectronic retailers?\b", "electronics retailers", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
    return normalized
