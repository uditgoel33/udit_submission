"""Target-free survey-text cleaning and cached OpenAI embeddings."""

import hashlib
import json
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd

CLEANING_VERSION = "survey-scaffolding-v3-no-category"
TEXT_FIELDS = ("category_name", "question_text", "qualifying_options")
STANDARD_OPTION_PATTERN = re.compile(
    r"^(?:yes|no|yes,?\s+i\s+(?:have|do|am)|"
    r"no,?\s+but\s+(?:(?:i|we)\s+)?(?:(?:am|are|would\s+be)\s+)?"
    r"(?:open(?:\s+or\s+likely)?|likely|(?:would\s+)?consider)"
    r"(?:\s+to)?(?:\s+(?:buy|purchase|use|visit|rent|purchasing|buying|using|visiting|renting|it|this|"
    r"(?:these|such)\s+(?:products|services)|doing\s+so))?"
    r"(?:\s+in)?(?:\s+the)?(?:\s+near)?(?:\s+future)?)[.!?]*$", re.I,
)
TIME_PATTERN = re.compile(
    r"\b(?:in|over|within|during)\s+the\s+(?:past|last)\s+"
    r"(?:(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|a|an)"
    r"\s+)?(?:days?|weeks?|months?|years?)\b", re.I,
)
STEM_PATTERNS = (
    r"^(?:which|what)\s+(?:one\s+)?of\s+the\s+following\s+(?:statements?\s+)?"
    r"(?:best\s+)?(?:describes?\s+you|appl(?:y|ies)(?:\s+to\s+you)?)\s*[:?]?\s*",
    r"^which\s+(?:best\s+)?describes?\s+you\s*[:?]?\s*",
    r"^(?:which|what)\s+(?:one\s+)?of\s+the\s+following(?:\s+|$)",
    r"^(?:(?:have|did|do|are)\s+you(?:\s*/\s*|\s+|[.?!:]+|$))+",
)


def normalize_text(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Embedding source fields must be nonempty strings")
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).replace("\u2019", "'")).strip()


def clean_record(category: str, question: str, qualifying_options: str) -> dict:
    """Clean question/answer content; category is validated but never embedded."""
    normalize_text(category)
    question = normalize_text(question)
    qualifying_options = normalize_text(qualifying_options)
    timeframes = [re.sub(r"^(?:in|over|within|during)\s+the\s+(?:past|last)\s+", "past ",
                         match.group(0), flags=re.I) for match in TIME_PATTERN.finditer(question)]
    cleaned = TIME_PATTERN.sub(" ", question)
    typical = re.match(r"^thinking about (a typical (?:day|week|month|year)),?\s*", cleaned, re.I)
    if typical:
        timeframes.append(typical.group(1))
        cleaned = cleaned[typical.end():]
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,:?")
    for pattern in STEM_PATTERNS:
        cleaned = re.sub(pattern, "", cleaned, flags=re.I)
    cleaned = re.sub(r"\b(?:any|one|some) of the following\b", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\bnote\s*:\s*", "", cleaned, flags=re.I)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,:?")
    options = sorted(dict.fromkeys(part.strip() for part in qualifying_options.split("|") if part.strip()),
                     key=str.casefold)
    if not options:
        raise ValueError("At least one qualifying option is required")
    removed_options = [option for option in options if STANDARD_OPTION_PATTERN.fullmatch(option)]
    cleaned_options = " | ".join(option for option in options if not STANDARD_OPTION_PATTERN.fullmatch(option))
    timeframe = "; ".join(dict.fromkeys(timeframes))
    lines = []
    if cleaned:
        lines.append(f"Question: {cleaned}")
    if timeframe:
        lines.append(f"Timeframe: {timeframe}")
    if cleaned_options:
        lines.append(f"Qualifying answers: {cleaned_options}")
    empty_after_cleaning = not lines
    text = "\n".join(lines) if lines else "No substantive question or qualifying-answer content."
    return {"cleaned_question": cleaned, "cleaned_qualifying_options": cleaned_options,
            "removed_standard_options": " | ".join(removed_options),
            "empty_after_cleaning": empty_after_cleaning,
            "recall_context": timeframe, "embedding_text": text,
            "embedding_text_id": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def build_text_table(data: pd.DataFrame, source: str) -> pd.DataFrame:
    missing = set(TEXT_FIELDS) - set(data.columns)
    if missing or data.empty:
        raise ValueError(f"Missing embedding fields or empty input: {sorted(missing)}")
    columns = [column for column in ("prediction_id", *TEXT_FIELDS, "geography_name") if column in data]
    result = data[columns].reset_index(drop=True).copy()
    result.insert(0, "source_row_id", np.arange(len(result)))
    result.insert(0, "source", source)
    records = [clean_record(*values) for values in data[list(TEXT_FIELDS)].itertuples(index=False, name=None)]
    return pd.concat([result, pd.DataFrame(records)], axis=1)


def _vector(value, dimensions: int | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
        raise ValueError("Embedding must be a nonzero finite vector")
    if dimensions is not None and len(vector) != dimensions:
        raise ValueError("Embedding dimensions are inconsistent")
    return vector


def embed_texts(texts: list[str], model: str, cache_directory: Path,
                client=None, batch_size: int = 32) -> tuple[np.ndarray, dict]:
    """Reuse exact model/text cache identities; request only missing unique texts."""
    if not texts or not model.strip() or batch_size < 1:
        raise ValueError("Embedding inputs, model and positive batch size are required")
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding inputs must be nonempty text")
    unique = list(dict.fromkeys(texts))
    cache_directory.mkdir(parents=True, exist_ok=True)
    vectors, paths = {}, {}
    summary = {"model": model, "unique_texts": len(unique), "cache_hits": 0,
               "api_calls": 0, "total_tokens": 0}
    dimensions = None
    for text in unique:
        identity = json.dumps({"model": model, "text": text}, sort_keys=True, ensure_ascii=False)
        path = cache_directory / (hashlib.sha256(identity.encode("utf-8")).hexdigest() + ".json")
        paths[text] = path
        if path.exists():
            record = json.loads(path.read_text())
            if record["model"] != model or record["text"] != text:
                raise ValueError("Embedding cache identity mismatch")
            vectors[text] = _vector(record["embedding"], dimensions)
            dimensions = len(vectors[text])
            summary["cache_hits"] += 1
    missing = [text for text in unique if text not in vectors]
    if missing and client is None:
        from openai import OpenAI
        client = OpenAI(max_retries=2, timeout=120)
    for start in range(0, len(missing), batch_size):
        batch = missing[start:start + batch_size]
        response = client.embeddings.create(model=model, input=batch, encoding_format="float")
        outputs = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in outputs] != list(range(len(batch))):
            raise ValueError("Embedding response indices do not match the batch")
        for text, item in zip(batch, outputs):
            vectors[text] = _vector(item.embedding, dimensions)
            dimensions = len(vectors[text])
            record = {"model": model, "text": text, "embedding": vectors[text].tolist()}
            temporary = paths[text].with_suffix(".tmp")
            temporary.write_text(json.dumps(record) + "\n")
            temporary.replace(paths[text])
        summary["api_calls"] += 1
        summary["total_tokens"] += int(response.usage.total_tokens) if response.usage else 0
    matrix = np.stack([vectors[text] for text in texts])
    summary["dimensions"] = matrix.shape[1]
    return matrix, summary