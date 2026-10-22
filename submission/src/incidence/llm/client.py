"""Official OpenAI structured outputs with isolated caches, retries and run usage."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import TypeVar
import warnings

from openai import (
    APIConnectionError, APITimeoutError, BadRequestError, InternalServerError, LengthFinishReasonError,
    OpenAI, RateLimitError,
)
from pydantic import BaseModel, ValidationError

from incidence.config import (
    CACHE_DIR, OPENAI_INPUT_USD_PER_MILLION, OPENAI_MODEL,
    OPENAI_OUTPUT_USD_PER_MILLION, OPENAI_TEMPERATURE_MODE, OPENAI_WORKERS,
)

PROMPT_DIR = Path(__file__).parent / "prompts"
SchemaType = TypeVar("SchemaType", bound=BaseModel)
_lock = threading.Lock()
_sdk_lock = threading.Lock()
_sdk_client = None
_default_temperature_models = set()
_usage = {"api_calls": 0, "responses_with_usage": 0, "cache_hits": 0,
          "failed_attempts": 0, "prompt_tokens": 0, "completion_tokens": 0,
          "known_estimated_cost_usd": 0.0, "unpriced_responses": 0}


def prompt_hash(prompt_name: str) -> str:
    """Version one prompt file without coupling unrelated feature sets."""
    if not re.fullmatch(r"[a-z0-9_]+", prompt_name):
        raise ValueError("Prompt name must use lowercase letters, digits or underscores")
    return hashlib.sha256((PROMPT_DIR / f"{prompt_name}.txt").read_bytes()).hexdigest()


def _request_identity(prompt_name, filled_prompt, schema, temperature, sample_index,
                       validation_context=None):
    """Include template and schema content so edits invalidate only affected cached requests."""
    model = os.environ.get("OPENAI_MODEL", OPENAI_MODEL).strip()
    if not model:
        raise ValueError("Set OPENAI_MODEL to a Structured Outputs-compatible OpenAI model")
    mode = os.environ.get("OPENAI_TEMPERATURE_MODE", OPENAI_TEMPERATURE_MODE)
    if mode not in {"auto", "explicit", "model_default"}:
        raise ValueError("OPENAI_TEMPERATURE_MODE must be auto, explicit or model_default")
    identity = {"model": model, "prompt_name": prompt_name, "prompt": filled_prompt,
                "prompt_hash": prompt_hash(prompt_name), "schema_name": schema.__name__,
                "schema": schema.model_json_schema(), "temperature": temperature,
                "temperature_mode": mode,
                "sample_index": sample_index, "validation_context": validation_context}
    encoded = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return identity, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _get_sdk_client():
    """Initialize only for cache misses, never expose or persist the API key."""
    global _sdk_client
    with _sdk_lock:
        if _sdk_client is None:
            if not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("Set OPENAI_API_KEY in your terminal; do not put it in source or chat")
            _sdk_client = OpenAI(max_retries=0, timeout=120)
    return _sdk_client


def _estimated_cost(prompt_tokens: int, completion_tokens: int):
    """Use configurable per-million-token prices; never report unknown API cost as zero."""
    input_price = os.environ.get("OPENAI_INPUT_USD_PER_MILLION", OPENAI_INPUT_USD_PER_MILLION)
    output_price = os.environ.get("OPENAI_OUTPUT_USD_PER_MILLION", OPENAI_OUTPUT_USD_PER_MILLION)
    if input_price is None or output_price is None:
        return None
    rates = [float(input_price), float(output_price)]
    if any(not math.isfinite(rate) or rate < 0 for rate in rates):
        raise ValueError("OpenAI token prices must be finite and nonnegative")
    return (prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1_000_000


def _record_completion(response, identity: dict, request_hash: str, directory: Path,
                       send_temperature: bool) -> dict:
    """Log returned token usage without storing credentials, prompts, targets or old labels."""
    usage = response.usage
    prompt_tokens = int(usage.prompt_tokens) if usage is not None else 0
    completion_tokens = int(usage.completion_tokens) if usage is not None else 0
    cost = _estimated_cost(prompt_tokens, completion_tokens) if usage is not None else None
    record = {"model": identity["model"], "prompt_name": identity["prompt_name"],
              "prompt_hash": identity["prompt_hash"], "request_hash": request_hash,
              "sample_index": identity["sample_index"], "prompt_tokens": prompt_tokens,
              "completion_tokens": completion_tokens, "estimated_cost_usd": cost,
              "requested_temperature": identity["temperature"],
              "temperature_mode": identity["temperature_mode"],
              "temperature_sent": send_temperature,
              "sampling_mode": "explicit" if send_temperature else "model_default"}
    with _lock:
        _usage["responses_with_usage"] += int(usage is not None)
        _usage["prompt_tokens"] += prompt_tokens
        _usage["completion_tokens"] += completion_tokens
        _usage["known_estimated_cost_usd"] += cost or 0.0
        _usage["unpriced_responses"] += int(cost is None)
        with (directory / "usage.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")
    return record


def _request_response(identity: dict, schema, request_hash: str, directory: Path):
    """Retry transient or parse failures at most three times, counting every API attempt."""
    sdk = _get_sdk_client()
    _estimated_cost(0, 0)
    correction = ""
    with _lock:
        send_temperature = identity["temperature_mode"] == "explicit" or (
            identity["temperature_mode"] == "auto" and identity["model"] not in _default_temperature_models
        )
    retryable = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError,
                 LengthFinishReasonError, ValidationError, ValueError)
    for attempt in range(3):
        with _lock:
            _usage["api_calls"] += 1
        try:
            parameters = {"model": identity["model"], "response_format": schema,
                          "messages": [{"role": "user", "content": identity["prompt"] + correction}]}
            if send_temperature:
                parameters["temperature"] = identity["temperature"]
            response = sdk.chat.completions.parse(**parameters)
            usage = _record_completion(response, identity, request_hash, directory, send_temperature)
            message = response.choices[0].message
            if message.refusal:
                raise RuntimeError("OpenAI refused the structured-output request; no label was cached")
            if message.parsed is None:
                raise ValueError("OpenAI returned no parsed structured output")
            payload = message.parsed.model_dump() if isinstance(message.parsed, BaseModel) else message.parsed
            return schema.model_validate(payload, context=identity["validation_context"]), usage
        except BadRequestError as error:
            with _lock:
                _usage["failed_attempts"] += 1
            details = error.body if isinstance(error.body, dict) else {}
            details = details.get("error", details)
            unsupported_temperature = isinstance(details, dict) and (
                details.get("param") == "temperature"
                and details.get("code") in {"unsupported_value", "unsupported_parameter"}
            )
            if not (unsupported_temperature and send_temperature
                    and identity["temperature_mode"] == "auto" and attempt < 2):
                raise
            send_temperature = False
            with _lock:
                _default_temperature_models.add(identity["model"])
            warnings.warn(
                f"{identity['model']} rejected temperature={identity['temperature']}. "
                "Retrying with the model default; requested-temperature sampling is not available. "
                "The cache records this change. Set OPENAI_TEMPERATURE_MODE=explicit to fail instead.",
                UserWarning, stacklevel=2,
            )
        except retryable as error:
            with _lock:
                _usage["failed_attempts"] += 1
            if attempt == 2:
                raise RuntimeError(f"Structured output failed after 3 attempts ({type(error).__name__})") from error
            if isinstance(error, (ValidationError, ValueError)):
                correction = f"\nPrevious output did not match the schema: {error}. Return a corrected JSON object."
            time.sleep(2 ** attempt)


def ask(prompt_name: str, filled_prompt: str, schema: type[SchemaType],
    temperature: float = 0.0, sample_index: int = 0,
    validation_context: dict | None = None) -> SchemaType:
    """Return schema-validated output from an isolated cache or official OpenAI Structured Outputs."""
    if sample_index < 0 or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ValueError("Sample index must be nonnegative and temperature must lie in [0, 2]")
    identity, request_hash = _request_identity(prompt_name, filled_prompt, schema, temperature, sample_index, validation_context)
    directory = CACHE_DIR / prompt_name
    cache_path = directory / f"{request_hash}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached["request_hash"] != request_hash or cached["prompt_hash"] != identity["prompt_hash"]:
            raise ValueError("Cached request identity does not match")
        result = schema.model_validate(cached["output"], context=validation_context)
        with _lock:
            _usage["cache_hits"] += 1
        return result
    directory.mkdir(parents=True, exist_ok=True)
    result, usage = _request_response(identity, schema, request_hash, directory)
    record = {**usage, "schema_name": schema.__name__, "temperature": temperature,
              "request_hash": request_hash, "output": result.model_dump(mode="json")}
    temporary = cache_path.with_suffix(f".{threading.get_ident()}.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(cache_path)
    return result


def ask_many(prompt_name: str, prompts: list[str], schema: type[SchemaType],
             temperature: float = 0.0, sample_index: int = 0, workers: int = OPENAI_WORKERS,
             validation_context: dict | None = None) -> list[SchemaType]:
    """Use a small thread pool while deduplicating identical requests and preserving row order."""
    if workers < 1:
        raise ValueError("Worker count must be positive")
    unique_prompts = list(dict.fromkeys(prompts))

    def complete(prompt):
        """Keep batch calls identical to the single-request interface."""
        return ask(prompt_name, prompt, schema, temperature, sample_index, validation_context)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = dict(zip(unique_prompts, pool.map(complete, unique_prompts)))
    return [results[prompt] for prompt in prompts]


def usage_summary(reset: bool = False) -> dict:
    """Report this run's calls, tokens and cost, including incomplete billing information."""
    with _lock:
        summary = dict(_usage)
        complete = summary["api_calls"] == summary["responses_with_usage"] and not summary["unpriced_responses"]
        summary["estimated_cost_usd"] = summary["known_estimated_cost_usd"] if complete else None
        if reset:
            for key in _usage:
                _usage[key] = 0.0 if key == "known_estimated_cost_usd" else 0
    return summary