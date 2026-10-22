"""Category-isolated cosine retrieval and target-blind LLM neighbor selection."""

import json

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field
from sklearn.metrics.pairwise import cosine_similarity

from incidence.data import normalize_category_name
from incidence.llm.client import PROMPT_DIR, ask

MATCH_FIELDS = ("category_name", "geography_name", "question_text", "qualifying_options", "disqualifying_options")
LLM_TEXT_FIELDS = ("question_text", "qualifying_options", "disqualifying_options")
MIN_SIMILARITY = None


class NeighborChoice(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: int | None
    reason: str = Field(min_length=1)

    def model_post_init(self, context) -> None:
        if context is not None and self.candidate_id is not None and self.candidate_id not in context["candidate_ids"]:
            raise ValueError("Selection must be a supplied candidate ID or null")


def candidate_table(queries: pd.DataFrame, references: pd.DataFrame,
                    query_vectors: np.ndarray, reference_vectors: np.ndarray,
                    top_k: int = 5, min_similarity: float | None = MIN_SIMILARITY) -> pd.DataFrame:
    """Find top-k rows only in the supplied reference pool, excluding query category."""
    if top_k < 1 or queries.empty or references.empty:
        raise ValueError("Nonempty query/reference pools and positive top_k are required")
    if min_similarity is not None and (not np.isfinite(min_similarity) or not -1 <= min_similarity <= 1):
        raise ValueError("Minimum similarity must be finite and within [-1, 1]")
    if not queries.query_id.is_unique or not references.source_row_id.is_unique:
        raise ValueError("Query IDs and reference source-row IDs must be unique")
    if any(set(MATCH_FIELDS) - set(frame.columns) for frame in (queries, references)):
        raise ValueError("Retrieval requires raw category, country, question and option fields")
    query_vectors = np.asarray(query_vectors)
    reference_vectors = np.asarray(reference_vectors)
    if (query_vectors.ndim != 2 or reference_vectors.ndim != 2
            or query_vectors.shape[0] != len(queries) or reference_vectors.shape[0] != len(references)
            or query_vectors.shape[1] != reference_vectors.shape[1]):
        raise ValueError("Embedding arrays must align to query/reference rows with matching dimensions")
    for matrix in (query_vectors, reference_vectors):
        if not np.isfinite(matrix).all() or (np.linalg.norm(matrix, axis=1) == 0).any():
            raise ValueError("Retrieval embeddings must be finite and nonzero")
    query_groups = queries.category_name.map(normalize_category_name).to_numpy()
    reference_groups = references.category_name.map(normalize_category_name).to_numpy()
    similarities = cosine_similarity(query_vectors, reference_vectors)
    reference_ids = references.source_row_id.to_numpy()
    records = []
    for position, query in enumerate(queries.itertuples(index=False)):
        eligible = reference_groups != query_groups[position]
        if min_similarity is not None:
            eligible &= similarities[position] >= min_similarity
        allowed = np.flatnonzero(eligible)
        ordered = allowed[np.lexsort((reference_ids[allowed], -similarities[position, allowed]))][:top_k]
        for rank, candidate_position in enumerate(ordered, start=1):
            candidate = references.iloc[candidate_position]
            records.append({"query_id": query.query_id, "rank": rank,
                            "candidate_row_id": int(candidate.source_row_id),
                            "cosine_similarity": float(similarities[position, candidate_position]),
                            **{field: candidate[field] for field in MATCH_FIELDS}})
    return pd.DataFrame(records, columns=["query_id", "rank", "candidate_row_id",
                                          "cosine_similarity", *MATCH_FIELDS])


def match_prompt(query: dict, candidates: pd.DataFrame) -> str:
    """Expose only original question/options and opaque local candidate IDs."""
    if not 1 <= len(candidates) <= 5 or candidates.candidate_row_id.duplicated().any():
        raise ValueError("LLM reranking requires one to five distinct eligible candidate rows")
    payload = {"query": {field: query[field] for field in LLM_TEXT_FIELDS},
               "candidates": [{"candidate_id": position, **{field: row[field] for field in LLM_TEXT_FIELDS}}
                              for position, row in enumerate(candidates.to_dict("records"), start=1)]}
    template = (PROMPT_DIR / "retrieval_match.txt").read_text(encoding="utf-8")
    return template.replace("{comparison}", json.dumps(payload, ensure_ascii=False))


def choose_neighbor(query: dict, candidates: pd.DataFrame, references: pd.DataFrame,
                    chooser=None, min_similarity: float | None = MIN_SIMILARITY) -> dict:
    """Select blind to incidence; attach the allowed reference target only after selection."""
    if not references.source_row_id.is_unique:
        raise ValueError("Reference source-row IDs must be unique")
    if min_similarity is not None and (not np.isfinite(min_similarity) or not -1 <= min_similarity <= 1):
        raise ValueError("Minimum similarity must be finite and within [-1, 1]")
    if min_similarity is not None:
        candidates = candidates.loc[candidates.cosine_similarity.ge(min_similarity)].copy()
    if candidates.empty:
        return {"query_id": query["query_id"], "retrieved_incidence": None,
                "selected_reference_row_id": None, "selected_similarity": None,
                "selected_rank": None, "selection_reason": "No candidates in the authorized retrieval pool",
                "retrieval_status": "no_candidates"}
    lookup = references.set_index("source_row_id")
    allowed_ids = candidates.candidate_row_id.astype(int).tolist()
    if not set(allowed_ids).issubset(lookup.index):
        raise ValueError("Candidate row is outside the authorized reference pool")
    query_group = normalize_category_name(query["category_name"])
    if lookup.loc[allowed_ids, "category_name"].map(normalize_category_name).eq(query_group).any():
        raise ValueError("A candidate shares the query category")
    context = {"candidate_ids": list(range(1, len(allowed_ids) + 1))}
    chooser = ask if chooser is None else chooser
    choice = chooser("retrieval_match", match_prompt(query, candidates), NeighborChoice,
                     temperature=0.0, validation_context=context)
    choice = NeighborChoice.model_validate(choice.model_dump(), context=context)
    if choice.candidate_id is None:
        return {"query_id": query["query_id"], "retrieved_incidence": None,
                "selected_reference_row_id": None, "selected_similarity": None,
                "selected_rank": None, "selection_reason": choice.reason,
                "retrieval_status": "llm_no_relevant_match"}
    selected_row_id = allowed_ids[choice.candidate_id - 1]
    value = float(lookup.at[selected_row_id, "percentage_3"])
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Reference incidence must be finite and in [0, 1]")
    candidate = candidates.set_index("candidate_row_id").loc[selected_row_id]
    return {"query_id": query["query_id"], "retrieved_incidence": value,
            "selected_reference_row_id": selected_row_id,
            "selected_similarity": float(candidate.cosine_similarity),
            "selected_rank": int(candidate["rank"]), "selection_reason": choice.reason,
            "retrieval_status": "selected"}