"""Feature set A1: build a fixed taxonomy from raw category names, then stop for review."""

import json
import os
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from incidence.config import DATA_DIR, OPENAI_MODEL, OUTPUT_DIR
from incidence.llm.client import PROMPT_DIR, ask, ask_many, prompt_hash
from incidence.llm.questions import text_hash

ASSIGNMENT_FIELDS = ("category_name", "question_text", "qualifying_options", "disqualifying_options")


class TaxonomyAssignment(BaseModel):
    """Reject any returned labels that are not an exact pair in the supplied taxonomy."""

    model_config = ConfigDict(extra="forbid")
    sector: str = Field(min_length=1)
    subsector: str = Field(min_length=1)

    def model_post_init(self, context) -> None:
        """Validate the sector/subsector relationship before the client caches a result."""
        if context is not None:
            allowed = context["subsector_to_sector"]
            if self.subsector not in allowed or allowed[self.subsector] != self.sector:
                raise ValueError("sector/subsector must be an exact allowed pair from the supplied taxonomy")


def load_fixed_taxonomy(path: Path = DATA_DIR / "taxonomy.csv") -> pd.DataFrame:
    """Read the approved CSV without inventing or merging any of its labels."""
    taxonomy = pd.read_csv(path)
    if taxonomy.columns.tolist() != ["sector", "subsector"] or taxonomy.empty:
        raise ValueError("Fixed taxonomy must contain nonempty sector,subsector columns")
    for column in ("sector", "subsector"):
        if taxonomy[column].isna().any() or not taxonomy[column].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise ValueError(f"Taxonomy {column} labels must be nonempty text")
    if taxonomy.duplicated(["sector", "subsector"]).any():
        raise ValueError("Fixed taxonomy contains duplicate pairs")
    if taxonomy["subsector"].duplicated().any():
        raise ValueError("Every fixed subsector must belong to exactly one sector")
    return taxonomy


def assignment_prompt(raw_fields: dict, taxonomy: pd.DataFrame) -> str:
    """Pass only the four requested raw fields and the fixed list, never targets or other labels."""
    question = {field: raw_fields[field] for field in ASSIGNMENT_FIELDS}
    if any(not isinstance(value, str) or not value.strip() for value in question.values()):
        raise ValueError("Taxonomy assignment requires four nonempty raw text fields")
    template = (PROMPT_DIR / "taxonomy_assign.txt").read_text(encoding="utf-8")
    return template.format(taxonomy_pairs=taxonomy.to_json(orient="records"),
                           question_fields=json.dumps(question, ensure_ascii=False))


def assign_fixed_taxonomy(data: pd.DataFrame, taxonomy: pd.DataFrame, limit=None) -> tuple:
    """Label unique raw screeners and append validated pairs while preserving input rows and columns."""
    missing = sorted(set(ASSIGNMENT_FIELDS) - set(data.columns))
    if missing:
        raise ValueError(f"Taxonomy assignment is missing raw columns: {missing}")
    result = data.copy()
    if {"tax_sector", "tax_subsector"}.intersection(result.columns):
        raise ValueError("Input already contains tax_sector/tax_subsector; use the original raw CSV")
    keys = [text_hash(list(values)) for values in result[list(ASSIGNMENT_FIELDS)].itertuples(index=False, name=None)]
    unique = result[list(ASSIGNMENT_FIELDS)].copy()
    unique["taxonomy_input_id"] = keys
    unique = unique.drop_duplicates("taxonomy_input_id", keep="first")
    if limit is not None:
        if limit < 1:
            raise ValueError("Taxonomy assignment limit must be positive")
        unique = unique.head(limit)
    context = {"subsector_to_sector": taxonomy.set_index("subsector")["sector"].to_dict()}
    prompts = [assignment_prompt(row, taxonomy) for row in unique.to_dict("records")]
    outputs = ask_many("taxonomy_assign", prompts, TaxonomyAssignment, temperature=0.0, validation_context=context)
    validated = [TaxonomyAssignment.model_validate(output.model_dump(), context=context) for output in outputs]
    sector_lookup = dict(zip(unique["taxonomy_input_id"], [output.sector for output in validated]))
    subsector_lookup = dict(zip(unique["taxonomy_input_id"], [output.subsector for output in validated]))
    if limit is not None:
        result = result.iloc[[index for index, key in enumerate(keys) if key in sector_lookup]].copy()
        keys = [key for key in keys if key in sector_lookup]
    result["tax_sector"] = [sector_lookup[key] for key in keys]
    result["tax_subsector"] = [subsector_lookup[key] for key in keys]
    return result, unique


class Subsector(BaseModel):
    """A short subsector name with its written inclusion definition."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)


class Sector(BaseModel):
    """A broad sector containing at least one defined subsector."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1)
    subsectors: list[Subsector] = Field(min_length=1)


class Taxonomy(BaseModel):
    """Constrain taxonomy sizes and names before accepting an LLM response."""

    model_config = ConfigDict(extra="forbid")
    sectors: list[Sector] = Field(min_length=12, max_length=16)

    def model_post_init(self, context) -> None:
        """Reject ambiguous node names and invalid taxonomy sizes during schema validation."""
        sector_names = [sector.name.casefold() for sector in self.sectors]
        subsectors = [subsector for sector in self.sectors for subsector in sector.subsectors]
        subsector_names = [subsector.name.casefold() for subsector in subsectors]
        if not 40 <= len(subsectors) <= 70:
            raise ValueError("Taxonomy must contain 40 to 70 subsectors")
        if len(set(sector_names)) != len(sector_names) or len(set(subsector_names)) != len(subsector_names):
            raise ValueError("Sector names and subsector names must each be globally unique")
        if any(not 1 <= len(name.split()) <= 4 for name in sector_names + subsector_names):
            raise ValueError("Taxonomy names must contain 1 to 4 words")


def build_prompt(training: pd.DataFrame, prediction: pd.DataFrame) -> tuple[str, list[str]]:
    """Use only sorted raw category names, never targets or another feature set's outputs."""
    names = pd.concat([training["category_name"], prediction["category_name"]], ignore_index=True)
    if names.isna().any() or not names.map(lambda name: isinstance(name, str) and bool(name.strip())).all():
        raise ValueError("Taxonomy needs nonempty raw category names")
    categories = sorted(names.unique().tolist())
    template = (PROMPT_DIR / "taxonomy_build.txt").read_text(encoding="utf-8")
    return template.replace("{category_list}", json.dumps(categories, ensure_ascii=False, indent=2)), categories


def build_taxonomy(training: pd.DataFrame, prediction: pd.DataFrame,
                   output_dir: Path = OUTPUT_DIR) -> Taxonomy:
    """Generate one schema-validated taxonomy and mark it as awaiting human approval."""
    prompt, categories = build_prompt(training, prediction)
    taxonomy = ask("taxonomy_build", prompt, Taxonomy, temperature=0.0)
    metadata = {"stage": "A1", "approved": False, "category_count": len(categories),
                "prompt_hash": prompt_hash("taxonomy_build"),
                "model": os.environ.get("OPENAI_MODEL", OPENAI_MODEL),
                "minimum_node_size_verified": False}
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {**taxonomy.model_dump(mode="json"), "_metadata": metadata}
    path = output_dir / "taxonomy.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    print_taxonomy(taxonomy)
    print("A1 complete. Review taxonomy.json and approve before A2 assignments.")
    print("Actual subsector sizes and semantic overlap are not verified until assignment and review.")
    return taxonomy


def print_taxonomy(taxonomy: Taxonomy) -> None:
    """Show all sectors, subsectors and definitions for the required approval checkpoint."""
    for sector in taxonomy.sectors:
        print(sector.name)
        for subsector in sector.subsectors:
            print(f"  - {subsector.name}: {subsector.definition}")