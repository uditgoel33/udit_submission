# Tracksuit Incidence Prediction

**This folder is the complete submission.** No parent repository folders, external
feature files or API credentials are needed to score or retrain the selected model.

## Deliverables

- [predictions.csv](predictions.csv): 110 rows in the required four-column format.
- [METHOD_AND_VALIDATION.md](METHOD_AND_VALIDATION.md): method, feature construction,
  evaluation, intervals, limitations, error analysis and the application.
- [predictions_detailed.csv](predictions_detailed.csv): original screeners, all
  features, retrieval lineage, predictions and interval bounds.
- [Air Conditioners scenario](evaluation/air_conditioners_country_scenario.csv):
  US prediction **46.66%**, nominal 80% interval **31.43%-62.53%**, with
  low-to-moderate trust under the documented country-transfer assumptions.

The final model is **ridge regression, alpha 10, with 19 predictors**, including
retrieved incidence and its missingness indicator. We chose it for the lower
holdout median absolute error, accepting slightly worse MAE and interval score.
This was a post-hoc choice on an already inspected holdout, not a blind test.

| Holdout Metric | Without Retrieval | Selected Retrieval |
|---|---:|---:|
| Mean absolute error | 7.82 pp | 7.94 pp |
| Median absolute error | 6.37 pp | 5.50 pp |
| RMSE | 10.21 pp | 10.36 pp |
| 80% interval coverage | 80.8% | 80.8% |
| Mean interval width | 27.85 pp | 27.24 pp |
| Mean interval score | 0.3968 | 0.3982 |

The evaluation artifact fits **708 development rows**, excluding 125 category-group
holdout rows. The separate deployment artifact fits **all 833 labeled rows** and
reuses the development-calibrated radius. Holdout scores describe only the former;
deployment coverage is not independently evaluated. Planning prediction-set error
is approximately 7-9 pp MAE, not a guarantee.

## Contents

| Location | Purpose |
|---|---|
| [features/](features/) | All frozen features, embeddings, cleaned text, selected neighbours and context-specific reference pools. |
| [models/](models/) | Selected evaluation and full-data deployment pipelines. |
| [evaluation/](evaluation/) | Frozen split, CV/holdout results, large-error analysis and application output. |
| [src/incidence/](src/incidence/) | Final feature preparation, ridge training/calibration/inference and exact LLM feature-generation code/prompts. No obsolete regression-family experiments. |
| [scripts/](scripts/) | Scoring, training, application, verification and optional API-backed feature generation. |
| [tests/](tests/) | Focused checks for cleaning, retrieval isolation, copied-target lineage, split behavior and feature generation. |
| [data/taxonomy.csv](data/taxonomy.csv) | The fixed approved taxonomy. |
| [environment.json](environment.json) | Python and direct-package versions used for the saved artifacts. |

The original [training data](incidence_dataset.csv), [prediction set](prediction_set.csv),
[format example](predictions_example.csv) and [task instructions](Instructions.md)
are included. [manifest.json](manifest.json) records checksums and final-model provenance.

## Set Up

Run commands **inside this folder**. Use Python 3.13, matching the recorded training
environment, and install only the required final-pipeline dependencies:

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Saved scikit-learn artifacts should be loaded with the versions in
[requirements.txt](requirements.txt). The frozen feature tables permit all commands
below without an LLM API key.

## Reproduce Final Results

```sh
# Check frozen-file integrity and independently reproduce the prediction values.
.venv/bin/python scripts/verify_submission.py

# Score the 110 rows with the saved deployment artifact; writes predictions.csv.
.venv/bin/python scripts/predict.py

# Retrain the same five-fold comparison and full-data model into a separate folder.
.venv/bin/python scripts/train_retrieval_model.py --output-dir reproduced_results

# Reproduce the documented AU-to-US Air Conditioners scenario.
.venv/bin/python scripts/air_conditioners_us.py

# Run the focused tests.
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests
```

Retraining reads frozen context-specific retrieval features and excludes validation
and holdout outcomes from their reference pools. **Do not train CV using the
single deployment retrieval column** in `features/training_features.csv`.
`features/context_features.csv`, `features/retrieval_features.csv` and
`features/reference_pools.json` retain the correct fold-specific views.

## Optional Feature Regeneration

The code and prompts used to construct the features are included. Regeneration
requires a locally configured `OPENAI_API_KEY` and the same `OPENAI_MODEL` recorded
for training (`gpt-6-astra` in the original cache records). Embeddings default to
`text-embedding-3-small`. Never store credentials in this folder.

```sh
.venv/bin/python scripts/build_llm_features.py --only taxonomy
.venv/bin/python scripts/build_llm_features.py --only profile
.venv/bin/python scripts/build_llm_features.py --only estimate
.venv/bin/python scripts/build_prediction_features.py
.venv/bin/python scripts/build_text_embeddings.py
.venv/bin/python scripts/build_retrieval_features.py
```

These optional commands write working files under `outputs/` and a new API cache
under `cache/` inside this folder; they do not require copies of the original
workspace directories. LLM regeneration can be costly and nondeterministic, so
the submission uses the frozen features for exact reproducibility. API caches
and irrelevant experimental pipelines are deliberately not bundled.

The reranking LLM sees only original question/options and anonymous candidate IDs;
incidence is attached locally after selection. Null abstentions remain missing
and are imputed within training folds with an explicit missingness indicator.
Category names remain metadata, not embedding inputs. Dense embedding dimensions
are used for retrieval, not directly as ridge predictors.

## Submission

This directory can be shared on its own. Following the task's original delivery
instructions, add `tracksuit-technical-test` as a collaborator to the repository
containing it and share the repository link with the talent manager.