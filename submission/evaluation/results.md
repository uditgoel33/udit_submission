# Retrieval-Feature Ridge

Five-fold grouped development CV compares identical ridge grids with and without retrieved incidence.
Every retrieval value is checked against its authorized reference pool and excludes the query category.
Null matches remain missing in exported features; fold-local median imputation and a missingness indicator enter the model.
No raw embeddings, similarity scores, LLM reasons or reference IDs are predictors.

| Model | CV MAE (pp) | Holdout MAE (pp) | Median Error (pp) | Coverage | Width (pp) | Interval Score |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 7.33 | 7.82 | 6.37 | 80.8% | 27.85 | 0.3968 |
| retrieval | 7.36 | 7.94 | 5.50 | 80.8% | 27.24 | 0.3982 |

Selection uses development-CV interval score, not holdout results.
Evaluation artifact fits development only; deployment artifact fits all 833 labeled rows with the same alpha and development radius.
The holdout was previously examined; these results are diagnostic. Intervals use approximate category-max OOF calibration, not exact independent split-conformal inference.
The final selected model and the median-versus-mean error trade-off are documented in the root README.
