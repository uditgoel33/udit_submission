# Method and Validation

## 1. Summary

- **Task:** predict the incidence rate a survey would return for a new category, with an 80% interval.
- **Model:** ridge regression (alpha = 10) on logit(incidence). Predictions are converted back to proportions.
- **Expected error on the prediction set:** about **7–9 pp MAE**, based on grouped CV and the holdout.
- **Intervals:** a single width on the logit scale, calibrated on out-of-fold errors. They hit **80.8% coverage on the holdout**.

Why the logit: incidence is bounded between 0 and 1. On the logit scale, predictions and intervals always stay inside that range, and intervals are naturally narrower near 0% and 100%.

## 2. Features

There are **19 predictors**. 🤖 = created with an LLM.

**No incidence values or survey counts were ever sent to the LLM.** Every prompt saw only the question text, the answer options and, where needed, the country.

| # | Feature | Source | What it captures |
|---|---|---|---|
| 1 | Recall window (months) | Rule | The window stated in the question (3m, 12m, 2y…). Missing if none is stated. |
| 2 | Typical frequency (months) | 🤖 | How often a typical buyer repeats the behaviour (bucket midpoint) |
| 3 | Country | Data | Survey geography |
| 4–5 | Sector, subsector | 🤖 | Two-level taxonomy, assigned from a fixed list I reviewed and approved |
| 6 | Behaviour | 🤖 | Purchase / use / visit / donate / status / intent |
| 7 | Eligibility gate | 🤖 | Who can qualify at all (pet owner, parent, vehicle owner…) |
| 8 | Frequency band | 🤖 | The category behind #2 |
| 9 | Price band | 🤖 | Typical cost per purchase, in fixed USD bands |
| 10 | Eligible share | 🤖 | % of all adults who could do this |
| 11 | Active share | 🤖 | % of **eligible** adults who did it in the window |
| 12 | Future openness | 🤖 | % of the **remaining** adults who would pick "likely in future" |
| 13 | Direct total | 🤖 | The LLM's overall % expected to qualify |
| 14 | Country × sector | Derived | Sector levels can differ by country |
| 15 | Price × frequency | Derived | Cheap-and-frequent vs expensive-and-rare |
| 16 | Gate × eligible share | Derived | How restrictive the gate really is |
| 17 | Subsector × direct total | Derived | Corrects the LLM estimate per subsector |
| 18 | Neighbour incidence | Embeddings + 🤖 | Incidence of the most relevant similar survey (Section 3) |
| 19 | Missing-neighbour flag | Rule | No valid neighbour found |

### How the LLM features were made

- **Separate prompts per feature set** (taxonomy, question profile, population estimate). No prompt sees another prompt's output.
- **Fixed answer lists with written definitions.** The LLM picks a label; it never invents a score.
- **Population features (10–13)** are asked **5 times**, and the most common answer is kept. Answers are bands of 20 percentage points, from `very_low` to `very_high`. The four use different bases (all adults, eligible adults, remaining adults), so they are separate features, not interchangeable probabilities.
- All responses are validated against a schema and cached, so reruns are identical. Notes, vote counts and agreement fields are kept for auditing only; they are not model inputs.
- Encoding, scaling and imputation are fitted inside each training fold. Categories with fewer than 3 training rows are pooled.

## 3. Neighbour feature: embeddings + LLM matching

**Idea:** a new category probably behaves like the most similar survey we already have. For example, air conditioners may resemble fridges.

**Step 1: clean the text.**
- Remove wording repeated across almost every question ("in the past … have you", "which best describes you", "Note:", generic Yes/No answers). I found it by counting common 3–7-word phrases.
- Keep what makes a question distinct: product names, definitions, specific answer options and the recall window.
- The category name and country are **not** embedded.
- Result: 645 unique cleaned texts across 943 rows.

**Step 2: embed and shortlist.**
- Embed with `text-embedding-3-small`.
- For each row, take the 5 most similar rows by cosine similarity. **Rows from the same category, in any country, are excluded.**

**Step 3: the LLM picks the best match.**
- A separate LLM sees only the original question and answer text of the row and the 5 candidates, labelled with anonymous IDs.
- It sees **no incidence, country, category name, similarity score or rank**.
- It picks one candidate, or "none" with a reason.
- Code then attaches the chosen row's incidence. "None" stays missing; it is never treated as 0%.

**Leakage controls:**
- The feature is **rebuilt for every CV fold**. Training and validation rows can only match to that fold's training rows.
- Holdout rows match to development rows only. Final predictions match to all 833 labelled rows.
- Checks confirm that every copied value came from the allowed pool and never from the row's own category.

## 4. Validation

**Split.** Rows are split by category, so the same category never appears on both sides, in any country. A random row split would leak a category's result from one country to another.

| Set | Rows | Category groups |
|---|---|---|
| Development | 708 | 441 |
| Holdout (~15%) | 125 | 78 |

- The holdout was balanced on country and taxonomy shares without looking at the target: country shares are within 0.3 pp, sector shares within 1.4 pp.
- Model choice used **5-fold grouped CV** on the development set. Ridge alpha was tried at 0.1, 1, 10 and 100, and picked by interval score. Both variants chose alpha = 10.

**Holdout results:**

| Metric | Without neighbour | With neighbour (selected) |
|---|---:|---:|
| MAE | 7.82 pp | 7.94 pp |
| Median absolute error | 6.37 pp | **5.50 pp** |
| RMSE | 10.21 pp | 10.36 pp |
| 80% coverage | 80.8% | 80.8% |
| Mean interval width | 27.85 pp | 27.24 pp |
| Interval score (lower is better) | 0.3968 | 0.3982 |

I kept the neighbour feature because it cuts the **typical (median) error by about 14%**. The cost is slightly worse average and tail errors.

**Being transparent:** this choice was made after seeing holdout results, so the holdout isn't a fully blind test. The early EDA also looked at all targets. Development CV MAE is 7.36 pp.

## 5. Intervals

1. Take each development row's **out-of-fold error** on the logit scale.
2. Keep the **largest error per category**, so repeated rows of the same category don't count twice.
3. Set the radius **r** to the ~80th percentile of those errors: **r = 0.646**.
4. For a predicted logit z: point = sigmoid(z), interval = [sigmoid(z − r), sigmoid(z + r)].

On the probability scale this gives wider intervals near 50% and narrower ones near 0% or 100%.

**What the interval covers:** everything the model gets wrong on an unseen category. That includes survey sampling noise, category and country differences the features miss, and LLM label errors. These sources are not separated.

**Caveat:** the final model is refit on all data and reuses the same radius; its own coverage can't be measured independently.

Interval score: IS = (U − L) + 10·max(L − y, 0) + 10·max(y − U, 0). It penalises both width and misses.

## 6. Error analysis

- I reviewed every holdout row with an error above 15 pp, with its question, features and chosen neighbour.
- Ridge contributions (value × coefficient) show what drove each prediction; these describe the model, not causes.
- Common themes:
  - no neighbour, or an irrelevant one
  - "premium" wording
  - future-intent options
  - eligibility-limited categories.

## 7. Air Conditioners in the US

The question asks whether someone **bought an air conditioner for their home in the past 2 years** (or is likely to soon). This is not AC ownership.

| Country | Prediction | 80% interval |
|---|---:|---:|
| Australia (observed: 34.6%) | 38.3% | 24.6–54.3% |
| **United States** | **46.7%** | **31.4–62.5%** |

**Method:** same question, labels and neighbour; only the country (and its interactions) changes.

**Trust: low to moderate.**
- There's no US AC data, and the LLM labels weren't re-made for the US.
- The data is from Jul–Sep, which is **US summer and AU winter**. The model can't see seasonality, and climate, housing stock and replacement cycles differ.
- **Use it for planning; confirm with a small US incidence test** if the decision is near the sales threshold.

## 8. Further work

- Test on genuinely new categories or later waves; use nested CV for wider model searches.
- Audit neighbour choices, and measure how much retrieval and LLM variation add to uncertainty.
- Model eligibility × behaviour explicitly, and treat future-intent options as their own structured feature.
- Check errors near the sales threshold and for thin groups (Canada has only 12 rows).
- Try interval widths that vary per row, e.g. wider when there's no good neighbour.

## 9. Files

- The **evaluation model** is fitted on the development set only.
- The **deployment model** is fitted on all 833 rows and scores the 110 prediction rows.
- Frozen features, fold-specific neighbour pools and lineage checks are in `features/`.
- The README has the reproduction commands and output locations.