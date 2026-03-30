# Exclusion Criteria - Full Documentation

This document describes all exclusion criteria applied during quality assurance
and data cleaning, at three levels:
(1) dialog generation, (2) Study A (rater level), (3) Study B (participant level).

All confirmatory criteria were preregistered on OSF prior to data collection
(see preregistration links in `README.md`).


## 1. Exclusion Criteria at Dialog Level (Generation Pipeline)

### 1.1 Automated Check (Python/pandas)

Each generated dialog was first checked automatically for formal correctness.
A dialog was automatically excluded and regenerated if **at least one** of the
following criteria was met:

| Criterion | Description |
|---|---|
| JSON validity | The dialog could not be parsed as valid JSON |
| Turn count | The dialog did not contain exactly 20 turns |
| Speaker alternation | Two consecutive turns were assigned to the same speaker |
| Error token | The dialog contained the token `[ERROR]` or unwanted speaker tags |
| Minimum character count | At least one turn fell below the defined minimum character count |

**Notebook:** `Systematic_Evaluation_of_the_pipeline.ipynb`


### 1.2 Manual Review (Criteria R1-R7)

Dialogs that passed the automated check were subsequently reviewed manually
using seven exclusion criteria. A dialog was excluded and regenerated if
**at least one** applied:

| Code | Criterion | Description |
|---|---|---|
| **R1** | Thematic frame break | The dialog leaves the recipe context; agents discuss topics unrelated to the assigned recipe |
| **R2** | Structural errors despite passing automated check | Structural errors not detected by the automated check (e.g., implicit role confusion, merged turns) |
| **R3** | Factual errors and hallucinations | Factually incorrect claims about the recipe (e.g., wrong ingredient names, fabricated nutrition values) |
| **R4** | Uninformative or repetitive contributions | Turns are too short, content-free, or verbatim repetitions of prior content without argumentative value |
| **R5** | Citation of CSV metadata | The dialog contains explicit nutrition values in grams or percent taken directly from the CSV data source, not derivable by real users |
| **R6** | Severe role violation | An agent acts strongly contrary to its assigned communication role (GOOD agent communicates like BAD or vice versa) |
| **R7** | Other errors | All further qualitative deficiencies that preclude use as stimulus material |

**Documentation:**
- `data/dialogs/manual_evaluation/manual_evaluation_GOOD_run1.txt`
- `data/dialogs/manual_evaluation/manual_evaluation_GOOD_run2.txt`
- `data/dialogs/manual_evaluation/manual_evaluation_BAD_run1.txt`
- `data/dialogs/manual_evaluation/manual_evaluation_BAD_run2.txt`

**Result:** Of 91 initially generated dialogs, 11 were excluded
(exclusion rate: 12.1 %). Sole sources of error were R5 (7 cases)
and R4 (4 cases). No dialog was excluded due to R6.

---

### 1.3 Criteria for Gold and IMC Dialog Selection (Study A)

From the validated pool of 80 dialogs, two subsets were isolated for Study A:

**16 Gold dialogs** (for measuring discriminant validity):
- Dialogs with particularly clear, extreme expression of the respective
  communication quality
- Selected based on exploratory LLM evaluation (1 run) and rule-based metric profile
- Documented in manual evaluation files (see above)

**10 IMC dialogs** (Instructional Manipulation Checks):
- Dialogs with unambiguous quality characteristics immediately recognizable to raters
- Used to check attention and instruction compliance in Study A
- Documented in manual evaluation files (see above)


## 2. Exclusion Criteria at Rater Level (Study A)

### 2.1 Preregistered Exclusion Criteria (Rater Level)

QA flags are computed in `StudyA_Analysis_Human_LLM.ipynb`
(function `compute_rater_flags_prereg`) and stored in
`audit_PRIMARY_rater_level.csv`.

| Flag column | Code | Criterion | Threshold |
|---|---|---|---|
| `flag_incomplete` | **QA-A1** | Wrong number of dialogs/gold/IMC | `n_total ≠ 10` or `n_gold ≠ 1` or `n_imc ≠ 1` |
| `flag_imc_fail` | **QA-A2** | IMC failure | `imc_dialog_pass == 0` for ≥ 1 IMC dialog |
| `flag_fast_median` | **QA-A3** | Implausibly short response time | Median `duration_sec` < 20.0 s |
| `flag_low_variance` | **QA-A4** | Very low response variance | SD across all scale responses < 0.30 |
| `flag_straightlining` | **QA-A5** | Straightlining | Share of identical responses ≥ 0.80 |
| `flag_gold_fail` | **QA-A6** | Gold standard failure | `overall` outside `[gold_expected_min, gold_expected_max]` (= `round(llm_overall) ± 1`) |

**PRIMARY core exclusions** (`flag_rater_excluded_core`):
QA-A1 OR QA-A2 OR QA-A3 OR QA-A4 OR QA-A5.
QA-A6 (`flag_gold_fail`) is a separate sensitivity toggle, excluded in PRIMARY,
included in SENS.

**Additional row-level exclusions:**
- `flag_long_duration`: `duration_sec > 300 s` (PRIMARY: excluded; SENS: included)
- `flag_missing_scales`: ≥ 1 scale missing --> listwise deletion (always active)
- Only **regular rows** (`is_gold == 0` AND `is_imc == 0`) enter confirmatory datasets

**PRIMARY result:** N = 50 valid, aggregated dialog rows (regular only,
M = 2.08 independent ratings per dialog), after enforcing exactly 2 raters
per dialog via `enforce_exactly_two_raters_per_dialog()`.

**Notebook:** `StudyA_Analysis_Human_LLM.ipynb`
**R script:** `studyA_analysis.R`

---

### 2.2 Sensitivity Dataset Study A (SENS, N = 54)

For robustness analysis, a less restrictive SENS dataset was created:
all QA flags (QA-A1-A6) and `flag_long_duration` lifted;
`flag_missing_scales` (listwise deletion) remains active.
Only regular rows remain primary.


## 3. Exclusion Criteria at Participant Level (Study B)

### 3.1 Preregistered Exclusion Criteria (Participant Level)

The QA pipeline is implemented in `StudyB_QA_and_Merge.ipynb`
(function `run_qa_pipeline()`) and applied sequentially. Each criterion
applies to the rows remaining after the previous step.

| Flag / column | Code | Criterion | Threshold / code |
|---|---|---|---|
| `imc_pass == 0` | **QA-B1** | IMC failure | Global IMC not passed at Welcome screen |
| `min_time_ok_dialog == 0` | **QA-B2** | Dialog time too short | `dialog_duration_sec < 10 s` --> flag set in HF Space |
| `nutrient_outlier_flag == 1` | **QA-B3** | Extreme nutrient outliers | Any estimate > 1,000 g (or kcal > 10,000) --> flag set in HF Space |
| `compute_straightlining(...)` | **QA-B4** | Straightlining | ≥ 80% identical scale values across 10 items; minimum 3 valid values required |
| Listwise deletion | **QA-B5** | Missing primary outcomes | `post_diet_suitability`, `post_recipe_stars`, `post_cook_intent`, `post_save_intent` |

**Straightlining items (QA-B4):** `post_diet_suitability`, `post_recipe_stars`,
`post_cook_intent`, `post_save_intent`, `dq_clarity`, `dq_relevance`, `dq_respect`,
`dq_logic`, `dq_coherence`, `manip_check`

**PRIMARY result:** N = 139 valid participants (n = 70 GOOD, n = 69 BAD)
Starting sample: N = 153

**Notebook:** `StudyB_QA_and_Merge.ipynb`
**R script:** `studyB_analysis.R`

---

### 3.2 Deviation from Preregistration (Study B)

One preregistered covariate was **not** included in the confirmatory models,
deviating from the preregistered specification:

> **Preference match (`preference_match`)** was not included as a covariate
> because this variable reflects a technical allocation criterion of the slot
> system and does not represent a substantively valid dietary preference.

This deviation is transparently documented in the results section of the thesis.


### 3.3 Sensitivity Dataset Study B (SENS, N = 153)

Function `run_qa_pipeline_sens()` in the same notebook:

- **QA-B1 (IMC)**: still excluded (validity criterion)
- **QA-B2-B4**: lifted (dialog time, nutrient outliers, straightlining)
- **QA-B5 (listwise)**: still active

This re-includes 14 previously excluded participants.

**Qualitative note:** Six of the 14 SENS-only participants left open-ended
comments indicating content incompatibility with their assigned recipe
(e.g., vegetarianism, food aversions). Their exclusion patterns therefore
reflect structural preference incompatibility rather than inattention.

---

## 4. If-Then Rules (Study B, Preregistered)

The following preregistered contingency rules were evaluated prior to
confirmatory interpretation of H1-H3:

| Condition | Consequence |
|---|---|
| Manipulation check (H4): p > .05 **or** d < .30 | H1-H3 interpreted as inconclusive rather than null |
| IMC failure rate > 20% | Separate analysis with and without IMC exclusion required |
| Missing values on primary outcomes > 5% | Multiple imputation sensitivity analysis required |
| ICC at recipe level > .05 (primary models H1-H3) | Cluster-robust standard errors additionally reported |

**Results of rule evaluation:**
- H4 passed (d = -2.71, p < .001) --> H1-H3 interpreted confirmatorily
- IMC failure rate: 0% --> no separate IMC analysis required
- Missing values ≤ 5% (fat: 5.0%; carbs/kcal: 3.6%) --> no MI required
- ICC > .05 for H3 outcomes (fat: .116; carbs: .123; kcal: .297) --> cluster-robust SEs reported
- ICC ≈ 0 for H1, H2a, H2b → cluster-robust SEs not required
