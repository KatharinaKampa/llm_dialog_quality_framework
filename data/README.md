# data/ - Data Structure and Privacy

This folder contains all raw data, processed datasets, and metadata for the
project "Können LLMs gut diskutieren?" (Kampa, 2026).

---

## Folder Overview

```
data/
├── raw/                        ← Recipe master lists
├── dialogs/                    ← Generated dialogs (JSON, CSV, TXT), OSF only
│   ├── good_dialogs/
│   ├── bad_dialogs/
│   ├── practice_dialogs/
│   └── manual_evaluation/
├── studyA/                     ← Study A: stimuli, slots, exports
│   └── studyA_exports/
├── studyB/                     ← Study B: stimuli and validation metadata
└── processed/                  ← Cleaned ratings, both studies (public)
    ├── studyA/
    └── studyB/
```


## Legal Classification of Datasets

### Basis

Data collection was carried out in accordance with GDPR and the data protection
guidelines of the University of Regensburg. All participants provided informed
consent prior to participation. Collected data are **pseudonymized**: only a
self-chosen or automatically generated participant ID was stored, no names,
email addresses, or other directly identifying information.

Tiered publication rules apply because timestamps and combinations of response
patterns could theoretically be re-identifying.


### `data/raw/` - Recipe Data

| File | Content | GitHub | OSF | Reason |
|---|---|---|---|---|
| `recipe_masterlist.csv` | 40 recipes (title, type, ingredients, nutrition) | ✓ | ✓ | Self-selected from university-internal dataset; no personal data |
| `masterlist_GOLD.csv` | 6 practice recipes | ✓ | ✓ | Identical; already included in OSF preregistration Study A |

**Not publishable:** `all_recipe.csv` (full raw dataset, 58,424 recipes,
provided by the university, redistribution not permitted).


### `data/studyB/` - Study B Stimuli

| File | Content | GitHub | OSF | Reason |
|---|---|---|---|---|
| `studyB_stimuli.csv` | 34 dialogs + nutrition data (Study B main experiment) | ✓ | ✓ | Included in OSF preregistration Study B |
| `studyB_top_pool_pairs.csv` | 18 candidate pairs after pre-check | ✓ | ✓ | Metadata without personal data |


### `data/dialogs/` - Generated Dialogs

| File group | Content | GitHub | OSF | Reason |
|---|---|---|---|---|
| `good_dialogs/`, `bad_dialogs/` | LLM-generated dialog texts + sub-metric scores | ✗ | ✓ | LLM output; no personal data, but copyright status open. OSF upload recommended |
| `practice_dialogs/` | 12 practice dialogs (6 GOOD + 6 BAD) | ✗ | ✓ | Identical |
| `manual_evaluation/*.txt` | Manual quality review decision logs | ✓ | ✓ | No personal data; methodological transparency |

**Note on LLM output:** The copyright situation for machine-generated content
in Germany has not yet been definitively clarified (cf. § 2 UrhG: no protection
without human creative contribution). For academic purposes (master's thesis,
OSF archiving) publication with an appropriate note is unproblematic.
GitHub publication is not recommended due to file size (JSON, several MB).


### `data/studyA/` - Stimuli and Metadata Study A

| File | Content | GitHub | OSF | Reason |
|---|---|---|---|---|
| `studyA_dialogs_from_goodbad.csv` | 80 stimuli with Gold/IMC flags | ✓ | ✓ | Included in OSF preregistration Study A |
| `studyA_assignment_slots.csv` | Slot assignment (2 slots/dialog) | ✓ | ✓ | Included in OSF preregistration Study A |
| `gold_practice.csv` | 12 practice dialogs with gold intervals | ✓ | ✓ | Included in OSF preregistration Study A |
| `studyA_exports/llm_runs_all_dialogs.csv` | All 3 LLM runs × 80 dialogs × 8 scales | ✓ | ✓ | No personal data; pure model outputs |
| `studyA_exports/llm_median_all_dialogs.csv` | Medians of 3 runs | ✓ | ✓ | Identical |
| `studyA_exports/llm_PRIMARY/SENS_median_subset.csv` | LLM subsets for analysis | ✓ | ✓ | Identical |
| `studyA_exports/audit_PRIMARY/SENS_*.csv` | QA audit tables (row/rater/dialog) | ✓ | ✓ | Contain rater IDs (pseudonymized, P001-P168); re-identification not possible |


### `data/processed/` - Processed Ratings

All files in `data/processed/` are **cleaned, anonymized datasets** from the
QA pipelines. They contain **no timestamps, no IP addresses, no raw
session-level data**, only the variables relevant for statistical analysis.

Rater IDs have been replaced with sequential anonymous codes (P001-P168 across
both studies, MP001-MP034 for the mini-pilot).

#### `data/processed/studyA/`

```
processed/studyA/
├── human_PRIMARY_rows_regular_only_kept.csv    ← regular rows after QA (PRIMARY)
├── human_PRIMARY_gold_dialog_level.csv         ← dialog-level human gold (PRIMARY)
├── human_SENS_rows_regular_only_kept.csv       ← regular rows after QA (SENS)
└── human_SENS_gold_dialog_level.csv            ← dialog-level human gold (SENS)
```

| File | Content | Personal data? | GitHub | OSF |
|---|---|---|---|---|
| `human_PRIMARY_rows_regular_only_kept.csv` | Cleaned individual ratings (regular only); columns: `rater_id`, `dialog_id`, `condition`, Likert scales | Pseudonymized (`rater_id` = P001–P168; no timestamp) | ✓ | ✓ |
| `human_PRIMARY_gold_dialog_level.csv` | Aggregated human gold (mean of 2 raters per dialog); columns: `dialog_id`, `human_gold_*` | No personal data (fully aggregated) | ✓ | ✓ |
| `human_SENS_*` | SENS analogues (N=54 dialogs) | Identical to PRIMARY | ✓ | ✓ |

**Not publishable:** Raw HF Space exports (`ratings_*.csv` per rater from
`KKam799/studyA-storage`) contain timestamps and must remain in the
private HF dataset.

#### `data/processed/studyB/`

```
processed/studyB/
├── studyB_all_ratings_cleaned.csv              ← PRIMARY (N=139, after QA pipeline)
├── studyB_all_ratings_cleaned_SENS.csv         ← SENS (N=153, IMC + listwise only)
├── studyB_merged_with_llm.csv                  ← PRIMARY + LLM sub-metrics (H6)
└── studyB_merged_with_llm_SENS.csv             ← SENS + LLM sub-metrics (H6)
```

| File | Content | Personal data? | GitHub | OSF |
|---|---|---|---|---|
| `studyB_all_ratings_cleaned.csv` | Cleaned PRIMARY ratings; columns: `rater_id`, `condition`, `dialog_id`, pre/post scales, DQ items, nutrient estimates, QA flags | Pseudonymized (`rater_id` = P001-P167); no timestamp | ✓ | ✓ |
| `studyB_all_ratings_cleaned_SENS.csv` | SENS analogue (N=153) | Identical | ✓ | ✓ |
| `studyB_merged_with_llm.csv` | PRIMARY + LLM medians (left join `dialog_id`) | No additional personal data | ✓ | ✓ |
| `studyB_merged_with_llm_SENS.csv` | SENS + LLM medians | Identical | ✓ | ✓ |

**Not publishable:** Raw data from `KKam799/studyB-storage`
(`ratings/ratings_*.csv` per participant) contain `timestamp` (Unix time)
and `reservation_token`. These must remain in the private HF dataset.


## Publication Rules Summary

| Category | GitHub | OSF (open) | Private only |
|---|---|---|---|
| Recipe master lists | ✓ | ✓ | - |
| Study B stimuli + pair metadata | ✓ | ✓ | - |
| Stimuli metadata (IDs, flags, slots) | ✓ | ✓ | - |
| Generated dialog texts (JSON/TXT) | ✗ | ✓ | - |
| Manual evaluation logs | ✓ | ✓ | - |
| LLM runs + medians (model output) | ✓ | ✓ | - |
| QA audit tables (pseudonymized) | ✓ | ✓ | - |
| Cleaned ratings Study A (pseudonymized, no timestamp) | ✓ | ✓ | - |
| Cleaned ratings Study B (pseudonymized, no timestamp) | ✓ | ✓ | - |
| Raw HF Space data (with timestamp/token) | ✗ | ✗ | ✓ |
| `all_recipe.csv` (university-internal) | ✗ | ✗ | ✓ |


## OSF Archiving Notes

The following steps were completed before uploading to OSF:

1. **Timestamps removed:** `timestamp` and `start_time` columns deleted from
   all rating files (prevents theoretical re-identification via participation time).
2. **Reservation token removed:** `reservation_token` deleted from Study B files
   (technical artifact, no analytical value).
3. **Source file column removed:** `__source_file` deleted from Study A audit
   files (technical artifact referencing raw per-rater CSV filenames).
4. **Rater IDs anonymized:** All original rater IDs (including names entered
   by participants) replaced with sequential codes P001-P168 (Studies A+B) and
   MP001-MP034 (mini-pilot) using a consistent mapping applied across all files
   in a single script run.
5. **Codebooks:** `docs/codebook_metrics.md` + `Codebook HF-Space.pdf` (Study A)
   + `Codebook_StudyB_Main.pdf` (Study B) are archived alongside the preregistrations.
6. **License:** Data published under **CC BY 4.0**; code under **MIT License**.


## Variable Overview: Processed Ratings

### Study A - `human_PRIMARY_gold_dialog_level.csv`

| Column | Type | Description |
|---|---|---|
| `dialog_id` | str | Dialog identifier (D001-D080) |
| `human_gold_truthfulness` | float | Mean of 2 raters (1-7) |
| `human_gold_relevance` | float | Mean of 2 raters (1-7) |
| `human_gold_clarity` | float | Mean of 2 raters (1-7) |
| `human_gold_logic_coherence` | float | Mean of 2 raters (1-7) |
| `human_gold_feedback_depth` | float | Mean of 2 raters (1-7) |
| `human_gold_relational_appropriateness` | float | Mean of 2 raters (1-7) |
| `human_gold_respect_appreciation` | float | Mean of 2 raters (1-7) |
| `overall` | float | Overall rating, mean of 2 raters (1-7) |
| `n_human_raters` | int | Number of raters (always 2 in PRIMARY) |

### Study B - `studyB_all_ratings_cleaned.csv` (selection)

| Column | Type | Description |
|---|---|---|
| `rater_id` | str | Anonymized participant ID (P001-P167) |
| `condition` | str | `good` / `bad` |
| `dialog_id` | str | Assigned dialog |
| `recipe_title` | str | Recipe name |
| `pre_diet_suitability` | int | Diet suitability before dialog (1-7) |
| `post_diet_suitability` | int | Diet suitability after dialog (1-7) |
| `pre_recipe_stars` | int | Star rating before dialog (1-5) |
| `post_recipe_stars` | int | Star rating after dialog (1-5) |
| `pre_cook_intent` | int | Cooking intention before dialog (1-7) |
| `post_cook_intent` | int | Cooking intention after dialog (1-7) |
| `pre_save_intent` | int | Saving intention before dialog (1-7) |
| `post_save_intent` | int | Saving intention after dialog (1-7) |
| `pre_est_fat_g` | float | Fat estimate before dialog (g) |
| `post_est_fat_g` | float | Fat estimate after dialog (g) |
| `pre_est_carb_g` | float | Carbohydrate estimate before dialog (g) |
| `post_est_carb_g` | float | Carbohydrate estimate after dialog (g) |
| `pre_est_kcal` | float | Calorie estimate before dialog (kcal) |
| `post_est_kcal` | float | Calorie estimate after dialog (kcal) |
| `abs_err_fat_post` | float | Absolute estimation error fat (post) |
| `abs_err_carb_post` | float | Absolute estimation error carbohydrates (post) |
| `dq_clarity` | int | Dialog quality: clarity (1-7) |
| `dq_relevance` | int | Dialog quality: relevance (1-7) |
| `dq_respect` | int | Dialog quality: respect (1-7) |
| `dq_logic` | int | Dialog quality: logic (1-7) |
| `dq_coherence` | int | Dialog quality: coherence (1-7) |
| `dq_mean` | float | Mean of 5 DQ items |
| `manip_check` | int | Manipulation check (1-7) |
| `involvement` | int | Nutrition involvement (1-7) |
| `nutrition_knowledge` | int | Nutrition knowledge (1-7) |
| `imc_pass` | int | IMC passed (1=yes) |
| `min_time_ok_dialog` | int | Dialog time ≥ 10s (1=yes) |
| `nutrient_outlier_flag` | int | Nutrient outlier flag (1=yes) |
