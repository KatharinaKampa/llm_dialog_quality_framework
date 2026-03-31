# Können LLMs gut diskutieren? - Research Repository

**Master's Thesis | Katharina Kampa | University of Regensburg, 2026**
Chair of Digital Humanities, Institute for Information and Media, Language and Culture (I:IMSK)


## Project Overview

This thesis investigates the dialogic quality of Large Language Models (LLMs) using
a multi-agent debate framework. Two LLM agents (Fat Expert, Carbohydrate Expert)
discuss recipes; a third LLM communication expert evaluates dialog quality based on
metrics derived from five communication theories (Grice, Schulz von Thun, Watzlawick,
Rogers, Hargie).

The multi-stage research design comprises:

- **Study A** - Validation study: Comparison of LLM evaluator ratings against a
  human gold standard (N = 50 dialogs, 16 raters)
- **Study B** — Online experiment: Causal effect of dialog quality on
  human users (N = 139)


## Repository Structure

```
llm-dialogue-quality-framework/
│
├── README.md
├── requirements.txt                         ← Python packages (notebooks + HF Spaces)
├── LICENSE
│
├── data/
│   ├── README.md                            ← Data documentation, privacy rules, variables
│   ├── raw/
│   │   ├── recipe_masterlist.csv            ← 40 recipes (random_state=42)
│   │   └── masterlist_GOLD.csv              ← 6 recipes for practice dialogs
│   │
│   ├── dialogs/                             ← OSF only (LLM-generated content)
│   │   ├── good_dialogs/
│   │   │   ├── all_dialogs_generated.json
│   │   │   ├── dialogs_with_evaluations.csv
│   │   │   ├── submetric_scores.csv
│   │   │   └── generated_dialogs.txt
│   │   ├── bad_dialogs/
│   │   │   ├── all_bad_dialogs_generated.json
│   │   │   ├── bad_dialogs_with_evaluations.csv
│   │   │   ├── bad_submetric_scores.csv
│   │   │   └── generated_bad_dialogs.txt
│   │   ├── practice_dialogs/
│   │   │   ├── all_gold_dialogs_generated.json
│   │   │   ├── gold_dialogs_with_evaluations.csv
│   │   │   ├── gold_submetric_scores.csv
│   │   │   ├── generated_gold_dialogs.txt
│   │   │   ├── all_bad_gold_dialogs_generated.json
│   │   │   ├── bad_gold_dialogs_with_evaluations.csv
│   │   │   ├── bad_gold_submetric_scores.csv
│   │   │   └── generated_bad_gold_dialogs.txt
│   │   └── manual_evaluation/
│   │       ├── manual_evaluation_GOOD_run1.txt
│   │       ├── manual_evaluation_GOOD_run2.txt
│   │       ├── manual_evaluation_BAD_run1.txt
│   │       └── manual_evaluation_BAD_run2.txt
│   │
│   ├── studyA/
│   │   ├── studyA_dialogs_from_goodbad.csv  ← 80 stimuli (D001–D080) with Gold/IMC flags
│   │   ├── studyA_assignment_slots.csv      ← slot seed (2 slots/dialog)
│   │   ├── gold_practice.csv               ← 12 practice dialogs
│   │   └── studyA_exports/
│   │       ├── human_PRIMARY_rows_regular_only_kept.csv
│   │       ├── human_PRIMARY_gold_dialog_level.csv
│   │       ├── human_SENS_rows_regular_only_kept.csv
│   │       ├── human_SENS_gold_dialog_level.csv
│   │       ├── llm_runs_all_dialogs.csv
│   │       ├── llm_median_all_dialogs.csv
│   │       ├── llm_PRIMARY_median_subset.csv
│   │       ├── llm_SENS_median_subset.csv
│   │       ├── audit_PRIMARY_row_level.csv
│   │       ├── audit_PRIMARY_rater_level.csv
│   │       ├── audit_PRIMARY_dialog_level.csv
│   │       ├── audit_SENS_row_level.csv
│   │       ├── audit_SENS_rater_level.csv
│   │       └── audit_SENS_dialog_level.csv
│   │
│   ├── studyB/
│   │   ├── studyB_stimuli.csv              ← 34 dialogs (17 pairs) with nutrition info
│   │   └── studyB_top_pool_pairs.csv       ← 18 candidate pairs after pre-check
│   │
│   └── processed/                          ← Cleaned ratings (anonymized, no timestamps)
│       ├── studyA/
│       │   ├── human_PRIMARY_rows_regular_only_kept.csv
│       │   ├── human_PRIMARY_gold_dialog_level.csv
│       │   ├── human_SENS_rows_regular_only_kept.csv
│       │   └── human_SENS_gold_dialog_level.csv
│       └── studyB/
│           ├── studyB_all_ratings_cleaned.csv        (PRIMARY, N=139)
│           ├── studyB_all_ratings_cleaned_SENS.csv   (SENS, N=153)
│           ├── studyB_merged_with_llm.csv            (PRIMARY + LLM sub-metrics)
│           └── studyB_merged_with_llm_SENS.csv       (SENS + LLM sub-metrics)
│
├── notebooks/
│   ├── dialog_generation/
│   │   ├── GOOD_Dialogs_MA.ipynb
│   │   ├── BAD_Dialogs_MA.ipynb
│   │   ├── Practice-Dialogs_GOOD.ipynb
│   │   └── Practice-Dialogs_BAD.ipynb
│   ├── quality_assurance/
│   │   ├── Systematische_Evaluation_der_Pipeline.ipynb
│   │   └── Automated_Practice_Dialog_Check.ipynb
│   ├── study_preparation/
│   │   ├── Study_A_Stimuli_Seed_Practice_Builder.ipynb
│   │   ├── Mini_Pilot_StudyA.ipynb
│   │   ├── Prepare_Stimuli_StudyB.ipynb
│   │   ├── Study_B_Stimuli_and_Seed_Builder.ipynb
│   │   └── Mini_Pilot_StudyB_Auswertung.ipynb
│   └── analysis/
│       ├── StudyA_Analysis_Human_LLM.ipynb
│       └── StudyB_QA_and_Merge.ipynb
│
├── r_scripts/
│   ├── studyA_analysis.R
│   ├── studyB_analysis.R
│   └── session_info.txt
│
├── hf_spaces/
│   ├── README.md
│   ├── study_a_rater_interface/
│   │   └── app.py
│   ├── study_b_mini_pilot/
│   │   └── app.py
│   └── study_b_main_experiment/
│       └── app.py
│
└── docs/
    ├── codebook_metrics.md
    ├── exclusion_criteria.md
    └── hf_spaces_README.md
```


## Data

> Full data protection rules, variable descriptions, and OSF archiving notes: **`data/README.md`**

### Recipe Dataset
The full raw dataset (58,424 recipes) was provided by the University of Regensburg
and is not publicly available. The 40 selected recipes are available as
`data/raw/recipe_masterlist.csv` (selection: `random_state=42`).

### Dialogs and Practice Dialogs

| Type | Count | Source | Purpose |
|---|---|---|---|
| GOOD dialogs (main corpus) | 40 | `recipe_masterlist.csv` | Stimuli Studies A & B |
| BAD dialogs (main corpus) | 40 | `recipe_masterlist.csv` | Stimuli Studies A & B |
| GOOD practice dialogs | 6 | `masterlist_GOLD.csv` | Rater calibration Study A |
| BAD practice dialogs | 6 | `masterlist_GOLD.csv` | Rater calibration Study A |
| Gold dialogs | 16 (8G+8B) | Main corpus | LLM vs. human comparison |
| IMC dialogs | 8 (4G+4B) | Main corpus | Attention check |

### Study A - Data Flow

```
HF Space (main data collection)
        ↓
ratings_*.csv (per rater)  +  llm_runs_all_dialogs.csv (3 runs, temperature=0.2)
        ↓
StudyA_Analysis_Human_LLM.ipynb
        ↓
human_PRIMARY_gold_dialog_level.csv   +   llm_PRIMARY_median_subset.csv
human_SENS_gold_dialog_level.csv      +   llm_SENS_median_subset.csv
audit_PRIMARY/SENS_*.csv
        ↓
02_studyA_primary_and_sens_analysis.R
        ↓
outputs/H1–H9_primary.csv, H1–H9_sens.csv, LLM_intra_model_ICC.csv
figures/Figure_StudyA_*.png
```

**PRIMARY dataset (N=50 dialogs):**
All exclusion criteria applied (QA-A1–A5, flag_long_duration).

**SENS dataset (N=54 dialogs):**
All QA criteria lifted (except listwise deletion for missing values).

### Study B - Stimulus Selection and Preparation

```
1) Pre-Check (Study A data)
human_PRIMARY_gold_dialog_level.csv  +  llm_PRIMARY_median_subset.csv
        ↓
Prepare_Stimuli_StudyB.ipynb
        ↓
studyB_recipe_pairs_with_deltas.csv     ← all pairs with deltas + rank_score
studyB_pass_pool_pairs.csv              ← pairs passing pre-check gate
studyB_top_pool_pairs.csv               ← top 18 by rank_score

2) Merge stimuli and nutrition data
all_recipe.csv (university-internal)  +  recipe_masterlist.csv
        ↓
Study_B_Stimuli_and_Seed_Builder.ipynb
        ↓
all_recipe_filtered_by_masterlist.csv   ← 40 recipes with all nutrition columns
studyB_stimuli_information.csv          ← 80 dialogs + nutrition data merged
studyB_stimuli.csv                      ← final 34 dialogs (17 pairs)

3) Seed files
studyB_assignment_seed.csv              ← main Study B (4+7 slots/dialog)
studyB_pilot_assignment_seed.csv        ← mini-pilot (18 pairs, 4 slots/dialog)

4) Mini-Pilot (2nd validation step)
HF Space (N=34) + studyB_pilot_assignment_seed.csv + studyB_stimuli.csv
        ↓
pilot_all_ratings.csv                   ← mini-pilot raw data
        ↓
Mini_Pilot_StudyB_Auswertung.ipynb
        ↓
Decision: all 17 pairs retain (100% choice agreement, D=4.68)
```

**Pre-Check Gate (Study B, Step 1):**

| Criterion | Threshold |
|---|---|
| `human_delta_overall` | ≥ 0.70 |
| `llm_delta_overall` | ≥ 0.70 |
| `human_subscale_pos_frac` | ≥ 0.70 (at least 5/6 subscales correct direction) |
| `llm_subscale_pos_frac` | ≥ 0.70 |

Ranking score: sum of standardized human and LLM overall deltas.


## Notebooks

All notebooks were developed in **Google Colaboratory** (Python 3).
Full package list: `requirements.txt`.

### Dialog Generation

| Notebook | Content | Chapter |
|---|---|---|
| `GOOD_Dialogs_MA.ipynb` | 40 GOOD dialogs; single-API-call design; exploratory LLM evaluation (1 run); 15 rule-based sub-metrics (NLI, BERTScore, cosine, zero-shot, sentiment, Flesch) | 4.2, 5.2, 5.3 |
| `BAD_Dialogs_MA.ipynb` | 40 BAD dialogs; identical structure | 4.2, 5.2, 5.3 |
| `Practice-Dialogs_GOOD.ipynb` | 6 GOOD practice dialogs based on `masterlist_GOLD.csv`; simplified submetric implementation† | 5.3 |
| `Practice-Dialogs_BAD.ipynb` | 6 BAD practice dialogs; identical simplification† | 5.3 |

> † Deviating implementation in practice notebooks:
> `grice_truth` = BERTScore F1 (instead of NLI vs. DGE),
> `rogers_respect` = full-text sentiment (instead of NLI cross-agent),
> `watzlawick_disturbance` = upper-triangle cosine (instead of turn-to-turn sequence).

### Quality Assurance

| Notebook | Content | Chapter |
|---|---|---|
| `Systematische_Evaluation_der_Pipeline.ipynb` | Structure check; BERTopic (3 topics, KMeans, n=80); interaction markers (Mann-Whitney U) | 7.1 |
| `Automated_Practice_Dialog_Check.ipynb` | Structure check for 12 practice dialogs | 5.3 |

### Study Preparation

| Notebook | Content | Chapter |
|---|---|---|
| `Study_A_Stimuli_Seed_Practice_Builder.ipynb` | D001-D080; recipe type assignment; Gold/IMC IDs; slot file; `gold_practice.csv` | 8.1 |
| `Mini_Pilot_StudyA.ipynb` | N=3; functionality test; no statistical analysis | 8.1 |
| `Prepare_Stimuli_StudyB.ipynb` | Pre-check gate; pair table with human/LLM deltas + rank_score; `studyB_top_pool_pairs.csv` (top 18) | 9.1 |
| `Study_B_Stimuli_and_Seed_Builder.ipynb` | Merge recipe nutrition data from `all_recipe.csv` via `(title_key, ing_key)` join → `studyB_stimuli_information.csv`; final 34 dialogs → `studyB_stimuli.csv`; `studyB_assignment_seed.csv` (base_slots=4, extra_recipes=7, smallest_human_delta); `studyB_pilot_assignment_seed.csv` (18 pairs, 4 slots) | 9.1 |
| `Mini_Pilot_StudyB_Auswertung.ipynb` | Mini-pilot Study B evaluation (N=34); decision rules per pair_index: mean_diff≥0.5 + forced_good_prop≥0.6 → RETAIN; result: all 17 pairs RETAIN (mean_diff=4.68, 100% forced_good_prop) | 9.1 |

### Analysis

| Notebook | Content | Chapter |
|---|---|---|
| `StudyA_Analysis_Human_LLM.ipynb` | Human ratings pipeline (QA flags, PRIMARY/SENS export, audit tables); LLM evaluation (3 runs, resume-capable, exponential backoff on 429); hypothesis overview (H1–H9); dataset inventory | 8.1-8.2 |
| `StudyB_QA_and_Merge.ipynb` | QA pipeline: merge individual `ratings_*.csv` → `studyB_all_ratings_merged.csv`; preregistered exclusion criteria PRIMARY (QA-B1–B4 + listwise) → `studyB_all_ratings_cleaned.csv`; SENS (IMC + listwise only) → `studyB_all_ratings_cleaned_SENS.csv`; LLM merge (left join on `dialog_id`) → `studyB_merged_with_llm.csv` | 11.1 |

**Outputs of `StudyA_Analysis_Human_LLM.ipynb`:**

| File | Content |
|---|---|
| `human_PRIMARY_rows_regular_only_kept.csv` | Cleaned regular rows (PRIMARY) |
| `human_PRIMARY_gold_dialog_level.csv` | Dialog-level human gold (mean of 2 raters) |
| `human_SENS_*` | Analogous SENS variants |
| `llm_runs_all_dialogs.csv` | All 3 runs × 80 dialogs × 8 scales |
| `llm_median_all_dialogs.csv` | Median of 3 runs per dialog/scale |
| `llm_PRIMARY_median_subset.csv` | LLM medians for PRIMARY dialog set |
| `llm_SENS_median_subset.csv` | LLM medians for SENS dialog set |
| `audit_PRIMARY_row_level.csv` | QA flags + keep_row + exclusion_reasons |
| `audit_PRIMARY_rater_level.csv` | Status (KEPT/EXCLUDED) + reasons per rater |
| `audit_PRIMARY_dialog_level.csv` | kept_for_human_gold + reasons per dialog |


## R Scripts

### `studyA_analysis.R`

Implements all preregistered hypothesis tests for Study A
(PRIMARY N=50 + SENS N=54, stratified bootstrap R=5,000):

| Hypothesis | Test | Benchmark |
|---|---|---|
| H1 | Pearson r + Spearman ρ (overall LLM vs. human) | r ≥ .40 |
| H2 | Pearson r + Spearman ρ per subscale + BH-FDR | r ≥ .40 |
| H3 | ICC(2,2) absolute agreement (human interrater) | ICC ≥ .75 |
| H4 | ROC-AUC + Cohen's d (human discriminability) | AUC ≥ .70, d ≥ .30 |
| H5 | ROC-AUC + Cohen's d (LLM discriminability) | AUC ≥ .70, d ≥ .30 |
| H6/H7 | Correlations (rule-based vs. human/LLM) | exploratory |
| H8 | Bland-Altman bias + LoA + bootstrap CI | \|bias\| ≤ 0.50 |
| H9 | Partial correlations + regression (optional) | robustness |
| LLM-ICC | ICC(2,1) and ICC(2,3) across 3 runs | descriptive |

**R packages:** tidyverse, readr, janitor, boot, pROC, irr

**Outputs (`outputs/`):**

| File | Content |
|---|---|
| `H1_primary/sens.csv` | Overall correlation |
| `H2_primary/sens.csv` | Subscale correlations + BH-FDR p-values |
| `H3_ICC_primary/sens.csv` | ICC(2,2) per scale |
| `H4_primary/sens_human.csv` | AUC + Cohen's d (human, overall) |
| `H4_primary/sens_human_subscales.csv` | AUC + Cohen's d (human, subscales) |
| `H5_primary/sens_llm.csv` | AUC + Cohen's d (LLM, overall) |
| `H5_primary/sens_llm_subscales.csv` | AUC + Cohen's d (LLM, subscales) |
| `H8_primary/sens_bland_altman.csv` | Bias, LoA, SD, n |
| `H9_primary/sens_optional.csv` | Partial correlations (if covariates present) |
| `LLM_intra_model_ICC.csv` | ICC(2,1) + ICC(2,3) per LLM scale |

**Figures (`figures/`):**

| File | Content |
|---|---|
| `Figure_StudyA_Heatmap_Correlations.png` | Convergent validity heatmap (H2) |
| `Figure_StudyA_ICC_Forest.png` | Forest plot ICC(2,2) (H3) |

### `studyB_analysis.R`

Implements all preregistered hypothesis tests for Study B
(PRIMARY N=139 + SENS N=153). All analyses are encapsulated in
`run_all_analyses(df, dataset_label, output_dir)`, called once for PRIMARY
and once for SENS.

**Design:** 2-group randomized between-subjects experiment (good vs. bad)
with pre-post measurement; blocking by recipe. ANCOVA as primary model:
`Post ~ Condition + Pre + recipe_title (fixed effects)`.

| Hypothesis | Test | Benchmark |
|---|---|---|
| H1 | ANCOVA: `post_diet_suitability ~ condition + pre + recipe_title` (primary) | α = .05 |
| H2a | ANCOVA: `post_recipe_stars` | FDR (BH) |
| H2b | ANCOVA: `post_cook_intent` | FDR (BH) |
| H3 | ANCOVA: `abs_err_fat_post`, `abs_err_carb_post` | FDR (BH) |
| H4 | Welch t-test + ANCOVA: `dq_mean` (good > bad) | d ≥ .30, p < .05 |
| H5a–c | Bootstrap mediation (5,000 sims, BCa): `condition → dq_mean → Δ-outcome` | CI excludes 0 |
| H6 | LLM quality index regression: `delta ~ llm_quality_index + pre + recipe_title + involvement` | FDR (BH) |

**LLM Quality Index** (H6): mean of 6 communicative process metrics
(clarity, relevance, logic_coherence, respect_appreciation,
relational_appropriateness, feedback_depth). `truthfulness` (factual
correctness) and `overall_quality` (aggregate) explicitly excluded.

**R packages:** tidyverse, car, emmeans, effectsize, mediation, lme4,
performance, parameters, boot, rstatix, ggpubr, patchwork, sandwich, lmtest

**Outputs (`output_PRIMARY/` and `output_SENS/`):**

| File | Content |
|---|---|
| `results_summary_PRIMARY/SENS.csv` | Results table H1-H3 |
| `results_PRIMARY/SENS.txt` | Full console log |
| `studyB_plot_prepost.png` | Pre-/post-trajectories H1, H2a, H2b (±SE) |
| `studyB_plot_dq_subscales.png` | Dialog quality subscales (H4) |
| `studyB_plot_deltas.png` | Violin+boxplot Δ-scores |
| `studyB_plot_h6_scatter.png` | LLM quality index vs. Δ-diet suitability (H6) |
| `output_comparison_PRIMARY_vs_SENS_StudyB.csv` | Direct PRIMARY vs. SENS comparison |

For package versions: `r_scripts/session_info.txt`.


## Preregistrations (OSF)

| Study | OSF Link | Associated Files |
|---|---|---|
| Study A | [https://osf.io/8e957] | `Preregistration_StudyA.pdf`, `LLM_EvaluationPlan.pdf`, `Study Plan Study A.pdf`, `Codebook HF-Space.pdf`, `Informed Consent.pdf`, `Debriefing End Text.pdf`, `Instructions.pdf`, `Recruitment Text.pdf`, `studyA_assignment_slots.csv`, `studyA_dialogs_from_goodbad.csv`, `gold_practice.csv` |
| Study B | [https://osf.io/akgx9] | `Preregistration_StudyB.pdf`, `Codebook_StudyB_Main.pdf`, `InformedConsent_Instructions_Debrief.pdf`, `Summary of Stimulus Validation for Study B.pdf`, `studyB_stimuli.csv` |

**Note:** All CSV files referenced in the Study A OSF preregistration
(`studyA_assignment_slots.csv`, `studyA_dialogs_from_goodbad.csv`, `gold_practice.csv`)
are available in this repository under `data/studyA/`.


## Hugging Face Spaces

| Folder | Purpose | N | Study |
|---|---|---|---|
| `study_a_rater_interface/` | Rater interface (practice 3 + main 10 dialogs, slot system, consent, debriefing) | 16 raters | Study A |
| `study_b_mini_pilot/` | Mini-pilot (2 pairs/person, forced choice + Likert, reservation system) | 34 | Study B |
| `study_b_main_experiment/` | Main experiment (pre-post design, recipe-blocked randomization) | 153 | Study B |

Full technical documentation: `docs/hf_spaces_README.md`.

### Study B — Mini-Pilot (`study_b_mini_pilot/app.py`)

**Flow:** Welcome (IMC: "Strongly agree") → Consent → Instructions → Pilot (2 pairs/person) → End

**Per pair:** Recipe card + Dialog A + Dialog B, then:
- `dqA_overall` + `dqB_overall` (Likert 1-7)
- Forced choice: "Which dialog communicated better overall?" (Dialog A / Dialog B)

**Reservation system:** Slots reserved on page load (timeout 20 min),
decremented only on submit. Prevents double allocation in parallel sessions.

**Pair assignment:** Deterministic via `stable_hash(rater_id) % n_recipes`,
fallback to next available pair by rank_score.
Presentation order good/bad: randomized per rater (`rng_for_rater(rater_id)`).

**Result:** N=34, all 17 pairs RETAIN (100% forced_good_prop, D=4.68).

**Analysis:** `notebooks/study_preparation/Mini_Pilot_StudyB_Auswertung.ipynb`


## Model and API Configuration

| Parameter | Value |
|---|---|
| Base model | `llama-3.3-70b-instruct` |
| API | AcademicCloud SAIA (`https://chat-ai.academiccloud.de/v1`) |
| temperature (dialog generation) | 0.35 |
| temperature (LLM evaluation, all runs) | 0.2 |
| max_tokens (generation) | 1100 |
| max_tokens (evaluation) | 400 |
| LLM evaluation runs | 3 (median as final score) |
| Aggregation | Median across 3 runs |
| GLOBAL_SEED | 42 |
| Bootstrap R=5,000 (stratified) | all inference estimates |
| Gradio | 5.49.1 |
| MIN_DIALOG_TIME (Study A) | 20.0 s |
| MAX_MAIN_DIALOGS (Study A) | 10 |
| PAIRS_PER_PARTICIPANT (Mini-Pilot B) | 2 |
| TARGET_PAIR_RATINGS_PER_PAIR (Mini-Pilot B) | 4 |
| RESERVATION_TIMEOUT (Mini-Pilot B) | 20 min |
| base_slots (Study B main seed) | 4 |
| extra_recipes (Study B main seed) | 7 (smallest human_delta_overall) |


## Reproducibility

- **API key** (AcademicCloud SAIA): `api_key = ""` in all notebooks — insert at runtime.
- **HF token** (private dataset): set as `HF_TOKEN` Space Secret, not in code.
- **HF cache**: default `"/content/drive/MyDrive/hf_cache"`, adjust for local execution.
- **R seed**: `set.seed(1234)` at the start of `studyA_analysis.R`.
- **R seed**: `set.seed(42)` inside `run_ancova()` in `studyB_analysis.R`.
- **File paths in R scripts**: relative paths expected (`data/` and `outputs/`).
  For Google Drive: adjust to `/content/drive/MyDrive/Masterarbeit/Dialoge/StudyA/studyA_exports/`.


## Citation

```
Kampa, K. (2026). Können LLMs gut diskutieren? [Master's thesis]. University of Regensburg.
```
