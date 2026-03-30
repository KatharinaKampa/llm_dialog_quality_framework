# Hugging Face Spaces — Documentation

This folder contains the source code of all data collection interfaces developed
and used for the master's thesis "Können LLMs gut diskutieren?" (Kampa, 2026).

All interfaces were implemented with **Gradio 5.49.1** (Python) and hosted on
Hugging Face Spaces.


## Space Overview

| Folder | Purpose | Study | Status |
|---|---|---|---|
| `study_a_rater_interface/` | Human gold standard: raters evaluate dialogs on 7-point Likert scales | Study A (main data collection) | Completed |
| `study_b_mini_pilot/` | Mini-pilot for stimulus validation (N=34, forced choice + Likert) | Study B (pre-study) | Completed |
| `study_b_main_experiment/` | Main experiment: pre-post design with randomized dialog assignment | Study B (main data collection) | Completed |


## `study_a_rater_interface/` - Full Documentation

### Flow

```
Welcome → Consent → Instructions → Practice (3 dialogs) → Main (10 dialogs) → End (Debriefing)
```

### Stages in Detail

**Welcome:**
- Rater ID input (optional; if empty: auto-generated `R{random 5-digit}`)
- English proficiency confirmation (`english_ok_cb`)
- Global IMC: radio item "To show that you read carefully, please choose 'Strongly disagree'"
  --> only `Strongly disagree` activates the Start button
- `imc_pass_global` stored as state and included with each rating row

**Consent:**
- Full informed consent text (purpose, voluntary participation, data protection, risks)
- Checkbox confirmation required
- Dialog assignment (slot consumption) occurs only after consent confirmation

**Instructions:**
- Task description, scale explanation (1/4/7 principle), note on practice phase
- Checkbox confirmation required

**Practice (3 dialogs):**
- Dialogs from `gold_practice.csv` (12 practice dialogs; 3 randomly selected)
- 4 scales: clarity, relevance, respect_appreciation, overall
- After each practice dialog: feedback text with ±1 tolerance against gold reference value
- Practice ratings do **not** enter the analysis

**Main (10 dialogs):**
- 8 scales (7-point Likert): truthfulness, relevance, clarity,
  relation_appropriateness, logic_coherence, respect_appreciation,
  feedback_depth, overall
- Scales in 3 groups: Content & Reasoning / Interpersonal & Tone / Overall
- Comment field (optional)
- Progress display: "Main: X / 10"
- Scroll-to-top after each transition (JavaScript)

**End:**
- Debriefing text (purpose, disclosure of manipulation, contact details)
- Display of rater ID

---

### Slot System (Technical Details)

The slot system ensures that exactly 2 independent raters are assigned per dialog
and that assignments remain stable across sessions.

**Persistence:** The runtime copy of the assignment file is stored at
`/data/studyA/studyA_assignment_slots.runtime.csv` (Space Persistent Storage).
On each Space restart this file is loaded preferentially; only on the very first
start is the seed file from the HF dataset initialized.

**Slot assignment per rater (STRICT MODE, default):**
1. 1 IMC dialog (from `is_imc==1` pool) is inserted at position [2] or [3]
2. 1 Gold dialog (from `is_gold==1` pool, excluding the chosen IMC) is inserted
   at position [6] or [7]
3. The remaining 8 slots are filled with regular dialogs
   (balanced: ~4 good + ~4 bad where possible)
4. All positions are set deterministically via `RANDOM_SEED=42 + rater_id` hash

**RELAXED MODE** (only when STRICT is not possible, e.g., regular pool exhausted):
Regular dialogs prioritized; GOLD/IMC mixed in to fill remaining slots.

**Slot decrement:** After assignment, `remaining_slots` is decremented by 1 for
all chosen `dialog_id`s and the runtime file is saved. Additionally,
`ratings/assignment_status.csv` in the HF dataset is updated.

**Rater ID stability:** The rater ID is created in the Welcome stage and stored
in `pending_rater_id_state`. Slot consumption occurs only after consent.

---

### Ratings Persistence

One CSV file is created per rater: `ratings/ratings_<rater_id>.csv`
in the private HF dataset (`KKam799/studyA-storage`).

**Stored fields per row:**

| Field | Description |
|---|---|
| `start_time` | Unix timestamp at first dialog page load |
| `timestamp` | Unix timestamp at submit |
| `rater_id` | Pseudonymized rater ID |
| `dialog_id` | Dialog ID (D001-D080) |
| `recipe_title` | Recipe name |
| `recipe_type` | savory / sweet |
| `condition` | good / bad |
| `is_gold` | 0/1 |
| `gold_expected_min` / `gold_expected_max` | Expected value range (±1 from LLM overall) |
| `is_imc` | 0/1 |
| `imc_expected_overall` | Expected overall value (exact) |
| `imc_pass_global` | Global IMC passed (0/1) |
| `imc_dialog_pass` | Embedded IMC passed (0/1) |
| `min_time_ok_dialog` | ≥ 20s on dialog page (0/1) |
| `gold_pass` | Gold tolerance range met (0/1) |
| `duration_sec` | Time spent on dialog page |
| `comment` | Free-text comment |
| `truthfulness` ... `overall` | 8 Likert ratings (1-7) |

A local backup is additionally stored at
`/data/studyA/results/ratings_<rater_id>.csv`.

---

### Private HF Dataset (`KKam799/studyA-storage`)

| File / Path | Content |
|---|---|
| `studyA_dialogs_from_goodbad.csv` | 80 stimuli (D001–D080) with columns: `dialog_id`, `recipe_title`, `recipe_type`, `condition`, `dialog_text`, `is_gold`, `gold_expected_min`, `gold_expected_max`, `is_imc`, `imc_expected_overall` |
| `studyA_assignment_slots.csv` | Seed file with `remaining_slots=2` per dialog |
| `gold_practice.csv` | 12 practice dialogs (6 GOOD + 6 BAD) based on `masterlist_GOLD.csv`; columns as stimuli file plus gold reference values |
| `ratings/ratings_<rater_id>.csv` | Ratings per rater (append logic) |
| `ratings/assignment_status.csv` | Current slot consumption (updated after each submit) |

---

### Environment Variables (Space Secrets)

| Variable | Required | Description |
|---|---|---|
| `HF_TOKEN` | Yes | HuggingFace token with READ + WRITE access to `KKam799/studyA-storage` |
| `DATASET_ID` | No | Default: `KKam799/studyA-storage` |
| `MAX_MAIN_DIALOGS` | No | Default: `10` |
| `MIN_DIALOG_TIME` | No | Default: `20.0` (seconds) |
| `RANDOM_SEED` | No | Default: `42` |


## `study_b_mini_pilot/` - Full Documentation

### Purpose
Second validation step (manipulation check): verifying whether the 17-18 stimulus
pairs from the pre-check are perceived as differing in communication quality
by human participants.

### Flow

```
Welcome (IMC) → Consent → Instructions → Pilot (2 pairs/person) → End
```

### Stages in Detail

**Welcome:**
- Participant ID (optional; fallback: `P{random 5-digit}`)
- English proficiency confirmation
- Global IMC: radio "To show that you read carefully, please choose 'Strongly agree'"
  --> only `Strongly agree` activates the Start button

**Consent:**
- Short informed consent text (purpose, duration ~6-12 min, voluntary participation)
- Checkbox confirmation required

**Pilot (2 pairs):**
Per pair:
1. Recipe card (title, type, ingredients, description, directions)
2. Dialog A (bubble format: Fat left / Carb right)
3. `dqA_overall` — Likert 1-7: "How would you rate the quality of the communication?"
4. Dialog B
5. `dqB_overall` — Likert 1-7 (same question)
6. Forced choice: "Which dialog communicated better overall?" (Dialog A / Dialog B)
7. Comment field (optional)

---

### Reservation System (Technical Details)

The reservation system prevents two participants from working on the same pair
simultaneously and consuming slots beyond the target value.

**Procedure:**
1. **Allocation:** Pair assigned to participant + reservation created
   (locally at `/data/studyB_pilot/pilot_ratings/pilot_reservations.csv`)
2. **Timeout:** Reservation expires after 20 minutes, releasing the pair
3. **Submit:** `confirm_and_finalize()` - remove reservation + decrement slots
4. **Rejection:** Expired reservations are rejected on submit
   (no data saved, participant sees timeout message)

**Slot logic:**
- `assignment_status_df` reloaded from local CSV on each submit
  (no stale in-memory state)
- After submit: single HF commit for ratings + status (via `create_commit()` with
  `CommitOperationAdd` - prevents parallel commit conflicts)
- Local backup: `/data/studyB_pilot/results/ratings_<rater_id>.csv`

**Pair assignment:**
- Deterministic: `stable_hash(rater_id) % n_recipes` --> next available pair
  (ordered by rank_score, reservations considered)
- Dialog A/B order: `rng_for_rater(rater_id).random() < 0.5`
  (random per rater, deterministic across sessions)
- Second pair: excludes already-seen `pair_index` values

---

### Evaluation Decision Rules (Preregistered)

Per `pair_index` (0-16):

| Condition | Decision |
|---|---|
| `n_evaluations` < 3 | EXCLUDE (insufficient data) |
| `mean_diff` ≥ 0.50 AND `forced_good_prop` ≥ 0.60 | RETAIN |
| `mean_diff` < 0.50 AND `forced_good_prop` < 0.60 | EXCLUDE |
| Borderline with correct direction + `within_consistency` ≥ 0.60 | RETAIN |

Definitions:
- `mean_diff` = mean(good_rating - bad_rating) across all evaluations
- `forced_good_prop` = proportion of evaluations where the good dialog was chosen
- `within_consistency` = proportion of evaluations where good_rating > bad_rating

**Result:** All 17 pairs retain (mean_diff=4.68, forced_good_prop=1.00, N=34)

---

### Private HF Dataset (`KKam799/studyB_Mini-Pilot-storage`)

| File / Path | Content |
|---|---|
| `studyB_stimuli.csv` | 34 dialogs (17 pairs) with all columns: `dialog_id`, `recipe_title`, `recipe_type`, `condition`, `dialog_text`, `fat`, `carbs`, `calories`, `protein`, `servings`, `description`, `directions`, `ingredients_list` |
| `studyB_pilot_assignment_seed.csv` | 18 pairs, 4 slots/dialog; columns: `dialog_id`, `recipe_title`, `condition`, `remaining_slots`, `initial_slots` |
| `studyB_top_pool_pairs.csv` | Pair metadata: `recipe_title`, `dialog_id_good`, `dialog_id_bad`, `pass_precheck`, `rank_score`, deltas |
| `pilot_ratings/pilot_all_ratings.csv` | All ratings (append); schema below |
| `pilot_ratings/pilot_assignment_status.csv` | Current slot consumption |

**Stored fields per rating row:**

| Field | Description |
|---|---|
| `timestamp` | Unix timestamp |
| `rater_id` | Pseudonymized participant ID |
| `imc_pass` | Global IMC passed (0/1) |
| `pair_no` | Within-person position (1 or 2) |
| `pair_index` | Global stimulus pair index (0-16) |
| `rank_score` | Pair rank_score from pre-check |
| `recipe_title` | Recipe name |
| `order` | good_first / bad_first |
| `dialog_id_good` / `dialog_id_bad` | Dialog IDs |
| `dialogA_true` / `dialogB_true` | True condition of Dialog A/B |
| `dialog_id_A` / `dialog_id_B` | Actually displayed dialog IDs |
| `dqA_overall` | Likert 1-7 for Dialog A |
| `dqB_overall` | Likert 1-7 for Dialog B |
| `forced_choice` | "Dialog A" or "Dialog B" |
| `comment` | Free-text comment |

---

### Environment Variables (Space Secrets)

| Variable | Required | Default |
|---|---|---|
| `HF_TOKEN` | Yes | - |
| `DATASET_ID` | No | `KKam799/studyB_Mini-Pilot-storage` |
| `TOP_X` | No | `17` |
| `PAIRS_PER_PARTICIPANT` | No | `2` |
| `TARGET_PAIR_RATINGS_PER_PAIR` | No | `4` |
| `RESERVATION_TIMEOUT_MINUTES` | No | `20` |
| `RANDOM_SEED` | No | `42` |


## `study_b_main_experiment/` - Full Documentation

### Purpose
Main data collection for Study B: randomized between-subjects experiment
(N=153 raw, N=139 PRIMARY) on the causal effect of dialog quality on recipe
evaluations. Preregistered on OSF prior to data collection.

### Flow

```
Welcome (IMC) → Consent → Instructions → Screening/Preferences
        ↓
Continue to recipe → Reservation (slot reserved, not yet decremented)
        ↓
Pre-measures (diet suitability, stars, cooking intention, saving intention,
              nutrient estimates)
        ↓
Dialog (timer running)
        ↓
Post-measures + dialog quality (5 items) + manipulation check
        ↓
Submit → Commit + slot decrement + persist → Debriefing (nutrition facts revealed)
```

### Randomization (Recipe-Blocked Quota)

Assignment via `pick_dialog_recipe_blocked()`:
1. Preference matching: recipes with maximum match score to stated preferences
   (lower-fat ≤5.98g, lower-carb ≤10.57g, lower-calorie ≤106kcal, sweet/savory)
2. Recipe blocking: recipes with remaining slots in both conditions preferred,
   weighted by remaining slots
3. Deterministic RNG: `hashlib.sha256(f"{RANDOM_SEED}::{rater_id}")` — reproducible
   but not predictable for participants
4. Slot system: `studyB_assignment_seed.csv` (base_slots=4, extra_recipes=7) defines
   initial quotas; reservation on assignment, decrement only on submit

### Reservation System

Identical to mini-pilot but with `fcntl`-based file lock for concurrent requests
and TTL of 30 minutes. Reservation status: `reserved | committed | expired | cancelled`.

### Collected Variables (one row per participant)

**Onboarding:** `rater_id`, `imc_pass`, `preferences`, `involvement`,
`nutrition_knowledge`, `english_skill`, `match_score`, `prefs_satisfied`,
`fallback_used`

**Stimulus:** `dialog_id`, `recipe_title`, `recipe_type`, `condition`,
`true_fat_g`, `true_carb_g`, `true_kcal`

**Pre-measurement:** `pre_diet_suitability`, `pre_recipe_stars`, `pre_cook_intent`,
`pre_save_intent`, `pre_est_fat_g`, `pre_est_carb_g`, `pre_est_kcal`

**Dialog:** `dialog_duration_sec`, `min_time_ok_dialog` (≥ 10s = 1)

**Post-measurement:** `post_diet_suitability`, `post_recipe_stars`, `post_cook_intent`,
`post_save_intent`, `post_est_fat_g`, `post_est_carb_g`, `post_est_kcal`,
`abs_err_fat_post`, `abs_err_carb_post`, `abs_err_kcal_post`, `nutrient_outlier_flag`

**Dialog quality:** `dq_clarity`, `dq_relevance`, `dq_respect`, `dq_logic`,
`dq_coherence`, `dq_mean` (mean), `manip_check`

**Outlier flag:** `nutrient_outlier_flag = 1` if any estimate > 1,000g or kcal > 10,000

### Debriefing
After submit: disclosure of true nutrition values (fat, carbs, kcal) and explanation
of the manipulation (good vs. bad communication quality).

### Private HF Dataset (`KKam799/studyB-storage`)

| File / Path | Content |
|---|---|
| `studyB_stimuli.csv` | 34 dialogs (17 pairs) with nutrition data |
| `studyB_assignment_seed.csv` | Slot quotas (4-5 slots/dialog) |
| `ratings/ratings_<rater_id>.csv` | Raw data per participant (append) |
| `ratings/assignment_status.csv` | Slot consumption (throttled, every 10 commits or 3 min) |

### QA and Analysis

**QA pipeline** (Python notebook `StudyB_QA_and_Merge.ipynb`):

| Step | Function |
|---|---|
| Merge | Individual `ratings_*.csv` --> `studyB_all_ratings_merged.csv` |
| PRIMARY | `run_qa_pipeline()`: IMC --> dialog time → nutrient outlier → straightlining --> listwise |
| SENS | `run_qa_pipeline_sens()`: IMC + listwise only |
| LLM merge | Left join on `dialog_id` from `llm_median_all_dialogs.csv` |

**Statistical analysis:** `r_scripts/studyB_analysis.R`

### Environment Variables (Space Secrets)

| Variable | Required | Default |
|---|---|---|
| `HF_TOKEN` | Yes | - |
| `DATASET_ID` | No | `KKam799/studyB-storage` |
| `STIMULI_FILE` | No | `studyB_stimuli.csv` |
| `ASSIGN_FILE` | No | `studyB_assignment_seed.csv` |
| `MIN_DIALOG_TIME` | No | `10.0` |
| `RESERVATION_TTL_SEC` | No | `1800` (30 min) |
| `RANDOM_SEED` | No | `42` |
| `ASSIGN_UPLOAD_EVERY_N` | No | `10` |
| `ASSIGN_UPLOAD_MIN_INTERVAL` | No | `180` (3 min) |


## Reproducibility

- **Private HF datasets** are not publicly accessible (require `HF_TOKEN`).
  The slot logic is fully traceable in the respective `app.py` and can be
  reconstructed from the raw data (`data/processed/`).
- The Spaces are **no longer active** after data collection was completed.
  For replication they must be deployed locally or in a new HF Space.
  All necessary files are contained in this folder.
- Gradio version and further dependencies: `requirements.txt` in the root folder.
