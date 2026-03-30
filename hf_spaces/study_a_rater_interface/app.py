# app.py - Study A (HF Space)
# This file defines the Hugging Face Spaces version of the Study A Gradio application.
# The application loads stimuli, practice items, and assignment metadata from a private Hugging Face dataset.
# Runtime data is handled as follows:
# - The assignment runtime copy is persisted to /data to keep slot usage consistent across sessions.
# - Ratings are appended to a per-rater CSV file in the dataset under ratings/ratings_<rater_id>.csv.

# This import section provides standard-library modules that are used for configuration, text processing,
# timing, numeric checks, randomness, logging, hashing, and file system paths.
import os
import re
import time
import math
import random
import logging
import hashlib
from pathlib import Path
from typing import Optional

# These imports provide the external libraries for data handling (pandas), UI rendering (gradio),
# and downloading dataset files from Hugging Face Hub (huggingface_hub).
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download, HfApi


# ------------------------------------------------------------
# Global configuration

# This section defines global configuration values for the Space runtime environment.
# It configures logging, reads environment variables, and defines study parameters and local paths.

# configures the global logging format and verbosity
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
# used for structured log messages throughout the app
logger = logging.getLogger("studyA_space")

# stores the Hugging Face dataset repository ID (contains stimuli and runtime outputs)
DATASET_ID = os.getenv("DATASET_ID", "KKam799/studyA-storage")

# store filenames within the dataset repository for stimuli, practice items, and assignment slots
STIMULI_FILE = os.getenv("STIMULI_FILE", "studyA_dialogs_from_goodbad.csv")
PRACTICE_FILE = os.getenv("PRACTICE_FILE", "gold_practice.csv")
ASSIGN_FILE = os.getenv("ASSIGN_FILE", "studyA_assignment_slots.csv")

# reads Hugging Face token from the environment (required for private dataset access)
HF_TOKEN = os.getenv("HF_TOKEN")

# dataset output folder (in the dataset repo)
DATASET_RATINGS_DIR = os.getenv("DATASET_RATINGS_DIR", "ratings")

# print active configuration values to make runtime behavior transparent
logger.info(f"[LOAD] DATASET_ID={DATASET_ID}")
logger.info(f"[LOAD] STIMULI_FILE={STIMULI_FILE}")
logger.info(f"[LOAD] PRACTICE_FILE={PRACTICE_FILE}")
logger.info(f"[LOAD] ASSIGN_FILE={ASSIGN_FILE}")
logger.info(f"[LOAD] HF_TOKEN present={bool(HF_TOKEN)}")
logger.info(f"[LOAD] DATASET_RATINGS_DIR={DATASET_RATINGS_DIR}")

# Study parameters
MAX_MAIN_DIALOGS = int(os.getenv("MAX_MAIN_DIALOGS", "10")) # number of dialogs shown in main phase
MIN_DIALOG_TIME = float(os.getenv("MIN_DIALOG_TIME", "20.0")) # min. time (in seconds) required for dialog to count as sufficiently read

# Assignment config
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42")) # to derive stable per-rater random generators
# allowed insertion positions (0-based indices) for IMC and GOLD dialogs
IMC_POS_SET = [2, 3]    # 0-based -> 3rd or 4th dialog
GOLD_POS_SET = [6, 7]   # 0-based -> 7th or 8th dialog

# Persistent storage paths (Space)
SPACE_DATA_DIR = Path(os.getenv("SPACE_DATA_DIR", "/data/studyA"))
SPACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

RUNTIME_ASSIGN_PATH = SPACE_DATA_DIR / "studyA_assignment_slots.runtime.csv"
ASSIGN_STATUS_PATH = SPACE_DATA_DIR / "studyA_assignment_slots.status.csv"

# Optional local backup
RESULTS_DIR = SPACE_DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Scroll-to-top JS

# This JavaScript function scrolls the viewport to the top anchor after UI transitions.
# It returns the original input arguments so that Gradio continues the normal event flow.

SCROLL_TOP_JS = """
(...args) => {
  try {
    const el = document.getElementById("page-top-anchor");
    if (el) {
      el.scrollIntoView({ block: "start", behavior: "auto" });
    } else {
      window.scrollTo(0, 0);
    }
  } catch (e) {
    try { window.scrollTo(0, 0); } catch (e2) {}
  }
  // IMPORTANT: return original inputs so Gradio continues
  return args;
}
"""


# ------------------------------------------------------------
# Small utilities (clean text, newlines, safe conversions)

# This section defines small helper functions that normalize dialog formatting
# and perform safe type conversions for robust downstream processing.

def fix_dialog_newlines(text):
    """Normalizes newline markers before 'Fat:' and 'Carb:' turns to enforce consistent turn boundaries."""
    if not isinstance(text, str):
        return text
    text = re.sub(r"\*sFat:", r"\nFat:", text)
    text = re.sub(r"\*sCarb:", r"\nCarb:", text)
    return text.strip()


def safe_text(x):
    """Converts an input into a clean string and maps None/NaN/'nan'/empty values to an empty string."""
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if (s == "" or s.lower() == "nan") else s


def _to_int_safe(x):
    """Converts an input into an integer and returns None when conversion is not possible."""
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except Exception:
        pass
    try:
        return int(x)
    except Exception:
        try:
            return int(float(str(x)))
        except Exception:
            return None


# ------------------------------------------------------------
# HF dataset download / upload helpers

# This section defines helper functions for reading and writing files in the Hugging Face dataset repository.
# It enforces the presence of HF_TOKEN because the dataset is private.

def hf_download_csv(filename: str) -> pd.DataFrame:
    """Downloads a CSV file from the Hugging Face dataset repository and returns it as a pandas DataFrame."""
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it in Space Settings -> Secrets as HF_TOKEN "
            "(token needs at least READ access to the private dataset)."
        )

    local_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=filename,
        token=HF_TOKEN,
    )
    return pd.read_csv(local_path)


def hf_download_text(path_in_repo: str) -> Optional[str]:
    """Downloads a text file from the dataset repository and returns its contents, or None if it is not available."""
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is missing. Add it in Space Settings -> Secrets as HF_TOKEN.")

    try:
        p = hf_hub_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            filename=path_in_repo,
            token=HF_TOKEN,
        )
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.info(f"[hf_download_text] could not download {path_in_repo} (maybe first write): {e!r}")
        return None


def upload_text_to_dataset(text: str, path_in_repo: str, commit_message: str):
    """Uploads a text payload to the dataset repository and overwrites an existing file at the same path."""
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it in Space Settings -> Secrets as HF_TOKEN "
            "(token needs READ + WRITE access to upload)."
        )

    api = HfApi()
    api.upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo=path_in_repo,
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=commit_message,
    )


def append_row_to_rater_csv_in_dataset(rater_id: str, row: dict):
    """
    Appends one row to a per-rater CSV file in the dataset under ratings/.
    Creates the file including a header if it does not exist yet.
    """
    rid = safe_text(rater_id) or "RATER_NONE"
    path_in_repo = f"{DATASET_RATINGS_DIR}/ratings_{rid}.csv"

    df_new = pd.DataFrame([row])
    new_csv = df_new.to_csv(index=False, encoding="utf-8")

    existing = hf_download_text(path_in_repo)
    if existing and existing.strip():
        # Append without header
        appended = existing.rstrip("\n") + "\n" + "\n".join(new_csv.splitlines()[1:]) + "\n"
        upload_text_to_dataset(appended, path_in_repo, commit_message=f"Append rating row for {rid}")
    else:
        # Create with header
        upload_text_to_dataset(new_csv, path_in_repo, commit_message=f"Create ratings file for {rid}")

    logger.info(f"[UPLOAD] appended rating to dataset file: {DATASET_ID}/{path_in_repo}")


# ------------------------------------------------------------
# Load stimuli, practice and assignment seed

# This section defines in-memory caches for stimuli and practice data and provides loader functions.
# It uses lazy loading so that the dataset is downloaded only when needed.

dialogs_df = pd.DataFrame()
dialogs_list = []
practice_gold_pool = []
HAS_PRACTICE_GOLD = False

def load_assets_or_raise():
    """Lazily loads stimuli and practice data from the dataset and caches them in global variables."""
    global dialogs_df, dialogs_list, practice_gold_pool, HAS_PRACTICE_GOLD

    if not dialogs_df.empty and dialogs_list:
        return

    logger.info(f"[LOAD] DATASET_ID={DATASET_ID}")
    logger.info(f"[LOAD] STIMULI_FILE={STIMULI_FILE}")
    logger.info(f"[LOAD] PRACTICE_FILE={PRACTICE_FILE}")

    # Stimuli
    df = hf_download_csv(STIMULI_FILE)

    if "dialog_text" in df.columns:
        df["dialog_text"] = df["dialog_text"].apply(fix_dialog_newlines)
    elif "discussion" in df.columns:
        df["dialog_text"] = df["discussion"].apply(fix_dialog_newlines)

    if "recipe_title" not in df.columns and "recipe" in df.columns:
        df["recipe_title"] = df["recipe"]

    dialogs_df = df.copy()
    dialogs_list = dialogs_df.to_dict(orient="records")
    logger.info(f"[LOAD] stimuli loaded shape={dialogs_df.shape}")

    # Practice (Gold practice)
    # attempts to load gold-labeled practice dialogs, falls back if the file is not available
    try:
        df_pr = hf_download_csv(PRACTICE_FILE)

        if "dialog_text" in df_pr.columns:
            df_pr["dialog_text"] = df_pr["dialog_text"].apply(fix_dialog_newlines)
        elif "discussion" in df_pr.columns:
            df_pr["dialog_text"] = df_pr["discussion"].apply(fix_dialog_newlines)

        if "recipe_title" not in df_pr.columns and "recipe" in df_pr.columns:
            df_pr["recipe_title"] = df_pr["recipe"]

        practice_gold_pool = df_pr.to_dict(orient="records")
        HAS_PRACTICE_GOLD = len(practice_gold_pool) > 0
        logger.info(f"[PRACTICE_GOLD] loaded {len(practice_gold_pool)} practice dialogs from {PRACTICE_FILE}")
    except Exception:
        logger.exception("[PRACTICE_GOLD] Could not load practice gold CSV; fallback to generic practice.")
        practice_gold_pool = []
        HAS_PRACTICE_GOLD = False


def write_assignment_status(assign_df: pd.DataFrame):
    """
    Write a human-readable slot status table to disk and optionally upload it to the dataset.
    """
    df = assign_df.copy()

    if "initial_slots" not in df.columns:
        df["initial_slots"] = df.get("remaining_slots", 0)

    df["initial_slots"] = pd.to_numeric(df["initial_slots"], errors="coerce").fillna(0).astype(int)
    df["remaining_slots"] = pd.to_numeric(df["remaining_slots"], errors="coerce").fillna(0).astype(int)

    df["used_slots"] = (df["initial_slots"] - df["remaining_slots"]).clip(lower=0).astype(int)

    preferred = ["dialog_id", "initial_slots", "remaining_slots", "used_slots"]
    extra_cols = [c for c in df.columns if c not in preferred]
    df = df[[*preferred, *extra_cols]]

    df.to_csv(ASSIGN_STATUS_PATH, index=False, encoding="utf-8")
    logger.info(f"[ASSIGN_STATUS] wrote status file to: {ASSIGN_STATUS_PATH}")

    old = hf_download_text(f"{DATASET_RATINGS_DIR}/assignment_status.csv")
    new = df.to_csv(index=False, encoding="utf-8")
    logger.info(f"[ASSIGN_STATUS_DEBUG] status_changed={old != new if old is not None else True}")
    
    # upload to dataset repo for external visibility
    try:
        text = df.to_csv(index=False, encoding="utf-8")
        upload_text_to_dataset(
            text=text,
            path_in_repo=f"{DATASET_RATINGS_DIR}/assignment_status.csv",
            commit_message="Update assignment slot status",
        )
        logger.info("[ASSIGN_STATUS] uploaded assignment_status.csv to dataset")
    except Exception:
        logger.exception("[ASSIGN_STATUS] failed to upload assignment_status.csv to dataset")
        

def load_or_init_assignment_runtime() -> pd.DataFrame:
    """Loads the runtime assignment file from /data or initializes it from the dataset assignment seed file."""
    if RUNTIME_ASSIGN_PATH.exists():
        logger.info(f"[ASSIGN] Using existing runtime assignment: {RUNTIME_ASSIGN_PATH}")
        df = pd.read_csv(RUNTIME_ASSIGN_PATH)

        # normalize ids and numeric columns
        df["dialog_id"] = df["dialog_id"].astype(str).str.strip()

        if "remaining_slots" not in df.columns:
            df["remaining_slots"] = 2
        df["remaining_slots"] = pd.to_numeric(df["remaining_slots"], errors="coerce").fillna(0).clip(lower=0).astype(int)

        # ensure initial_slots exists once and persists (important for stable used_slots)
        if "initial_slots" not in df.columns:
            df["initial_slots"] = df["remaining_slots"].copy()
            df["initial_slots"] = pd.to_numeric(df["initial_slots"], errors="coerce").fillna(0).clip(lower=0).astype(int)

            # persist the upgraded runtime so future loads keep initial_slots
            df.to_csv(RUNTIME_ASSIGN_PATH, index=False, encoding="utf-8")

        write_assignment_status(df)
        return df

    logger.info(f"[ASSIGN] Initializing runtime assignment from dataset file: {ASSIGN_FILE}")
    seed_assign = hf_download_csv(ASSIGN_FILE)

    if "dialog_id" not in seed_assign.columns:
        raise ValueError("Assignment CSV is missing required column: dialog_id")

    seed_assign["dialog_id"] = seed_assign["dialog_id"].astype(str).str.strip()

    if "remaining_slots" not in seed_assign.columns:
        logger.warning("[ASSIGN] remaining_slots missing; initializing to 2")
        seed_assign["remaining_slots"] = 2

    seed_assign["remaining_slots"] = pd.to_numeric(seed_assign["remaining_slots"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    if "initial_slots" not in seed_assign.columns:
        seed_assign["initial_slots"] = seed_assign["remaining_slots"].copy()
    seed_assign["initial_slots"] = pd.to_numeric(seed_assign["initial_slots"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    seed_assign.to_csv(RUNTIME_ASSIGN_PATH, index=False, encoding="utf-8")
    write_assignment_status(seed_assign)
    logger.info(f"[ASSIGN] Runtime assignment saved to: {RUNTIME_ASSIGN_PATH}")
    return seed_assign


def save_assignment_runtime(assign_df: pd.DataFrame):
    """Persists an updated assignment DataFrame to the runtime assignment CSV under /data."""
    assign_df.to_csv(RUNTIME_ASSIGN_PATH, index=False, encoding="utf-8")


# ------------------------------------------------------------
# Dialog rendering (Fat/Carb bubbles)

# This section parses raw dialog text into speaker turns and renders the turns as simple HTML "bubbles"
# that are later displayed in the Gradio interface.

def parse_dialog_turns(raw: str):
    """Splits a raw dialog string into an ordered list of (speaker, content) turns."""
    if not isinstance(raw, str):
        return []
    text = re.sub(r"\s*(Fat:)", r"\n\1", raw)
    text = re.sub(r"\s*(Carb:)", r"\n\1", text)
    text = text.strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    turns = []
    for line in lines:
        if line.startswith("Fat:"):
            speaker = "Fat"
            content = line[len("Fat:"):].strip()
        elif line.startswith("Carb:"):
            speaker = "Carb"
            content = line[len("Carb:"):].strip()
        else:
            speaker = "Other"
            content = line
        turns.append((speaker, content))
    return turns


def dialog_to_bubbles(raw: str) -> str:
    """Converts a raw dialog string into HTML that uses CSS classes for speaker-specific bubbles."""
    turns = parse_dialog_turns(raw)
    if not turns:
        return ""
    html = ["<div class='dialog-container'>"]
    for speaker, content in turns:
        if speaker == "Fat":
            bubble_class = "bubble bubble-fat"
            name = "Fat"
        elif speaker == "Carb":
            bubble_class = "bubble bubble-carb"
            name = "Carb"
        else:
            bubble_class = "bubble bubble-other"
            name = ""
        speaker_html = f"<div class='speaker'>{name}</div>" if name else ""
        html.append(
            f"<div class='{bubble_class}'>{speaker_html}<div class='bubble-text'>{content}</div></div>"
        )
    html.append("</div>")
    return "\n".join(html)


def show_current_dialog(main_index, dialogs):
    """Returns the currently selected dialog (by index) as rendered HTML."""
    if dialogs is None or main_index is None:
        return "No dialogs (state is None)."
    if main_index >= len(dialogs):
        return "No more dialogs."
    raw = dialogs[main_index]["dialog_text"]
    return dialog_to_bubbles(raw)


# ------------------------------------------------------------
# Scales and practice feedback

# This section defines the rating scales used in practice and main study phases, including grouping metadata
# and mappings needed to generate practice feedback against gold reference ratings.

# Defines each scale key and its explanatory anchor text shown in the UI
SUBSCALES = [
    ("truthfulness", "The dialog is internally consistent, without contradictions or misleading statements."),
    ("relevance", "Each turn in the dialog responds meaningfully to the previous one and stays on topic."),
    ("clarity", "The dialog is clear, easy to understand, and expressed in unambiguous language."),
    ("relation_appropriateness", "The dialog handles the interpersonal relationship appropriately, without tension, dominance, or relational violations."),
    ("logic_coherence", "The dialog presents coherent reasoning, with arguments that follow a logical structure."),
    ("respect_appreciation", "The dialog maintains a respectful tone, acknowledging the partner’s contributions appropriately."),
    ("feedback_depth", "The dialog builds meaningfully on the partner’s statements and contains sufficient conversational depth."),
    ("overall", "Overall, the dialog shows high communication quality."),
]

# structure groups scale keys into UI sections to present related scales together
SCALE_GROUPS = [
    ("Content & Reasoning", ["truthfulness", "relevance", "clarity", "logic_coherence", "feedback_depth"]),
    ("Interpersonal & Tone", ["relation_appropriateness", "respect_appreciation"]),
    ("Overall", ["overall"]),
]

# allowed Likert response options shown in the radio buttons
LIKERT_LABELS = [1, 2, 3, 4, 5, 6, 7]
# maps each subscale key to its anchor text
SUBSCALES_DICT = {key: label for key, label in SUBSCALES}

# links internal scale keys to column names used in the practice gold CSV file
GOLD_PRACTICE_COLS = {
    "truthfulness": "truthfulness",
    "relevance": "relevance",
    "clarity": "clarity",
    "relation_appropriateness": "relational_appropriateness",
    "logic_coherence": "logic_coherence",
    "respect_appreciation": "respect_appreciation",
    "feedback_depth": "feedback_depth",
    "overall": "overall_quality",
}

# scales included in the practice feedback summary
PRACTICE_FEEDBACK_KEYS = ["clarity", "relevance", "respect_appreciation", "overall"]

PRACTICE_SHORT_LABELS = {
    "clarity": "Comprehensibility",
    "relevance": "Thematic relevance",
    "respect_appreciation": "Tone of voice",
    "overall": "Overall quality",
}


def parse_likert(choice):
    """Converts a UI selection into an integer Likert value and returns None for missing or invalid inputs."""
    if choice is None or choice == "":
        return None
    try:
        return int(choice)
    except Exception:
        return None


def main_progress_text(index, total):
    """Formats human-readable progress indicator for the main study phase."""
    if total <= 0:
        return ""
    current = min(max(index + 1, 1), total)
    return f"Main: {current} / {total}"


def practice_progress_text(index, total):
    """Formats human-readable progress indicator for the practice phase."""
    if total <= 0:
        return ""
    current = min(max(index + 1, 1), total)
    return f"Practice: {current} / {total}"


def build_practice_feedback(dialog_record, rating_vals):
    """Generates a feedback text by comparing participant ratings to gold reference values with ±1 tolerance."""
    if dialog_record is None:
        return ""

    idx_map = {key: i for i, (key, _) in enumerate(SUBSCALES)}

    lines = []
    lines.append("**Feedback for the previous practice dialog**\n")
    lines.append(
        "The communication expert provided target ratings for some scales.\n"
        "Your rating is compared to this target (tolerance ±1 point)."
    )

    for key in PRACTICE_FEEDBACK_KEYS:
        if key not in idx_map:
            continue

        i = idx_map[key]
        long_label = SUBSCALES_DICT.get(key, key)
        label = PRACTICE_SHORT_LABELS.get(key, long_label)

        participant = rating_vals[i] if i < len(rating_vals) else None

        gold_col = GOLD_PRACTICE_COLS.get(key)
        gold_raw = dialog_record.get(gold_col) if gold_col else None

        gold_val = None
        if gold_raw is not None:
            try:
                gold_val = float(gold_raw)
            except Exception:
                gold_val = None

        try:
            if gold_val is not None and math.isnan(gold_val):
                gold_val = None
        except Exception:
            pass

        if participant is None or gold_val is None:
            lines.append(f"- {label}: no reference available for this scale.")
            continue

        diff = abs(participant - gold_val)
        gold_int = int(round(gold_val))

        if diff <= 1:
            prefix = "✓ close"
        elif diff <= 2:
            prefix = "! somewhat different"
        else:
            prefix = "X clearly different"

        lines.append(f"- {label}: {prefix} — your rating **{participant}**, expert **{gold_int}**.")

    return "\n".join(lines)


# ------------------------------------------------------------
# IMC/GOLD position validation

# This section validates the configured IMC and GOLD insertion position sets against the main study length.
# It raises errors early if configuration values are inconsistent or out of range.

def validate_imc_gold_pos_sets(n_main: int, imc_pos_set, gold_pos_set):
    """Validates IMC_POS_SET and GOLD_POS_SET for a given number of main dialogs."""
    n_main = int(n_main)
    if n_main < 2:
        raise ValueError(f"n_main must be >= 2 (got {n_main})")

    if not imc_pos_set or not gold_pos_set:
        raise ValueError("IMC_POS_SET and GOLD_POS_SET must be non-empty")

    imc_pos_set = [int(x) for x in imc_pos_set]
    gold_pos_set = [int(x) for x in gold_pos_set]

    bad_imc = [p for p in imc_pos_set if p < 0 or p >= n_main]
    bad_gold = [p for p in gold_pos_set if p < 0 or p >= n_main]
    if bad_imc:
        raise ValueError(f"IMC_POS_SET contains out-of-range positions for n_main={n_main}: {bad_imc}")
    if bad_gold:
        raise ValueError(f"GOLD_POS_SET contains out-of-range positions for n_main={n_main}: {bad_gold}")

    has_distinct_pair = any(i != g for i in imc_pos_set for g in gold_pos_set)
    if not has_distinct_pair:
        raise ValueError(f"IMC_POS_SET and GOLD_POS_SET do not allow distinct placement (n_main={n_main}).")

    return True

validate_imc_gold_pos_sets(MAX_MAIN_DIALOGS, IMC_POS_SET, GOLD_POS_SET)


# ------------------------------------------------------------
# Slot-based assignment (runtime copy in /data)

# This section assigns a fixed number of main dialogs to each rater while enforcing slot limits.
# It uses a runtime copy of the assignment table stored in /data so that remaining slots persist across sessions.

def rng_for_rater(rater_id: str) -> random.Random:
    """Creates a stable per-rater random generator based on RANDOM_SEED and the rater_id."""
    rid = (str(rater_id).strip() if rater_id is not None else "RATER_NONE")
    s = f"{RANDOM_SEED}::{rid}".encode("utf-8")
    seed_int = int(hashlib.sha256(s).hexdigest()[:16], 16)
    return random.Random(seed_int)


def _decrement_slots(assign_df, used_ids):
    """Decrements remaining_slots for the selected dialog_ids and clamps values to be non-negative."""
    # normalize ids on both sides
    assign_df["dialog_id"] = assign_df["dialog_id"].astype(str).str.strip()
    used_ids = [str(x).strip() for x in used_ids]

    mask = assign_df["dialog_id"].isin(used_ids)
    matched = int(mask.sum())

    # DEBUG: if this is 0, nothing will ever update
    logger.info(f"[ASSIGN_DEBUG] decrement matched_rows={matched} / used_ids={len(used_ids)}")

    before_sum = int(pd.to_numeric(assign_df["remaining_slots"], errors="coerce").fillna(0).sum())

    assign_df.loc[mask, "remaining_slots"] = (
        pd.to_numeric(assign_df.loc[mask, "remaining_slots"], errors="coerce").fillna(0) - 1
    )
    assign_df.loc[mask, "remaining_slots"] = assign_df.loc[mask, "remaining_slots"].clip(lower=0).astype(int)

    after_sum = int(assign_df["remaining_slots"].sum())
    logger.info(f"[ASSIGN_DEBUG] remaining_slots sum before={before_sum} after={after_sum}")

    return assign_df


def make_main_dialogs_for_rater(rater_id, n_main=MAX_MAIN_DIALOGS):
    """
    Selects up to n_main dialogs for a given rater and updates the runtime remaining_slots accordingly.

    Guarantees:
    - STRICT MODE (default if feasible): places exactly 1 IMC dialog and exactly 1 GOLD dialog (distinct dialog_ids),
      and fills the remaining (n_main - 2) with neutral dialogs.
    - RELAXED MODE (only if STRICT is not feasible, e.g., neutral pool insufficient): assigns exactly n_main dialogs
      by filling with as many neutral dialogs as possible, then mixing GOLD and IMC (no longer enforcing "max 1 IMC"
      or "max 1 GOLD" per rater)
    - returns exactly n_main (or fewer if the remaining pool is smaller)
    - tries to include both conditions (good and bad) among fillers in STRICT MODE if possible
    - logs pool sizes and final length for debugging

    Important robustness changes:
    - n_main is reduced dynamically to the number of available dialogs with remaining_slots > 0
    - STRICT is used only when it can be satisfied (neutral >= n_main-2 and IMC+GOLD exist)
    - RELAXED kicks in only when STRICT is impossible
    """
    load_assets_or_raise()
    rng = rng_for_rater(rater_id)

    # load assignment runtime and available pool
    assign_df = load_or_init_assignment_runtime()
    available = assign_df[assign_df["remaining_slots"] > 0].copy()
    if available.empty:
        raise ValueError("No dialogs with remaining slots left in runtime assignment.")

    # LOG: assignment table sizes
    logger.info(
        f"[ASSIGN_LEN] rater={rater_id} | assign_df={len(assign_df)} | available_remaining_slots>0={len(available)}"
    )

    # merge assignment metadata into stimuli
    merge_cols = ["dialog_id", "remaining_slots"]
    for c in ["recipe_title", "recipe_type", "condition", "is_gold", "is_imc"]:
        if c in available.columns:
            merge_cols.append(c)

    merged = dialogs_df.merge(
        available[merge_cols],
        on="dialog_id",
        how="inner",
        suffixes=("", "_assign"),
    )
    if merged.empty:
        raise ValueError("Merged assignment with dialogs_df is empty.")

    # harmonize duplicated cols
    for col in ["recipe_title", "recipe_type", "condition", "is_gold", "is_imc"]:
        a = f"{col}_assign"
        if col not in merged.columns and a in merged.columns:
            merged = merged.rename(columns={a: col})
        if col in merged.columns and a in merged.columns:
            merged[col] = merged[col].fillna(merged[a])

    # ensure required cols exist
    if "is_gold" not in merged.columns:
        merged["is_gold"] = 0
    if "is_imc" not in merged.columns:
        merged["is_imc"] = 0
    if "condition" not in merged.columns:
        merged["condition"] = "good"

    if "recipe_title" not in merged.columns:
        merged["recipe_title"] = ""
    merged["recipe_title"] = merged["recipe_title"].apply(safe_text)

    if "recipe_type" not in merged.columns:
        merged["recipe_type"] = ""
    merged["recipe_type"] = merged["recipe_type"].apply(safe_text)

    merged["is_gold"] = pd.to_numeric(merged["is_gold"], errors="coerce").fillna(0).astype(int)
    merged["is_imc"] = pd.to_numeric(merged["is_imc"], errors="coerce").fillna(0).astype(int)
    merged["condition"] = merged["condition"].apply(safe_text)

    # dynamic n_main
    available_n = int(len(merged))
    if available_n < 1:
        raise ValueError("No dialogs available after merge (should not happen if available was non-empty).")

    requested_n_main = int(n_main)
    n_main = min(requested_n_main, available_n)
    if n_main != requested_n_main:
        logger.warning(
            f"[ASSIGN_LEN] rater={rater_id} | reducing n_main from {requested_n_main} to {n_main} "
            f"because only {available_n} dialogs are available."
        )

    if n_main <= 0:
        raise ValueError(f"n_main must be >= 1 (got {n_main}).")

    # derive pools
    imc_pool = merged[merged["is_imc"] == 1].copy()
    gold_pool = merged[merged["is_gold"] == 1].copy()
    neutral_pool = merged[(merged["is_imc"] == 0) & (merged["is_gold"] == 0)].copy()

    # strict rule is only possible if can still fill (n_main-2) neutrals and have IMC + GOLD
    remaining_needed_strict = max(0, n_main - 2)
    strict_mode = (
        (n_main >= 2)
        and (len(neutral_pool) >= remaining_needed_strict)
        and (not imc_pool.empty)
        and (not gold_pool.empty)
    )

    logger.info(
        f"[ASSIGN_LEN] rater={rater_id} | merged_candidates={len(merged)} | "
        f"neutral_pool={len(neutral_pool)} | imc_pool={len(imc_pool)} | gold_pool={len(gold_pool)} | "
        f"strict_mode={strict_mode} | remaining_needed_strict={remaining_needed_strict}"
    )

    # STRICT mode: exactly 1 IMC + 1 GOLD + rest neutrals (if feasible)
    # RELAXED mode: fill n_main from neutral (as much as possible) + GOLD + IMC mixed

    if strict_mode:
        # pick IMC and GOLD (distinct)
        imc_row = imc_pool.sample(n=1, random_state=rng.randint(0, 10**9)).iloc[0]
        gold_pool2 = gold_pool[gold_pool["dialog_id"] != imc_row["dialog_id"]]
        if gold_pool2.empty:
            raise ValueError("Gold pool only contains the selected IMC dialog. Need separate dialogs.")
        gold_row = gold_pool2.sample(n=1, random_state=rng.randint(0, 10**9)).iloc[0]

        chosen_ids = {imc_row["dialog_id"], gold_row["dialog_id"]}

        # robust insertion positions (works even if n_main < sets)
        def pick_positions(n_main_: int, rng_: random.Random):
            imc_candidates = [p for p in IMC_POS_SET if 0 <= int(p) < n_main_]
            gold_candidates = [p for p in GOLD_POS_SET if 0 <= int(p) < n_main_]

            if not imc_candidates:
                imc_candidates = list(range(min(3, n_main_)))  # 0..2 (or fewer)
            if not gold_candidates:
                gold_candidates = list(range(n_main_))

            imc_pos_ = rng_.choice(imc_candidates)
            gold_pos_choices_ = [p for p in gold_candidates if p != imc_pos_]
            if not gold_pos_choices_:
                gold_pos_choices_ = [p for p in range(n_main_) if p != imc_pos_]
            gold_pos_ = rng_.choice(gold_pos_choices_)
            return int(imc_pos_), int(gold_pos_)

        imc_pos, gold_pos = pick_positions(n_main, rng)

        final = [None] * n_main
        final[imc_pos] = imc_row.to_dict()
        final[gold_pos] = gold_row.to_dict()

        # fillers (STRICT --> neutral only)
        remaining_needed = n_main - 2

        # candidate pool excluding chosen IDs
        remaining_pool_all = merged[~merged["dialog_id"].isin(chosen_ids)].copy()

        # neutral-only fillers
        remaining_pool = remaining_pool_all[
            (remaining_pool_all["is_imc"] == 0) & (remaining_pool_all["is_gold"] == 0)
        ].copy()

        # LOG: filler pool (neutral)
        cond_counts = (
            remaining_pool["condition"].value_counts(dropna=False).to_dict() if not remaining_pool.empty else {}
        )
        logger.info(
            f"[ASSIGN_LEN] rater={rater_id} | filler_pool_neutral={len(remaining_pool)} | "
            f"remaining_needed={remaining_needed} | filler_condition_counts={cond_counts}"
        )

        if len(remaining_pool) < remaining_needed:
            # should not happen because strict_mode checks it (keep hard fail for safety)
            raise ValueError(
                f"STRICT mode invariant violated: need {remaining_needed} neutral fillers, have {len(remaining_pool)}."
            )

        # Condition mixing guarantee (good + bad) if possible
        pool_good = remaining_pool[remaining_pool["condition"] == "good"].copy()
        pool_bad = remaining_pool[remaining_pool["condition"] == "bad"].copy()

        must_take_good = 1 if len(pool_good) > 0 else 0
        must_take_bad = 1 if len(pool_bad) > 0 else 0

        picked_rows = []

        if must_take_good:
            picked_rows.append(pool_good.sample(n=1, random_state=rng.randint(0, 10**9)).iloc[0].to_dict())
        if must_take_bad:
            pool_bad2 = pool_bad if not picked_rows else pool_bad[pool_bad["dialog_id"] != picked_rows[0]["dialog_id"]]
            if len(pool_bad2) > 0:
                picked_rows.append(pool_bad2.sample(n=1, random_state=rng.randint(0, 10**9)).iloc[0].to_dict())
            else:
                must_take_bad = 0

        already_ids = {r["dialog_id"] for r in picked_rows}
        remaining_k = remaining_needed - len(picked_rows)

        target_good = remaining_needed // 2
        target_bad = remaining_needed - target_good

        cur_good = sum(1 for r in picked_rows if safe_text(r.get("condition")) == "good")
        cur_bad = sum(1 for r in picked_rows if safe_text(r.get("condition")) == "bad")

        need_good_more = max(0, target_good - cur_good)
        need_bad_more = max(0, target_bad - cur_bad)

        pool_rest = remaining_pool[~remaining_pool["dialog_id"].isin(already_ids)].copy()

        add_rows = []
        if remaining_k > 0 and not pool_rest.empty:
            pool_rest_good = pool_rest[pool_rest["condition"] == "good"]
            pool_rest_bad = pool_rest[pool_rest["condition"] == "bad"]

            take_good = min(need_good_more, remaining_k, len(pool_rest_good))
            if take_good > 0:
                add_rows.extend(
                    pool_rest_good.sample(n=take_good, random_state=rng.randint(0, 10**9)).to_dict(orient="records")
                )
                remaining_k -= take_good

            if add_rows:
                taken_ids = {r["dialog_id"] for r in add_rows}
                pool_rest = pool_rest[~pool_rest["dialog_id"].isin(taken_ids)]
                pool_rest_bad = pool_rest[pool_rest["condition"] == "bad"]

            take_bad = min(need_bad_more, remaining_k, len(pool_rest_bad))
            if take_bad > 0:
                add_rows.extend(
                    pool_rest_bad.sample(n=take_bad, random_state=rng.randint(0, 10**9)).to_dict(orient="records")
                )
                remaining_k -= take_bad

            if add_rows:
                taken_ids = {r["dialog_id"] for r in add_rows}
                pool_rest = pool_rest[~pool_rest["dialog_id"].isin(taken_ids)]

            if remaining_k > 0:
                if len(pool_rest) < remaining_k:
                    raise ValueError(
                        f"Internal error: filler selection underflow. Need remaining_k={remaining_k}, "
                        f"but pool_rest has {len(pool_rest)}."
                    )
                add_rows.extend(
                    pool_rest.sample(n=remaining_k, random_state=rng.randint(0, 10**9)).to_dict(orient="records")
                )
                remaining_k = 0

        rest_records = picked_rows + add_rows

        if len(rest_records) != remaining_needed:
            raise ValueError(
                f"Filler selection failed: expected {remaining_needed} fillers, got {len(rest_records)}."
            )

        # insert fillers into free positions
        free_positions = [i for i, x in enumerate(final) if x is None]
        rng.shuffle(free_positions)

        for pos, rec in zip(free_positions, rest_records):
            final[pos] = rec

    else:
        # RELAXED mode: fill n_main with as many neutral as possible, then fill remaining with GOLD/IMC mixed

        pool_neutral = neutral_pool.copy()
        pool_non_neutral = merged[(merged["is_imc"] == 1) | (merged["is_gold"] == 1)].copy()

        take_neutral = min(len(pool_neutral), n_main)
        remaining_needed = n_main - take_neutral

        neutral_picks = []
        if take_neutral > 0:
            neutral_picks = pool_neutral.sample(
                n=take_neutral, random_state=rng.randint(0, 10**9)
            ).to_dict(orient="records")

        extra_picks = []
        if remaining_needed > 0:
            used_ids_tmp = {r["dialog_id"] for r in neutral_picks}
            pool_non_neutral = pool_non_neutral[~pool_non_neutral["dialog_id"].isin(used_ids_tmp)].copy()

            if len(pool_non_neutral) >= remaining_needed:
                extra_picks = pool_non_neutral.sample(
                    n=remaining_needed, random_state=rng.randint(0, 10**9)
                ).to_dict(orient="records")
            else:
                # allow any remaining dialogs (still no duplicates within rater)
                pool_any = merged[~merged["dialog_id"].isin(used_ids_tmp)].copy()
                if len(pool_any) < remaining_needed:
                    raise ValueError(
                        f"Not enough dialogs available in relaxed mode for n_main={n_main}. "
                        f"Need remaining_needed={remaining_needed}, have {len(pool_any)}."
                    )
                extra_picks = pool_any.sample(
                    n=remaining_needed, random_state=rng.randint(0, 10**9)
                ).to_dict(orient="records")

        final = neutral_picks + extra_picks
        rng.shuffle(final)

        logger.warning(
            f"[ASSIGN_LEN] rater={rater_id} | RELAXED_MODE active: "
            f"neutral_taken={len(neutral_picks)} | extra_taken={len(extra_picks)} | final_len={len(final)}"
        )

    # ensure full length
    if any(x is None for x in final):
        missing = sum(1 for x in final if x is None)
        raise ValueError(f"Assignment produced None entries (missing={missing}). This should not happen.")

    used_ids = [x["dialog_id"] for x in final]
    if len(used_ids) != n_main:
        raise ValueError(f"Final assignment length mismatch: expected {n_main}, got {len(used_ids)}.")

    # decrement slots and persist
    assign_df = _decrement_slots(assign_df, used_ids)
    save_assignment_runtime(assign_df)
    write_assignment_status(assign_df)

    # LOG: final outcome
    final_conds = pd.Series([safe_text(x.get("condition")) for x in final]).value_counts().to_dict()
    logger.info(
        f"[ASSIGN_LEN] rater={rater_id} | final_len={len(final)} | used_ids_len={len(used_ids)} | "
        f"final_condition_counts={final_conds}"
    )
    logger.info(f"[ASSIGN] rater_id={rater_id} used_ids={used_ids}")

    return final


# ------------------------------------------------------------
# Ratings persistence (local backup and per-rater dataset CSV)

# This section writes one rating row per dialog to a local CSV backup and appends the same row
# to the per-rater CSV stored in the Hugging Face dataset repository.

def append_rating_row(
    rater_id,
    dialog_record,
    ratings,
    comment,
    duration,
    imc_pass_global,
    imc_dialog_pass,
    start_time,
    min_time_ok_dialog,
    gold_pass,
):
    """Appends a single rating row to the local backup and to the per-rater dataset CSV."""
    out_local = RESULTS_DIR / f"ratings_{rater_id}.csv"

    recipe_title = safe_text(dialog_record.get("recipe_title")) or safe_text(dialog_record.get("recipe"))
    recipe_type = safe_text(dialog_record.get("recipe_type"))

    # Defines the fixed metadata fields recorded for each rated dialog
    row = {
        "start_time": start_time,
        "timestamp": time.time(),
        "rater_id": rater_id,
        "dialog_id": dialog_record.get("dialog_id"),
        "recipe_title": recipe_title,
        "recipe_type": recipe_type,
        "condition": dialog_record.get("condition"),
        "is_gold": dialog_record.get("is_gold", 0),
        "gold_expected_min": dialog_record.get("gold_expected_min"),
        "gold_expected_max": dialog_record.get("gold_expected_max"),
        "is_imc": dialog_record.get("is_imc", 0),
        "imc_expected_overall": dialog_record.get("imc_expected_overall"),
        "imc_pass_global": imc_pass_global,
        "imc_dialog_pass": imc_dialog_pass,
        "min_time_ok_dialog": min_time_ok_dialog,
        "gold_pass": gold_pass,
        "duration_sec": duration,
        "comment": comment or "",
    }
    for k, v in ratings.items():
        row[k] = v

    # Local backup (append mode)
    df = pd.DataFrame([row])
    write_header = not out_local.exists()
    df.to_csv(out_local, mode="a", index=False, header=write_header, encoding="utf-8")

    # Dataset per-rater append
    try:
        append_row_to_rater_csv_in_dataset(rater_id, row)
    except Exception:
        logger.exception("[UPLOAD] failed to append rating row to per-rater dataset CSV")


# ------------------------------------------------------------
# UI logic: Welcome → Consent → Instructions → Practice → Main → End

# This section defines small UI helper functions that control whether navigation buttons are interactive
# based on the current participant inputs.

def toggle_welcome_start_button(english_ok: bool, imc_answer: str):
    """Enables the Start button only when the English checkbox is checked and the IMC answer is correct."""
    enabled = bool(english_ok and imc_answer == "Strongly disagree")
    return gr.update(interactive=enabled)

def toggle_consent_continue_button(consent: bool):
    """Enables the Continue button only when the consent checkbox is checked."""
    return gr.update(interactive=bool(consent))

def toggle_instructions_button(confirmed: bool):
    """Enables the Continue-to-Practice button only when the instructions confirmation checkbox is checked."""
    return gr.update(interactive=bool(confirmed))


def handle_welcome(rater_id_input, english_ok, imc_answer):
    """
    Validates the Welcome-Page inputs and switches the UI to consent page. 
    Creates stable rater_id and stores it in pending state without assigning any dialogs.
    """
    load_assets_or_raise()

    imc_pass_global = int(imc_answer == "Strongly disagree")

    if not (english_ok and imc_answer == "Strongly disagree"):
        msg = "Please confirm your English skills. To show that you read carefully, please select 'Strongly disagree' below."
        return (
            msg,
            None,                      # pending_rater_id_state
            gr.update(visible=True),   # welcome_group
            gr.update(visible=False),  # consent_group
            gr.update(visible=False),  # instructions_group
            imc_pass_global,           # imc_pass_state
        )

    # Create rater_id now (stable), but do not burn slots yet
    if not rater_id_input or str(rater_id_input).strip() == "":
        rater_id = f"R{random.randint(10000, 99999)}"
    else:
        rater_id = str(rater_id_input).strip()

    msg = f"Rater ID: **{rater_id}**. Please review the consent information on the next page."

    # Stores the rater ID in the pending state and routes the UI to the consent page
    return (
        msg,
        rater_id,                  # pending_rater_id_state
        gr.update(visible=False),  # hide welcome
        gr.update(visible=True),   # show consent
        gr.update(visible=False),  # hide instructions
        imc_pass_global,
    )


def handle_consent_continue(consent_checked: bool, pending_rater_id: str):
    """
    Checks whether informed consent is given and then finalizes the rater setup.
    Assigns main dialogs only after consent is confirmed and consumes remaining slots only in that case.

    Robustness change:
    - If slot-based assignment fails, do not fall back to dialogs_list (would ignore slots).
      Instead, return an empty assignment and show a clear status message.
    """
    load_assets_or_raise()

    if not consent_checked:
        return (
            "Please check the consent box to continue.",
            None,                     # rater_id_state
            dialogs_list,             # dialogs_state (safe fallback)
            gr.update(visible=True),  # consent_group stays
            gr.update(visible=False), # instructions_group stays hidden
        )

    # finalize rater id or generate fallback ID
    rater_id = safe_text(pending_rater_id) or f"R{random.randint(10000, 99999)}"

    try:
        dialogs_for_rater = make_main_dialogs_for_rater(rater_id=rater_id, n_main=MAX_MAIN_DIALOGS)
        status = ""
    except Exception as e:
        logger.exception("[handle_consent_continue] Slot-based assignment failed (no fallback to dialogs_list).")
        dialogs_for_rater = []
        status = (
            "Sorry — not enough dialog slots are available at the moment to start a new session. "
            "Please try again later."
        )

    return (
        status,                    # consent_status_md
        rater_id,                  # rater_id_state
        dialogs_for_rater,          # dialogs_state
        gr.update(visible=False),  # hide consent
        gr.update(visible=True),   # show instructions
    )
    

def start_practice_from_instructions(rater_id, dialogs):
    """Initializes the practice phase after the participant confirms the instructions."""
    load_assets_or_raise()

    # ensures that valid rater_id is present before practice begins
    if not rater_id:
        rater_id = f"R{random.randint(10000, 99999)}"

    # chooses practice pool from gold practice data when available
    if HAS_PRACTICE_GOLD and practice_gold_pool:
        pool = practice_gold_pool
    # uses stimuli dialogs as a fallback practice pool when gold practice is unavailable
    else:
        pool = dialogs_df.to_dict(orient="records") if not dialogs_df.empty else []

    # handles case where no practice dialogs are available and routes directly to main as fallback
    if not pool:
        status = "No dialogs available for practice."
        return (
            status,
            [],                       # practice_dialogs_state
            0,                        # practice_index_state
            "No practice dialogs.",   # practice_dialog_md
            "",                       # practice_progress_md
            "",                       # practice_title_md
            gr.update(visible=False), # instructions_group
            gr.update(visible=False), # practice_group
            gr.update(visible=True),  # main_group fallback
        )

    # samples 3 practice dialogs or uses full pool if it contains 3 or fewer
    practice_dialogs = pool if len(pool) <= 3 else random.sample(pool, 3)
    practice_index = 0

    first = practice_dialogs[practice_index]
    title_raw = first.get("recipe_title") or first.get("recipe") or ""
    title = f"**{title_raw}**" if title_raw else ""
    first_html = dialog_to_bubbles(first["dialog_text"])
    practice_progress = practice_progress_text(practice_index, len(practice_dialogs))

    status = (
        f"Rater ID: **{rater_id}**. This is a short practice round. "
        f"You will see {len(practice_dialogs)} example dialogues."
    )

    # initializes practice-related states and switches visibility from instructions to practice
    return (
        status,
        practice_dialogs,
        practice_index,
        first_html,
        practice_progress,
        title,
        gr.update(visible=False),  # instructions_group off
        gr.update(visible=True),   # practice_group on
        gr.update(visible=False),  # main_group off
    )


def next_practice_dialog(
    p_radio_clarity,
    p_radio_relevance,
    p_radio_respect,
    p_radio_overall,
    p_comment,
    rater_id,
    practice_dialogs,
    practice_index,
    dialogs,
):
    """Advances the practice phase by one dialog and transitions to the main phase when practice ends."""
    practice_raw_vals = [p_radio_clarity, p_radio_relevance, p_radio_respect, p_radio_overall]
    practice_parsed_vals = [parse_likert(v) for v in practice_raw_vals]

    # aligns each practice feedback key with the parsed rating value
    key_to_val = {k: v for k, v in zip(PRACTICE_FEEDBACK_KEYS, practice_parsed_vals)}

    full_rating_vals = []
    for key, _ in SUBSCALES:
        full_rating_vals.append(key_to_val.get(key, None))

    total_practice = len(practice_dialogs) if practice_dialogs is not None else 0
    current_practice_dialog = (
        practice_dialogs[practice_index] if practice_dialogs and 0 <= practice_index < total_practice else None
    )

    feedback_text = ""
    try:
        if current_practice_dialog is not None:
            feedback_text = build_practice_feedback(current_practice_dialog, full_rating_vals)
    except Exception:
        logger.exception("[next_practice_dialog] build_practice_feedback failed")
        feedback_text = ""

    practice_index += 1

    # transitions from practice to main when the practice index reaches the end of the practice set
    if practice_index >= total_practice:
        main_index = 0
        start_time = time.time()

        total_dialogs = len(dialogs) if dialogs else 0
        total_main = min(total_dialogs, MAX_MAIN_DIALOGS)

        first_main_html = show_current_dialog(main_index, dialogs)
        main_progress = main_progress_text(main_index, total_main)

        first_record = dialogs[main_index] if dialogs and total_main > 0 else {}
        rtype = (first_record.get("recipe_type") or "").strip()
        suffix = f" ({rtype})" if rtype else ""
        main_title_raw = (first_record.get("recipe_title") or first_record.get("recipe") or "") + suffix
        main_title = f"**{main_title_raw}**" if main_title_raw else ""

        status = (
            (feedback_text + "\n\n" if feedback_text else "")
            + "Practice finished. You will now start the main study. "
            "Please continue rating as you did during practice."
        )

        # ends practice, initializes main states, and switches visibility to the main group
        return (
            status,                                         # practice_status_md
            practice_index,                                 # practice_index_state
            "No more practice dialogs.",                    # practice_dialog_md
            practice_progress_text(total_practice, total_practice),  # practice_progress_md
            "",                                             # practice_title_md
            main_index,                                     # main_index_state
            start_time,                                     # dialog_start_state
            first_main_html,                                # dialog_md
            main_progress,                                  # progress_md
            False,                                          # finished_state
            gr.update(visible=False),                       # practice_group off
            gr.update(visible=True),                        # main_group on
            main_title,                                     # main_title_md
        )

    # renders next practice dialog when practice is not finished
    next_practice = practice_dialogs[practice_index]
    next_html = dialog_to_bubbles(next_practice["dialog_text"])
    practice_progress = practice_progress_text(practice_index, total_practice)
    status = (feedback_text + "\n\n" if feedback_text else "")
    
    next_title_raw = next_practice.get("recipe_title") or next_practice.get("recipe") or ""
    next_title = f"**{next_title_raw}**" if next_title_raw else ""

    # update practice-related outputs while keeping the participant in the practice group
    return (
        status,                                  # practice_status_md
        practice_index,                          # practice_index_state
        next_html,                               # practice_dialog_md
        practice_progress,                       # practice_progress_md
        next_title,                              # practice_title_md
        gr.update(),                             # main_index_state
        gr.update(),                             # dialog_start_state
        gr.update(),                             # dialog_md
        gr.update(),                             # progress_md
        gr.update(),                             # finished_state
        gr.update(visible=True),                 # practice_group stays on
        gr.update(visible=False),                # main_group stays off
        gr.update(),                             # main_title_md
    )


def submit_main_core(*args):
    """Processes one main-dialog submission, persists ratings and advances to the next dialog or end."""
    n_scales = len(rating_keys_in_ui_order)
    expected_min = n_scales + 6
    if len(args) < expected_min:
        raise RuntimeError(f"submit_main_core expected >= {expected_min} args, got {len(args)}")

    radio_vals = args[:n_scales]
    comment, rater_id, dialogs, main_index, dialog_start_time, imc_pass_global = args[n_scales:n_scales + 6]

    # parses the raw radio values into integers or None (0)
    rating_vals = [parse_likert(v) for v in radio_vals]
    processed_vals = [v if v is not None else 0 for v in rating_vals]
    ratings = {k: processed_vals[i] for i, k in enumerate(rating_keys_in_ui_order)}

    total_dialogs = len(dialogs) if dialogs is not None else 0
    total_main = min(total_dialogs, MAX_MAIN_DIALOGS)

    # handles missing or out-of-range state and routes directly to the end page
    if dialogs is None or main_index is None or main_index >= total_main or total_main == 0:
        end_text = (f"No dialogs available.\n\nYour rater code: **{rater_id}**" if rater_id else "No dialogs available.")
        return (
            "",                                            # status_md
            gr.update(value="No more dialogs.", visible=False),  # dialog_md
            main_index or 0,
            dialog_start_time,
            True,
            main_progress_text(0, total_main),
            gr.update(visible=False),                       # main_group
            gr.update(visible=True),                        # end_group
            end_text,
            "",
        )

    # select current dialog record based on main index
    current = dialogs[main_index]

    # duration spent
    duration = time.time() - dialog_start_time if dialog_start_time else None
    min_time_ok_dialog = int(duration >= MIN_DIALOG_TIME) if duration is not None else None

    # stores IMC pass indicator for current dialog when it is IMC item
    imc_dialog_pass = None
    try:
        # computes IMC pass only when current dialog is labeled as IMC
        if int(current.get("is_imc", 0)) == 1:
            # read expected IMC overall value and convert to int
            expected_val = _to_int_safe(current.get("imc_expected_overall", None))
            # read participant's overall rating and encode = 1/0
            overall_val = ratings.get("overall", None)
            if expected_val is not None and overall_val is not None:
                imc_dialog_pass = int(overall_val == expected_val)
    except Exception:
        logger.exception("[submit_main_core] error computing imc_dialog_pass")
        imc_dialog_pass = None

    # stores Gold pass indicator for current dialog when it is Gold item
    gold_pass = None
    try:
        # computes GOLD pass only when current dialog is labeled as GOLD
        if int(current.get("is_gold", 0)) == 1:
            # read expected GOLD interval bounds and convert to int
            exp_min = _to_int_safe(current.get("gold_expected_min", None))
            exp_max = _to_int_safe(current.get("gold_expected_max", None))
            # read participant's overall rating from rating dictionary
            overall_val = ratings.get("overall", None)
            # checks whether overall rating falls within expected interval
            if overall_val is not None and exp_min is not None and exp_max is not None:
                gold_pass = int(exp_min <= overall_val <= exp_max)
    except Exception:
        logger.exception("[submit_main_core] error computing gold_pass")
        gold_pass = None

    # persist current rating row both locally and in the dataset repository
    append_rating_row(
        rater_id=rater_id,
        dialog_record=current,
        ratings=ratings,
        comment=comment,
        duration=duration,
        imc_pass_global=imc_pass_global,
        imc_dialog_pass=imc_dialog_pass,
        start_time=dialog_start_time,
        min_time_ok_dialog=min_time_ok_dialog,
        gold_pass=gold_pass,
    )

    main_index += 1

    # end screen when the participant completes all main dialogs
    if main_index >= total_main:
        end_text = (
            f"Thank you for participating!\n\n"
            f"You have completed **{total_main}** dialogs.\n\n"
            f"Your rater code: **{rater_id}**\n\n"
            f"---\n\n"
            f"### Debriefing\n\n"
            f"The **purpose of this research** is to examine how communication quality in expert dialogs is perceived and evaluated by humans, "
            f"and how these evaluations compare to assessments produced by an artificial intelligence system.\n\n"
            f"Some of the dialogs you rated were intentionally designed to differ in their communication quality, while others were designed to be more effective and cooperative.\n\n"
            f"This information was not provided earlier to avoid influencing your judgments.\n"
            f"Your ratings help assess whether automated systems can evaluate communication in a way that aligns with human perception.\n\n"
            f"All data is analyzed in anonymized form and used exclusively for academic research.\n\n"
            f"If you have any questions about the study or would like additional information, you may contact the researcher after completion [Katharina.Kampa@stud.uni-regensburg.de].\n\n"
            f"Thank you for contributing to research on communication and artificial intelligence.\n\n"
            f"**You can now close the window.**"
            if rater_id else
            f"Thank you for participating!\n\n"
            f"You have completed **{total_main}** dialogues.\n\n"
            f"---\n\n"
            f"### Debriefing\n\n"
            f"The **purpose of this research** is to examine how communication quality in expert dialogs is perceived and evaluated by humans, "
            f"and how these evaluations compare to assessments produced by an artificial intelligence system.\n\n"
            f"Some of the dialogs you rated were intentionally designed to differ in their communication quality, while others were designed to be more effective and cooperative.\n\n"
            f"This information was not provided earlier to avoid influencing your judgments.\n"
            f"Your ratings help assess whether automated systems can evaluate communication in a way that aligns with human perception.\n\n"
            f"All data is analyzed in anonymized form and used exclusively for academic research.\n\n"
            f"If you have any questions about the study or would like additional information, you may contact the researcher after completion [Katharina.Kampa@stud.uni-regensburg.de].\n\n"
            f"Thank you for contributing to research on communication and artificial intelligence.\n\n"
            f"**You can now close the window.**"
        )

        # hide main group, show end group and freeze dialog display
        return (
            "",
            gr.update(value="No more dialogs to show.", visible=False),
            main_index,
            None,
            True,
            main_progress_text(total_main - 1, total_main),
            gr.update(visible=False),
            gr.update(visible=True),
            end_text,
            "",
        )

    # render next dialog for display
    next_text = show_current_dialog(main_index, dialogs)
    # reset start time so duration measurement begins for next dialog
    new_start_time = time.time()

    next_record = dialogs[main_index]
    rtype = (next_record.get("recipe_type") or "").strip()
    suffix = f" ({rtype})" if rtype else ""
    main_title_raw = (next_record.get("recipe_title") or next_record.get("recipe") or "") + suffix
    main_title = f"**{main_title_raw}**" if main_title_raw else ""

    # update dialog display and keep participant in the main group
    return (
        "",
        gr.update(value=next_text, visible=True),
        main_index,
        new_start_time,
        False,
        main_progress_text(main_index, total_main),
        gr.update(visible=True),
        gr.update(visible=False),
        "",
        main_title,
    )


def reset_main_controls():
    """Resets all main phase rating inputs and clears the main comment box."""
    updates = [gr.update(value=None) for _ in range(len(rating_keys_in_ui_order))]
    updates.append(gr.update(value=""))
    return updates


def reset_practice_controls():
    """Resets all practice phase rating inputs and clears the practice comment box."""
    updates = [gr.update(value=None) for _ in range(len(PRACTICE_FEEDBACK_KEYS))]
    updates.append(gr.update(value=""))
    return updates


# ------------------------------------------------------------
# CSS (visual style)

# Defines the visual style of the Gradio app, including theme-aware text colors, dialog speech bubbles for different speakers, 
# button styles and layout adjustments.

custom_css = """
/* Scale label text: black in light mode, white in dark mode */
@media (prefers-color-scheme: light) {
  .gradio-container label,
  .gradio-container .wrap label,
  .gradio-container .block-label {
    color: #000 !important;
  }
}

@media (prefers-color-scheme: dark) {
  .gradio-container label,
  .gradio-container .wrap label,
  .gradio-container .block-label {
    color: #fff !important;
  }
}
.dialog-container {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
    max-width: 48rem;
    margin: 0.25rem auto;
}
.bubble {
    width: 22rem;
    padding: 0.25rem 0.5rem;
    border-radius: 0.75rem;
    font-size: 0.95rem;
    line-height: 1.35;
    word-wrap: break-word;
    overflow-wrap: break-word;
}
.bubble-fat {
    align-self: flex-start;
    background-color: #e3f2fd;
    border: 1px solid #bbdefb;
}
.bubble-carb {
    align-self: flex-end;
    background-color: #fff3e0;
    border: 1px solid #ffe0b2;
}
.bubble-other {
    align-self: center;
    background-color: #f5f5f5;
    border: 1px solid #ededed;
    color: #111 !important;
}
.speaker {
    font-weight: 600;
    font-size: 0.78rem;
    margin-bottom: 0.02rem;
    opacity: 0.8;
}
.bubble-text {
    white-space: normal;
    color: #111 !important;
}
button.green-btn {
    background-color: #4caf50 !important;
    color: white !important;
    border-color: #4caf50 !important;
}
button.green-btn[disabled] {
    background-color: #c8e6c9 !important;
    color: #555 !important;
    border-color: #c8e6c9 !important;
}
button.orange-btn {
    background-color: #ff9800 !important;
    color: white !important;
    border-color: #ff9800 !important;
}
button.orange-btn[disabled] {
    background-color: #ffe0b2 !important;
    color: #555 !important;
    border-color: #ffe0b2 !important;
}
.dialog-hint {
    font-size: 0.9rem;
    color: #555;
    margin-top: 0.25rem;
    margin-bottom: 0.25rem;
}

/* Scroll anchor offset so it doesn't hide under sticky header */
#page-top-anchor {
    scroll-margin-top: 90px;
}
"""


# ------------------------------------------------------------
# Build the Gradio UI

# This section defines the Gradio Blocks layout, initializes UI state variables
# and wires user interactions to the corresponding callback functions.

load_assets_or_raise()  # ensure dialogs_list exists for initial placeholders
load_or_init_assignment_runtime() # generates runtime and status at startup

# creates the Gradio app and applies the custom CSS stylesheet
with gr.Blocks(css=custom_css) as demo:
    gr.Markdown(
        "## Study – Dialog Communication Quality",
        elem_classes=["hint-text", "sticky-header"],
        elem_id="page-top-anchor",
    )

    # States
    # store values across UI interactions
    # control participant identity, dialog assignment, progress and IMC status
    rater_id_state = gr.State(value=None)            # finalized rater ID after consent
    dialogs_state = gr.State(value=dialogs_list)     # dialog list used in the main study
    main_index_state = gr.State(value=0)             # current index in the main dialog sequence
    dialog_start_state = gr.State(value=time.time()) # start time for the current dialog
    finished_state = gr.State(value=False)           # stores whether main study is completed
    imc_pass_state = gr.State(value=None)            # stores whether global IMC check is passed
    pending_rater_id_state = gr.State(value=None)    # provisional rater ID before consent

    # Practice states
    practice_dialogs_state = gr.State(value=[])      # list of practice dialogs shown to participant
    practice_index_state = gr.State(value=0)         # current index in practice dialog sequence

    # -------------------------------
    # WELCOME
    
    # defines the initial page where participants confirm language skills and complete the IMC gating item
    with gr.Group(visible=True) as welcome_group:
        gr.Markdown("### Welcome", elem_classes=["hint-text"])
        rater_id_input = gr.Textbox(label="Rater ID (optional)")
        english_ok_cb = gr.Checkbox(label="I have good English reading skills.")

        imc_radio = gr.Radio(
            ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
            label="To show that you read carefully, please choose 'Strongly disagree'.",
        )

        start_btn = gr.Button("Start", interactive=False, elem_classes=["green-btn"])
        welcome_info = gr.Markdown("Fill in the fields above to start.")

        # English and IMC
        english_ok_cb.change(toggle_welcome_start_button, inputs=[english_ok_cb, imc_radio], outputs=start_btn)
        imc_radio.change(toggle_welcome_start_button, inputs=[english_ok_cb, imc_radio], outputs=start_btn)

    # -------------------------------
    # CONSENT
    
    # displays informed consent information and requires an explicit agreement before proceeding
    with gr.Group(visible=False) as consent_group:
        gr.Markdown("## Informed Consent", elem_classes=["hint-text"])

        consent_text_md = gr.Markdown(
            """
Please read the following information carefully before continuing.

#### Purpose of the study
You are invited to participate in a research study conducted as part of a Master’s thesis.
The purpose of this study is to examine how people perceive and evaluate communication quality in written dialogs.

#### What you will do
- Read several short dialogs
- Rate communication quality using rating scales
- Optionally provide short comments

#### Duration
Approx. 30–50 minutes

#### Voluntary participation
Your participation is completely voluntary. 
You can stop at any time by closing the browser window, without any consequences.

#### Data protection and anonymity
No personally identifying data is collected. 
Your responses are stored under a pseudonymous participant ID and used exclusively for academic research purposes.

#### Risks and benefits
This study involves no known risks beyond those of everyday reading and judgment tasks.
There is no direct personal benefit, but your participation contributes to research on communication and artificial intelligence.

By checking the box below, you confirm that you are at least 18 years old and agree to participate.

If you have questions, please contact the researcher [Katharina.Kampa@stud.uni-regensburg.de].
            """,
            elem_classes=["hint-text"],
        )

        consent_cb = gr.Checkbox(label="I have read the information above and I agree to participate.", value=False)
        consent_continue_btn = gr.Button("Continue", interactive=False, elem_classes=["green-btn"])
        consent_status_md = gr.Markdown("", elem_classes=["hint-text"])

        # Enable continue only if consent checked
        consent_cb.change(toggle_consent_continue_button, inputs=[consent_cb], outputs=consent_continue_btn)

    # -------------------------------
    # INSTRUCTIONS
    
    # presents task instructions and requires confirmation before starting the practice phase
    with gr.Group(visible=False) as instructions_group:
        gr.Markdown(
            """
## Instructions

You will see a series of short dialogs between two nutrition experts.
Each dialog presents a discussion about a recipe from different nutritional perspectives.

Your task is to **evaluate the quality of the communication** in each dialog.
You do not need to know the recipe. 
Please focus on how the experts communicate with each other, not on whether you agree with their viewpoints or recommendations.

Please consider the **entire dialog** when rating, not individual statements.
There are no right or wrong answers. I am interested in your personal judgment.

#### Rating scale
You will use a **1 / 4 / 7 rating-principle**:
- **1** = strongly disagree
- **4** = neutral
- **7** = strongly agree

Short explanations (anchors) are provided to help you interpret the scales consistently.
Try to use the full range of the scale where appropriate.

#### Practice Phase
Before the main part of the study, you will complete a short practice phase. 
This phase helps you become familiar with the interface and the rating scale principle.
Ratings from the practice phase are **not included** in the final analysis.

Please take your time to read each dialog carefully before rating.
            """,
            elem_classes=["hint-text"],
        )

        instructions_confirm_cb = gr.Checkbox(
            label="I have read and understood the instructions.",
            value=False,
        )
        instructions_continue_btn = gr.Button("Continue to practice", interactive=False, elem_classes=["green-btn"])

        # enables Continue button only when instructions confirmation is checked
        instructions_confirm_cb.change(
            toggle_instructions_button,
            inputs=[instructions_confirm_cb],
            outputs=instructions_continue_btn,
        )

    # -------------------------------
    # PRACTICE / MAIN / END
    
    # defines the UI components for the practice phase, the main rating phase and the final debrief page

    # --- PRACTICE ---
    with gr.Group(visible=False) as practice_group:
        gr.Markdown("### Practice", elem_classes=["hint-text"])
        practice_status_md = gr.Markdown("", elem_classes=["hint-text"])
        practice_progress_md = gr.Markdown("", elem_classes=["hint-text"])
        practice_title_md = gr.Markdown("", elem_classes=["title-text"])
        practice_dialog_md = gr.Markdown("", elem_classes=["dialog-text"], elem_id="practice-dialog")

        practice_rating_inputs = []
        for key in PRACTICE_FEEDBACK_KEYS:
            label = SUBSCALES_DICT.get(key, key)
            practice_rating_inputs.append(
                gr.Radio(
                    choices=LIKERT_LABELS,
                    label=f"{label} (1 = strongly disagree, 4 = neutral, 7 = strongly agree)",
                )
            )

        practice_comment_box = gr.Textbox(label="Comment (optional)", lines=2)

        practice_next_btn = gr.Button("Next practice dialog", elem_classes=["orange-btn"])

    # --- MAIN ---
    with gr.Group(visible=False) as main_group:
        gr.Markdown("### Main dialog rating", elem_classes=["hint-text"])
        progress_md = gr.Markdown("", elem_classes=["hint-text"])
        main_title_md = gr.Markdown("", elem_classes=["title-text"])

        first_dialog_html = show_current_dialog(0, dialogs_list)
        dialog_md = gr.Markdown(first_dialog_html, elem_classes=["dialog-text"], elem_id="main-dialog")

        rating_inputs = []
        rating_keys_in_ui_order = []

        for group_title, keys in SCALE_GROUPS:
            gr.Markdown(f"#### {group_title}")
            for key in keys:
                lbl = SUBSCALES_DICT[key]
                rating_inputs.append(
                    gr.Radio(
                        choices=LIKERT_LABELS,
                        label=f"{lbl} (1 = strongly disagree, 4 = neutral, 7 = strongly agree)",
                    )
                )
                rating_keys_in_ui_order.append(key)

        comment_box = gr.Textbox(label="Comment (optional)", lines=2)

        next_btn = gr.Button("Next dialog", elem_classes=["orange-btn"])
        status_md = gr.Markdown("Status will appear here.", elem_classes=["hint-text"])

    # --- END ---
    with gr.Group(visible=False) as end_group:
        end_md = gr.Markdown("Thank you for participating.", elem_classes=["hint-text"])

    # -------------------------------
    # Wiring
    
    # connects UI events to callback functions and defines the page flow between UI groups

    # Welcome -> Consent
    # validates welcome inputs and moves the UI from Welcome to Consent
    start_btn.click(
        handle_welcome,
        inputs=[rater_id_input, english_ok_cb, imc_radio],
        outputs=[
            welcome_info,
            pending_rater_id_state,
            welcome_group,
            consent_group,
            instructions_group,
            imc_pass_state,
        ],
        js=SCROLL_TOP_JS,
    )

    # Consent -> Instructions
    # validates consent, assigns dialogs and moves the UI from Consent to Instructions
    consent_continue_btn.click(
        handle_consent_continue,
        inputs=[consent_cb, pending_rater_id_state],
        outputs=[
            consent_status_md, 
            rater_id_state, 
            dialogs_state, 
            consent_group, 
            instructions_group,
        ],
        js=SCROLL_TOP_JS,
    )

    # Instructions -> Practice
    # starts practice phase and updates UI from Instructions to Practice
    instructions_continue_btn.click(
        start_practice_from_instructions,
        inputs=[rater_id_state, dialogs_state],
        outputs=[
            practice_status_md,
            practice_dialogs_state,
            practice_index_state,
            practice_dialog_md,
            practice_progress_md,
            practice_title_md,
            instructions_group,
            practice_group,
            main_group,
        ],
        js=SCROLL_TOP_JS,
    )

    # Practice -> Practice / Main
    # submits practice ratings and either advances practice or transitions to main phase
    practice_submit_evt = practice_next_btn.click(
        next_practice_dialog,
        inputs=[
            *practice_rating_inputs,
            practice_comment_box,
            rater_id_state,
            practice_dialogs_state,
            practice_index_state,
            dialogs_state,
        ],
        outputs=[
            practice_status_md,
            practice_index_state,
            practice_dialog_md,
            practice_progress_md,
            practice_title_md,
            main_index_state,
            dialog_start_state,
            dialog_md,
            progress_md,
            finished_state,
            practice_group,
            main_group,
            main_title_md,
        ],
        js=SCROLL_TOP_JS,
    )
    practice_submit_evt.then(reset_practice_controls, inputs=[], outputs=[*practice_rating_inputs, practice_comment_box])

    # Main -> Main / End
    # submits main ratings, persists them and advances to the next dialog or ends the study
    submit_evt = next_btn.click(
        submit_main_core,
        inputs=[
            *rating_inputs,
            comment_box,
            rater_id_state,
            dialogs_state,
            main_index_state,
            dialog_start_state,
            imc_pass_state,
        ],
        outputs=[
            status_md,
            dialog_md,
            main_index_state,
            dialog_start_state,
            finished_state,
            progress_md,
            main_group,
            end_group,
            end_md,
            main_title_md,
        ],
        js=SCROLL_TOP_JS,
    )
    submit_evt.then(reset_main_controls, inputs=[], outputs=[*rating_inputs, comment_box])

demo.queue()
demo.launch()