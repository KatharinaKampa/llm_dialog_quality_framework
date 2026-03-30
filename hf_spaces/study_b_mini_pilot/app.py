# app.py — Study B MINI-PILOT (HF Space) — Gradio 5.49.1 compatible
#
# Purpose:
#  - Manipulation check on Top-X recipe pairs that passed pre-check
#  - Each participant completes PAIRS_PER_PARTICIPANT pairs sequentially (each pair = good+bad dialogs for one recipe)
#  - Order is randomized per participant (deterministic RNG from rater_id)
#
# Inputs:
#  - studyB_top_pool_pairs.csv  (columns include: recipe_title, dialog_id_good, dialog_id_bad, pass_precheck, rank_score, ...)
#  - studyB_stimuli.csv         (contains dialog_text + recipe fields per dialog_id)
#
# Slots:
# - Slots are reserved upon allocation, deducted upon submission, and released upon timeout
#
# Output:
#  - one row per completed pair saved locally and appended to HF dataset under pilot_ratings/

import os
import re
import time
import math
import random
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta

import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub import CommitOperationAdd

# -----------------------------
# Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("studyB_pilot_space")

DATASET_ID = os.getenv("DATASET_ID", "KKam799/studyB_Mini-Pilot-storage")
HF_TOKEN = os.getenv("HF_TOKEN")

STIMULI_FILE = os.getenv("STIMULI_FILE", "studyB_stimuli.csv")
TOP_PAIRS_FILE = os.getenv("TOP_PAIRS_FILE", "studyB_top_pool_pairs.csv")
ASSIGNMENT_SEED_FILE = os.getenv("ASSIGNMENT_SEED_FILE", "studyB_pilot_assignment_seed.csv")

SPACE_DATA_DIR = Path(os.getenv("SPACE_DATA_DIR", "/data/studyB_pilot"))
SPACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_DIR = SPACE_DATA_DIR / "pilot_ratings"
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = SPACE_DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATASET_PILOT_DIR = os.getenv("DATASET_PILOT_DIR", "pilot_ratings")

# Remote paths (in HF dataset repo)
SHARED_CSV = f"{DATASET_PILOT_DIR}/pilot_all_ratings.csv"
ASSIGNMENT_STATUS_FILE = f"{DATASET_PILOT_DIR}/pilot_assignment_status.csv"
RESERVATIONS_FILE = f"{DATASET_PILOT_DIR}/pilot_reservations.csv"  # optional remote (we will not write frequently)

LOCAL_SHARED_CSV = LOCAL_DIR / "pilot_all_ratings.csv"
LOCAL_STATUS_CSV = LOCAL_DIR / "pilot_assignment_status.csv"
# reservations local
LOCAL_RESERVATIONS_CSV = LOCAL_DIR / "pilot_reservations.csv"

TOP_X = int(os.getenv("TOP_X", "17"))

# Participant flow
PAIRS_PER_PARTICIPANT = int(os.getenv("PAIRS_PER_PARTICIPANT", "2"))

# Target: pair-ratings per pair (e.g., 4 = 34 participants * 2 pairs = 68 pair-ratings; 68/17=4)
TARGET_PAIR_RATINGS_PER_PAIR = int(os.getenv("TARGET_PAIR_RATINGS_PER_PAIR", "4"))

# IMPORTANT: slots are tracked at dialog-level in assignment_status_df
# Each completed pair-rating consumes 1 slot for the good dialog and 1 slot for the bad dialog
# Therefore: slots per dialog should equal TARGET_PAIR_RATINGS_PER_PAIR
TARGET_SLOTS_PER_DIALOG = TARGET_PAIR_RATINGS_PER_PAIR

# Optional derived counts
MAX_TOTAL_PAIR_RATINGS = TOP_X * TARGET_PAIR_RATINGS_PER_PAIR         # e.g., 17*4 = 68
MAX_TOTAL_DIALOG_SLOTS = TOP_X * 2 * TARGET_SLOTS_PER_DIALOG          # e.g., 17*2*4 = 136

# Reservation expires after X minutes (if someone cancels)
RESERVATION_TIMEOUT_MINUTES = int(os.getenv("RESERVATION_TIMEOUT_MINUTES", "20"))

RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

RANK_BY_RECIPE: Dict[str, float] = {}

SCROLL_TOP_JS = """
(...args) => {
  try {
    const el = document.getElementById("page-top-anchor");
    if (el) el.scrollIntoView({ block: "start", behavior: "auto" });
    else window.scrollTo(0, 0);
  } catch (e) {
    try { window.scrollTo(0, 0); } catch (e2) {}
  }
  return args;
}
"""

# -----------------------------
# Utils

def safe_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        if isinstance(x, float) and math.isnan(x):
            return ""
    except Exception:
        pass
    s = str(x).strip()
    return "" if (s == "" or s.lower() == "nan") else s

def parse_likert(choice):
    if choice is None or choice == "":
        return None
    try:
        return int(choice)
    except Exception:
        return None

def rng_for_rater(rater_id: str) -> random.Random:
    rid = safe_text(rater_id) or "RATER_NONE"
    s = f"{RANDOM_SEED}::{rid}".encode("utf-8")
    seed_int = int(hashlib.sha256(s).hexdigest()[:16], 16)
    return random.Random(seed_int)

def stable_index(rater_id: str, n: int) -> int:
    rid = safe_text(rater_id) or "RATER_NONE"
    h = hashlib.sha256(rid.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % max(n, 1)

def fix_dialog_newlines(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\*sFat:", r"\nFat:", text)
    text = re.sub(r"\*sCarb:", r"\nCarb:", text)
    return text.strip()

def get_timestamp() -> float:
    """Current Unix timestamp."""
    return time.time()

def format_time_remaining(expires_at: float) -> str:
    """Formatted remaining time for display."""
    remaining = expires_at - get_timestamp()
    if remaining <= 0:
        return "expired"
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    return f"{minutes}:{seconds:02d}"

# -----------------------------
# Reservation management

def load_reservations() -> pd.DataFrame:
    """Loads active reservations (local only)."""
    if LOCAL_RESERVATIONS_CSV.exists():
        df = pd.read_csv(LOCAL_RESERVATIONS_CSV)
        df["reserved_at"] = pd.to_numeric(df.get("reserved_at"), errors="coerce")
        df["expires_at"] = pd.to_numeric(df.get("expires_at"), errors="coerce")
        df["status"] = df.get("status", "active")
        return df
    return pd.DataFrame(
        columns=[
            "rater_id", "pair_index", "dialog_id_good", "dialog_id_bad",
            "reserved_at", "expires_at", "status"
        ]
    )


def save_reservations(df: pd.DataFrame):
    """Save reservations (local only, no HF commits to avoid rate limiting)."""
    df.to_csv(LOCAL_RESERVATIONS_CSV, index=False, encoding="utf-8")


def cleanup_expired_reservations() -> pd.DataFrame:
    """Removes expired active reservations and returns DataFrame."""
    df = load_reservations()
    if df.empty:
        return df

    now = get_timestamp()

    # normalize status
    df["status"] = df["status"].astype(str).fillna("active")

    # expire only active reservations
    expired = df[(df["status"] == "active") & (df["expires_at"] < now)]

    if not expired.empty:
        logger.info(f"[RESERVATIONS] Cleaning up {len(expired)} expired reservations")

        # keep:
        # all non-active rows (if they ever exist)
        # active rows that are not expired
        df = df[(df["status"] != "active") | (df["expires_at"] >= now)].copy()
        save_reservations(df)

    return df

def has_active_reservation(rater_id: str) -> Optional[Dict[str, Any]]:
    """Checks whether the rater already has an active reservation."""
    df = cleanup_expired_reservations()
    if df.empty:
        return None

    df["status"] = df["status"].astype(str).fillna("active")
    mask = (df["rater_id"] == rater_id) & (df["status"] == "active")

    if mask.any():
        return df[mask].iloc[0].to_dict()
    return None

def create_reservation(rater_id: str, pair_index: int, dialog_id_good: str, dialog_id_bad: str) -> Dict[str, Any]:
    """Creates a new reservation."""
    df = cleanup_expired_reservations()

    if not df.empty:
        df["status"] = df["status"].astype(str).fillna("active")
        if ((df["rater_id"] == rater_id) & (df["status"] == "active")).any():
            raise RuntimeError(f"Rater {rater_id} already has an active reservation")

    now = get_timestamp()
    expires = now + (RESERVATION_TIMEOUT_MINUTES * 60)

    new_res = {
        "rater_id": rater_id,
        "pair_index": pair_index,
        "dialog_id_good": dialog_id_good,
        "dialog_id_bad": dialog_id_bad,
        "reserved_at": now,
        "expires_at": expires,
        "status": "active"
    }

    df = pd.concat([df, pd.DataFrame([new_res])], ignore_index=True)
    save_reservations(df)
    logger.info(f"[RESERVATIONS] Created for {rater_id}, pair {pair_index}, expires in {RESERVATION_TIMEOUT_MINUTES}min")

    return new_res

def confirm_reservation(rater_id: str) -> bool:
    """Confirms reservation by removing it (submit finalization)."""
    df = load_reservations()
    mask = df["rater_id"] == rater_id

    if not mask.any():
        logger.warning(f"[RESERVATIONS] No reservation found for {rater_id} to confirm")
        return False

    # Remove reservation entirely so the pair can be assigned again if slots remain
    df = df[~mask].copy()
    save_reservations(df)
    return True

def release_reservation(rater_id: str) -> bool:
    """Releases reservation (manual cancellation)."""
    df = load_reservations()
    mask = df["rater_id"] == rater_id
    
    if not mask.any():
        return False
    
    df = df[~mask].copy()
    save_reservations(df)
    logger.info(f"[RESERVATIONS] Released for {rater_id}")
    return True

def get_reserved_pair_indices() -> List[int]:
    df = cleanup_expired_reservations()
    if df.empty:
        return []
    df["status"] = df["status"].astype(str).fillna("active")
    df = df[df["status"] == "active"].copy()
    return df["pair_index"].astype(int).tolist()

# -----------------------------
# Assignment Status

def load_assignment_status() -> pd.DataFrame:
    """
    Load assignment status:
    - Prefer local status for instant display / runtime truth
    - Otherwise build from seed (HF download), initialize slots, and save local
    """
    # Prefer local
    if LOCAL_STATUS_CSV.exists():
        return pd.read_csv(LOCAL_STATUS_CSV)

    # Create from seed
    seed_df = hf_download_csv(ASSIGNMENT_SEED_FILE)
    required = ["dialog_id", "recipe_title", "condition"]
    missing = [c for c in required if c not in seed_df.columns]
    if missing:
        raise ValueError(f"Seed file missing columns: {missing}")

    status_df = seed_df[["dialog_id", "recipe_title", "condition"]].copy()
    status_df["dialog_id"] = status_df["dialog_id"].astype(str).str.strip()
    status_df["recipe_title"] = status_df["recipe_title"].apply(safe_text)
    status_df["condition"] = status_df["condition"].apply(lambda x: safe_text(x).lower())

    # Enforce slots per dialog from config
    status_df["initial_slots"] = int(TARGET_SLOTS_PER_DIALOG)
    status_df["remaining_slots"] = int(TARGET_SLOTS_PER_DIALOG)

    # pair_index will be enforced later in load_data_or_raise()
    if "pair_index" not in status_df.columns:
        status_df["pair_index"] = -1

    save_assignment_status(status_df)  # local save
    return status_df


def save_assignment_status(df: pd.DataFrame):
    """Saves assignment status (local only)."""
    df.to_csv(LOCAL_STATUS_CSV, index=False, encoding="utf-8")


def decrement_slot(status_df: pd.DataFrame, dialog_id: str) -> pd.DataFrame:
    """Removes one slot (only with Submit!). Local save only."""
    mask = status_df["dialog_id"] == dialog_id
    if not mask.any():
        raise ValueError(f"Dialog {dialog_id} not found")

    current = int(status_df.loc[mask, "remaining_slots"].iloc[0])
    if current <= 0:
        raise RuntimeError(f"No remaining slots for {dialog_id}")

    status_df.loc[mask, "remaining_slots"] = current - 1
    save_assignment_status(status_df)  # LOCAL persist immediately
    logger.info(f"[STATUS] Slot decremented for {dialog_id}: {current} -> {current-1}")

    return status_df


def get_pair_slot_info(status_df: pd.DataFrame, pair_index: int, 
                      reserved_indices: List[int] = None) -> Dict[str, Any]:
    """Returns slot information, taking reservations into account."""
    pair_dialogs = status_df[status_df["pair_index"] == pair_index]
    if len(pair_dialogs) != 2:
        raise ValueError(f"Expected 2 dialogs for pair {pair_index}")
    
    good_row = pair_dialogs[pair_dialogs["condition"] == "good"].iloc[0]
    bad_row = pair_dialogs[pair_dialogs["condition"] == "bad"].iloc[0]
    
    good_remaining = int(good_row["remaining_slots"])
    bad_remaining = int(bad_row["remaining_slots"])
    
    # Consider active reservations
    is_reserved = pair_index in (reserved_indices or [])
    
    # Available if both slots > 0 AND not reserved
    pair_available = (good_remaining > 0) and (bad_remaining > 0) and not is_reserved
    
    return {
        "pair_index": pair_index,
        "recipe_title": good_row["recipe_title"],
        "dialog_id_good": good_row["dialog_id"],
        "dialog_id_bad": bad_row["dialog_id"],
        "good_remaining": good_remaining,
        "bad_remaining": bad_remaining,
        "is_reserved": is_reserved,
        "pair_available": pair_available,
        "min_remaining": min(good_remaining, bad_remaining),
    }

def get_available_pairs(status_df: pd.DataFrame, reserved_indices: List[int] = None) -> List[int]:
    """Returns available pairs (not reserved, with remaining slots)."""
    available: List[int] = []
    reserved_set = set(reserved_indices or [])

    # Defensive: ignore any invalid pair_index values (e.g., -1)
    all_pairs = sorted([int(x) for x in status_df["pair_index"].unique() if pd.notna(x) and int(x) >= 0])

    for pair_idx in all_pairs:
        info = get_pair_slot_info(status_df, pair_idx, reserved_indices=list(reserved_set))
        if info["pair_available"]:
            available.append(pair_idx)

    return available

def get_total_progress(status_df: pd.DataFrame) -> Dict[str, int]:
    """Returns overall progress."""
    total_initial = int(status_df["initial_slots"].sum())
    total_remaining = int(status_df["remaining_slots"].sum())
    total_completed = total_initial - total_remaining
    
    return {
        "total_completed": total_completed,
        "total_remaining": total_remaining,
        "total_target": total_initial,
        "is_complete": total_remaining == 0
    }

# -----------------------------
# HF helpers

def hf_download_csv(filename: str) -> pd.DataFrame:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing. Add HF_TOKEN in Space Secrets.")
    p = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=filename,
        token=HF_TOKEN,
    )
    return pd.read_csv(p)

def hf_download_text(path_in_repo: str) -> Optional[str]:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing.")
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
        logger.info(f"[hf_download_text] could not download {path_in_repo}: {e!r}")
        return None

def upload_text_to_dataset(text: str, path_in_repo: str, commit_message: str):
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing. Needs WRITE access for uploads.")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo=path_in_repo,
        repo_id=DATASET_ID,
        repo_type="dataset",
        token=HF_TOKEN,
        commit_message=commit_message,
    )

def append_row_to_shared_csv(row: Dict[str, Any]):
    """Append row to shared CSV."""
    df_new = pd.DataFrame([row])
    new_csv = df_new.to_csv(index=False, encoding="utf-8")

    existing = hf_download_text(SHARED_CSV)
    if existing and existing.strip():
        appended = existing.rstrip("\n") + "\n" + "\n".join(new_csv.splitlines()[1:]) + "\n"
        upload_text_to_dataset(appended, SHARED_CSV, 
                              commit_message=f"Append pilot row {row.get('rater_id')}")
    else:
        upload_text_to_dataset(new_csv, SHARED_CSV, 
                              commit_message="Create pilot ratings file")

def hf_commit_ratings_and_status():
    if not HF_TOKEN:
        logger.warning("HF_TOKEN missing; skip HF commit.")
        return

    if not LOCAL_SHARED_CSV.exists() or not LOCAL_STATUS_CSV.exists():
        logger.warning("Local ratings/status missing; skip HF commit.")
        return

    api = HfApi()
    ratings_bytes = LOCAL_SHARED_CSV.read_bytes()
    status_bytes = LOCAL_STATUS_CSV.read_bytes()

    ops = [
        CommitOperationAdd(path_in_repo=SHARED_CSV, path_or_fileobj=ratings_bytes),
        CommitOperationAdd(path_in_repo=ASSIGNMENT_STATUS_FILE, path_or_fileobj=status_bytes),
    ]

    try:
        api.create_commit(
            repo_id=DATASET_ID,
            repo_type="dataset",
            token=HF_TOKEN,
            operations=ops,
            commit_message="Pilot: update ratings+status (single commit)",
        )
    except Exception as e:
        logger.error(f"[HF COMMIT] failed (kept local): {e}")


# -----------------------------
# Load data

stimuli_df = pd.DataFrame()
pairs_df = pd.DataFrame()
stim_by_id: Dict[str, Dict[str, Any]] = {}
assignment_status_df: pd.DataFrame = pd.DataFrame()

def load_data_or_raise():
    global stimuli_df, pairs_df, stim_by_id, assignment_status_df, RANK_BY_RECIPE

    # pair_index and rank mapping are consistent
    # assignment_status_df has a valid pair_index mapping
    if (
        not stimuli_df.empty
        and not pairs_df.empty
        and stim_by_id
        and not assignment_status_df.empty
        and "pair_index" in assignment_status_df.columns
        and assignment_status_df["pair_index"].notna().all()
        and (assignment_status_df["pair_index"].astype(int) >= 0).all()
    ):
        return

    # Load stimuli
    s = hf_download_csv(STIMULI_FILE)
    required_s = [
        "dialog_id", "recipe_title", "recipe_type", "condition", "dialog_text",
        "fat", "carbs", "calories", "servings", "description", "directions", "ingredients_list"
    ]
    missing_s = [c for c in required_s if c not in s.columns]
    if missing_s:
        raise ValueError(f"Stimuli file missing columns: {missing_s}")

    s["dialog_id"] = s["dialog_id"].astype(str).str.strip()
    s["recipe_title"] = s["recipe_title"].apply(safe_text)
    s["recipe_type"] = s["recipe_type"].apply(lambda x: safe_text(x).lower())
    s["condition"] = s["condition"].apply(lambda x: safe_text(x).lower())
    s["dialog_text"] = s["dialog_text"].apply(fix_dialog_newlines)
    for col in ["description", "directions", "ingredients_list"]:
        s[col] = s[col].apply(safe_text)

    # Load top pairs
    p = hf_download_csv(TOP_PAIRS_FILE)
    required_p = ["recipe_title", "dialog_id_good", "dialog_id_bad", "pass_precheck", "rank_score"]
    missing_p = [c for c in required_p if c not in p.columns]
    if missing_p:
        raise ValueError(f"Top-pairs file missing columns: {missing_p}")

    p["recipe_title"] = p["recipe_title"].apply(safe_text)
    p["dialog_id_good"] = p["dialog_id_good"].astype(str).str.strip()
    p["dialog_id_bad"] = p["dialog_id_bad"].astype(str).str.strip()

    p["pass_precheck"] = p["pass_precheck"].astype(str).str.lower().isin(["true","1","yes"])
    p = p[p["pass_precheck"] == True].copy()
    p["rank_score"] = pd.to_numeric(p["rank_score"], errors="coerce")

    # Rank order = pair_index order
    p = (
        p.sort_values("rank_score", ascending=False)
         .head(max(TOP_X, 1))
         .reset_index(drop=True)
         .copy()
    )
    p["pair_index"] = range(len(p))  # enforce rank-based pair_index

    # Build lookup maps (rank_score by recipe_title)
    RANK_BY_RECIPE = {
        rt: (None if pd.isna(rs) else float(rs))
        for rt, rs in zip(p["recipe_title"].tolist(), p["rank_score"].tolist())
    }
    recipe_to_pairindex = {rt: int(i) for i, rt in enumerate(p["recipe_title"].tolist())}

    # Commit to globals
    stimuli_df = s.copy()
    pairs_df = p.copy()
    stim_by_id = {row["dialog_id"]: row.to_dict() for _, row in stimuli_df.iterrows()}

    # Load assignment status (seed or existing) and ENFORCE pair_index mapping
    assignment_status_df = load_assignment_status()

    # Ensure column exists (seed may not have it or existing might be stale)
    if "pair_index" not in assignment_status_df.columns:
        assignment_status_df["pair_index"] = -1

    assignment_status_df["pair_index"] = assignment_status_df["recipe_title"].map(recipe_to_pairindex)

    # Safety: every recipe_title in status must exist in top pairs
    if assignment_status_df["pair_index"].isna().any():
        missing = (
            assignment_status_df.loc[assignment_status_df["pair_index"].isna(), "recipe_title"]
            .dropna()
            .unique()
            .tolist()
        )
        raise ValueError(f"[STATUS] recipe_title(s) in seed/status not found in top pairs: {missing}")

    assignment_status_df["pair_index"] = assignment_status_df["pair_index"].astype(int)

    logger.info(
        f"[LOAD] stimuli={stimuli_df.shape} | pairs={pairs_df.shape} | status={assignment_status_df.shape}"
    )

# -----------------------------
# Reservation-based allocation

def assign_pair_for_rater(rater_id: str, exclude_pair_indices: Optional[List[int]] = None) -> Dict[str, Any]:
    """
    Assign pair with reservation.

    Procedure:
    1. Check whether a reservation already exists
    2. Load fresh data/status (pair_index mapping enforced in load_data_or_raise)
    3. Check available pairs (excluding active reservations)
    4. Create reservation
    5. On submit: confirm reservation (remove it) and deduct slots
    """
    global assignment_status_df

    # Check existing active reservation
    existing = has_active_reservation(rater_id)
    if existing:
        logger.info(f"[ASSIGN] Reusing existing reservation for {rater_id}")
        pair_index = int(existing["pair_index"])

        # still valid?
        if float(existing.get("expires_at", 0)) > get_timestamp():
            # ensure that consistent status loaded
            load_data_or_raise()

            # IMPORTANT: pass reserved indices so is_reserved is computed correctly
            reserved = get_reserved_pair_indices()
            pair_info = get_pair_slot_info(assignment_status_df, pair_index, reserved_indices=reserved)

            return _build_assignment_response(
                rater_id=rater_id,
                pair_index=pair_index,
                pair_info=pair_info,
                assignment_type="existing_reservation",
                expires_at=float(existing["expires_at"]),
            )
        else:
            logger.info(f"[ASSIGN] Existing reservation expired for {rater_id}")

    # Load fresh data/status (includes assignment_status_df reload and enforced pair_index mapping to top pairs)
    load_data_or_raise()

    # Always reload latest status before allocation (avoid stale global df)
    assignment_status_df = load_assignment_status()

    # Re-enforce pair_index mapping (status file might be stale / missing mapping)
    recipe_to_pairindex = {rt: int(i) for i, rt in enumerate(pairs_df["recipe_title"].tolist())}
    assignment_status_df["pair_index"] = assignment_status_df["recipe_title"].map(recipe_to_pairindex)

    if assignment_status_df["pair_index"].isna().any():
        missing = (
            assignment_status_df.loc[assignment_status_df["pair_index"].isna(), "recipe_title"]
            .dropna().unique().tolist()
        )
        raise ValueError(f"[STATUS] recipe_title(s) in status not found in top pairs: {missing}")

    assignment_status_df["pair_index"] = assignment_status_df["pair_index"].astype(int)
    
    # Check Pilot-Status
    progress = get_total_progress(assignment_status_df)
    if progress["is_complete"]:
        raise RuntimeError(f"Mini-Pilot complete. All {progress['total_target']} slots assigned.")

    # Available pairs (taking active reservations into account)
    reserved = get_reserved_pair_indices()
    available = get_available_pairs(assignment_status_df, reserved)

    exclude_set = set(int(x) for x in (exclude_pair_indices or []))
    if exclude_set:
        available = [p for p in available if int(p) not in exclude_set]

    if not available:
        # Try cleanup and check again
        cleanup_expired_reservations()
        reserved = get_reserved_pair_indices()
        available = get_available_pairs(assignment_status_df, reserved)

        if exclude_set:
            available = [p for p in available if int(p) not in exclude_set]

        if not available:
            raise RuntimeError("No available pairs. All slots reserved or filled.")

    # Deterministic selection (based on rank-based pair_index space 0..TOP_X-1)
    deterministic_i = stable_index(rater_id, len(pairs_df))

    if deterministic_i in available:
        chosen_i = deterministic_i
        assignment_type = "deterministic"
    else:
        # pick closest available pair_index to deterministic target
        available_sorted = sorted(available, key=lambda x: abs(int(x) - int(deterministic_i)))
        chosen_i = int(available_sorted[0])
        assignment_type = "fallback"

    # Get pair info (with reservation awareness)
    pair_info = get_pair_slot_info(assignment_status_df, chosen_i, reserved_indices=reserved)

    # Create reservation (will block this pair for other users until submit/timeout)
    expires_at = get_timestamp() + (RESERVATION_TIMEOUT_MINUTES * 60)
    try:
        create_reservation(
            rater_id=rater_id,
            pair_index=chosen_i,
            dialog_id_good=pair_info["dialog_id_good"],
            dialog_id_bad=pair_info["dialog_id_bad"],
        )
    except RuntimeError:
        # If conflict occurs, try again (rare but possible with concurrent allocations)
        logger.warning(f"ASSIGN Reservation conflict for {rater_id}, retrying...")
        return assign_pair_for_rater(rater_id, exclude_pair_indices=exclude_pair_indices)

    return _build_assignment_response(
        rater_id=rater_id,
        pair_index=chosen_i,
        pair_info=pair_info,
        assignment_type=assignment_type,
        expires_at=expires_at,
    )

def _pack_to_renderables(rid: str, pack: Dict[str, Any]) -> Tuple[str, str, str, str, str, str, Dict[str, Any]]:
    """
    Turns an assignment 'pack' into (progress_info, recipe_html, A_title, A_html, B_title, B_html, new_state)
    """
    recipe_html = build_recipe_card_html(pack["stim_good"])

    if pack["order"] == "good_first":
        stim1, stim2 = pack["stim_good"], pack["stim_bad"]
        true1, true2 = "good", "bad"
        id1, id2 = pack["dialog_id_good"], pack["dialog_id_bad"]
    else:
        stim1, stim2 = pack["stim_bad"], pack["stim_good"]
        true1, true2 = "bad", "good"
        id1, id2 = pack["dialog_id_bad"], pack["dialog_id_good"]

    dialog1_html = dialog_to_bubbles(safe_text(stim1.get("dialog_text")))
    dialog2_html = dialog_to_bubbles(safe_text(stim2.get("dialog_text")))

    # progress text in UI
    progress_info = "Please read the recipe and both dialogs carefully."

    # Note: pair_no and seen_pair_indices are set by caller
    state = {
        "pair_index": pack["pair_index"],
        "recipe_title": pack["recipe_title"],
        "order": pack["order"],
        "dialog_id_good": pack["dialog_id_good"],
        "dialog_id_bad": pack["dialog_id_bad"],
        "dialogA_true": true1,
        "dialogB_true": true2,
        "dialog_id_A": id1,
        "dialog_id_B": id2,
        "rank_score": pack.get("rank_score"),
        "reservation_expires_at": pack["reservation_expires_at"],
    }

    return (
        progress_info,
        recipe_html,
        "## Dialog A",
        dialog1_html,
        "## Dialog B",
        dialog2_html,
        state,
    )


def _build_assignment_response(
    rater_id: str,
    pair_index: int,
    pair_info: Dict,
    assignment_type: str,
    expires_at: float
) -> Dict[str, Any]:
    """Builds allocation response."""

    stim_good = stim_by_id.get(pair_info["dialog_id_good"])
    stim_bad = stim_by_id.get(pair_info["dialog_id_bad"])

    if not stim_good or not stim_bad:
        raise ValueError(f"Missing stimuli for pair {pair_index}")

    rng = rng_for_rater(rater_id)
    order = "good_first" if rng.random() < 0.5 else "bad_first"

    progress = get_total_progress(assignment_status_df)
    time_remaining = format_time_remaining(expires_at)

    # rank_score must be consistent with recipe_title (not iloc[pair_index])
    recipe_title = pair_info["recipe_title"]
    rank_score = None
    try:
        rank_score = RANK_BY_RECIPE.get(recipe_title, None)
        if rank_score is not None:
            rank_score = float(rank_score)
    except Exception:
        rank_score = None

    return {
        "pair_index": pair_index,
        "recipe_title": recipe_title,
        "dialog_id_good": pair_info["dialog_id_good"],
        "dialog_id_bad": pair_info["dialog_id_bad"],
        "order": order,
        "stim_good": stim_good,
        "stim_bad": stim_bad,
        "rank_score": rank_score,
        "assignment_type": assignment_type,
        "reservation_expires_at": expires_at,
        "time_remaining": time_remaining,
        "total_completed": progress["total_completed"],
        "total_target": progress["total_target"],
    }

def confirm_and_finalize(rater_id: str, pair_index: int, dialog_id_good: str, dialog_id_bad: str):
    """
    Confirms reservation and deducts slots (called when Submit is clicked).
    Robust against parallel submits by reloading assignment_status from HF first.
    """
    global assignment_status_df

    # Ensure globals/data are initialized (safe no-op if already loaded)
    load_data_or_raise()

    # Always reload latest status to avoid stale in-memory state under concurrency
    assignment_status_df = load_assignment_status()

    try:
        # Deduct slots first (authoritative)
        assignment_status_df = decrement_slot(assignment_status_df, dialog_id_good)
        assignment_status_df = decrement_slot(assignment_status_df, dialog_id_bad)

        # Now remove reservation (submit finalization)
        if not confirm_reservation(rater_id):
            logger.warning(f"FINALIZE Could not confirm reservation for {rater_id}")

        logger.info(f"FINALIZE Slots decremented for pair {pair_index}")
        return True

    except Exception as e:
        logger.error(f"FINALIZE Failed to decrement slots: {e}")
        return False

# -----------------------------
# Rendering helpers

def parse_dialog_turns(raw: str):
    if not isinstance(raw, str):
        return []
    text = re.sub(r"\s*(Fat:)", r"\n\1", raw)
    text = re.sub(r"\s*(Carb:)", r"\n\1", text)
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    turns = []
    for line in lines:
        if line.startswith("Fat:"):
            turns.append(("Fat", line[len("Fat:"):].strip()))
        elif line.startswith("Carb:"):
            turns.append(("Carb", line[len("Carb:"):].strip()))
        else:
            turns.append(("Other", line))
    return turns

def dialog_to_bubbles(raw: str) -> str:
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
        html.append(f"<div class='{bubble_class}'>{speaker_html}<div class='bubble-text'>{content}</div></div>")
    html.append("</div>")
    return "\n".join(html)

def build_recipe_card_html(stim: Dict[str, Any]) -> str:
    def esc(s: str) -> str:
        s = "" if s is None else str(s)
        return (s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
                 .replace('"',"&quot;").replace("'","&#39;"))

    title = esc(safe_text(stim.get("recipe_title")) or "Recipe")
    rtype = esc(safe_text(stim.get("recipe_type")).lower())
    servings = safe_text(stim.get("servings"))
    desc = safe_text(stim.get("description"))
    ingred = safe_text(stim.get("ingredients_list"))
    directions = safe_text(stim.get("directions"))

    badges = []
    if rtype:
        badges.append(f"<span class='recipe-badge'>{rtype}</span>")
    if servings:
        badges.append(f"<span class='recipe-badge'>{esc(servings)} servings</span>")
    badge_html = "".join(badges)

    def para_block(raw: str) -> str:
        raw = safe_text(raw)
        if not raw:
            return ""
        raw = re.sub(r"<\s*br\s*/?\s*>", "\n", raw, flags=re.IGNORECASE)
        parts = [p.strip() for p in raw.split("\n") if p.strip()]
        return "".join(f"<p class='recipe-text'>{esc(p)}</p>" for p in parts)

    def ingredient_chips(raw: str) -> str:
        s = safe_text(raw).strip()
        if not s:
            return ""
        items: List[str] = []
        if s.startswith("[") and s.endswith("]"):
            inner = s[1:-1].strip()
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            for p in parts:
                p2 = p.strip().strip('"').strip("'").strip()
                if p2:
                    items.append(p2)
        else:
            parts = [p.strip() for p in s.split(",") if p.strip()]
            items = parts if len(parts) >= 2 else [s]
        items = items[:60]
        return "<div class='ing-chips'>" + "".join(f"<span class='ing-chip'>{esc(it)}</span>" for it in items) + "</div>"

    def box(title_: str, inner: str) -> str:
        if not inner:
            return ""
        return (
            "<div class='recipe-box'>"
            f"<div class='recipe-box-title'>{esc(title_)}</div>"
            f"{inner}"
            "</div>"
        )

    return (
        "<div class='recipe-card'>"
        "<div class='recipe-header'>"
        f"<div class='recipe-title'>{title}</div>"
        f"{badge_html}"
        "</div>"
        "<div class='recipe-grid-boxes-2'>"
        f"{box('Description', para_block(desc))}"
        f"{box('Ingredients', ingredient_chips(ingred))}"
        "</div>"
        "<div class='recipe-boxes-1'>"
        f"{box('Directions', para_block(directions))}"
        "</div>"
        "</div>"
    )

# -----------------------------
# Persist

def append_row_local_shared(row: Dict[str, Any]):
    """Append one row to local shared CSV (instant availability in Space)."""
    df_new = pd.DataFrame([row])
    write_header = not LOCAL_SHARED_CSV.exists()
    df_new.to_csv(
        LOCAL_SHARED_CSV,
        mode="a",
        index=False,
        header=write_header,
        encoding="utf-8"
    )

def persist_row(rater_id: str, row: Dict[str, Any]):
    """Persist per-rater file and shared file (local only)."""
    rid = safe_text(rater_id) or "RATER_NONE"

    # per-rater log
    out_local = RESULTS_DIR / f"ratings_{rid}.csv"
    df = pd.DataFrame([row])
    write_header = not out_local.exists()
    df.to_csv(out_local, mode="a", index=False, header=write_header, encoding="utf-8")

    # shared ratings file for instant display
    append_row_local_shared(row)

# -----------------------------
# UI callbacks

LIKERT_1_7 = [1,2,3,4,5,6,7]

def toggle_start(english_ok: bool, imc_answer: str):
    return gr.update(interactive=bool(english_ok and imc_answer == "Strongly agree"))

def toggle_consent(consent: bool):
    return gr.update(interactive=bool(consent))

def get_status_text() -> str:
    try:
        global assignment_status_df
        load_data_or_raise()

        # always reload latest status from HF (avoid stale cached df)
        assignment_status_df = load_assignment_status()
        progress = get_total_progress(assignment_status_df)

        if progress["is_complete"]:
            return "**Mini-Pilot complete!** Thank you for your interest."
        else:
            return "**Mini-Pilot active.** Please continue."
    except Exception:
        return "Loading..."

def handle_welcome(rater_id_input, english_ok, imc_answer):
    try:
        global assignment_status_df
        load_data_or_raise()

        assignment_status_df = load_assignment_status()
        progress = get_total_progress(assignment_status_df)
        if progress["is_complete"]:
            return (
                "The mini-pilot is already complete. All assignments have been filled.",
                None,
                gr.update(visible=True),
                gr.update(visible=False),
                0,
            )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
    
    imc_pass = int(imc_answer == "Strongly agree")
    if not (english_ok and imc_answer == "Strongly agree"):
        return (
            "Please confirm your English skills and choose 'Strongly agree'.",
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            imc_pass,
        )
    rid = safe_text(rater_id_input) or f"P{random.randint(10000,99999)}"
    return (
        f"Participant ID: **{rid}**\n\n{get_status_text()}",
        rid,
        gr.update(visible=False),
        gr.update(visible=True),
        imc_pass,
    )

def handle_consent_continue(consent_checked: bool, rater_id: str):
    if not consent_checked:
        return ("Please check the consent box to continue.", gr.update(visible=True), gr.update(visible=False))
    return ("", gr.update(visible=False), gr.update(visible=True))

def handle_instructions_continue():
    return ("", gr.update(visible=False), gr.update(visible=True))

def start_pilot(rater_id: str):
    rid = safe_text(rater_id) or f"P{random.randint(10000,99999)}"

    # ensure data/status are loaded and pair_index enforced
    load_data_or_raise()
    global assignment_status_df
    assignment_status_df = load_assignment_status()

    try:
        progress = get_total_progress(assignment_status_df)
        if progress["is_complete"]:
            return (
                "Mini-pilot is complete.",
                "", "", "", "", "", {}, gr.update(visible=False), gr.update(visible=False)
            )
    except Exception:
        pass

    try:
        # Pair 1 assignment
        pack = assign_pair_for_rater(rid)
    except RuntimeError as e:
        return (str(e), "", "", "", "", "", {}, gr.update(visible=False), gr.update(visible=False))

    # Renderables via helper (recommended)
    progress_info, recipe_html, a_title, a_html, b_title, b_html, state = _pack_to_renderables(rid, pack)

    # Pair tracking for 2-pairs-per-participant flow
    state["pair_no"] = 1
    state["seen_pair_indices"] = [int(pack["pair_index"])]

    # show pair counter (uses global/const PAIRS_PER_PARTICIPANT)
    progress_info = f"Pair 1/{PAIRS_PER_PARTICIPANT} - please read the recipe and both dialogs carefully."

    return (
        progress_info,
        recipe_html,
        a_title,
        a_html,
        b_title,
        b_html,
        state,
        gr.update(visible=False),
        gr.update(visible=True),
    )

def submit_pilot(
    dqA_overall,
    dqB_overall,
    forced_choice,
    comment,
    rater_id,
    imc_pass,
    pilot_state: dict
):
    # require answers
    if dqA_overall is None or dqB_overall is None or forced_choice is None:
        return (
            "", "", "", "", "", "", pilot_state,
            dqA_overall, dqB_overall, forced_choice, comment,
            "Please answer all questions before submitting.",
            gr.update(visible=True),
            gr.update(visible=False),
            ""
        )

    rid = safe_text(rater_id) or "RATER_NONE"
    st = pilot_state or {}

    pair_no = int(st.get("pair_no", 1))
    seen = st.get("seen_pair_indices") or []
    seen = [int(x) for x in seen if x is not None]

    # Timeout-Check
    expires_at = float(st.get("reservation_expires_at", 0) or 0)
    if get_timestamp() > expires_at:
        logger.warning(f"[SUBMIT] Reservation expired for {rid} (pair_no={pair_no})")

    # finalize reservation and decrement slots (only if reservation is still active and matches)
    pair_index = st.get("pair_index")
    dialog_id_good = st.get("dialog_id_good")
    dialog_id_bad = st.get("dialog_id_bad")

    active_res = has_active_reservation(rid)
    can_finalize = (
        active_res is not None
        and pair_index is not None
        and int(active_res.get("pair_index", -1)) == int(pair_index)
    )

    if can_finalize and dialog_id_good and dialog_id_bad:
        try:
            ok = confirm_and_finalize(rid, pair_index, dialog_id_good, dialog_id_bad)
            if not ok:
                logger.warning(f"[SUBMIT] finalize returned False for {rid} (pair_no={pair_no})")
        except Exception as e:
            logger.error(f"[SUBMIT] Failed to finalize: {e}")
    else:
        msg = (
            "Your session timed out before submission (reservation expired). "
            "Please restart the task to receive a fresh assignment.\n\n"
            "No data were submitted."
        )
        logger.warning(
            f"[SUBMIT] Rejecting submit (no matching active reservation). "
            f"rid={rid}, pair_no={pair_no}, pair_index={pair_index}, active_res={active_res}"
        )
        return (
            "", "", "", "", "", "", st,
            dqA_overall, dqB_overall, forced_choice, comment,
            msg,
            gr.update(visible=True),
            gr.update(visible=False),
            ""
        )

    # persist data row (one row per pair)
    row = {
        "timestamp": time.time(),
        "rater_id": rid,
        "imc_pass": int(imc_pass) if imc_pass is not None else None,
        "pair_no": pair_no,
        "pair_index": st.get("pair_index"),
        "rank_score": st.get("rank_score"),
        "recipe_title": st.get("recipe_title"),
        "order": st.get("order"),
        "dialog_id_good": dialog_id_good,
        "dialog_id_bad": dialog_id_bad,
        "dialogA_true": st.get("dialogA_true"),
        "dialogB_true": st.get("dialogB_true"),
        "dialog_id_A": st.get("dialog_id_A"),
        "dialog_id_B": st.get("dialog_id_B"),
        "dqA_overall": parse_likert(dqA_overall),
        "dqB_overall": parse_likert(dqB_overall),
        "forced_choice": safe_text(forced_choice),
        "comment": safe_text(comment),
    }
    persist_row(rid, row)

    # ensure latest local status is written
    try:
        global assignment_status_df
        save_assignment_status(assignment_status_df)
    except Exception as e:
        logger.warning(f"Could not save local status: {e}")

    # single HF commit for both files (ratings and status)
    # requires implementation of hf_commit_ratings_and_status() + CommitOperationAdd import
    try:
        hf_commit_ratings_and_status()
    except Exception as e:
        logger.warning(f"[HF COMMIT] skipped/failed (local kept): {e}")

    # If still owe another pair, load it now
    if pair_no < PAIRS_PER_PARTICIPANT:
        try:
            pack2 = assign_pair_for_rater(rid, exclude_pair_indices=seen)

            progress_info, recipe_html, a_title, a_html, b_title, b_html, state2 = _pack_to_renderables(rid, pack2)

            state2["pair_no"] = pair_no + 1
            state2["seen_pair_indices"] = seen + [int(pack2["pair_index"])]

            progress_info = f"Pair {state2['pair_no']}/{PAIRS_PER_PARTICIPANT} — please read the recipe and both dialogs carefully."

            return (
                progress_info,
                recipe_html,
                a_title,
                a_html,
                b_title,
                b_html,
                state2,
                None,  # dqA_overall reset
                None,  # dqB_overall reset
                None,  # forced_choice reset
                "",    # comment reset
                "",    # pilot_status clear
                gr.update(visible=True),
                gr.update(visible=False),
                ""
            )

        except Exception as e:
            logger.error(f"[SUBMIT] Failed to assign next pair for {rid}: {e}")

            end_text = (
                "Thank you for participating!\n\n"
                f"Your participant code: **{rid}**\n\n"
                "(We could not assign another pair due to capacity constraints.)\n\n"
                "**You can now close the window.**"
            )
            return (
                "", "", "", "", "", "", st,
                dqA_overall, dqB_overall, forced_choice, comment,
                "",
                gr.update(visible=False),
                gr.update(visible=True),
                end_text
            )

    end_text = (
        "Thank you for participating!\n\n"
        f"Your participant code: **{rid}**\n\n"
        "**You can now close the window.**"
    )

    return (
        "", "", "", "", "", "", st,
        dqA_overall, dqB_overall, forced_choice, comment,
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        end_text
    )

# -----------------------------
# CSS

custom_css = """
html, body { height: auto; }
body { overflow-x: hidden; }
.gradio-container { overflow-x: hidden !important; }

#page-top-anchor { scroll-margin-top: 90px; }

.dialog-container {
  display:flex; flex-direction:column; gap:0.1rem;
  max-width:48rem; margin:0.25rem auto; padding:0 0.40rem;
  box-sizing:border-box;
}
.bubble {
  max-width:100%;
  width:fit-content;
  padding:0.5rem 0.75rem;
  border-radius:0.9rem;
  font-size:1rem;
  line-height:1.4;
  overflow-wrap:break-word;
  box-sizing:border-box;
}
@media (min-width: 541px) { .bubble { max-width: 34rem; } }
.bubble-fat  { align-self:flex-start; background:#e3f2fd; border:1px solid #bbdefb; }
.bubble-carb { align-self:flex-end;   background:#fff3e0; border:1px solid #ffe0b2; }
.bubble-other{ align-self:center;     background:#f5f5f5; border:1px solid #ededed; }
.speaker { font-weight:600; font-size:0.8rem; margin-bottom:0.1rem; opacity:0.85; }

.recipe-card {
  max-width:48rem; margin:0.5rem auto 1rem auto;
  padding:0.9rem 1rem;
  border:1px solid rgba(0,0,0,0.12);
  border-radius:1rem;
  background:rgba(235,250,225,0.75);
}
.recipe-header { display:flex; justify-content:space-between; align-items:baseline; gap:0.75rem; margin-bottom:0.35rem; }
.recipe-title { font-size:1.25rem; font-weight:700; }
.recipe-badge {
  font-size:0.85rem; padding:0.15rem 0.55rem; border-radius:999px;
  border:1px solid rgba(0,0,0,0.15); background:rgba(0,0,0,0.04);
}
.recipe-grid-boxes-2 { display:grid; grid-template-columns:1fr 1fr; gap:0.9rem; margin-top:0.75rem; }
@media (max-width: 720px) { .recipe-grid-boxes-2 { grid-template-columns: 1fr; } }
.recipe-box { border:1px solid rgba(0,0,0,0.12); border-radius:0.9rem; padding:0.75rem 0.85rem; background:rgba(0,0,0,0.02); }
.recipe-box-title { font-weight:700; margin:0 0 0.35rem 0; opacity:0.95; }
.recipe-text { margin:0; opacity:0.92; }
.ing-chips { display:flex; flex-wrap:wrap; gap:0.35rem; margin-top:0.35rem; }
.ing-chip { font-size:0.85rem; padding:0.12rem 0.5rem; border-radius:999px; border:1px solid rgba(0,0,0,0.12); background:rgba(0,0,0,0.03); }

button.green-btn, button.orange-btn { border-radius: 0.8rem !important; }
@media (max-width: 540px) { button.green-btn, button.orange-btn { width: 100% !important; } }
button.green-btn { background:#4caf50 !important; color:white !important; border-color:#4caf50 !important; }
button.orange-btn { background:#ff9800 !important; color:white !important; border-color:#ff9800 !important; }
"""

# -----------------------------
# Build UI

load_data_or_raise()

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## Study B – Mini Pilot (Dialog Manipulation Check)", elem_id="page-top-anchor")
    
    status_md = gr.Markdown(get_status_text())

    rater_id_state = gr.State(value=None)
    imc_pass_state = gr.State(value=None)
    pilot_state = gr.State(value={})

    with gr.Group(visible=True) as welcome_group:
        gr.Markdown("### Welcome")
        rater_id_input = gr.Textbox(label="Participant ID (optional)")
        english_ok_cb = gr.Checkbox(label="I confirm that I can read and understand English well enough to complete this study.")
        imc_radio = gr.Radio(
            ["Strongly disagree", "Disagree", "Neutral", "Agree", "Strongly agree"],
            label="To show that you read carefully, please choose 'Strongly agree'.",
        )
        start_btn = gr.Button("Start", interactive=False, elem_classes=["green-btn"])
        welcome_info = gr.Markdown("Fill in the fields above to start.")

        english_ok_cb.change(toggle_start, inputs=[english_ok_cb, imc_radio], outputs=start_btn)
        imc_radio.change(toggle_start, inputs=[english_ok_cb, imc_radio], outputs=start_btn)

    with gr.Group(visible=False) as consent_group:
        gr.Markdown("## Informed Consent (Mini Pilot)")
        gr.Markdown(
            """
**Purpose:** 
This mini-pilot checks whether two versions of a dialog (good vs. bad communication) are perceived differently.

**What you will do:**  
You will complete **2 short tasks**. Each task includes:  
1) Read one recipe (context)  
2) Read two short dialogs about it  
3) Rate both dialogs and choose which dialog communicated better overall

**Duration:** 
~6–12 minutes

Participation is voluntary. You can stop anytime by closing the browser.
No directly identifying information is collected.
            """
        )
        consent_cb = gr.Checkbox(label="I agree to participate.", value=False)
        consent_continue_btn = gr.Button("Continue", interactive=False, elem_classes=["green-btn"])
        consent_status = gr.Markdown("")
        consent_cb.change(toggle_consent, inputs=[consent_cb], outputs=consent_continue_btn)

    with gr.Group(visible=False) as instructions_group:
        gr.Markdown("## Instructions")
        gr.Markdown(
            """
Please read carefully and do not rush.

- You will see **one recipe** for context
- Then you will read **two dialogs** (Dialog A and Dialog B)
- After each dialog, answer **two short questions** about communication quality
- At the end, choose **which dialog had better communication overall**

Click **Continue** to start.
            """
        )
        instructions_continue_btn = gr.Button("Continue", elem_classes=["green-btn"])
        instructions_status = gr.Markdown("")

    with gr.Group(visible=False) as pilot_group:
        gr.Markdown("### Recipe (Context)")
        recipe_md = gr.Markdown("")
        progress_md = gr.Markdown("")

        dialogA_title = gr.Markdown("## Dialog A")
        dialogA_md = gr.Markdown("")

        dqA_overall = gr.Radio(LIKERT_1_7, label="Dialog A: How would you rate the quality of the communication in this dialog? (1 = very poor, 7 = excellent)")

        dialogB_title = gr.Markdown("## Dialog B")
        dialogB_md = gr.Markdown("")

        dqB_overall = gr.Radio(LIKERT_1_7, label="Dialog B: How would you rate the quality of the communication in this dialog? (1 = very poor, 7 = excellent)")

        forced_choice = gr.Radio(
            ["Dialog A", "Dialog B"],
            label="Which of the two dialogs communicated better overall? (Please focus only on communication, not on the recipe itself.)"
        )

        comment = gr.Textbox(label="Comment (optional)", lines=2)

        submit_btn = gr.Button("Submit", elem_classes=["orange-btn"])
        pilot_status = gr.Markdown("")

    with gr.Group(visible=False) as end_group:
        end_md = gr.Markdown("Thank you!")

    start_btn.click(
        handle_welcome,
        inputs=[rater_id_input, english_ok_cb, imc_radio],
        outputs=[welcome_info, rater_id_state, welcome_group, consent_group, imc_pass_state],
        js=SCROLL_TOP_JS,
    )

    consent_continue_btn.click(
        handle_consent_continue,
        inputs=[consent_cb, rater_id_state],
        outputs=[consent_status, consent_group, instructions_group],
        js=SCROLL_TOP_JS,
    )

    instructions_continue_btn.click(
        handle_instructions_continue,
        inputs=[],
        outputs=[instructions_status, instructions_group, pilot_group],
        js=SCROLL_TOP_JS,
    ).then(
        start_pilot,
        inputs=[rater_id_state],
        outputs=[
            progress_md,
            recipe_md,
            dialogA_title,
            dialogA_md,
            dialogB_title,
            dialogB_md,
            pilot_state,
            instructions_group,
            pilot_group,
        ],
        js=SCROLL_TOP_JS,
    )

    submit_btn.click(
        submit_pilot,
        inputs=[
            dqA_overall,
            dqB_overall,
            forced_choice,
            comment,
            rater_id_state,
            imc_pass_state,
            pilot_state,
        ],
        outputs=[
            progress_md,
            recipe_md,
            dialogA_title,
            dialogA_md,
            dialogB_title,
            dialogB_md,
            pilot_state,
            dqA_overall,
            dqB_overall,
            forced_choice,
            comment,
            pilot_status,
            pilot_group,
            end_group,
            end_md,
        ],
        js=SCROLL_TOP_JS,
    )

demo.queue()
demo.launch()