# app.py — Study B (HF Space) — Gradio 5.49.1 compatible
#
# Flow:
#  - Welcome (English + IMC)
#  - Instructions
#  - Consent (no reservation)
#  - Preferences & screening
#  - CLICK "Continue to recipe" --> reserve dialog_id based on preferences (no slot decrement yet)
#  - Pre measures
#  - Dialog
#  - Post measures
#  - CLICK "Submit" --> commit reservation + decrement remaining_slots by 1 + persist row
#
# Mini-safety against double reservation:
#  - If reservation_token_state is already present in the current session, reuse it (no new reservation)
#  - If session is reloaded, but user re-enters same rater_id, reuse active reservation server-side

import os
import re
import time
import math
import random
import logging
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager

import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download, HfApi


# ------------------------------------------------------------
# Global configuration

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("studyB_space")

DATASET_ID = os.getenv("DATASET_ID", "KKam799/studyB-storage")
HF_TOKEN = os.getenv("HF_TOKEN")

STIMULI_FILE = os.getenv("STIMULI_FILE", "studyB_stimuli.csv")
ASSIGN_FILE = os.getenv("ASSIGN_FILE", "studyB_assignment_seed.csv")
DATASET_RATINGS_DIR = os.getenv("DATASET_RATINGS_DIR", "ratings")

# QA / timing
MIN_DIALOG_TIME = float(os.getenv("MIN_DIALOG_TIME", "10.0")) # on dialog page
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))

# Reservation behavior
RESERVATION_TTL_SEC = int(os.getenv("RESERVATION_TTL_SEC", str(20 * 60)))  # 20 minutes default

SPACE_DATA_DIR = Path(os.getenv("SPACE_DATA_DIR", "/data/studyB"))
SPACE_DATA_DIR.mkdir(parents=True, exist_ok=True)

RUNTIME_ASSIGN_PATH = SPACE_DATA_DIR / "studyB_assignment.runtime.csv"
ASSIGN_STATUS_PATH = SPACE_DATA_DIR / "studyB_assignment.status.csv"
RESERVATIONS_PATH = SPACE_DATA_DIR / "studyB_reservations.csv"

# ------------------------------------------------------------
# Assignment status upload throttling

ASSIGN_UPLOAD_EVERY_N = int(os.getenv("ASSIGN_UPLOAD_EVERY_N", "10"))  # upload every 10 commits
ASSIGN_UPLOAD_MIN_INTERVAL = int(os.getenv("ASSIGN_UPLOAD_MIN_INTERVAL", "180"))  # 3 minutes

_last_assign_upload_ts = 0
_assign_commit_counter = 0

RESULTS_DIR = SPACE_DATA_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LOCK_PATH = SPACE_DATA_DIR / "studyB.lock"

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

# ------------------------------------------------------------
# Utilities

def fix_dialog_newlines(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\*sFat:", r"\nFat:", text)
    text = re.sub(r"\*sCarb:", r"\nCarb:", text)
    return text.strip()

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

def _to_float_safe(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
    except Exception:
        pass
    try:
        return float(str(x).strip())
    except Exception:
        return None

def _to_int_safe(x: Any) -> Optional[int]:
    f = _to_float_safe(x)
    if f is None:
        return None
    try:
        return int(round(f))
    except Exception:
        return None

def rng_for_rater(rater_id: str) -> random.Random:
    rid = (safe_text(rater_id) or "RATER_NONE")
    s = f"{RANDOM_SEED}::{rid}".encode("utf-8")
    seed_int = int(hashlib.sha256(s).hexdigest()[:16], 16)
    return random.Random(seed_int)

def parse_likert(choice):
    if choice is None or choice == "":
        return None
    try:
        return int(choice)
    except Exception:
        return None

def now_ts() -> float:
    return time.time()


# ------------------------------------------------------------
# Simple file lock (Linux HF Spaces)

@contextmanager
def file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+", encoding="utf-8")
    try:
        import fcntl
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        try:
            f.close()
        except Exception:
            pass


# ------------------------------------------------------------
# Preference thresholds (quartiles-based from previous version)

def _is_num(x):
    try:
        return x is not None and not (isinstance(x, float) and math.isnan(x))
    except Exception:
        return x is not None

# Preference thresholds (median split based on Study B stimuli)
THRESHOLDS = {
    "lower-fat":      lambda r: _is_num(r.get("fat")) and float(r["fat"]) <= 5.98,
    "lower-carb":     lambda r: _is_num(r.get("carbs")) and float(r["carbs"]) <= 10.57,
    "lower-calorie":  lambda r: _is_num(r.get("calories")) and float(r["calories"]) <= 106,
    "sweet":          lambda r: safe_text(r.get("recipe_type")).lower() == "sweet",
    "savory":         lambda r: safe_text(r.get("recipe_type")).lower() == "savory",
}

def prefs_match_score(stim_row: Dict[str, Any], prefs: List[str]) -> Tuple[int, List[str]]:
    prefs = [safe_text(p).lower() for p in (prefs or []) if safe_text(p)]
    sat = [p for p in prefs if p in THRESHOLDS and THRESHOLDS[p](stim_row)]
    return len(sat), sat


# ------------------------------------------------------------
# HF dataset helpers

def hf_download_csv(filename: str) -> pd.DataFrame:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN missing. Add HF_TOKEN in Space Secrets (read access).")
    local_path = hf_hub_download(
        repo_id=DATASET_ID,
        repo_type="dataset",
        filename=filename,
        token=HF_TOKEN,
    )
    return pd.read_csv(local_path)

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

def append_row_to_rater_csv_in_dataset(rater_id: str, row: Dict[str, Any]):
    rid = safe_text(rater_id) or "RATER_NONE"
    path_in_repo = f"{DATASET_RATINGS_DIR}/ratings_{rid}.csv"

    df_new = pd.DataFrame([row])
    new_csv = df_new.to_csv(index=False, encoding="utf-8")

    existing = hf_download_text(path_in_repo)
    if existing and existing.strip():
        appended = existing.rstrip("\n") + "\n" + "\n".join(new_csv.splitlines()[1:]) + "\n"
        upload_text_to_dataset(appended, path_in_repo, commit_message=f"Append StudyB row for {rid}")
    else:
        upload_text_to_dataset(new_csv, path_in_repo, commit_message=f"Create StudyB ratings for {rid}")


# ------------------------------------------------------------
# Load stimuli

stimuli_df = pd.DataFrame()

def load_stimuli_or_raise():
    global stimuli_df
    if not stimuli_df.empty:
        return

    df = hf_download_csv(STIMULI_FILE)

    required = [
        "dialog_id", "recipe_title", "recipe_type", "condition", "dialog_text",
        "fat", "carbs", "calories", "servings", "description", "directions", "ingredients_list"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Stimuli file missing columns: {missing}. Present={list(df.columns)}")

    df["dialog_id"] = df["dialog_id"].astype(str).str.strip()
    df["recipe_title"] = df["recipe_title"].apply(safe_text)
    df["recipe_type"] = df["recipe_type"].apply(lambda x: safe_text(x).lower())
    df["condition"] = df["condition"].apply(lambda x: safe_text(x).lower())
    df["dialog_text"] = df["dialog_text"].apply(fix_dialog_newlines)

    df["fat"] = pd.to_numeric(df["fat"], errors="coerce")
    df["carbs"] = pd.to_numeric(df["carbs"], errors="coerce")
    df["calories"] = pd.to_numeric(df["calories"], errors="coerce")
    df["servings"] = pd.to_numeric(df["servings"], errors="coerce")

    for col in ["description", "directions", "ingredients_list"]:
        df[col] = df[col].apply(safe_text)

    stimuli_df = df.copy()
    logger.info(f"[LOAD] stimuli_df shape={stimuli_df.shape}")


# ------------------------------------------------------------
# Assignment runtime (dialog_id with remaining_slots)

def write_assignment_status(assign_df: pd.DataFrame):
    global _last_assign_upload_ts, _assign_commit_counter

    df = assign_df.copy()
    df["dialog_id"] = df["dialog_id"].astype(str).str.strip()

    if "initial_slots" not in df.columns:
        df["initial_slots"] = df.get("remaining_slots", 0)

    df["initial_slots"] = pd.to_numeric(df["initial_slots"], errors="coerce").fillna(0).astype(int)
    df["remaining_slots"] = pd.to_numeric(df["remaining_slots"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    df["used_slots"] = (df["initial_slots"] - df["remaining_slots"]).clip(lower=0).astype(int)

    # Always update locally
    df.to_csv(ASSIGN_STATUS_PATH, index=False, encoding="utf-8")

    # Throttled HF upload
    _assign_commit_counter += 1
    now = time.time()

    should_upload = False

    if _assign_commit_counter >= ASSIGN_UPLOAD_EVERY_N:
        should_upload = True
    elif (now - _last_assign_upload_ts) >= ASSIGN_UPLOAD_MIN_INTERVAL:
        should_upload = True

    if should_upload:
        try:
            upload_text_to_dataset(
                df.to_csv(index=False, encoding="utf-8"),
                f"{DATASET_RATINGS_DIR}/assignment_status.csv",
                commit_message="Update StudyB assignment status (throttled)",
            )
            _last_assign_upload_ts = now
            _assign_commit_counter = 0
        except Exception:
            logger.exception("[ASSIGN_STATUS] upload failed")


def load_or_init_assignment_runtime() -> pd.DataFrame:
    if RUNTIME_ASSIGN_PATH.exists():
        df = pd.read_csv(RUNTIME_ASSIGN_PATH)
        if "dialog_id" not in df.columns:
            raise ValueError("Runtime assignment missing dialog_id.")
        df["dialog_id"] = df["dialog_id"].astype(str).str.strip()

        if "remaining_slots" not in df.columns:
            df["remaining_slots"] = 1
        df["remaining_slots"] = pd.to_numeric(df["remaining_slots"], errors="coerce").fillna(0).clip(lower=0).astype(int)

        if "initial_slots" not in df.columns:
            df["initial_slots"] = df["remaining_slots"].copy().astype(int)
            df.to_csv(RUNTIME_ASSIGN_PATH, index=False, encoding="utf-8")

        write_assignment_status(df)
        return df

    seed = hf_download_csv(ASSIGN_FILE)
    needed = ["dialog_id", "remaining_slots"]
    missing = [c for c in needed if c not in seed.columns]
    if missing:
        raise ValueError(f"Assignment seed missing columns: {missing}. Present={list(seed.columns)}")

    seed["dialog_id"] = seed["dialog_id"].astype(str).str.strip()
    seed["remaining_slots"] = pd.to_numeric(seed["remaining_slots"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    if "initial_slots" not in seed.columns:
        seed["initial_slots"] = seed["remaining_slots"].copy().astype(int)

    seed.to_csv(RUNTIME_ASSIGN_PATH, index=False, encoding="utf-8")
    write_assignment_status(seed)
    logger.info(f"[ASSIGN] initialized runtime: {RUNTIME_ASSIGN_PATH}")
    return seed

def save_assignment_runtime(assign_df: pd.DataFrame):
    assign_df.to_csv(RUNTIME_ASSIGN_PATH, index=False, encoding="utf-8")


# ------------------------------------------------------------
# Reservations

def _init_reservations_if_missing():
    if not RESERVATIONS_PATH.exists():
        df = pd.DataFrame(columns=[
            "reservation_token",
            "rater_id",
            "dialog_id",
            "recipe_title",
            "condition",
            "reserved_at",
            "expires_at",
            "status",      # reserved | committed | expired | cancelled
        ])
        df.to_csv(RESERVATIONS_PATH, index=False, encoding="utf-8")

def _load_reservations() -> pd.DataFrame:
    _init_reservations_if_missing()
    df = pd.read_csv(RESERVATIONS_PATH)
    if df.empty:
        return df
    for col in ["reservation_token", "rater_id", "dialog_id", "recipe_title", "condition", "status"]:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("").str.strip()
    for col in ["reserved_at", "expires_at"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _save_reservations(df: pd.DataFrame):
    df.to_csv(RESERVATIONS_PATH, index=False, encoding="utf-8")

def _expire_old_reservations(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    t = now_ts()
    mask_active = (df["status"] == "reserved") & (pd.to_numeric(df["expires_at"], errors="coerce") <= t)
    if mask_active.any():
        df.loc[mask_active, "status"] = "expired"
    return df

def _active_reservation_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    t = now_ts()
    active = df[(df["status"] == "reserved") & (pd.to_numeric(df["expires_at"], errors="coerce") > t)]
    if active.empty:
        return pd.Series(dtype=int)
    return active.groupby("dialog_id").size()

def _make_reservation_token(rater_id: str, dialog_id: str, salt: str = "") -> str:
    s = f"{rater_id}::{dialog_id}::{now_ts()}::{random.random()}::{salt}".encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:24]

def _extend_reservation_ttl(res_df: pd.DataFrame, rater_id: str, token: str) -> pd.DataFrame:
    """
    Mini-safety: if user navigates back/reloads within session, we reuse same token and extend TTL.
    """
    rid = safe_text(rater_id)
    tok = safe_text(token)
    if res_df.empty or not (rid and tok):
        return res_df
    t = now_ts()
    mask = (
        (res_df["status"] == "reserved") &
        (res_df["rater_id"] == rid) &
        (res_df["reservation_token"] == tok) &
        (pd.to_numeric(res_df["expires_at"], errors="coerce") > t)
    )
    if mask.any():
        res_df.loc[mask, "expires_at"] = t + float(RESERVATION_TTL_SEC)
        res_df.loc[mask, "reserved_at"] = t
    return res_df


# ------------------------------------------------------------
# Recipe-blocked quota assignment with preference scoring

def _build_candidates_with_effective_slots(assign_df: pd.DataFrame, reservations_df: pd.DataFrame) -> pd.DataFrame:
    load_stimuli_or_raise()

    active_counts = _active_reservation_counts(reservations_df)
    a = assign_df.copy()
    a["dialog_id"] = a["dialog_id"].astype(str).str.strip()
    a["remaining_slots"] = pd.to_numeric(a["remaining_slots"], errors="coerce").fillna(0).clip(lower=0).astype(int)

    a["active_reserved"] = a["dialog_id"].map(active_counts).fillna(0).astype(int)
    a["effective_slots"] = (a["remaining_slots"] - a["active_reserved"]).clip(lower=0).astype(int)

    cand = stimuli_df.merge(
        a[["dialog_id", "remaining_slots", "effective_slots"]],
        on="dialog_id",
        how="inner",
    )
    return cand

def pick_dialog_recipe_blocked(
    rater_id: str,
    preferences: Optional[List[str]],
    assign_df: pd.DataFrame,
    reservations_df: pd.DataFrame
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    rng = rng_for_rater(rater_id)

    candidates = _build_candidates_with_effective_slots(assign_df, reservations_df)
    candidates = candidates[candidates["effective_slots"] > 0].copy()
    if candidates.empty:
        raise ValueError("No remaining slots available (including active reservations).")

    g = candidates.groupby(["recipe_title", "condition"], as_index=False)["effective_slots"].sum()
    pivot = g.pivot_table(index="recipe_title", columns="condition", values="effective_slots", fill_value=0)
    pivot["total_slots"] = pivot.sum(axis=1)

    if ("good" in pivot.columns) and ("bad" in pivot.columns):
        both = pivot[(pivot["good"] > 0) & (pivot["bad"] > 0)].copy()
    else:
        both = pivot[pivot["total_slots"] > 0].copy()

    recipe_pool = both if not both.empty else pivot[pivot["total_slots"] > 0].copy()
    if recipe_pool.empty:
        raise ValueError("No recipe-level slots available.")

    prefs = [safe_text(p).lower() for p in (preferences or []) if safe_text(p)]
    if prefs:
        rep = candidates.sort_values(["recipe_title", "condition"]).drop_duplicates("recipe_title")
        rep = rep.set_index("recipe_title")
        scores = []
        for rt in recipe_pool.index:
            if rt in rep.index:
                sc, _ = prefs_match_score(rep.loc[rt].to_dict(), prefs)
            else:
                sc = 0
            scores.append(sc)
        recipe_pool = recipe_pool.copy()
        recipe_pool["pref_score"] = scores
        max_sc = int(recipe_pool["pref_score"].max()) if len(recipe_pool) else 0
        if max_sc > 0:
            recipe_pool = recipe_pool[recipe_pool["pref_score"] == max_sc]

    recipe_titles = recipe_pool.index.tolist()
    weights = recipe_pool["total_slots"].astype(float).tolist()
    chosen_recipe = rng.choices(recipe_titles, weights=weights, k=1)[0]

    sub = candidates[candidates["recipe_title"] == chosen_recipe].copy()
    by_cond = sub.groupby("condition")["effective_slots"].sum().to_dict()
    conds = [c for c in ["good", "bad"] if by_cond.get(c, 0) > 0]
    if not conds:
        conds = sub["condition"].unique().tolist()

    cond_weights = [float(by_cond.get(c, 1)) for c in conds]
    chosen_cond = rng.choices(conds, weights=cond_weights, k=1)[0]

    final = sub[sub["condition"] == chosen_cond].copy()
    if final.empty:
        final = sub.copy()

    if len(final) > 1:
        w = final["effective_slots"].astype(float).tolist()
        chosen = final.sample(n=1, random_state=rng.randint(0, 1_000_000), weights=w).iloc[0]
    else:
        chosen = final.iloc[0]

    stim = chosen.to_dict()
    sc, sat = prefs_match_score(stim, prefs)

    meta = {
        "recipe_blocked": 1,
        "match_score": int(sc),
        "prefs_satisfied": sat,
        "fallback_used": 0,
    }
    return stim, meta


def reserve_dialog_for_rater(rater_id: str, preferences: Optional[List[str]] = None) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
    """
    Atomically creates a reservation row for a selected dialog_id.
    Does not decrement remaining_slots.
    Reuses an existing active reservation for the same rater_id (server-side reload safety).
    """
    rid = safe_text(rater_id)

    with file_lock(LOCK_PATH):
        load_stimuli_or_raise()
        assign_df = load_or_init_assignment_runtime()

        res_df = _load_reservations()
        res_df = _expire_old_reservations(res_df)

        # Reuse active reservation for same rater_id
        t = now_ts()
        active_existing = res_df[
            (res_df["status"] == "reserved") &
            (res_df["rater_id"] == rid) &
            (pd.to_numeric(res_df["expires_at"], errors="coerce") > t)
        ]
        if not active_existing.empty:
            row = active_existing.sort_values("reserved_at", ascending=False).iloc[0]
            did = safe_text(row["dialog_id"])
            stim_row = stimuli_df[stimuli_df["dialog_id"] == did]
            if stim_row.empty:
                res_df.loc[active_existing.index, "status"] = "cancelled"
            else:
                stim = stim_row.iloc[0].to_dict()
                sc, sat = prefs_match_score(stim, preferences or [])
                meta = {"recipe_blocked": 1, "match_score": int(sc), "prefs_satisfied": sat, "fallback_used": 0}
                # extend TTL
                res_df = _extend_reservation_ttl(res_df, rid, safe_text(row["reservation_token"]))
                _save_reservations(res_df)
                return stim, meta, safe_text(row["reservation_token"])

        stim, meta = pick_dialog_recipe_blocked(rid, preferences, assign_df, res_df)
        dialog_id = safe_text(stim.get("dialog_id"))
        recipe_title = safe_text(stim.get("recipe_title"))
        condition = safe_text(stim.get("condition")).lower()

        token = _make_reservation_token(rid, dialog_id, salt="reserve")

        reserved_at = now_ts()
        expires_at = reserved_at + float(RESERVATION_TTL_SEC)

        new_row = {
            "reservation_token": token,
            "rater_id": rid,
            "dialog_id": dialog_id,
            "recipe_title": recipe_title,
            "condition": condition,
            "reserved_at": reserved_at,
            "expires_at": expires_at,
            "status": "reserved",
        }
        res_df = pd.concat([res_df, pd.DataFrame([new_row])], ignore_index=True)
        _save_reservations(res_df)

        return stim, meta, token


def fetch_reserved_stim_by_token(rater_id: str, reservation_token: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Session mini-safety: validate token, return corresponding stimulus and lightweight meta.
    Extends TTL if still active.
    """
    rid = safe_text(rater_id)
    tok = safe_text(reservation_token)
    if not (rid and tok):
        return None, None

    with file_lock(LOCK_PATH):
        load_stimuli_or_raise()
        res_df = _load_reservations()
        res_df = _expire_old_reservations(res_df)

        t = now_ts()
        mask = (
            (res_df["status"] == "reserved") &
            (res_df["rater_id"] == rid) &
            (res_df["reservation_token"] == tok) &
            (pd.to_numeric(res_df["expires_at"], errors="coerce") > t)
        )
        if not mask.any():
            _save_reservations(res_df)
            return None, None

        row = res_df[mask].sort_values("reserved_at", ascending=False).iloc[0]
        did = safe_text(row["dialog_id"])
        stim_row = stimuli_df[stimuli_df["dialog_id"] == did]
        if stim_row.empty:
            res_df.loc[mask, "status"] = "cancelled"
            _save_reservations(res_df)
            return None, None

        # extend TTL
        res_df = _extend_reservation_ttl(res_df, rid, tok)
        _save_reservations(res_df)

        stim = stim_row.iloc[0].to_dict()
        meta = {"recipe_blocked": 1, "match_score": 0, "prefs_satisfied": [], "fallback_used": 0}
        return stim, meta


def commit_reservation_and_decrement(rater_id: str, reservation_token: str, dialog_id: str) -> Tuple[bool, str]:
    """
    Atomically:
      - verifies reservation is active for (rater_id, token, dialog_id)
      - decrements remaining_slots by 1 (if available)
      - marks reservation committed
    """
    rid = safe_text(rater_id)
    tok = safe_text(reservation_token)
    did = safe_text(dialog_id)

    if not (rid and tok and did):
        return False, "Missing reservation info."

    with file_lock(LOCK_PATH):
        assign_df = load_or_init_assignment_runtime()
        res_df = _load_reservations()
        res_df = _expire_old_reservations(res_df)

        t = now_ts()
        mask = (
            (res_df["status"] == "reserved") &
            (res_df["rater_id"] == rid) &
            (res_df["reservation_token"] == tok) &
            (res_df["dialog_id"] == did) &
            (pd.to_numeric(res_df["expires_at"], errors="coerce") > t)
        )
        if not mask.any():
            return False, "Reservation not found or expired (please restart)."

        assign_df["dialog_id"] = assign_df["dialog_id"].astype(str).str.strip()
        m2 = (assign_df["dialog_id"] == did)
        if not m2.any():
            return False, "Dialog not found in assignment runtime."

        cur = int(pd.to_numeric(assign_df.loc[m2, "remaining_slots"], errors="coerce").fillna(0).iloc[0])
        if cur <= 0:
            return False, "No remaining slots for this dialog (race condition)."

        assign_df.loc[m2, "remaining_slots"] = max(cur - 1, 0)
        save_assignment_runtime(assign_df)
        write_assignment_status(assign_df)

        res_df.loc[mask, "status"] = "committed"
        _save_reservations(res_df)

        return True, "Committed."


# ------------------------------------------------------------
# Dialog rendering (bubbles)

def parse_dialog_turns(raw: str):
    if not isinstance(raw, str):
        return []
    text = re.sub(r"\s*(Fat:)", r"\n\1", raw)
    text = re.sub(r"\s*(Carb:)", r"\n\1", text)
    text = text.strip()
    lines = [line.strip() for line in text.split("\n") if line.strip()]
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


# ------------------------------------------------------------
# Measures (Study B)

LIKERT_1_7 = [1, 2, 3, 4, 5, 6, 7]
STARS_1_5 = [1, 2, 3, 4, 5]

DIALOG_QUALITY_ITEMS = [
    ("dq_clarity", "The dialog was clear and easy to follow."),
    ("dq_relevance", "The dialog focused on relevant aspects of the recipe."),
    ("dq_logic", "The dialog was logically structured and coherent."),
    ("dq_respect", "The discussants treated each other respectfully."),
    ("dq_coherence", "The discussants responded to each other in a meaningful way."),
]
MANIP_ITEM = ("manip_check", "How good was the communication overall? (1 = very poor, 7 = excellent)")


# ------------------------------------------------------------
# Persist one-row-per-participant

def persist_participant_row(rater_id: str, row: Dict[str, Any]):
    out_local = RESULTS_DIR / f"ratings_{rater_id}.csv"
    df = pd.DataFrame([row])
    write_header = not out_local.exists()
    df.to_csv(out_local, mode="a", index=False, header=write_header, encoding="utf-8")

    try:
        append_row_to_rater_csv_in_dataset(rater_id, row)
    except Exception:
        logger.exception("UPLOAD failed to append StudyB row to dataset")


# ------------------------------------------------------------
# Page flow callbacks

def toggle_welcome_start_button(english_ok: bool, imc_answer: str):
    enabled = bool(english_ok and imc_answer == "Strongly agree")
    return gr.update(interactive=enabled)

def toggle_consent_continue_button(consent: bool):
    return gr.update(interactive=bool(consent))

def toggle_continue_button(confirmed: bool):
    return gr.update(interactive=bool(confirmed))


def handle_welcome(rater_id_input, english_ok, imc_answer):
    load_stimuli_or_raise()
    imc_pass = int(imc_answer == "Strongly agree")

    if not (english_ok and imc_answer == "Strongly agree"):
        msg = "Please confirm your English skills and choose 'Strongly agree', to show that you read carefully."
        return (
            msg,
            None,
            gr.update(visible=True),
            gr.update(visible=False),
            imc_pass,
        )

    rater_id = safe_text(rater_id_input) or f"R{random.randint(10000, 99999)}"
    msg = f"Participant ID: **{rater_id}**. Please review the consent information on the next page."
    return (
        msg,
        rater_id,
        gr.update(visible=False),
        gr.update(visible=True),
        imc_pass,
    )


def handle_consent_continue(consent_checked: bool, pending_rater_id: str):
    """
    Consent -> Instructions
    No reservation here (reservation happens after preferences/screening).
    """
    if not consent_checked:
        return (
            "Please check the consent box to continue.",
            None,
            gr.update(visible=True),
            gr.update(visible=False),
        )

    rater_id = safe_text(pending_rater_id) or f"R{random.randint(10000, 99999)}"
    return (
        "",
        rater_id,
        gr.update(visible=False),
        gr.update(visible=True),
    )

def handle_instructions_continue(rater_id: str):
    """
    Instructions -> Screening
    """
    rid = safe_text(rater_id) or f"R{random.randint(10000, 99999)}"
    return (
        "",
        rid,
        gr.update(visible=False),
        gr.update(visible=True),
    )


# ---- Screening / assignment ----
PREFERENCE_OPTIONS = ["lower-fat", "lower-carb", "lower-calorie", "sweet", "savory"]

def _prefs_to_state(x):
    return x if isinstance(x, list) else []


def build_recipe_card_html(stim: Dict[str, Any]) -> Tuple[str, str, str]:
    title = safe_text(stim.get("recipe_title"))
    rtype = safe_text(stim.get("recipe_type")).lower()

    desc = safe_text(stim.get("description"))
    ingred = safe_text(stim.get("ingredients_list"))
    servings = _to_int_safe(stim.get("servings"))
    directions = safe_text(stim.get("directions"))

    def _html_escape(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        return (
            s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&#39;")
        )

    def _clean_quoted_text(s: str) -> str:
        s = safe_text(s)
        if len(s) >= 2:
            if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
                return s[1:-1].strip()
        return s

    def _render_multiline_text(raw: str) -> str:
        s = safe_text(raw)
        if not s:
            return ""
        s = re.sub(r"<\s*br\s*/?\s*>", "\n", s, flags=re.IGNORECASE)
        parts = [p.strip() for p in s.split("\n") if p.strip()]
        if not parts:
            return ""
        return "".join(f"<p class='recipe-text'>{_html_escape(p)}</p>" for p in parts)

    def _ingredients_to_chips(raw: str) -> str:
        s = safe_text(raw)
        if not s:
            return ""
        cleaned = s.strip()
        items: List[str] = []
        if cleaned.startswith("[") and cleaned.endswith("]"):
            inner = cleaned[1:-1].strip()
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            for p in parts:
                p2 = p.strip().strip('"').strip("'").strip()
                if p2:
                    items.append(p2)
        else:
            parts = [p.strip() for p in cleaned.split(",") if p.strip()]
            items = parts if len(parts) >= 2 else [cleaned]
        items = items[:60]
        chips = [f"<span class='ing-chip'>{_html_escape(it)}</span>" for it in items]
        return "<div class='ing-chips'>" + "".join(chips) + "</div>"

    def _boxed_section(title_: str, inner_html: str) -> str:
        if not inner_html:
            return ""
        return (
            "<div class='recipe-box'>"
            f"<div class='recipe-box-title'>{_html_escape(title_)}</div>"
            f"{inner_html}"
            "</div>"
        )

    badges = []
    if rtype:
        badges.append(f"<span class='recipe-badge'>{_html_escape(rtype)}</span>")
    if servings is not None:
        badges.append(f"<span class='recipe-badge'>{_html_escape(str(servings))} servings</span>")
    badge_html = "".join(badges)

    desc_box_html = _boxed_section("Description", _render_multiline_text(_clean_quoted_text(desc))) if desc else ""
    ingred_box_html = _boxed_section("Ingredients", _ingredients_to_chips(ingred)) if ingred else ""
    directions_box_html = _boxed_section("Directions", _render_multiline_text(directions)) if directions else ""

    recipe_md = (
        "<div class='recipe-card'>"
        "<div class='recipe-header'>"
        f"<div class='recipe-title'>{_html_escape(title) if title else 'Recipe'}</div>"
        f"{badge_html}"
        "</div>"
        "<div class='recipe-grid-boxes-2'>"
        f"{desc_box_html}"
        f"{ingred_box_html}"
        "</div>"
        "<div class='recipe-boxes-1'>"
        f"{directions_box_html}"
        "</div>"
        "</div>"
    )

    dialog_raw = safe_text(stim.get("dialog_text"))
    dialog_html = dialog_to_bubbles(dialog_raw)
    cond = safe_text(stim.get("condition")).lower()

    return recipe_md, dialog_html, cond


def start_screening_and_reserve_then_prepare(
    rater_id: str,
    reservation_token: str,
    assigned_stim: dict,
    pref_state,
    involvement,
    nutrition_knowledge,
    english_skill,
    imc_pass: int,
):
    rid = safe_text(rater_id) or f"R{random.randint(10000, 99999)}"

    prefs = pref_state if isinstance(pref_state, list) else []
    prefs = [safe_text(p).lower() for p in prefs if safe_text(p)]

    inv = parse_likert(involvement)
    nk  = parse_likert(nutrition_knowledge)
    es  = parse_likert(english_skill)

    if inv is None or nk is None or es is None:
        return (
            "Please answer all three questions (involvement, knowledge, English skill).",
            None, None, reservation_token, None, prefs,
            None, None, None,
            "", "",
            gr.update(visible=True),
            gr.update(visible=False),
        )

    logger.info(f"[SCREENING] rid={rid} inv={inv} nk={nk} es={es} prefs={prefs}")

    tok = safe_text(reservation_token)
    stim = None
    meta = None

    if tok:
        stim2, meta2 = fetch_reserved_stim_by_token(rid, tok)
        if stim2:
            stim = stim2
            meta = meta2 or {}
            sc, sat = prefs_match_score(stim, prefs)
            meta.update({"match_score": int(sc), "prefs_satisfied": sat})

    if stim is None:
        try:
            stim, meta, tok = reserve_dialog_for_rater(rid, preferences=prefs)
            status = ""
        except Exception as e:
            logger.exception("RESERVE failed")
            return (
                f"Sorry, no stimuli available at the moment. ({e})",
                None, None, None, None, None, None, None, None,
                gr.update(visible=True),
                gr.update(visible=False),
            )
    else:
        status = ""

    recipe_md, dialog_html, cond = build_recipe_card_html(stim)
    title = safe_text(stim.get("recipe_title")) or "Recipe"
    dialog_title = f"## {title}"

    return (
        status,
        stim,
        meta,
        tok,
        cond,
        prefs,
        inv,
        nk,
        es,
        recipe_md,
        dialog_title,
        dialog_html,
        gr.update(visible=False),
        gr.update(visible=True),
    )


# ---- Pre -> Dialog ----
def go_to_dialog_page(
    pre_diet_suitability,
    pre_stars,
    pre_intent,
    pre_save_intent,
    pre_est_fat,
    pre_est_carb,
    pre_est_kcal,
    pre_comment,
    assigned_stim: dict,
):
    if not assigned_stim:
        return (
            "No assigned recipe/dialog found. Please restart.",
            gr.update(visible=True),
            gr.update(visible=False),
            None,
            {},
        )

    pre = {
        "pre_diet_suitability": parse_likert(pre_diet_suitability),
        "pre_recipe_stars": _to_int_safe(pre_stars),
        "pre_cook_intent": parse_likert(pre_intent),
        "pre_save_intent": parse_likert(pre_save_intent),

        "pre_est_fat_g": _to_float_safe(pre_est_fat),
        "pre_est_carb_g": _to_float_safe(pre_est_carb),
        "pre_est_kcal": _to_float_safe(pre_est_kcal),

        "pre_comment": safe_text(pre_comment),
    }

    return (
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        time.time(),
        pre,
    )


# ---- Dialog -> Post ----
def go_to_post_page(dialog_start_time: float, assigned_stim: dict, recipe_md_pre: str):
    dur = (time.time() - dialog_start_time) if dialog_start_time else None
    min_time_ok = int(dur >= MIN_DIALOG_TIME) if dur is not None else None

    # Rebuild recipe card from assigned stimulus (safe if recipe_md_pre is empty)
    recipe_md_built, _, _ = build_recipe_card_html(assigned_stim or {})

    # Prefer the already rendered recipe from the PRE page (avoids any drift),
    # but fallback to rebuilt HTML if needed.
    recipe_for_post = recipe_md_pre or recipe_md_built

    return (
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        dur,
        min_time_ok,
        recipe_for_post,
    )


# ---- Final submit (commit + persist) ----
def submit_post_and_finish(
    post_diet_suitability,
    post_stars,
    post_intent,
    post_save_intent,
    post_est_fat,
    post_est_carb,
    post_est_kcal,

    dq_clarity,
    dq_relevance,
    dq_respect,
    dq_logic,
    dq_coherence,
    manip_check,

    post_comment,

    # states
    rater_id,
    imc_pass,
    reservation_token: str,
    assigned_stim: dict,
    assign_meta: dict,
    condition: str,
    preferences: list,
    involvement: int,
    nutrition_knowledge: int,
    english_skill: int,
    pre_state: dict,
    dialog_duration_sec: float,
    min_time_ok_dialog: int,
):
    rid = safe_text(rater_id) or "RATER_NONE"
    stim = assigned_stim or {}
    meta = assign_meta or {}
    pre = pre_state or {}
    tok = safe_text(reservation_token)

    dialog_id = safe_text(stim.get("dialog_id"))

    # 1) COMMIT reservation (slot decrement happens here)
    ok, msg = commit_reservation_and_decrement(rid, tok, dialog_id)
    if not ok:
        return (
            f"Submission failed: {msg}\n\nPlease restart the study and try again.",
            gr.update(visible=True),
            gr.update(visible=False),
            "",
        )

    # 2) Compute outcome fields
    true_fat = _to_float_safe(stim.get("fat"))
    true_carb = _to_float_safe(stim.get("carbs"))
    true_kcal = _to_float_safe(stim.get("calories"))

    post_est_f = _to_float_safe(post_est_fat)
    post_est_c = _to_float_safe(post_est_carb)
    post_est_k = _to_float_safe(post_est_kcal)

    abs_err_fat = abs(post_est_f - true_fat) if (true_fat is not None and post_est_f is not None) else None
    abs_err_carb = abs(post_est_c - true_carb) if (true_carb is not None and post_est_c is not None) else None
    abs_err_kcal = abs(post_est_k - true_kcal) if (true_kcal is not None and post_est_k is not None) else None

    nutrient_outlier_flag = 0
    for v in [
        pre.get("pre_est_fat_g"), pre.get("pre_est_carb_g"), pre.get("pre_est_kcal"),
        post_est_f, post_est_c, post_est_k
    ]:
        if v is not None and v > 1000:
            nutrient_outlier_flag = 1
    if post_est_k is not None and post_est_k > 10000:
        nutrient_outlier_flag = 1

    dq_vals = list(map(parse_likert, [dq_clarity, dq_relevance, dq_respect, dq_logic, dq_coherence]))
    dq_clean = [v for v in dq_vals if v is not None]
    dq_mean = (sum(dq_clean) / len(dq_clean)) if dq_clean else None

    # 3) Persist row
    row = {
        "timestamp": time.time(),
        "rater_id": rid,
        "imc_pass": int(imc_pass) if imc_pass is not None else None,

        "reservation_token": tok,
        "preferences": "|".join([safe_text(x).lower() for x in (preferences or [])]),
        "involvement": parse_likert(involvement),
        "nutrition_knowledge": parse_likert(nutrition_knowledge),
        "english_skill": parse_likert(english_skill),

        "match_score": _to_int_safe(meta.get("match_score")),
        "prefs_satisfied": "|".join([safe_text(x).lower() for x in (meta.get("prefs_satisfied") or [])]),
        "fallback_used": _to_int_safe(meta.get("fallback_used")),

        "dialog_id": dialog_id,
        "recipe_title": safe_text(stim.get("recipe_title")),
        "recipe_type": safe_text(stim.get("recipe_type")),
        "condition": safe_text(condition),

        "true_fat_g": true_fat,
        "true_carb_g": true_carb,
        "true_kcal": true_kcal,

        **pre,

        "dialog_duration_sec": dialog_duration_sec,
        "min_time_ok_dialog": min_time_ok_dialog,

        "post_diet_suitability": parse_likert(post_diet_suitability),
        "post_recipe_stars": _to_int_safe(post_stars),
        "post_cook_intent": parse_likert(post_intent),
        "post_save_intent": parse_likert(post_save_intent),

        "post_est_fat_g": post_est_f,
        "post_est_carb_g": post_est_c,
        "post_est_kcal": post_est_k,

        "abs_err_fat_post": abs_err_fat,
        "abs_err_carb_post": abs_err_carb,
        "abs_err_kcal_post": abs_err_kcal,
        "nutrient_outlier_flag": nutrient_outlier_flag,

        "dq_clarity": parse_likert(dq_clarity),
        "dq_relevance": parse_likert(dq_relevance),
        "dq_respect": parse_likert(dq_respect),
        "dq_logic": parse_likert(dq_logic),
        "dq_coherence": parse_likert(dq_coherence),
        "dq_mean": dq_mean,
        "manip_check": parse_likert(manip_check),

        "post_comment": safe_text(post_comment),
    }

    persist_participant_row(rid, row)

    # Debriefing: disclose true nutrition facts transparently
    nutrition_debrief_lines = []
    if true_fat is not None:
        nutrition_debrief_lines.append(f"- **Fat:** {true_fat:.1f} g")
    if true_carb is not None:
        nutrition_debrief_lines.append(f"- **Carbohydrates:** {true_carb:.1f} g")
    if true_kcal is not None:
        nutrition_debrief_lines.append(f"- **Calories:** {true_kcal:.0f} kcal")

    nutrition_debrief = ""
    if nutrition_debrief_lines:
        nutrition_debrief = (
            "\n\n"
            "### Nutrition facts:\n"
            "For transparency, here are the actual nutrition facts of the recipe you evaluated:\n"
            + "\n".join(nutrition_debrief_lines)
            + "\n\nYou were asked for estimates before and after the dialog to measure changes in understanding."
        )

    end_text = (
        f"Thank you for participating!\n\n"
        f"Your participant code: **{rid}**\n"
        "### Debriefing\n"
        "This study investigates whether the **quality of an expert dialog** (e.g., clear and respectful vs. unclear or less constructive communication) "
        "changes how people evaluate a recipe and how accurately they estimate nutrition facts.\n\n"
        "Some elements of the dialog may have been designed to represent different communication qualities. \n\n"
        f"{nutrition_debrief}\n\n"
        "**You can now close the window.**"
    )


    return (
        "",
        gr.update(visible=False),
        gr.update(visible=True),
        end_text,
    )


# ------------------------------------------------------------
# CSS (responsive)

custom_css = """
html, body { height: auto; }
body { overflow-x: hidden; }
.gradio-container { overflow-x: hidden !important; }

/* Avoid accidental horizontal cropping on mobile */
.gradio-container {
  overflow-x: hidden !important;
}

/* =========================================================
   0) Define a single foreground color token (light/dark)
   ========================================================= */
:root { --kk-fg: #000; }
@media (prefers-color-scheme: dark) { :root { --kk-fg: #fff; } }

/* Also respect Gradio theme toggles (some builds use attributes/classes) */
html[data-theme="dark"], body.dark, .dark { --kk-fg: #fff; }
html[data-theme="light"], body.light, .light { --kk-fg: #000; }

/* =========================================================
   1) Force Gradio theme variables at multiple roots
   (Gradio reads vars from :root/html/body/.gradio-container depending on build)
   ========================================================= */
:root, html, body, .gradio-container {
  --body-text-color: var(--kk-fg) !important;
  --body-text-color-subdued: var(--kk-fg) !important;
  --block-label-text-color: var(--kk-fg) !important;
  --block-title-text-color: var(--kk-fg) !important;
  --block-info-text-color: var(--kk-fg) !important;
  --input-text-color: var(--kk-fg) !important;

  /* extra tokens used across versions */
  --color-text-primary: var(--kk-fg) !important;
  --color-text-secondary: var(--kk-fg) !important;
  --color-text-tertiary: var(--kk-fg) !important;
}

/* =========================================================
   2) Hard fallback: force color + kill "subdued" opacity in LIGHT MODE
   ========================================================= */
.gradio-container, .gradio-container * { color: var(--kk-fg) !important; }

/* Placeholders */
.gradio-container input::placeholder,
.gradio-container textarea::placeholder {
  color: var(--kk-fg) !important;
  opacity: 0.65 !important;
}

@media (prefers-color-scheme: light) {
  .gradio-container .info,
  .gradio-container .hint,
  .gradio-container .secondary,
  .gradio-container .subdued,
  .gradio-container .block-info,
  .gradio-container .block-info-text,
  .gradio-container .block-title,
  .gradio-container .block-label,
  .gradio-container small,
  .gradio-container .prose,
  .gradio-container .prose *,
  .gradio-container .markdown,
  .gradio-container .markdown * {
    opacity: 1 !important;
    filter: none !important;
    color: #000 !important; /* enforce pure black explicitly */
  }
}

/* Also cover theme-class light toggles */
body.light .gradio-container .info,
.light .gradio-container .info,
.gradio-container[data-theme="light"] .info,
body.light .gradio-container .hint,
.light .gradio-container .hint,
.gradio-container[data-theme="light"] .hint,
body.light .gradio-container .secondary,
.light .gradio-container .secondary,
.gradio-container[data-theme="light"] .secondary,
body.light .gradio-container .subdued,
.light .gradio-container .subdued,
.gradio-container[data-theme="light"] .subdued,
body.light .gradio-container .prose,
.light .gradio-container .prose,
.gradio-container[data-theme="light"] .prose,
body.light .gradio-container .prose *,
.light .gradio-container .prose *,
.gradio-container[data-theme="light"] .prose * {
  opacity: 1 !important;
  filter: none !important;
  color: #000 !important;
}

/* Dark mode: keep consistent */
@media (prefers-color-scheme: dark) {
  .gradio-container .info,
  .gradio-container .hint,
  .gradio-container .secondary,
  .gradio-container .subdued,
  .gradio-container .block-info,
  .gradio-container .block-info-text,
  .gradio-container .prose,
  .gradio-container .prose *,
  .gradio-container .markdown,
  .gradio-container .markdown * {
    opacity: 1 !important;
    filter: none !important;
    color: #fff !important;
  }
}

/* =========================================================
   3) layout/styling
   ========================================================= */

#page-top-anchor { scroll-margin-top: 90px; }

/* Dialog bubbles (responsive) */
.dialog-container {
  display: flex;
  flex-direction: column;
  gap: 0.1rem;
  max-width: 48rem;
  margin: 0.25rem auto;
  padding: 0 0.40rem;          
  box-sizing: border-box;
}


.bubble {
  max-width: 100%;           
  width: fit-content;        
  padding: 0.5rem 0.75rem;
  border-radius: 0.9rem;
  font-size: 1rem;
  line-height: 1.4;
  word-wrap: break-word;
  overflow-wrap: break-word;
  box-sizing: border-box;
}

/* "max 34rem" as the upper limit on larger screens */
@media (min-width: 541px) {
  .bubble { max-width: 34rem; }
}

/* Light-mode bubble styling */
.bubble-fat  { align-self:flex-start; background-color:#e3f2fd; border:1px solid #bbdefb; }
.bubble-carb { align-self:flex-end;   background-color:#fff3e0; border:1px solid #ffe0b2; }
.bubble-other{ align-self:center;     background-color:#f5f5f5; border:1px solid #ededed; }

.speaker { font-weight:600; font-size:0.8rem; margin-bottom:0.1rem; opacity:0.85; }
.bubble-text { white-space:normal; }

/* Dark-mode bubble backgrounds */
@media (prefers-color-scheme: dark) {
  .bubble-fat  { background-color: #153951; border: 1px solid #5da5d5; }
  .bubble-carb { background-color: #661a00; border: 1px solid #ff6633; }
  .bubble-other{ background-color: #222;    border: 1px solid #333; }
}

/* Buttons */
button.green-btn, button.orange-btn { border-radius: 0.8rem !important; }
@media (max-width: 540px) {
  button.green-btn, button.orange-btn { width: 100% !important; }
}
button.green-btn { background-color:#4caf50 !important; color:white !important; border-color:#4caf50 !important; }
button.green-btn[disabled] { background-color:#c8e6c9 !important; color:#555 !important; border-color:#c8e6c9 !important; }
button.orange-btn { background-color:#ff9800 !important; color:white !important; border-color:#ff9800 !important; }
button.orange-btn[disabled] { background-color:#ffe0b2 !important; color:#555 !important; border-color:#ffe0b2 !important; }

/* Recipe card */
.recipe-card {
  max-width: 48rem;
  margin: 0.5rem auto 1rem auto;
  padding: 0.9rem 1rem;
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 1rem;
  background: rgba(235,250,225,0.75);
}
@media (max-width: 540px) {
  .recipe-card { padding: 0.8rem 0.85rem; border-radius: 0.9rem; }
}
@media (prefers-color-scheme: dark) {
  .recipe-card { background: #003300; border: 1px solid #009900; }
}
.recipe-header { display:flex; justify-content: space-between; align-items: baseline; gap:0.75rem; margin-bottom:0.35rem; }
.recipe-title { font-size:1.25rem; font-weight:700; margin:0; }
@media (max-width: 540px) { .recipe-title { font-size:1.15rem; } }
.recipe-badge {
  font-size:0.85rem; padding:0.15rem 0.55rem; border-radius:999px;
  border:1px solid rgba(0,0,0,0.15); background:rgba(0,0,0,0.04);
}
@media (prefers-color-scheme: dark) {
  .recipe-badge { border: 1px solid rgba(0,179,0,0.80); background: rgba(0,77,0,0.80); }
}
.recipe-text { margin:0; opacity:0.92; }
.ing-chips { display:flex; flex-wrap:wrap; gap:0.35rem; margin-top:0.35rem; }
.ing-chip {
  font-size:0.85rem; padding:0.12rem 0.5rem; border-radius:999px;
  border:1px solid rgba(0,0,0,0.12); background:rgba(0,0,0,0.03);
}
@media (prefers-color-scheme: dark) {
  .ing-chip { border: 1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.06); }
}

/* Boxes */
.recipe-grid-boxes-2 { display:grid; grid-template-columns: 1fr 1fr; gap:0.9rem; margin-top:0.75rem; }
@media (max-width: 720px) { .recipe-grid-boxes-2 { grid-template-columns: 1fr; } }
.recipe-boxes-1 { margin-top:0.9rem; }
.recipe-box {
  border:1px solid rgba(0,0,0,0.12);
  border-radius:0.9rem;
  padding:0.75rem 0.85rem;
  background:rgba(0,0,0,0.02);
}
@media (prefers-color-scheme: dark) {
  .recipe-box { border: 1px solid rgba(255,255,255,0.18); background: rgba(255,255,255,0.06); }
}
.recipe-box-title { font-weight:700; margin:0 0 0.35rem 0; opacity:0.95; }
"""

# ------------------------------------------------------------
# Build UI

load_stimuli_or_raise()
with file_lock(LOCK_PATH):
    load_or_init_assignment_runtime()
    _init_reservations_if_missing()

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("## Recipe Evaluation Study", elem_id="page-top-anchor")

    # States
    pending_rater_id_state = gr.State(value=None)
    rater_id_state = gr.State(value=None)
    imc_pass_state = gr.State(value=None)

    reservation_token_state = gr.State(value=None)

    pref_input_state = gr.State(value=[])

    preferences_state = gr.State(value=[])
    involvement_state = gr.State(value=None)
    nutrition_knowledge_state = gr.State(value=None)
    english_skill_state = gr.State(value=None)
    # backend-safe mirrors for Radio values (fixes queue/SSR None on click)
    involvement_input_state = gr.State(value=None)
    nutrition_knowledge_input_state = gr.State(value=None)
    english_skill_input_state = gr.State(value=None)

    assigned_stim_state = gr.State(value=None)
    assign_meta_state = gr.State(value=None)
    condition_state = gr.State(value=None)
    pre_state = gr.State(value={})

    dialog_start_time_state = gr.State(value=None)
    dialog_duration_state = gr.State(value=None)
    min_time_ok_state = gr.State(value=None)

    # -------------------------------
    # WELCOME
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

        english_ok_cb.change(toggle_welcome_start_button, inputs=[english_ok_cb, imc_radio], outputs=start_btn)
        imc_radio.change(toggle_welcome_start_button, inputs=[english_ok_cb, imc_radio], outputs=start_btn)

    # -------------------------------
    # CONSENT
    with gr.Group(visible=False) as consent_group:
        gr.Markdown("## Informed Consent")
        consent_text_md = gr.Markdown(
        """
        Please read the information below carefully.
        
        **Purpose of the study:**  
        This study examines how people evaluate recipes and how an dialog may influence these evaluations.
        
        **What you will do:**  
        You will:
        1. Answer few questions about your dietary preferences and experience in nutrition
        2. View one recipe and answer a few questions
        3. Read one short dialog about the recipe
        4. Answer similar questions again
        
        **Study duration:**  
        Participation typically takes about **5–15 minutes** (may vary).
        
        **Voluntary participation and withdrawal:**  
        Your participation is voluntary. You may stop at any time by closing the browser window. There are no disadvantages for non-participation.
        
        **Risks and benefits:**  
        There are no known risks beyond those of everyday online activities. You may not benefit directly, but your participation helps research on human judgement and communication.
        
        **Data protection and confidentiality:**  
        No directly identifying information is collected. Your responses are stored in pseudonymized form (participant code). Data are used for scientific purposes only and may be reported in aggregated form.
        
         
        If you have questions about the study, please contact the researcher (E-Mail: Katharina.Kampa@stud.uni-regensburg.de).
        
        By checking the box below, you confirm that you are at least 18 years old, have read and understood the information above, and agree to participate.
        Next you will see the instructions.
        """
        )
        consent_cb = gr.Checkbox(label="I have read the information above and I agree to participate.", value=False)
        consent_continue_btn = gr.Button("Continue", interactive=False, elem_classes=["green-btn"])
        consent_status_md = gr.Markdown("")

        consent_cb.change(toggle_consent_continue_button, inputs=[consent_cb], outputs=consent_continue_btn)

    # -------------------------------
    # INSTRUCTIONS
    with gr.Group(visible=False) as instructions_group:
        gr.Markdown("## Instructions")
        gr.Markdown(
        """
        Please read the instructions carefully.
        
        ### What you will do
        1. **Answer a few short questions** about your dietary preferences and your experience with nutrition.
        2. **Read a recipe** and answer a few questions about your first impression.
        3. **Read a short dialog** about the recipe.
        4. **Answer the same questions again** after reading the dialog.
        5. Finally, you will answer **a few questions about the dialog’s communication quality**.
        
        ### Important notes
        - Please complete the study **in one sitting**.
        - There are **no right or wrong answers**. I am interested in your personal evaluation.
        - Please **read carefully** and do not rush.
        - **Note for mobile participation (smartphone, tablet, etc.)**: The user interface may start in the centre on some pages. If this is the case, please scroll to the top of the page and start from there.
        
        Click **Continue** to start with the preference and experience questions.
        """
        )
        instructions_continue_btn = gr.Button("Continue", elem_classes=["green-btn"])
        instructions_status_md = gr.Markdown("")

    
    # -------------------------------
    # SCREENING / PREFERENCES
    with gr.Group(visible=False) as screening_group:
        gr.Markdown("## Preferences and nutrition-related questions \n")
        gr.Markdown("\n\n Please **select your dietary preferences** and **answer the questions**. We will try to assign you a recipe that matches your preferences.\n\n")

        pref_choices = gr.CheckboxGroup(choices=PREFERENCE_OPTIONS, label="Select your dietary preferences - select all that apply", value=[])
        pref_choices.change(_prefs_to_state, inputs=[pref_choices], outputs=[pref_input_state], queue=False)

        involvement = gr.Radio(
            LIKERT_1_7,
            label="How would you rate your nutrition involvement? (1 = very low, 7 = excellent)",
            info="**Note:** *Nutrition involvement* means how important nutrition is to you and how often you pay attention to it in daily life."
        )

        nutrition_knowledge = gr.Radio(
            LIKERT_1_7,
            label="How would you rate your nutrition knowledge? (1 = very low, 7 = excellent)",
            info="**Note:** *Nutrition knowledge* means how well-informed you feel about nutrition topics."
        )

        english_skill = gr.Radio(
            LIKERT_1_7,
            label="How would you rate your English reading proficiency? (1 = very low, 7 = excellent)",
            info="**Note:** *English reading proficiency* means how well you can read and understand written English."
        )

        # Mirror radio values into backend-safe states (queue=False is important)
        involvement.change(lambda x: x, inputs=[involvement], outputs=[involvement_input_state], queue=False)
        nutrition_knowledge.change(lambda x: x, inputs=[nutrition_knowledge], outputs=[nutrition_knowledge_input_state], queue=False)
        english_skill.change(lambda x: x, inputs=[english_skill], outputs=[english_skill_input_state], queue=False)

        gr.Markdown("Next you will see the recipe and answer some questions.")
        screening_confirm_cb = gr.Checkbox(label="I confirm my answers and want to continue.", value=False)
        screening_continue_btn = gr.Button("Continue to recipe", interactive=False, elem_classes=["green-btn"])
        screening_status_md = gr.Markdown("")

        screening_confirm_cb.change(toggle_continue_button, inputs=[screening_confirm_cb], outputs=screening_continue_btn)

    # -------------------------------
    # RECIPE (PRE)
    with gr.Group(visible=False) as recipe_group:
        gr.Markdown("### Recipe Evaluation (Part 1)")
        gr.Markdown("Please **read the following recipe** and review the recipe information carefully. \n")
        recipe_md = gr.Markdown("")
        gr.Markdown("\n Please **answer the questions below** based on your current impression.")

        pre_diet = gr.Radio(LIKERT_1_7, label="How suitable is this recipe for your diet? (1 = very low, 7 = excellent)")
        pre_stars = gr.Radio(STARS_1_5, label="How would you rate this recipe? (1–5 stars)")
        pre_intent = gr.Radio(LIKERT_1_7, label="How likely would you cook this recipe? (1 = very low, 7 = excellent)")
        pre_save = gr.Radio(LIKERT_1_7, label="How likely would you save this recipe? (1 = very low, 7 = excellent)")

        with gr.Row():
            pre_est_fat = gr.Number(label="Please try to estimate the fat content in grams (g) per serving.", precision=2, 
                                   info="**Note:** You find the total number of servings on the recipe card. Please estimate for **one** serving.")
            pre_est_carb = gr.Number(label="Please try to estimate the carbohydrates content in grams (g) per serving", precision=2, 
                                    info="**Note:** You find the total number of servings on the recipe card. Please estimate for **one** serving.")
        pre_est_kcal = gr.Number(label="Please try to estimate the calories (kcal) per serving", precision=0, 
                                info="**Note:** You find the total number of servings on the recipe card. Please estimate for **one** serving.")

        pre_comment = gr.Textbox(label="Comment (optional)", lines=2)

        gr.Markdown("Next you will read a dialog about the recipe.")
        to_dialog_btn = gr.Button("Continue to dialog", elem_classes=["orange-btn"])
        recipe_status_md = gr.Markdown("")

    # -------------------------------
    # DIALOG
    with gr.Group(visible=False) as dialog_group:
        dialog_title_md = gr.Markdown("## Recipe Dialog")
        gr.Markdown("\n Please **read the dialog** carefully. Afterwards, you will answer the next set of questions. Do not rush. \n")
        dialog_md = gr.Markdown("", elem_id="main-dialog")
        to_post_btn = gr.Button("Continue to questions", elem_classes=["orange-btn"])
        dialog_status_md = gr.Markdown("")

    # -------------------------------
    # POST
    with gr.Group(visible=False) as post_group:
        gr.Markdown("## Recipe Evaluation (Part 2)")
        gr.Markdown("\n Now you see the same recipe again. \n")
        recipe_md_post = gr.Markdown("")
        gr.Markdown("\n Please **answer the questions below** based on your current impression. \n")

        post_diet = gr.Radio(LIKERT_1_7, label="How suitable is this recipe for your diet? (1 = very low, 7 = excellent)")
        post_stars = gr.Radio(STARS_1_5, label="How would you rate this recipe? (1–5 stars)")
        post_intent = gr.Radio(LIKERT_1_7, label="How likely would you cook this recipe? (1 = very low, 7 = excellent)")
        post_save = gr.Radio(LIKERT_1_7, label="How likely would you save this recipe? (1 = very low, 7 = excellent)")

        with gr.Row():
            post_est_fat = gr.Number(label="Please try to estimate the fat content in grams (g) per serving.", precision=2, 
                                    info="**Note:** You find the total number of servings on the recipe card. Please estimate for **one** serving.")
            post_est_carb = gr.Number(label="Please try to estimate the carbohydrates content in grams (g) per serving", precision=2, 
                                     info="**Note:** You find the total number of servings on the recipe card. Please estimate for **one** serving.")
        post_est_kcal = gr.Number(label="Please try to estimate the calories (kcal) per serving", precision=0, 
                                 info="**Note:** You find the total number of servings on the recipe card. Please estimate for **one** serving.")

        gr.Markdown("\n\n ### Dialog communication quality\n")
        gr.Markdown("Please **evaluate the dialog** you have read, using the following statements. \n")
        dq_inputs = []
        for key, text in DIALOG_QUALITY_ITEMS:
            dq_inputs.append(gr.Radio(LIKERT_1_7, label=f"{text} (1 = very poor, 7 = excellent)"))

        manip = gr.Radio(LIKERT_1_7, label=MANIP_ITEM[1])
        post_comment = gr.Textbox(label="Comment (optional)", lines=2)

        submit_btn = gr.Button("Submit", elem_classes=["orange-btn"])
        post_status_md = gr.Markdown("")

    # -------------------------------
    # END
    with gr.Group(visible=False) as end_group:
        end_md = gr.Markdown("Thank you for participating.")

    # ------------------------------------------------------------
    # Wiring

    # Welcome -> Consent
    start_btn.click(
        handle_welcome,
        inputs=[rater_id_input, english_ok_cb, imc_radio],
        outputs=[welcome_info, pending_rater_id_state, welcome_group, consent_group, imc_pass_state],
        js=SCROLL_TOP_JS,
    )

    # Consent -> Instructions
    consent_continue_btn.click(
        handle_consent_continue,
        inputs=[consent_cb, pending_rater_id_state],
        outputs=[consent_status_md, rater_id_state, consent_group, instructions_group],
        js=SCROLL_TOP_JS,
    )

    # Instructions -> Screening
    instructions_continue_btn.click(
        handle_instructions_continue,
        inputs=[rater_id_state],
        outputs=[instructions_status_md, rater_id_state, instructions_group, screening_group],
        js=SCROLL_TOP_JS,
    )

    # Screening -> Recipe (reserve here, with mini-safety via token reuse)
    screening_continue_btn.click(
        start_screening_and_reserve_then_prepare,
        inputs=[
            rater_id_state,
            reservation_token_state,
            assigned_stim_state,
            pref_input_state,
            involvement_input_state,
            nutrition_knowledge_input_state,
            english_skill_input_state,
            imc_pass_state,
        ],
        outputs=[
            screening_status_md,
            assigned_stim_state,
            assign_meta_state,
            reservation_token_state,
            condition_state,
            preferences_state,
            involvement_state,
            nutrition_knowledge_state,
            english_skill_state,
            recipe_md,
            dialog_title_md,
            dialog_md,
            screening_group,
            recipe_group,
        ],
        js=SCROLL_TOP_JS,
    )

    # Pre -> Dialog
    to_dialog_btn.click(
        go_to_dialog_page,
        inputs=[
            pre_diet, pre_stars, pre_intent, pre_save,
            pre_est_fat, pre_est_carb, pre_est_kcal,
            pre_comment,
            assigned_stim_state
        ],
        outputs=[
            recipe_status_md,
            recipe_group,
            dialog_group,
            dialog_start_time_state,
            pre_state,
        ],
        js=SCROLL_TOP_JS,
    )

    # Dialog -> Post
    to_post_btn.click(
        go_to_post_page,
        inputs=[dialog_start_time_state, assigned_stim_state, recipe_md],
        outputs=[
            dialog_status_md,
            dialog_group,
            post_group,
            dialog_duration_state,
            min_time_ok_state,
            recipe_md_post,
        ],
        js=SCROLL_TOP_JS,
    )

    # Submit -> Commit slots -> Persist -> End
    submit_btn.click(
        submit_post_and_finish,
        inputs=[
            post_diet, post_stars, post_intent, post_save,
            post_est_fat, post_est_carb, post_est_kcal,

            dq_inputs[0], dq_inputs[1], dq_inputs[2], dq_inputs[3], dq_inputs[4],
            manip,
            post_comment,

            rater_id_state,
            imc_pass_state,
            reservation_token_state,
            assigned_stim_state,
            assign_meta_state,
            condition_state,
            preferences_state,
            involvement_state,
            nutrition_knowledge_state,
            english_skill_state,
            pre_state,
            dialog_duration_state,
            min_time_ok_state,
        ],
        outputs=[post_status_md, post_group, end_group, end_md],
        js=SCROLL_TOP_JS,
    )

demo.queue()
demo.launch()