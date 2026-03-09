"""gaze_ml_prediction.py

Binary-classification ML for predicting self-reports from gaze/pupil features.

Targets (high/low binary):
- accuracy (uses existing binary column if available; otherwise thresholds continuous accuracy)
- confidence
- cognitive_load
- trust_info (trust in information)
- trust_sys (trust in system)

Validation modes:
- kfold: Stratified K-fold cross-validation
- loso: Leave-One-Subject-Out (LeaveOneGroupOut) using participant_id

Hyperparameter tuning (optional):
- none: fixed hyperparameters
- grid: GridSearchCV inside each outer fold (nested CV)
- random: RandomizedSearchCV inside each outer fold (nested CV)

Inputs:
- Option A (recommended): a single merged table via --input
    Must include participant_id, gaze feature columns, and self-report columns.
- Option B: build a merged table from --gaze-csv and --responses-csv
    Uses trial-level merge if both have trial_num; otherwise falls back to
    participant_id × condition aggregation.

Example:
    python gaze_ml_prediction.py --input analysis_df.csv --cv kfold --tuning grid
    python gaze_ml_prediction.py --gaze-csv eye_metrics_per_trial_all_participants_with_conditions.csv \
            --responses-csv responses.csv --cv loso --tuning random

Outputs:
- <outdir>/ml_binary_self_reports_metrics.csv        (summary: mean±std over outer folds)
- <outdir>/ml_binary_self_reports_folds.csv          (per-fold metrics + best_params)
- <outdir>/ml_binary_self_reports_used_features.json

Leakage control:
- All preprocessing (imputation/scaling) is inside sklearn Pipelines.
- For thresholded targets, thresholds default to per-fold training-only.
- With tuning enabled, hyperparameters are selected using only training data of each outer fold.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

def _json_default(obj: object) -> object:
    """Best-effort JSON serializer for metrics/params outputs.

    Some sklearn objects (e.g., GaussianProcess kernels like Product/Sum) and numpy
    scalars are not JSON serializable by default.
    """

    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # containers
    if isinstance(obj, (set, tuple)):
        return list(obj)

    # paths
    if isinstance(obj, Path):
        return str(obj)

    # fallback: stable string representation
    return str(obj)


def _safe_json_dumps(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default)

from joblib import parallel_backend

from sklearn.base import BaseEstimator
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut, RandomizedSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier  # type: ignore

    _XGB_AVAILABLE = True
except Exception:
    XGBClassifier = None  # type: ignore
    _XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier  # type: ignore

    _LGBM_AVAILABLE = True
except Exception:
    LGBMClassifier = None  # type: ignore
    _LGBM_AVAILABLE = False

try:
    from catboost import CatBoostClassifier  # type: ignore

    _CATBOOST_AVAILABLE = True
except Exception:
    CatBoostClassifier = None  # type: ignore
    _CATBOOST_AVAILABLE = False


METRIC_SUFFIXES: Tuple[str, ...] = (
    "gaze_duration",
    "fixation_duration",
    "fixation_count",
    "saccade_length",
    "saccade_count",
    "pupil_diameter",
    "pupil_diameter_fixation",
    "ttff_ms",
    "ffd_ms",
    "sfd_ms",
)

BOOL_TRUE = {"true", "t", "1", "yes", "y"}
BOOL_FALSE = {"false", "f", "0", "no", "n"}


CONDITION_CHOICES = ("all", "correct", "incorrect", "no_reasoning", "no_answer", "moe")
_CONDITION_VALUE_MAP: Dict[str, Tuple[str, ...]] = {
    # These map user-facing categories to values in `classified_condition`.
    "correct": ("correct_reasoning", "correct"),
    "incorrect": ("incorrect_reasoning", "incorrect"),
    "no_reasoning": ("no_reasoning",),
    # Replaces the legacy catch-all "other" bucket. We keep "other" for backward compatibility
    # with already-exported datasets.
    "no_answer": ("no_answer", "other"),
}


def _normalize_condition_series(cond_series: pd.Series) -> pd.Series:
    """Normalize raw condition labels (robust across sources)."""

    return (
        pd.Series(cond_series, index=cond_series.index)
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("-", "_", regex=False)
    )


def _standardize_classified_condition(
    df: pd.DataFrame,
    *,
    col: str = "classified_condition",
) -> pd.DataFrame:
    """Standardize the `classified_condition` column.

    Goal: remove the legacy bucket label `other` entirely from the working dataset.
    - Normalizes formatting (lowercase + underscores)
    - Maps legacy `other` -> `no_answer`

    This is intentionally limited to the condition label column and does not touch
    unrelated uses of the word "Other" (e.g., AOI group labels in plots).
    """

    if col not in df.columns:
        return df

    s = _normalize_condition_series(df[col])
    s = s.replace({"other": "no_answer"})
    out = df.copy()
    out[col] = s
    return out


def _bucketize_condition_series(cond_series: pd.Series) -> pd.Series:
    """Bucket raw condition strings into {correct, incorrect, no_reasoning, no_answer}."""

    s = _normalize_condition_series(cond_series)

    keep_correct = {
        str(v).strip().lower().replace("-", "_").replace(" ", "_") for v in _CONDITION_VALUE_MAP["correct"]
    }
    keep_incorrect = {
        str(v).strip().lower().replace("-", "_").replace(" ", "_") for v in _CONDITION_VALUE_MAP["incorrect"]
    }
    keep_no_reason = {
        str(v).strip().lower().replace("-", "_").replace(" ", "_") for v in _CONDITION_VALUE_MAP["no_reasoning"]
    }

    # Default: anything not recognized becomes no_answer.
    bucket = pd.Series("no_answer", index=s.index, dtype="string")
    bucket.loc[s.isin(keep_correct)] = "correct"
    bucket.loc[s.isin(keep_incorrect)] = "incorrect"
    bucket.loc[s.isin(keep_no_reason)] = "no_reasoning"
    return bucket


def _norm_pid(v: object) -> str:
    """Normalize participant_id across files."""

    if pd.isna(v):
        return ""

    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isfinite(v) and float(v).is_integer():
            return str(int(v))
        return str(v).strip()

    s = str(v).strip()
    if not s:
        return ""

    m = re.search(r"(\d+)\s*$", s)
    if m:
        return str(int(m.group(1)))

    return s


def _as_bool(v: object) -> Optional[bool]:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in BOOL_TRUE:
        return True
    if s in BOOL_FALSE:
        return False
    return None


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))

    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path, low_memory=False)

    raise ValueError(f"Unsupported input format: {path.suffix}. Use .csv or .parquet")


def _first_non_null(s: pd.Series) -> object:
    ss = s.dropna()
    if ss.empty:
        return np.nan
    return ss.iloc[0]


def _coalesce_str_cols(a: Optional[pd.Series], b: Optional[pd.Series]) -> Optional[pd.Series]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    out = a.astype("string")
    mask = out.isna() | (out.astype(str).str.len() == 0)
    out = out.copy()
    out.loc[mask] = b.astype("string").loc[mask]
    return out


def filter_by_condition(df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Filter rows by condition.

    Expects a column `classified_condition` when condition != 'all'.
    User-facing options are: all/correct/incorrect/no_reasoning/no_answer.
    The meta-condition 'moe' is treated as no filtering (useful for MoE routing).
    """

    cond = str(condition).strip().lower()
    if cond == "all":
        return df

    if cond == "moe":
        return df

    if "classified_condition" not in df.columns:
        raise ValueError(
            "Condition filtering requested but column 'classified_condition' is missing. "
            "Ensure inputs include condition fields or use gaze/responses CSVs so it can be derived."
        )

    keep_vals = _CONDITION_VALUE_MAP.get(cond)
    if not keep_vals:
        raise ValueError(f"Unknown condition: {condition}. Expected one of {CONDITION_CHOICES}")

    # Normalize to avoid brittleness across sources (e.g., "Correct reasoning" vs "correct_reasoning").
    s = _normalize_condition_series(df["classified_condition"])
    keep_norm = {str(v).strip().lower().replace("-", "_").replace(" ", "_") for v in keep_vals}
    return df[s.isin(keep_norm)].copy()


def _add_condition_one_hot_features(
    df: pd.DataFrame,
    *,
    col: str = "classified_condition",
    prefix: str = "cond_feat__",
) -> Tuple[pd.DataFrame, List[str]]:
    """Add condition one-hot columns derived from df[col].

    Produces numeric 0/1 columns suitable for sklearn pipelines.
    Categories are: correct / incorrect / no_reasoning / no_answer.
    """

    if col not in df.columns:
        raise ValueError(
            f"include_condition_as_feature=True requires column '{col}'. "
            "Ensure inputs include condition fields or use gaze/responses CSVs so it can be derived."
        )

    s = (
        df[col]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("-", "_", regex=False)
    )

    keep_correct = {str(v).strip().lower().replace("-", "_").replace(" ", "_") for v in _CONDITION_VALUE_MAP["correct"]}
    keep_incorrect = {str(v).strip().lower().replace("-", "_").replace(" ", "_") for v in _CONDITION_VALUE_MAP["incorrect"]}
    keep_no_reason = {str(v).strip().lower().replace("-", "_").replace(" ", "_") for v in _CONDITION_VALUE_MAP["no_reasoning"]}

    bucket = pd.Series("no_answer", index=df.index, dtype="string")
    bucket.loc[s.isin(keep_correct)] = "correct"
    bucket.loc[s.isin(keep_incorrect)] = "incorrect"
    bucket.loc[s.isin(keep_no_reason)] = "no_reasoning"

    dummies = pd.get_dummies(bucket, prefix=prefix.rstrip("_"), prefix_sep="", dtype=int)
    out = df.copy()
    for c in dummies.columns:
        out[c] = dummies[c].astype(int)

    return out, list(dummies.columns)


_CONDITION_FEATURE_PREFIXES: Tuple[str, ...] = ("cond_feat__", "condition_")
_CONDITION_FEATURE_EXACT: Tuple[str, ...] = (
    "classified_condition_numeric",
    "classified_condition",
    "classified_condition_resp",
    "classified_condition_gaze",
)


def _is_condition_indicator_feature(col: object) -> bool:
    s = str(col)
    if s in _CONDITION_FEATURE_EXACT:
        return True
    return any(s.startswith(p) for p in _CONDITION_FEATURE_PREFIXES)


def _prune_condition_indicator_features(feature_cols: Sequence[str], *, cond_name: object) -> List[str]:
    """Drop condition-encoding features unless condition == 'moe'.

    Policy:
      - For condition='moe', keep condition-indicator features (e.g., cond_feat__*).
      - For all other conditions (all/correct/incorrect/no_reasoning/no_answer/multi),
        remove them.

    This matches the requirement: only MoE runs include explicit condition features.
    """

    cond = str(cond_name).strip().lower()
    if cond == "moe":
        return list(feature_cols)
    return [c for c in feature_cols if not _is_condition_indicator_feature(c)]


def _union_feature_cols_across_conditions(feature_cols: Sequence[str], *, conditions: Sequence[object]) -> List[str]:
    """Stable feature list when running multiple conditions.

    Returns the union of per-condition feature lists after applying
    `_prune_condition_indicator_features()`.
    """

    out: List[str] = []
    seen: set[str] = set()
    for c in conditions:
        for f in _prune_condition_indicator_features(feature_cols, cond_name=c):
            if f not in seen:
                seen.add(f)
                out.append(f)
    return out


def _detect_gaze_feature_cols(df: pd.DataFrame) -> List[str]:
    feats: List[str] = []
    for col in df.columns:
        s = str(col)
        for suf in METRIC_SUFFIXES:
            if s.endswith("_" + suf):
                feats.append(s)
                break
    return sorted(set(feats))


def resolve_feature_columns(
    df_columns: Sequence[str],
    requested_features: Sequence[str],
) -> Tuple[List[str], Dict[str, str], List[str]]:
    """Resolve requested feature names to actual df columns.

    Supports common AOI naming aliases:
      - answer_<metric> -> answer_aoi_<metric>
      - reasoning_<metric> -> reasoning_aoi_<metric>

    Returns (resolved_list, mapping, missing_requested).
    """

    cols = set(map(str, df_columns))
    mapping: Dict[str, str] = {}
    resolved: List[str] = []
    missing: List[str] = []

    def _candidates(name: str) -> List[str]:
        s = str(name)
        cands = [s]
        # AOI aliasing
        if s.startswith("answer_") and not s.startswith("answer_aoi_"):
            cands.append("answer_aoi_" + s[len("answer_") :])
        if s.startswith("reasoning_") and not s.startswith("reasoning_aoi_"):
            cands.append("reasoning_aoi_" + s[len("reasoning_") :])
        # A bit of tolerance for accidental double underscores
        cands.extend([cc.replace("__", "_") for cc in list(cands)])
        # Deduplicate while preserving order
        out: List[str] = []
        seen: set = set()
        for cc in cands:
            if cc not in seen:
                out.append(cc)
                seen.add(cc)
        return out

    for raw in requested_features:
        key = str(raw)
        if not key:
            continue
        found = None
        for cand in _candidates(key):
            if cand in cols:
                found = cand
                break
        if found is None:
            missing.append(key)
            continue
        mapping[key] = found
        resolved.append(found)

    # keep stable order + remove duplicates
    resolved = list(dict.fromkeys(resolved))
    return resolved, mapping, missing


def classify_condition_from_response_row(row: pd.Series) -> str:
    cond = row.get("condition", None)
    trial_type = row.get("trial_type", None)
    reasoning_type = row.get("reasoning_type", None)

    if cond == "without_reasoning" and trial_type == "pre":
        return "no_answer"
    if cond == "without_reasoning" and trial_type == "main":
        return "no_reasoning"

    if cond == "with_reasoning" and reasoning_type == "correct":
        return "correct_reasoning"
    if cond == "with_reasoning" and pd.notna(reasoning_type) and reasoning_type != "correct":
        return "incorrect_reasoning"

    # Fall back to the "no_answer" bucket to avoid an extra catch-all category.
    return "no_answer"


def add_derived_self_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add/normalize self-report columns if raw items exist."""

    out = df.copy()

    if "participant_id" in out.columns:
        out["participant_id"] = out["participant_id"].map(_norm_pid)

    # Confidence
    if "confidence" in out.columns:
        out["confidence"] = pd.to_numeric(out["confidence"], errors="coerce")

    # Cognitive load (mean of items if present)
    if {"cognitive_load_1", "cognitive_load_2"}.issubset(out.columns) and "cognitive_load" not in out.columns:
        out["cognitive_load"] = (
            out[["cognitive_load_1", "cognitive_load_2"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        )
    if "cognitive_load" in out.columns:
        out["cognitive_load"] = pd.to_numeric(out["cognitive_load"], errors="coerce")

    # Trust in information
    if {"trust_1", "trust_2", "trust_3"}.issubset(out.columns) and "trust_info" not in out.columns:
        out["trust_info"] = out[["trust_1", "trust_2", "trust_3"]].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Trust in system
    trust_sys_cols = [c for c in ["trust_4", "trust_5", "trust_6", "trust_7", "trust_8", "trust_9"] if c in out.columns]
    if trust_sys_cols and "trust_sys" not in out.columns:
        out["trust_sys"] = out[trust_sys_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    # Allow alternative naming
    if "trust_in_info" in out.columns and "trust_info" not in out.columns:
        out["trust_info"] = pd.to_numeric(out["trust_in_info"], errors="coerce")
    if "trust_in_system" in out.columns and "trust_sys" not in out.columns:
        out["trust_sys"] = pd.to_numeric(out["trust_in_system"], errors="coerce")

    # Decision/accuracy variants
    if "decision" in out.columns and "answer" in out.columns and "decision_match" not in out.columns:
        dec = out["decision"].map(_as_bool)
        ans = out["answer"].map(_as_bool)
        out["decision_match"] = np.where(
            pd.notna(dec) & pd.notna(ans),
            (pd.Series(dec, index=out.index) == pd.Series(ans, index=out.index)).astype(float),
            np.nan,
        )

    # Normalize common accuracy-like columns to numeric
    for c in ["decision_acc", "accuracy", "decision_match", "reasoning_acc"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def load_from_input_table(path: Path) -> pd.DataFrame:
    df = _read_table(path)
    df = add_derived_self_report_columns(df)
    df = _standardize_classified_condition(df)
    if "participant_id" not in df.columns:
        raise ValueError("Input table must contain participant_id (required for LOSO and reporting)")
    return df


def load_gaze_table(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path, low_memory=False)

    if "participant_id" not in df.columns:
        raise ValueError("Gaze CSV missing required column: participant_id")

    df["participant_id"] = df["participant_id"].map(_norm_pid)

    if "classified_condition" not in df.columns and "experimental_condition" in df.columns:
        df = df.rename(columns={"experimental_condition": "classified_condition"})

    df = _standardize_classified_condition(df)

    feature_cols = _detect_gaze_feature_cols(df)
    if not feature_cols:
        raise ValueError(
            "No gaze feature columns detected. Expected <AOI>_<metric> columns with metric suffix in "
            f"{list(METRIC_SUFFIXES)}"
        )

    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Prefer trial-level if trial_num exists
    keep_cols = ["participant_id"]
    if "classified_condition" in df.columns:
        keep_cols.append("classified_condition")
    if "trial_num" in df.columns:
        keep_cols.append("trial_num")

    df = df[keep_cols + feature_cols].copy()

    return df, feature_cols


def load_responses_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df = add_derived_self_report_columns(df)

    if "participant_id" not in df.columns:
        raise ValueError("Responses CSV missing required column: participant_id")

    # Derive condition label if possible
    needed_for_cond = {"condition", "trial_type", "reasoning_type"}
    if needed_for_cond.issubset(df.columns):
        df["classified_condition"] = df.apply(classify_condition_from_response_row, axis=1)
    elif "experimental_condition" in df.columns:
        df = df.rename(columns={"experimental_condition": "classified_condition"})
    elif "classified_condition" not in df.columns:
        df["classified_condition"] = "ALL"

    df = _standardize_classified_condition(df)

    return df


def load_demographic_features(
    demographic_csv: Path,
    *,
    participant_id_col: str = "participant_id",
    skiprows: Sequence[int] = (1,),
) -> Tuple[pd.DataFrame, List[str]]:
    """Load participant-level demographic features.

    Designed for Qualtrics exports (commonly with a metadata row at index 1).
    Returns (df_demo, feature_cols) where df_demo has one row per participant_id.
    """

    demo = pd.read_csv(demographic_csv, low_memory=False, skiprows=list(skiprows))
    if participant_id_col not in demo.columns:
        raise ValueError(
            f"Demographic CSV missing required column '{participant_id_col}'. Columns: {demo.columns.tolist()[:20]}"
        )

    def _coerce_likert_1to5(s: pd.Series) -> pd.Series:
        """Coerce common Qualtrics Likert text to numeric 1..5.

        Supports: Strongly disagree/Disagree/Neutral/Agree/Strongly agree.
        Leaves numeric-like strings untouched.
        """

        ss = s.astype(str).str.strip()
        # remove obvious qualtrics metadata tokens
        ss = ss.mask(ss.str.contains(r"\{\s*\"ImportId\"\s*:\s*\"", regex=True, na=False))

        # First try numeric coercion
        num = pd.to_numeric(ss, errors="coerce")
        if num.notna().any():
            # Keep numeric where possible; map remaining via text
            out = num
        else:
            out = num

        txt = ss.str.lower()
        txt = txt.str.replace(r"\s+", " ", regex=True)

        mapping = {
            "strongly disagree": 1,
            "disagree": 2,
            "neutral": 3,
            "neither agree nor disagree": 3,
            "agree": 4,
            "strongly agree": 5,
        }
        mapped = txt.map(mapping)
        out = out.where(out.notna(), mapped)
        return out.astype(float)

    demo = demo.copy()

    # Drop Qualtrics ImportId metadata row(s) if present
    pid_raw = demo[participant_id_col].astype(str)
    demo = demo.loc[~pid_raw.str.contains(r"\{\s*\"ImportId\"\s*:\s*\"", regex=True, na=False)].copy()

    demo[participant_id_col] = demo[participant_id_col].map(_norm_pid)
    demo = demo[demo[participant_id_col].astype(str).str.len() > 0]

    # Prefer item-level numeric features if present.
    item_cols: List[str] = []
    for c in demo.columns:
        s = str(c)
        if re.match(r"^PPT[_\s-]*\d+$", s, flags=re.IGNORECASE):
            item_cols.append(s)
        elif re.match(r"^AI\s*literacy[_\s-]*\d+$", s, flags=re.IGNORECASE):
            item_cols.append(s)

    # Common categorical demographics from the existing workflow.
    cat_cols = [c for c in ["Demographic-2", "Demographic-3", "Demographic-4", "Demographic-5", "Demographic-6"] if c in demo.columns]

    # Numeric items
    numeric_df = pd.DataFrame({participant_id_col: demo[participant_id_col]})
    numeric_feature_cols: List[str] = []
    for c in item_cols:
        new_c = f"demo__{c}"
        numeric_df[new_c] = _coerce_likert_1to5(demo[c])
        numeric_feature_cols.append(new_c)

    # One-hot categorical
    cat_feature_cols: List[str] = []
    if cat_cols:
        cat_raw = demo[cat_cols].copy()
        for c in cat_cols:
            cat_raw[c] = cat_raw[c].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
        dummies = pd.get_dummies(cat_raw, prefix=[f"demo_cat__{c}" for c in cat_cols], prefix_sep="=", dtype=int)
        for c in dummies.columns:
            cat_feature_cols.append(str(c))
        out = pd.concat([numeric_df, dummies], axis=1)
    else:
        out = numeric_df

    # Deduplicate to one row per participant.
    out = out.groupby(participant_id_col, as_index=False).mean(numeric_only=True)
    feature_cols = numeric_feature_cols + cat_feature_cols
    feature_cols = [c for c in feature_cols if c in out.columns]

    # Hard sanity check: if we detected item columns but all are NaN, raise a clear error.
    if numeric_feature_cols:
        nn = int(out[numeric_feature_cols].notna().sum().sum())
        if nn == 0:
            raise ValueError(
                "Demographic features were detected but parsed as all-missing. "
                "This usually means the CSV stores Likert responses as text labels that were not recognized. "
                "Check the demographic CSV values for PPT_* / AI literacy_* columns."
            )
    return out, feature_cols


def build_merged_dataset(gaze_csv: Path, responses_csv: Path) -> Tuple[pd.DataFrame, List[str]]:
    gaze_df, feature_cols = load_gaze_table(gaze_csv)
    resp_df = load_responses_table(responses_csv)

    # Prefer trial-level merge if trial_num exists in both tables.
    # We aggregate both sides to ensure one row per key (prevents accidental duplicates).
    merge_keys: List[str]
    if "trial_num" in gaze_df.columns and "trial_num" in resp_df.columns:
        merge_keys = ["participant_id", "trial_num"]

        # Aggregate gaze features and preserve condition label if present
        gaze_feat = gaze_df.groupby(merge_keys, as_index=False)[feature_cols].mean(numeric_only=True)
        gaze_cond: Optional[pd.DataFrame] = None
        if "classified_condition" in gaze_df.columns:
            gaze_cond = (
                gaze_df.groupby(merge_keys, as_index=False)["classified_condition"]
                .agg(_first_non_null)
                .rename(columns={"classified_condition": "classified_condition_gaze"})
            )
        gaze_df = pd.merge(gaze_feat, gaze_cond, on=merge_keys, how="left") if gaze_cond is not None else gaze_feat

        resp_measures = [
            c
            for c in [
                "decision_acc",
                "accuracy",
                "decision_match",
                "reasoning_acc",
                "confidence",
                "cognitive_load",
                "trust_info",
                "trust_sys",
            ]
            if c in resp_df.columns
        ]

        # Aggregate responses and preserve condition label if present
        resp_feat = resp_df.groupby(merge_keys, as_index=False)[resp_measures].mean(numeric_only=True)
        resp_cond: Optional[pd.DataFrame] = None
        if "classified_condition" in resp_df.columns:
            resp_cond = (
                resp_df.groupby(merge_keys, as_index=False)["classified_condition"]
                .agg(_first_non_null)
                .rename(columns={"classified_condition": "classified_condition_resp"})
            )
        resp_df = pd.merge(resp_feat, resp_cond, on=merge_keys, how="left") if resp_cond is not None else resp_feat
    else:
        merge_keys = ["participant_id", "classified_condition"]

        # Aggregate gaze to participant × condition
        if "classified_condition" not in gaze_df.columns:
            gaze_df["classified_condition"] = "ALL"
        gaze_df = gaze_df.groupby(merge_keys, as_index=False)[feature_cols].mean(numeric_only=True)

        # Aggregate responses similarly
        if "classified_condition" not in resp_df.columns:
            resp_df["classified_condition"] = "ALL"

        resp_measures = [
            c
            for c in [
                "decision_acc",
                "accuracy",
                "decision_match",
                "reasoning_acc",
                "confidence",
                "cognitive_load",
                "trust_info",
                "trust_sys",
            ]
            if c in resp_df.columns
        ]
        resp_df = resp_df.groupby(merge_keys, as_index=False)[resp_measures].mean(numeric_only=True)

    merged = pd.merge(gaze_df, resp_df, on=merge_keys, how="inner")

    # If trial-level merge used, reconcile condition columns into `classified_condition`.
    if "classified_condition" not in merged.columns:
        cc = _coalesce_str_cols(
            merged["classified_condition_resp"] if "classified_condition_resp" in merged.columns else None,
            merged["classified_condition_gaze"] if "classified_condition_gaze" in merged.columns else None,
        )
        if cc is not None:
            merged["classified_condition"] = cc

    for extra in ["classified_condition_resp", "classified_condition_gaze"]:
        if extra in merged.columns:
            merged = merged.drop(columns=[extra])

    if merged.empty:
        raise ValueError(
            "Merged dataset is empty. Check merge keys and that participant IDs/trial numbers align between files."
        )

    merged = _standardize_classified_condition(merged)

    return merged, feature_cols


@dataclass(frozen=True)
class TargetSpec:
    name: str
    candidates: Tuple[str, ...]


TARGETS: Tuple[TargetSpec, ...] = (
    TargetSpec("accuracy", ("decision_acc", "accuracy", "decision_match", "reasoning_acc")),
    TargetSpec("confidence", ("confidence",)),
    TargetSpec("cognitive_load", ("cognitive_load",)),
    TargetSpec("trust_info", ("trust_info", "trust_in_info")),
    TargetSpec("trust_sys", ("trust_sys", "trust_in_system")),
)


def resolve_target_column(df: pd.DataFrame, spec: TargetSpec) -> Optional[str]:
    for c in spec.candidates:
        if c in df.columns:
            return c
    return None


def is_binary_series(y: pd.Series) -> bool:
    vals = sorted(set(v for v in y.dropna().unique().tolist()))
    if not vals:
        return False
    return set(vals).issubset({0, 1, 0.0, 1.0})


def binarize_y(y_cont: pd.Series, threshold: float) -> pd.Series:
    y_num = pd.to_numeric(y_cont, errors="coerce")
    y_bin = np.where(pd.notna(y_num), (y_num >= float(threshold)).astype(int), np.nan)
    return pd.Series(y_bin, index=y_cont.index, dtype="float").dropna().astype(int)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _safe_precision_recall(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    try:
        if len(np.unique(y_true)) < 2:
            return float("nan"), float("nan")
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        return prec, rec
    except Exception:
        return float("nan"), float("nan")


def _resolve_joblib_backend(backend: str) -> str:
    b = str(backend).strip().lower()
    if b in {"loky", "threading"}:
        return b
    if b != "auto":
        raise ValueError(f"Unknown joblib backend: {backend}. Use auto/loky/threading")

    # Safer default on macOS: avoid loky worker crashes (SIGBUS) by using threads.
    # Users can override to loky if they need process-based parallelism.
    return "threading" if platform.system() == "Darwin" else "loky"


def _resolve_joblib_temp_folder(temp_folder: str) -> Optional[str]:
    tf = str(temp_folder).strip()
    if tf.lower() == "none":
        return None
    if tf.lower() == "auto":
        # Keep joblib temp/memmap off the project folder (e.g., OneDrive) by default.
        p = Path(tempfile.gettempdir()) / "joblib"
        p.mkdir(parents=True, exist_ok=True)
        return str(p)

    p = Path(tf).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def build_models(random_state: int, *, rf_n_jobs: int) -> Dict[str, BaseEstimator]:
    """Minimal, readable model set (no redundancy)."""

    logreg = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state,n_jobs=1)),
        ]
    )

    rf = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=500,
                random_state=random_state,
                n_jobs=int(rf_n_jobs),
                class_weight="balanced",
            )),
        ]
    )

    svm = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", SVC(probability=True, random_state=random_state)),
        ]
    )

    gboost = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                GradientBoostingClassifier(random_state=random_state),
            ),
        ]
    )

    mlp = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(random_state=random_state, max_iter=1000, early_stopping=True)),
        ]
    )

    adaboost = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", AdaBoostClassifier(random_state=random_state)),
        ]
    )

    extratrees = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    random_state=random_state,
                    n_jobs=int(rf_n_jobs),
                    class_weight="balanced",
                ),
            ),
        ]
    )

    gpc = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", GaussianProcessClassifier(random_state=random_state)),
        ]
    )

    models: Dict[str, BaseEstimator] = {
        "logreg": logreg,
        "rf": rf,
        "svm": svm,
        "gboost": gboost,
        "mlp": mlp,
        "adaboost": adaboost,
        "extratrees": extratrees,
        "gpc": gpc,
    }

    # Optional external libraries
    if _XGB_AVAILABLE and XGBClassifier is not None:
        models["xgb"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    XGBClassifier(
                        random_state=random_state,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=1.0,
                        n_jobs=int(rf_n_jobs),
                        eval_metric="logloss",
                        # Much faster on CPU; keeps GridSearch logic unchanged.
                        tree_method="hist",
                        max_bin=256,
                    ),
                ),
            ]
        )

    if _LGBM_AVAILABLE and LGBMClassifier is not None:
        models["lgbm"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        random_state=random_state,
                        n_estimators=300,
                        learning_rate=0.1,
                        num_leaves=31,
                        verbosity=-1,
                        n_jobs=int(rf_n_jobs),
                        # Often speeds up on wide tables.
                        force_col_wise=True,
                    ),
                ),
            ]
        )

    if _CATBOOST_AVAILABLE and CatBoostClassifier is not None:
        models["catboost"] = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    CatBoostClassifier(
                        random_state=random_state,
                        verbose=False,
                        loss_function="Logloss",
                        thread_count=int(rf_n_jobs),
                        # Avoid filesystem overhead.
                        allow_writing_files=False,
                    ),
                ),
            ]
        )

    return models


def build_param_grids() -> Dict[str, Dict[str, List[object]]]:
    """Parameter grids keyed by model name.

    The parameter names must match the Pipeline step name 'model'.
    """

    # Logistic regression: keep grid small and solver-compatible.
    # We default to class_weight='balanced' already.
    # Keep this grid solver/penalty-compatible across sklearn versions.
    logreg_grid: Dict[str, List[object]] = {
        "model__solver": ["liblinear"],
        "model__penalty": ["l1", "l2"],
        "model__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
    }

    rf_grid: Dict[str, List[object]] = {
        'model__n_estimators': [100, 300, 600],
        'model__max_depth': [5, 10, 20],
        # 'model__min_samples_split': [2, 5, 10],
        # 'model__min_samples_leaf': [1, 2, 4],
        # 'model__max_features': ['sqrt', 'log2', 0.3, 0.5],
        # 'model__class_weight': ['balanced', 'balanced_subsample', None],
        # 'model__bootstrap': [True, False]
    }

    # SVM (expanded kernels and hyperparameters)
    svm_grid: Dict[str, List[object]] = {
        "model__C": [0.1, 1, 10, 100],
        "model__kernel": ["rbf"],
        "model__gamma": ["scale", "auto"],
    }

    gboost_grid: Dict[str, List[object]] = {
        "model__n_estimators": [100, 300, 500],
        "model__learning_rate": [0.01, 0.05, 0.15],
        "model__max_depth": [3, 5, 7],
        "model__min_samples_split": [2, 5, 10],
        # "model__min_samples_leaf": [1, 2, 4],
        # "model__subsample": [0.7, 0.9, 1.0],
        "model__max_features": [None, "sqrt", "log2"],
    }

    mlp_grid: Dict[str, List[object]] = {
        "model__hidden_layer_sizes": [(50, 50), (100, 50), (100, 100)],
        "model__activation": ["tanh", "relu"],
        "model__solver": ["adam"],
        "model__alpha": [0.001, 0.01, 0.1],
        "model__learning_rate": ["constant", "adaptive"],
    }

    adaboost_grid: Dict[str, List[object]] = {
        "model__n_estimators": [50, 100, 300],
        "model__learning_rate": [0.01, 0.1, 0.5, 1.5],
    }

    extratrees_grid: Dict[str, List[object]] = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__class_weight": ["balanced", None],
    }

    # Keep booster grids compact: GridSearchCV remains the same, but the cartesian
    # product is reduced to a size that is practical for LOSO/KFold.
    xgb_grid: Dict[str, List[object]] = {
        "model__n_estimators": [200, 500],
        "model__max_depth": [3, 6],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__gamma": [0.0, 0.1],
        "model__reg_alpha": [0.0, 0.1],
        "model__reg_lambda": [1.0],
    }

    lgbm_grid: Dict[str, List[object]] = {
        "model__n_estimators": [200, 500],
        "model__num_leaves": [31, 63],
        "model__max_depth": [-1, 10],
        "model__learning_rate": [0.05, 0.1],
        "model__feature_fraction": [0.8, 1.0],
        "model__bagging_fraction": [0.8, 1.0],
        "model__bagging_freq": [0],
        "model__min_child_samples": [10, 20],
        "model__class_weight": ["balanced", None],
    }

    catboost_grid: Dict[str, List[object]] = {
        "model__iterations": [200, 500],
        "model__depth": [4, 6, 8],
        "model__learning_rate": [0.05, 0.1],
        "model__l2_leaf_reg": [3, 7],
        "model__border_count": [64, 128],
    }

    gpc_grid: Dict[str, List[object]] = {
        "model__kernel": [
            1.0 * RBF(length_scale=1.0),
            1.0 * RBF(length_scale=0.1),
            1.0 * Matern(length_scale=1.0, nu=1.5),
            1.0 * Matern(length_scale=1.0, nu=2.5),
            1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
        ],
        "model__max_iter_predict": [100, 200],
        "model__n_restarts_optimizer": [0, 1, 3],
    }

    # # 3. Gradient Boosting (expanded hyperparameters)
    # models_params['GradientBoosting'] = {
    #     'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
    #     'params': {
    #         'n_estimators': [100, 200, 300, 500],
    #         'learning_rate': [0.01, 0.05, 0.1, 0.15],
    #         'max_depth': [3, 5, 7, 10],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 2, 4],
    #         'subsample': [0.7, 0.8, 0.9, 1.0],
    #         'max_features': ['sqrt', 'log2', None]
    #     }
    # }
    
    
    # # 5. Neural Network (MLP)
    # models_params['NeuralNet'] = {
    #     'model': MLPClassifier(random_state=RANDOM_STATE, max_iter=1000),
    #     'params': {
    #         'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
    #         'activation': ['tanh', 'relu', 'logistic'],
    #         'solver': ['adam', 'sgd', 'lbfgs'],
    #         'alpha': [0.0001, 0.001, 0.01, 0.1],
    #         'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #         'early_stopping': [True, False]
    #     }
    # }
    
    # # 6. AdaBoost
    # models_params['AdaBoost'] = {
    #     'model': AdaBoostClassifier(random_state=RANDOM_STATE),
    #     'params': {
    #         'n_estimators': [50, 100, 200, 300],
    #         'learning_rate': [0.01, 0.1, 0.5, 1.0, 1.5],
    #         'algorithm': ['SAMME', 'SAMME.R']
    #     }
    # }
    
    # # 7. Extra Trees
    # models_params['ExtraTrees'] = {
    #     'model': ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    #     'params': {
    #         'n_estimators': [100, 200, 300, 500],
    #         'max_depth': [None, 10, 20, 30],
    #         'min_samples_split': [2, 5, 10],
    #         'min_samples_leaf': [1, 2, 4],
    #         'class_weight': ['balanced', None]
    #     }
    # }
    
    # # 8. XGBoost (if available)
    # if XGB_AVAILABLE:
    #     models_params['XGBoost'] = {
    #         'model': XGBClassifier(random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
    #         'params': {
    #             'n_estimators': [100, 200, 300, 500],
    #             'max_depth': [3, 5, 7, 10],
    #             'learning_rate': [0.01, 0.05, 0.1, 0.3],
    #             'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    #             'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    #             'gamma': [0, 0.1, 0.2, 0.5],
    #             'reg_alpha': [0, 0.1, 0.5, 1],
    #             'reg_lambda': [0.5, 1, 1.5, 2]
    #         }
    #     }
    
    # # 9. LightGBM (if available)
    # if LGBM_AVAILABLE:
    #     models_params['LightGBM'] = {
    #         'model': LGBMClassifier(random_state=RANDOM_STATE, verbosity=-1),
    #         'params': {
    #             'n_estimators': [100, 200, 300, 500],
    #             'num_leaves': [15, 31, 63, 127],
    #             'max_depth': [-1, 5, 10, 15],
    #             'learning_rate': [0.01, 0.05, 0.1, 0.3],
    #             'feature_fraction': [0.5, 0.7, 0.9, 1.0],
    #             'bagging_fraction': [0.5, 0.7, 0.9, 1.0],
    #             'bagging_freq': [0, 5, 10],
    #             'min_child_samples': [5, 10, 20, 30],
    #             'class_weight': ['balanced', None]
    #         }
    #     }
    
    # # 10. CatBoost (if available)
    # if CATBOOST_AVAILABLE:
    #     models_params['CatBoost'] = {
    #         'model': CatBoostClassifier(random_state=RANDOM_STATE, verbose=False),
    #         'params': {
    #             'iterations': [100, 200, 300, 500],
    #             'depth': [4, 6, 8, 10],
    #             'learning_rate': [0.01, 0.05, 0.1, 0.3],
    #             'l2_leaf_reg': [1, 3, 5, 7, 9],
    #             'border_count': [32, 64, 128, 256]
    #         }
    #     }
    
    # # 11. Gaussian Process (useful for temporal structure)
    # models_params['GaussianProcess'] = {
    #     'model': GaussianProcessClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    #     'params': {
    #         'kernel': [
    #             1.0 * RBF(1.0),
    #             1.0 * RBF(0.1),
    #             1.0 * Matern(1.0, nu=1.5),
    #             1.0 * Matern(1.0, nu=2.5),
    #             1.0 * RBF(1.0) + WhiteKernel(1.0)
    #         ],
    #         'max_iter_predict': [100, 200],
    #         'n_restarts_optimizer': [0, 1, 3]
    #     }
    # }


    grids: Dict[str, Dict[str, List[object]]] = {
        "logreg": logreg_grid,
        "rf": rf_grid,
        "svm": svm_grid,
        "gboost": gboost_grid,
        "mlp": mlp_grid,
        "adaboost": adaboost_grid,
        "extratrees": extratrees_grid,
        "gpc": gpc_grid,
    }

    if _XGB_AVAILABLE:
        grids["xgb"] = xgb_grid
    if _LGBM_AVAILABLE:
        grids["lgbm"] = lgbm_grid
    if _CATBOOST_AVAILABLE:
        grids["catboost"] = catboost_grid

    return grids


def merge_param_grids(
    base_grids: Dict[str, Dict[str, List[object]]],
    overrides: Optional[Dict[str, Dict[str, List[object]]]] = None,
) -> Dict[str, Dict[str, List[object]]]:
    if not overrides:
        return dict(base_grids)

    merged: Dict[str, Dict[str, List[object]]] = {}
    for model_name, base_grid in base_grids.items():
        if model_name in overrides:
            merged[model_name] = overrides[model_name]
        else:
            merged[model_name] = base_grid
    # Include any extra overrides for models not in base (if user defines new model params)
    for model_name, override_grid in overrides.items():
        if model_name not in merged:
            merged[model_name] = override_grid
    return merged


def load_param_grid_config(path: Optional[Union[str, Path]]) -> Optional[Dict[str, Dict[str, List[Any]]]]:
    if not path:
        return None
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Parameter grid config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Parameter grid config must be a JSON object mapping model names to param grids")
    cleaned: Dict[str, Dict[str, List[Any]]] = {}
    for model_name, param_dict in data.items():
        if not isinstance(param_dict, dict):
            raise ValueError(f"Param grid for model '{model_name}' must be an object")
        cleaned_grid: Dict[str, List[Any]] = {}
        for param_name, values in param_dict.items():
            if isinstance(values, list):
                cleaned_grid[param_name] = values
            else:
                cleaned_grid[param_name] = [values]
        cleaned[model_name] = cleaned_grid
    return cleaned


def infer_feature_columns(
    df: pd.DataFrame,
    explicit_features: Optional[Sequence[str]],
    target_cols: Sequence[str],
) -> List[str]:
    if explicit_features:
        missing = [c for c in explicit_features if c not in df.columns]
        if missing:
            raise ValueError(f"Requested features not found in data: {missing[:10]}")
        return list(explicit_features)

    # Prefer gaze-like features by suffix
    feats = _detect_gaze_feature_cols(df)
    if feats:
        return feats

    # Fallback: numeric columns excluding obvious metadata/targets
    exclude = set(target_cols)
    exclude |= {"participant_id", "trial_num", "condition", "classified_condition", "experimental_condition"}

    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(numeric_cols)


def make_splits(
    X: pd.DataFrame,
    y_for_split: pd.Series,
    cv_mode: str,
    n_splits: int,
    random_state: int,
    groups: Optional[pd.Series],
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    if cv_mode == "kfold":
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return skf.split(X, y_for_split)

    if cv_mode == "loso":
        if groups is None:
            raise ValueError("LOSO requires --group-col (e.g., participant_id)")
        logo = LeaveOneGroupOut()
        return logo.split(X, y_for_split, groups=groups)

    raise ValueError(f"Unknown cv mode: {cv_mode}")


def _inner_cv_for_tuning(
    cv_mode: str,
    y_train_for_split: pd.Series,
    groups_train: Optional[pd.Series],
    inner_splits: int,
    random_state: int,
):
    """Choose an inner CV splitter for hyperparameter tuning.

    - For kfold: StratifiedKFold.
    - For loso: we use StratifiedKFold as default, but if groups are provided and
      there are enough unique groups, we still rely on StratifiedKFold because
      GroupKFold is not stratified. This keeps class balance stable.
    """

    # NOTE: If you prefer strictly group-separated tuning, we can switch to GroupKFold,
    # but class imbalance can make it unstable. Keeping it simple/robust here.
    n_splits = int(max(2, inner_splits))
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def _safe_n_splits_for_stratified(y: pd.Series, desired: int) -> int:
    """Avoid common StratifiedKFold errors when class counts are small."""

    yv = pd.Series(y).dropna().astype(int)
    if yv.empty:
        return 2
    counts = yv.value_counts()
    min_class = int(counts.min()) if not counts.empty else 2
    return int(max(2, min(int(desired), min_class)))


def _fit_with_optional_tuning(
    model: BaseEstimator,
    param_grid: Dict[str, List[object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuning: str,
    scoring: str,
    inner_splits: int,
    n_iter: int,
    random_state: int,
    n_jobs: int,
    joblib_backend: str,
    joblib_temp_folder: Optional[str],
) -> Tuple[BaseEstimator, Dict[str, object]]:
    if tuning == "none":
        fitted = model.fit(X_train, y_train)
        return fitted, {}

    inner_splits_safe = _safe_n_splits_for_stratified(y_train, desired=int(inner_splits))
    inner_cv = _inner_cv_for_tuning(
        cv_mode="kfold",
        y_train_for_split=y_train,
        groups_train=None,
        inner_splits=inner_splits_safe,
        random_state=random_state,
    )

    if tuning == "grid":
        search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            scoring=scoring,
            cv=inner_cv,
            n_jobs=int(n_jobs),
            refit=True,
            verbose=0,
        )
    elif tuning == "random":
        # For small grids this behaves like a shuffled subset.
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=int(max(1, n_iter)),
            scoring=scoring,
            cv=inner_cv,
            n_jobs=int(n_jobs),
            refit=True,
            random_state=random_state,
            verbose=0,
        )
    else:
        raise ValueError(f"Unknown tuning mode: {tuning}")

    backend_resolved = _resolve_joblib_backend(joblib_backend)
    if backend_resolved == "loky":
        with parallel_backend(
            backend_resolved,
            temp_folder=joblib_temp_folder,
            # Reduce thread oversubscription when using process-based backends.
            inner_max_num_threads=1,
        ):
            search.fit(X_train, y_train)
    else:
        with parallel_backend(
            backend_resolved,
            temp_folder=joblib_temp_folder,
        ):
            search.fit(X_train, y_train)
    best_params = dict(search.best_params_) if hasattr(search, "best_params_") else {}
    return search.best_estimator_, best_params


def evaluate_binary_classification(
    X: pd.DataFrame,
    y_cont: pd.Series,
    model: BaseEstimator,
    cv_mode: str,
    n_splits: int,
    random_state: int,
    threshold_quantile: float,
    threshold_mode: str,
    groups: Optional[pd.Series],
) -> Dict[str, float]:
    """Evaluate with fold-wise metrics.

    threshold_mode:
      - global: compute threshold on full y_cont
      - train: compute threshold per-fold from y_train_cont

    For kfold splitting, we stratify using global-threshold labels to keep folds balanced.
    """

    y_num = pd.to_numeric(y_cont, errors="coerce")

    if is_binary_series(y_num):
        y_global_bin = y_num.dropna().astype(int)
        threshold_global = float("nan")
    else:
        threshold_global = float(y_num.quantile(float(threshold_quantile)))
        y_global_bin = binarize_y(y_num, threshold_global)

    common_index = X.index.intersection(y_global_bin.index)
    X_use = X.loc[common_index]
    y_cont_use = y_num.loc[common_index]
    y_split = y_global_bin.loc[common_index]

    if len(X_use) < 10:
        raise ValueError("Too few samples after filtering missing labels.")

    if cv_mode == "loso":
        if groups is None:
            raise ValueError("groups required for loso")
        groups_use = groups.loc[common_index]
    else:
        groups_use = None

    accs: List[float] = []
    f1s: List[float] = []
    aucs: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    thresholds_used: List[float] = []

    splits = make_splits(X_use, y_split, cv_mode=cv_mode, n_splits=n_splits, random_state=random_state, groups=groups_use)

    for tr_idx, te_idx in splits:
        X_tr, X_te = X_use.iloc[tr_idx], X_use.iloc[te_idx]
        y_tr_cont = y_cont_use.iloc[tr_idx]
        y_te_cont = y_cont_use.iloc[te_idx]

        if is_binary_series(y_num):
            y_tr = y_split.iloc[tr_idx]
            y_te = y_split.iloc[te_idx]
            thr = float("nan")
        else:
            if threshold_mode == "train":
                thr = float(pd.to_numeric(y_tr_cont, errors="coerce").quantile(float(threshold_quantile)))
            else:
                thr = threshold_global

            y_tr = binarize_y(y_tr_cont, thr)
            y_te = binarize_y(y_te_cont, thr)

            # Align after binarization (drops NaNs)
            tr_common = X_tr.index.intersection(y_tr.index)
            te_common = X_te.index.intersection(y_te.index)
            X_tr = X_tr.loc[tr_common]
            X_te = X_te.loc[te_common]
            y_tr = y_tr.loc[tr_common]
            y_te = y_te.loc[te_common]

        if len(np.unique(y_tr.values)) < 2:
            # Cannot train a binary classifier with single-class training data
            continue

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_te)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_te)
            y_score = 1.0 / (1.0 + np.exp(-scores))
        else:
            y_score = y_pred.astype(float)

        accs.append(float(accuracy_score(y_te, y_pred)))
        f1s.append(float(f1_score(y_te, y_pred, zero_division=0)))
        aucs.append(safe_auc(y_te.values, y_score))
        prec, rec = _safe_precision_recall(y_te.values, y_pred)
        precs.append(prec)
        recs.append(rec)
        thresholds_used.append(thr)

    if not accs:
        raise ValueError("No valid folds produced metrics (check class balance / labels / groups).")

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.nanmean(precs)),
        "precision_std": float(np.nanstd(precs)),
        "recall_mean": float(np.nanmean(recs)),
        "recall_std": float(np.nanstd(recs)),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "folds_used": float(len(accs)),
        "threshold_mean": float(np.nanmean(thresholds_used)) if thresholds_used else float("nan"),
    }


def evaluate_binary_classification_moe(
    X: pd.DataFrame,
    y_cont: pd.Series,
    condition_series: pd.Series,
    model: BaseEstimator,
    cv_mode: str,
    n_splits: int,
    random_state: int,
    threshold_quantile: float,
    threshold_mode: str,
    groups: Optional[pd.Series],
) -> Dict[str, float]:
    """Mixture-of-Experts evaluation.

    For each fold, train separate expert models for condition buckets
    (correct/incorrect/no_reasoning/no_answer) using ONLY that bucket's training data.
    Test samples are routed to their bucket's expert.

    Metrics are computed on the routed subset of test samples for which a valid
    expert could be trained.
    """

    y_num = pd.to_numeric(y_cont, errors="coerce")

    if is_binary_series(y_num):
        y_global_bin = y_num.dropna().astype(int)
        threshold_global = float("nan")
    else:
        threshold_global = float(y_num.quantile(float(threshold_quantile)))
        y_global_bin = binarize_y(y_num, threshold_global)

    common_index = X.index.intersection(y_global_bin.index).intersection(condition_series.index)
    X_use = X.loc[common_index]
    y_cont_use = y_num.loc[common_index]
    y_split = y_global_bin.loc[common_index]
    cond_use = condition_series.loc[common_index]

    if len(X_use) < 10:
        raise ValueError("Too few samples after filtering missing labels.")

    if cv_mode == "loso":
        if groups is None:
            raise ValueError("groups required for loso")
        groups_use = groups.loc[common_index]
    else:
        groups_use = None

    accs: List[float] = []
    f1s: List[float] = []
    aucs: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    thresholds_used: List[float] = []

    splits = make_splits(
        X_use, y_split, cv_mode=cv_mode, n_splits=n_splits, random_state=random_state, groups=groups_use
    )
    expert_names = ["correct", "incorrect", "no_reasoning", "no_answer"]

    for tr_idx, te_idx in splits:
        X_tr, X_te = X_use.iloc[tr_idx], X_use.iloc[te_idx]
        y_tr_cont = y_cont_use.iloc[tr_idx]
        y_te_cont = y_cont_use.iloc[te_idx]
        cond_tr = cond_use.iloc[tr_idx]
        cond_te = cond_use.iloc[te_idx]

        if is_binary_series(y_num):
            y_tr = y_split.iloc[tr_idx]
            y_te = y_split.iloc[te_idx]
            thr = float("nan")
        else:
            if threshold_mode == "train":
                thr = float(pd.to_numeric(y_tr_cont, errors="coerce").quantile(float(threshold_quantile)))
            else:
                thr = threshold_global

            y_tr = binarize_y(y_tr_cont, thr)
            y_te = binarize_y(y_te_cont, thr)

            tr_common = X_tr.index.intersection(y_tr.index)
            te_common = X_te.index.intersection(y_te.index)
            X_tr = X_tr.loc[tr_common]
            X_te = X_te.loc[te_common]
            y_tr = y_tr.loc[tr_common]
            y_te = y_te.loc[te_common]
            cond_tr = cond_tr.loc[tr_common]
            cond_te = cond_te.loc[te_common]

        if len(np.unique(y_tr.values)) < 2:
            continue

        bucket_tr = _bucketize_condition_series(cond_tr)
        bucket_te = _bucketize_condition_series(cond_te)

        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        y_score_all: List[float] = []

        for expert in expert_names:
            te_mask = bucket_te == expert
            if not bool(te_mask.any()):
                continue
            tr_mask = bucket_tr == expert
            if not bool(tr_mask.any()):
                continue

            X_tr_e = X_tr.loc[tr_mask]
            y_tr_e = y_tr.loc[tr_mask]
            if len(np.unique(y_tr_e.values)) < 2:
                continue

            est = clone(model)
            est.fit(X_tr_e, y_tr_e)

            X_te_e = X_te.loc[te_mask]
            y_te_e = y_te.loc[te_mask]
            if X_te_e.empty:
                continue

            y_pred_e = est.predict(X_te_e)
            if hasattr(est, "predict_proba"):
                y_score_e = est.predict_proba(X_te_e)[:, 1]
            elif hasattr(est, "decision_function"):
                scores = est.decision_function(X_te_e)
                y_score_e = 1.0 / (1.0 + np.exp(-scores))
            else:
                y_score_e = y_pred_e.astype(float)

            y_true_all.extend([int(v) for v in y_te_e.values])
            y_pred_all.extend([int(v) for v in np.asarray(y_pred_e)])
            y_score_all.extend([float(v) for v in np.asarray(y_score_e)])

        if not y_true_all:
            continue

        y_true_arr = np.asarray(y_true_all, dtype=int)
        y_pred_arr = np.asarray(y_pred_all, dtype=int)
        y_score_arr = np.asarray(y_score_all, dtype=float)

        accs.append(float(accuracy_score(y_true_arr, y_pred_arr)))
        f1s.append(float(f1_score(y_true_arr, y_pred_arr, zero_division=0)))
        aucs.append(safe_auc(y_true_arr, y_score_arr))
        prec, rec = _safe_precision_recall(y_true_arr, y_pred_arr)
        precs.append(prec)
        recs.append(rec)
        thresholds_used.append(thr)

    if not accs:
        raise ValueError("No valid folds produced metrics (check class balance / labels / conditions).")

    return {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.nanmean(precs)),
        "precision_std": float(np.nanstd(precs)),
        "recall_mean": float(np.nanmean(recs)),
        "recall_std": float(np.nanstd(recs)),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "folds_used": float(len(accs)),
        "threshold_mean": float(np.nanmean(thresholds_used)) if thresholds_used else float("nan"),
    }


def evaluate_binary_classification_nested(
    X: pd.DataFrame,
    y_cont: pd.Series,
    model: BaseEstimator,
    param_grid: Dict[str, List[object]],
    cv_mode: str,
    n_splits: int,
    random_state: int,
    threshold_quantile: float,
    threshold_mode: str,
    groups: Optional[pd.Series],
    tuning: str,
    scoring: str,
    inner_splits: int,
    n_iter: int,
    n_jobs: int,
    joblib_backend: str,
    joblib_temp_folder: Optional[str],
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Nested CV evaluation with optional tuning.

    Returns (summary_metrics, per_fold_rows).
    """

    y_num = pd.to_numeric(y_cont, errors="coerce")

    if is_binary_series(y_num):
        y_global_bin = y_num.dropna().astype(int)
        threshold_global = float("nan")
    else:
        threshold_global = float(y_num.quantile(float(threshold_quantile)))
        y_global_bin = binarize_y(y_num, threshold_global)

    common_index = X.index.intersection(y_global_bin.index)
    X_use = X.loc[common_index]
    y_cont_use = y_num.loc[common_index]
    y_split = y_global_bin.loc[common_index]

    if X_use.empty or y_split.empty:
        raise ValueError("No samples available after filtering missing labels.")

    if cv_mode == "loso":
        if groups is None:
            raise ValueError("groups required for loso")
        groups_use = groups.loc[common_index]
    else:
        groups_use = None

    splits = make_splits(X_use, y_split, cv_mode=cv_mode, n_splits=n_splits, random_state=random_state, groups=groups_use)

    accs: List[float] = []
    f1s: List[float] = []
    aucs: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    thresholds_used: List[float] = []
    fold_rows: List[Dict[str, object]] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(splits, start=1):
        X_tr, X_te = X_use.iloc[tr_idx], X_use.iloc[te_idx]
        y_tr_cont = y_cont_use.iloc[tr_idx]
        y_te_cont = y_cont_use.iloc[te_idx]

        if is_binary_series(y_num):
            y_tr = y_split.iloc[tr_idx]
            y_te = y_split.iloc[te_idx]
            thr = float("nan")
        else:
            if threshold_mode == "train":
                thr = float(pd.to_numeric(y_tr_cont, errors="coerce").quantile(float(threshold_quantile)))
            else:
                thr = threshold_global

            y_tr = binarize_y(y_tr_cont, thr)
            y_te = binarize_y(y_te_cont, thr)

            tr_common = X_tr.index.intersection(y_tr.index)
            te_common = X_te.index.intersection(y_te.index)
            X_tr = X_tr.loc[tr_common]
            X_te = X_te.loc[te_common]
            y_tr = y_tr.loc[tr_common]
            y_te = y_te.loc[te_common]

        if len(np.unique(y_tr.values)) < 2 or len(np.unique(y_te.values)) < 2:
            continue

        best_estimator, best_params = _fit_with_optional_tuning(
            model=model,
            param_grid=param_grid,
            X_train=X_tr,
            y_train=y_tr,
            tuning=tuning,
            scoring=scoring,
            inner_splits=inner_splits,
            n_iter=n_iter,
            random_state=random_state,
            n_jobs=n_jobs,
            joblib_backend=joblib_backend,
            joblib_temp_folder=joblib_temp_folder,
        )

        y_pred = best_estimator.predict(X_te)

        if hasattr(best_estimator, "predict_proba"):
            y_score = best_estimator.predict_proba(X_te)[:, 1]
        elif hasattr(best_estimator, "decision_function"):
            scores = best_estimator.decision_function(X_te)
            y_score = 1.0 / (1.0 + np.exp(-scores))
        else:
            y_score = y_pred.astype(float)

        acc = float(accuracy_score(y_te, y_pred))
        f1v = float(f1_score(y_te, y_pred, zero_division=0))
        auc = safe_auc(y_te.values, y_score)
        prec, rec = _safe_precision_recall(y_te.values, y_pred)

        accs.append(acc)
        f1s.append(f1v)
        aucs.append(auc)
        precs.append(prec)
        recs.append(rec)
        thresholds_used.append(thr)

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
                "threshold": thr,
                "accuracy": acc,
                "f1": f1v,
                "precision": prec,
                "recall": rec,
                "auc": auc,
                "best_params": _safe_json_dumps(best_params),
            }
        )

    if not accs:
        raise ValueError("No valid folds produced metrics (check class balance / labels / groups).")

    summary = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.nanmean(precs)),
        "precision_std": float(np.nanstd(precs)),
        "recall_mean": float(np.nanmean(recs)),
        "recall_std": float(np.nanstd(recs)),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "folds_used": float(len(accs)),
        "threshold_mean": float(np.nanmean(thresholds_used)) if thresholds_used else float("nan"),
    }
    return summary, fold_rows


def evaluate_binary_classification_nested_moe(
    X: pd.DataFrame,
    y_cont: pd.Series,
    condition_series: pd.Series,
    model: BaseEstimator,
    param_grid: Dict[str, List[object]],
    cv_mode: str,
    n_splits: int,
    random_state: int,
    threshold_quantile: float,
    threshold_mode: str,
    groups: Optional[pd.Series],
    tuning: str,
    scoring: str,
    inner_splits: int,
    n_iter: int,
    n_jobs: int,
    joblib_backend: str,
    joblib_temp_folder: Optional[str],
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    """Nested CV MoE evaluation.

    Each outer fold trains per-condition experts; within each expert we optionally
    tune hyperparameters using only that expert's training data.
    """

    y_num = pd.to_numeric(y_cont, errors="coerce")

    if is_binary_series(y_num):
        y_global_bin = y_num.dropna().astype(int)
        threshold_global = float("nan")
    else:
        threshold_global = float(y_num.quantile(float(threshold_quantile)))
        y_global_bin = binarize_y(y_num, threshold_global)

    common_index = X.index.intersection(y_global_bin.index).intersection(condition_series.index)
    X_use = X.loc[common_index]
    y_cont_use = y_num.loc[common_index]
    y_split = y_global_bin.loc[common_index]
    cond_use = condition_series.loc[common_index]

    if X_use.empty or y_split.empty:
        raise ValueError("No samples available after filtering missing labels/conditions.")

    if cv_mode == "loso":
        if groups is None:
            raise ValueError("groups required for loso")
        groups_use = groups.loc[common_index]
    else:
        groups_use = None

    splits = make_splits(
        X_use, y_split, cv_mode=cv_mode, n_splits=n_splits, random_state=random_state, groups=groups_use
    )
    expert_names = ["correct", "incorrect", "no_reasoning", "no_answer"]

    accs: List[float] = []
    f1s: List[float] = []
    aucs: List[float] = []
    precs: List[float] = []
    recs: List[float] = []
    thresholds_used: List[float] = []
    fold_rows: List[Dict[str, object]] = []

    for fold_idx, (tr_idx, te_idx) in enumerate(splits, start=1):
        X_tr, X_te = X_use.iloc[tr_idx], X_use.iloc[te_idx]
        y_tr_cont = y_cont_use.iloc[tr_idx]
        y_te_cont = y_cont_use.iloc[te_idx]
        cond_tr = cond_use.iloc[tr_idx]
        cond_te = cond_use.iloc[te_idx]

        if is_binary_series(y_num):
            y_tr = y_split.iloc[tr_idx]
            y_te = y_split.iloc[te_idx]
            thr = float("nan")
        else:
            if threshold_mode == "train":
                thr = float(pd.to_numeric(y_tr_cont, errors="coerce").quantile(float(threshold_quantile)))
            else:
                thr = threshold_global

            y_tr = binarize_y(y_tr_cont, thr)
            y_te = binarize_y(y_te_cont, thr)

            tr_common = X_tr.index.intersection(y_tr.index)
            te_common = X_te.index.intersection(y_te.index)
            X_tr = X_tr.loc[tr_common]
            X_te = X_te.loc[te_common]
            y_tr = y_tr.loc[tr_common]
            y_te = y_te.loc[te_common]
            cond_tr = cond_tr.loc[tr_common]
            cond_te = cond_te.loc[te_common]

        if len(np.unique(y_tr.values)) < 2:
            continue

        bucket_tr = _bucketize_condition_series(cond_tr)
        bucket_te = _bucketize_condition_series(cond_te)

        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        y_score_all: List[float] = []
        best_params_by_expert: Dict[str, object] = {}

        for expert in expert_names:
            te_mask = bucket_te == expert
            if not bool(te_mask.any()):
                continue
            tr_mask = bucket_tr == expert
            if not bool(tr_mask.any()):
                continue

            X_tr_e = X_tr.loc[tr_mask]
            y_tr_e = y_tr.loc[tr_mask]
            if len(np.unique(y_tr_e.values)) < 2:
                continue

            best_estimator, best_params = _fit_with_optional_tuning(
                model=clone(model),
                param_grid=param_grid,
                X_train=X_tr_e,
                y_train=y_tr_e,
                tuning=str(tuning),
                scoring=str(scoring),
                inner_splits=int(inner_splits),
                n_iter=int(n_iter),
                random_state=int(random_state),
                n_jobs=int(n_jobs),
                joblib_backend=str(joblib_backend),
                joblib_temp_folder=joblib_temp_folder,
            )
            best_params_by_expert[expert] = best_params

            X_te_e = X_te.loc[te_mask]
            y_te_e = y_te.loc[te_mask]
            if X_te_e.empty:
                continue

            y_pred_e = best_estimator.predict(X_te_e)
            if hasattr(best_estimator, "predict_proba"):
                y_score_e = best_estimator.predict_proba(X_te_e)[:, 1]
            elif hasattr(best_estimator, "decision_function"):
                scores = best_estimator.decision_function(X_te_e)
                y_score_e = 1.0 / (1.0 + np.exp(-scores))
            else:
                y_score_e = y_pred_e.astype(float)

            y_true_all.extend([int(v) for v in y_te_e.values])
            y_pred_all.extend([int(v) for v in np.asarray(y_pred_e)])
            y_score_all.extend([float(v) for v in np.asarray(y_score_e)])

        if not y_true_all:
            continue

        y_true_arr = np.asarray(y_true_all, dtype=int)
        y_pred_arr = np.asarray(y_pred_all, dtype=int)
        y_score_arr = np.asarray(y_score_all, dtype=float)

        acc = float(accuracy_score(y_true_arr, y_pred_arr))
        f1v = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))
        auc = safe_auc(y_true_arr, y_score_arr)
        prec, rec = _safe_precision_recall(y_true_arr, y_pred_arr)

        accs.append(acc)
        f1s.append(f1v)
        aucs.append(auc)
        precs.append(prec)
        recs.append(rec)
        thresholds_used.append(thr)

        fold_rows.append(
            {
                "fold": fold_idx,
                "n_train": int(len(X_tr)),
                "n_test": int(len(X_te)),
                "n_test_used": int(len(y_true_arr)),
                "threshold": thr,
                "accuracy": acc,
                "f1": f1v,
                "precision": prec,
                "recall": rec,
                "auc": auc,
                "best_params": _safe_json_dumps(best_params_by_expert),
            }
        )

    if not accs:
        raise ValueError("No valid folds produced metrics (check class balance / labels / conditions).")

    summary = {
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "precision_mean": float(np.nanmean(precs)),
        "precision_std": float(np.nanstd(precs)),
        "recall_mean": float(np.nanmean(recs)),
        "recall_std": float(np.nanstd(recs)),
        "auc_mean": float(np.nanmean(aucs)),
        "auc_std": float(np.nanstd(aucs)),
        "folds_used": float(len(accs)),
        "threshold_mean": float(np.nanmean(thresholds_used)) if thresholds_used else float("nan"),
    }
    return summary, fold_rows


def parse_targets_arg(target: Optional[str], targets: Optional[str]) -> Optional[List[str]]:
    # If both provided, merge.
    chosen: List[str] = []
    if target:
        chosen.append(str(target).strip())
    if targets:
        chosen.extend([t.strip() for t in str(targets).split(",") if t.strip()])
    chosen = [c for c in chosen if c]
    return chosen or None


def _make_run_suffix(*, condition: str, target_names: Optional[Sequence[str]]) -> str:
    parts: List[str] = []
    cond = str(condition).strip().lower()
    if cond and cond != "all":
        parts.append(f"cond-{cond}")
    if target_names:
        if len(target_names) == 1:
            parts.append(f"target-{target_names[0]}")
        else:
            parts.append("target-multi")
    return "__" + "__".join(parts) if parts else ""


def _feature_report(df: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    n = int(df.shape[0])
    for c in feature_cols:
        s = pd.to_numeric(df[c], errors="coerce") if c in df.columns else pd.Series([], dtype=float)
        nn = int(s.notna().sum())
        rows.append(
            {
                "feature": c,
                "n_rows": n,
                "n_nonnull": nn,
                "missing_rate": float(1.0 - (nn / n)) if n > 0 else float("nan"),
                "mean": float(s.mean()) if nn > 0 else float("nan"),
                "std": float(s.std()) if nn > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_rate", "feature"], ascending=[True, True])


def _target_split_preview_table(
    *,
    X: pd.DataFrame,
    df: pd.DataFrame,
    resolved_targets: Dict[str, str],
    cv_mode: str,
    n_splits: int,
    random_state: int,
    quantile: float,
    threshold_mode: str,
    groups: Optional[pd.Series],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for target_name, col in resolved_targets.items():
        y = pd.to_numeric(df[col], errors="coerce")
        preview = _preview_threshold_splits(
            X=X,
            y_cont=y,
            cv_mode=cv_mode,
            n_splits=int(n_splits),
            random_state=int(random_state),
            quantile=float(quantile),
            threshold_mode=str(threshold_mode),
            groups=groups,
        )

        # Compute a global split (for interpretable overall means/ratios)
        if is_binary_series(y):
            yb = y.dropna().astype(int)
            thr = float("nan")
        else:
            thr = float(y.quantile(float(quantile)))
            yb = binarize_y(y, thr)

        y_raw = y.loc[yb.index]
        low_mask = yb == 0
        high_mask = yb == 1

        counts = yb.value_counts().to_dict()
        n0, n1 = int(counts.get(0, 0)), int(counts.get(1, 0))
        n_total = int(n0 + n1)

        rows.append(
            {
                "target": target_name,
                "target_column": col,
                "n": int(preview.get("n", n_total)),
                "is_binary": bool(preview.get("is_binary", False)),
                "quantile": float(quantile),
                "threshold_mode": str(threshold_mode),
                "threshold_global": float(thr) if np.isfinite(thr) else float("nan"),
                "threshold_mean": float(preview.get("threshold_mean", float("nan"))),
                "threshold_std": float(preview.get("threshold_std", float("nan"))),
                "overall_n0": n0,
                "overall_n1": n1,
                "overall_pos_rate": float(n1 / n_total) if n_total > 0 else float("nan"),
                "raw_mean": float(y_raw.mean()) if y_raw.notna().any() else float("nan"),
                "raw_mean_low": float(y_raw[low_mask].mean()) if low_mask.any() else float("nan"),
                "raw_mean_high": float(y_raw[high_mask].mean()) if high_mask.any() else float("nan"),
                "test_counts0_mean": float(preview.get("test_counts0_mean", float("nan"))),
                "test_counts1_mean": float(preview.get("test_counts1_mean", float("nan"))),
                "folds_used": int(preview.get("folds_used", 0)) if preview.get("folds_used") is not None else None,
            }
        )
    return pd.DataFrame(rows)


def parse_feature_list_arg(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    cols = [c.strip() for c in s.split(",") if c.strip()]
    return cols or None


def parse_name_list_arg(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    items = [x.strip() for x in s.split(",") if x.strip()]
    return items or None


def _preview_threshold_splits(
    *,
    X: pd.DataFrame,
    y_cont: pd.Series,
    cv_mode: str,
    n_splits: int,
    random_state: int,
    quantile: float,
    threshold_mode: str,
    groups: Optional[pd.Series],
) -> Dict[str, object]:
    """Compute threshold statistics and class split counts WITHOUT fitting models.

    For continuous targets and threshold_mode='train', thresholds vary per fold.
    This function summarizes what will happen during training.
    """

    y_num = pd.to_numeric(y_cont, errors="coerce")

    if is_binary_series(y_num):
        y_bin = y_num.dropna().astype(int)
        counts = y_bin.value_counts().to_dict()
        return {
            "is_binary": True,
            "n": int(y_bin.shape[0]),
            "threshold_mean": float("nan"),
            "overall_counts_0": int(counts.get(0, 0)),
            "overall_counts_1": int(counts.get(1, 0)),
        }

    # Use a global threshold ONLY for defining y_split for stratification.
    thr_global = float(y_num.quantile(float(quantile)))
    y_split = binarize_y(y_num, thr_global)

    # If there are no usable labels (all NaN), don't attempt to split.
    if y_split.empty:
        return {
            "is_binary": False,
            "n": 0,
            "threshold_global": thr_global,
            "threshold_mean": float("nan"),
            "threshold_std": float("nan"),
            "overall_counts_0": 0,
            "overall_counts_1": 0,
            "test_counts0_mean": float("nan"),
            "test_counts1_mean": float("nan"),
            "folds_used": 0,
        }

    common_index = X.index.intersection(y_split.index)
    X_use = X.loc[common_index]
    y_cont_use = y_num.loc[common_index]
    y_split_use = y_split.loc[common_index]

    if X_use.empty or y_split_use.empty:
        overall_counts = y_split_use.value_counts().to_dict() if not y_split_use.empty else {}
        return {
            "is_binary": False,
            "n": int(y_split_use.shape[0]),
            "threshold_global": thr_global,
            "threshold_mean": float("nan"),
            "threshold_std": float("nan"),
            "overall_counts_0": int(overall_counts.get(0, 0)),
            "overall_counts_1": int(overall_counts.get(1, 0)),
            "test_counts0_mean": float("nan"),
            "test_counts1_mean": float("nan"),
            "folds_used": 0,
        }

    if cv_mode == "loso":
        if groups is None:
            raise ValueError("groups required for loso")
        groups_use = groups.loc[common_index]
    else:
        groups_use = None

    try:
        splits = make_splits(
            X_use,
            y_split_use,
            cv_mode=cv_mode,
            n_splits=n_splits,
            random_state=random_state,
            groups=groups_use,
        )
    except Exception:
        # Not enough samples / classes for the requested CV.
        overall_counts = y_split_use.value_counts().to_dict()
        return {
            "is_binary": False,
            "n": int(y_split_use.shape[0]),
            "threshold_global": thr_global,
            "threshold_mean": float("nan"),
            "threshold_std": float("nan"),
            "overall_counts_0": int(overall_counts.get(0, 0)),
            "overall_counts_1": int(overall_counts.get(1, 0)),
            "test_counts0_mean": float("nan"),
            "test_counts1_mean": float("nan"),
            "folds_used": 0,
        }

    fold_thresholds: List[float] = []
    fold_counts: List[Tuple[int, int]] = []
    folds_used = 0

    for tr_idx, te_idx in splits:
        y_tr_cont = y_cont_use.iloc[tr_idx]
        y_te_cont = y_cont_use.iloc[te_idx]

        if threshold_mode == "train":
            thr = float(pd.to_numeric(y_tr_cont, errors="coerce").quantile(float(quantile)))
        else:
            thr = thr_global

        y_te = binarize_y(y_te_cont, thr)
        counts = y_te.value_counts().to_dict()
        c0, c1 = int(counts.get(0, 0)), int(counts.get(1, 0))

        fold_thresholds.append(thr)
        fold_counts.append((c0, c1))
        folds_used += 1

    overall_counts = y_split_use.value_counts().to_dict()
    mean_c0 = float(np.mean([c0 for c0, _ in fold_counts])) if fold_counts else float("nan")
    mean_c1 = float(np.mean([c1 for _, c1 in fold_counts])) if fold_counts else float("nan")

    return {
        "is_binary": False,
        "n": int(y_split_use.shape[0]),
        "threshold_global": thr_global,
        "threshold_mean": float(np.mean(fold_thresholds)) if fold_thresholds else float("nan"),
        "threshold_std": float(np.std(fold_thresholds)) if fold_thresholds else float("nan"),
        "overall_counts_0": int(overall_counts.get(0, 0)),
        "overall_counts_1": int(overall_counts.get(1, 0)),
        "test_counts0_mean": mean_c0,
        "test_counts1_mean": mean_c1,
        "folds_used": int(folds_used),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict self-reports from gaze features (binary classification).")

    src = parser.add_argument_group("Input")
    src.add_argument("--input", type=str, default=None, help="Merged input table (.csv/.parquet) containing features + targets")
    src.add_argument("--gaze-csv", type=str, default=None, help="Gaze trial metrics CSV")
    src.add_argument("--responses-csv", type=str, default=None, help="Responses CSV")

    feat = parser.add_argument_group("Features")
    feat.add_argument("--features", type=str, default=None, help="Comma-separated feature columns (optional). If omitted, auto-detect.")

    filt = parser.add_argument_group("Data filter")
    filt.add_argument(
        "--condition",
        choices=list(CONDITION_CHOICES),
        default="all",
        help="Which condition subset to include: all/correct/incorrect/no_reasoning/no_answer",
    )
    filt.add_argument(
        "--conditions",
        type=str,
        default=None,
        help="Comma-separated list of conditions to run (e.g., 'all,correct'). Overrides --condition.",
    )
    filt.add_argument(
        "--include-condition-as-feature",
        action="store_true",
        help="Add condition (classified_condition) as one-hot features for training.",
    )

    mdl = parser.add_argument_group("Models")
    mdl.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names to run. Available: logreg, rf. Default runs all.",
    )

    grid = parser.add_argument_group("Hyperparameters")
    grid.add_argument(
        "--grid-config",
        type=str,
        default=None,
        help="Path to JSON file overriding parameter grids per model.",
    )

    cv = parser.add_argument_group("Validation")
    cv.add_argument("--cv", choices=["kfold", "loso"], default="kfold")
    cv.add_argument("--n-splits", type=int, default=5, help="Number of folds for kfold")
    cv.add_argument("--group-col", type=str, default="participant_id", help="Group column for LOSO")

    tune = parser.add_argument_group("Tuning")
    tune.add_argument("--tuning", choices=["none", "grid", "random"], default="none")
    tune.add_argument("--inner-splits", type=int, default=3, help="Inner CV folds for tuning")
    tune.add_argument("--n-iter", type=int, default=25, help="n_iter for RandomizedSearchCV")
    tune.add_argument("--scoring", type=str, default="f1", help="Scoring metric for tuning")

    par = parser.add_argument_group("Parallelism")
    par.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for tuning (Grid/RandomizedSearchCV). Use 1 for stability; -1 uses all cores.",
    )
    par.add_argument(
        "--joblib-backend",
        choices=["auto", "loky", "threading"],
        default="auto",
        help="joblib backend for tuning parallelism. On macOS, auto defaults to threading.",
    )
    par.add_argument(
        "--joblib-temp-folder",
        type=str,
        default="auto",
        help="Temp folder for joblib memmaps (auto uses system temp; none disables custom temp folder).",
    )

    ycfg = parser.add_argument_group("Targets")
    ycfg.add_argument(
        "--target",
        type=str,
        default=None,
        help="Run only a single target (e.g., trust_info). If omitted, runs all available targets.",
    )
    ycfg.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated list of targets to run (alternative to --target).",
    )
    ycfg.add_argument("--quantile", type=float, default=0.6, help="Quantile for high/low split (for continuous targets)")
    ycfg.add_argument(
        "--threshold-mode",
        choices=["global", "train"],
        default="train",
        help="How to compute thresholds for continuous targets (train is stricter).",
    )

    out = parser.add_argument_group("Output")
    out.add_argument("--outdir", type=str, default="analysis_results/ml_prediction", help="Output directory")
    out.add_argument("--random-state", type=int, default=42)
    out.add_argument("--preview-only", action="store_true", help="Show threshold preview then exit before training")
    out.add_argument(
        "--pre-review",
        action="store_true",
        help="Pre-review mode: prints+saves available features and target split stats, then exits without training.",
    )

    args = parser.parse_args()

    joblib_backend = _resolve_joblib_backend(str(args.joblib_backend))
    joblib_temp_folder = _resolve_joblib_temp_folder(str(args.joblib_temp_folder))
    n_jobs = int(args.n_jobs)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- Load / build dataset ----------------
    if args.input:
        df = load_from_input_table(Path(args.input))
    else:
        if not (args.gaze_csv and args.responses_csv):
            raise SystemExit("Provide either --input OR both --gaze-csv and --responses-csv")
        df, _ = build_merged_dataset(Path(args.gaze_csv), Path(args.responses_csv))
        df = add_derived_self_report_columns(df)

    condition_feature_cols: List[str] = []
    if bool(getattr(args, "include_condition_as_feature", False)):
        df, condition_feature_cols = _add_condition_one_hot_features(df)

    if "participant_id" not in df.columns:
        raise SystemExit("participant_id column is required")

    # ---------------- Resolve which conditions to run ----------------
    if args.conditions:
        conditions_to_run = [c.strip().lower() for c in str(args.conditions).split(",") if c.strip()]
    else:
        conditions_to_run = [str(args.condition).strip().lower()]

    unknown_c = [c for c in conditions_to_run if c not in set(CONDITION_CHOICES)]
    if unknown_c:
        raise SystemExit(f"Unknown condition(s): {unknown_c}. Expected: {list(CONDITION_CHOICES)}")

    # ---------------- Resolve targets ----------------
    resolved_targets: Dict[str, str] = {}
    for spec in TARGETS:
        col = resolve_target_column(df, spec)
        if col is None:
            print(f"⚠️  Target '{spec.name}' not found (candidates={spec.candidates}); skipping")
            continue
        resolved_targets[spec.name] = col

    if not resolved_targets:
        raise SystemExit("No targets found in the data.")

    chosen_targets = parse_targets_arg(args.target, args.targets)
    if chosen_targets:
        unknown_t = [t for t in chosen_targets if t not in resolved_targets]
        if unknown_t:
            raise SystemExit(f"Unknown/unavailable target(s): {unknown_t}. Available: {sorted(resolved_targets.keys())}")
        resolved_targets = {k: v for k, v in resolved_targets.items() if k in set(chosen_targets)}

    if not resolved_targets:
        raise SystemExit("No targets selected/found after filtering.")

    target_cols = list(resolved_targets.values())

    # ---------------- Resolve features ----------------
    explicit_features = parse_feature_list_arg(args.features)
    selected_models = parse_name_list_arg(args.models)
    feature_cols = infer_feature_columns(df, explicit_features, target_cols=target_cols)

    if bool(getattr(args, "include_condition_as_feature", False)):
        for c in condition_feature_cols:
            if c not in feature_cols:
                feature_cols.append(c)

    # Ensure numeric X
    # Condition-indicator features are only kept for condition='moe'.
    feature_cols_base = _union_feature_cols_across_conditions(feature_cols, conditions=conditions_to_run)
    X_all = df[feature_cols_base].apply(pd.to_numeric, errors="coerce")

    group_series = None
    if args.cv == "loso":
        if args.group_col not in df.columns:
            raise SystemExit(f"LOSO requires group column '{args.group_col}' in the data")
        group_series = df[args.group_col].astype(str)

    # ---------------- Evaluate ----------------
    # Avoid oversubscription: when tuning in parallel, keep RF itself single-threaded.
    rf_n_jobs = 1 if (str(args.tuning) != "none" and n_jobs != 1) else n_jobs
    models_all = build_models(random_state=int(args.random_state), rf_n_jobs=rf_n_jobs)
    if selected_models:
        unknown = [m for m in selected_models if m not in models_all]
        if unknown:
            raise SystemExit(f"Unknown model(s): {unknown}. Available: {sorted(models_all.keys())}")
        models = {k: v for k, v in models_all.items() if k in selected_models}
    else:
        models = models_all
    base_grids = build_param_grids()
    user_grids = load_param_grid_config(args.grid_config)
    param_grids = merge_param_grids(base_grids, user_grids)

    # ---------------- Pre-review mode ----------------
    # Suffix is used for output filenames. If multiple conditions are run, store them in one file.
    suffix_condition = conditions_to_run[0] if len(conditions_to_run) == 1 else "multi"
    run_suffix = _make_run_suffix(condition=str(suffix_condition), target_names=list(resolved_targets.keys()))
    if args.pre_review:
        print("=" * 70)
        print(f"Pre-review | condition={args.condition} | targets={list(resolved_targets.keys())}")
        print(f"Detected features: {len(feature_cols)}")

        feat_report = _feature_report(df, feature_cols)
        out_feat = outdir / f"pre_review_available_features{run_suffix}.csv"
        feat_report.to_csv(out_feat, index=False)
        print("Saved feature list to:", out_feat)

        split_df = _target_split_preview_table(
            X=X_all,
            df=df,
            resolved_targets=resolved_targets,
            cv_mode=str(args.cv),
            n_splits=int(args.n_splits),
            random_state=int(args.random_state),
            quantile=float(args.quantile),
            threshold_mode=str(args.threshold_mode),
            groups=group_series,
        )
        out_split = outdir / f"pre_review_target_splits{run_suffix}.csv"
        split_df.to_csv(out_split, index=False)
        print("Saved target split stats to:", out_split)

        # Print a compact console summary
        for _, r in split_df.iterrows():
            print(
                f"  {r['target']}: n={int(r['n'])} pos_rate={r['overall_pos_rate']:.3f} "
                f"raw_mean={r['raw_mean']:.3f} (low={r['raw_mean_low']:.3f}, high={r['raw_mean_high']:.3f})"
            )

        print("Pre-review complete. Exiting without training.")
        return

    rows: List[Dict[str, object]] = []
    fold_rows_all: List[Dict[str, object]] = []

    # Always evaluate requested conditions independently.
    # (Condition features, if any, are only used in the MoE run.)
    conditions_iter = conditions_to_run
    df_by_condition = {}

    for cond_name in conditions_iter:
        # ---------------- Filter by condition (optional) ----------------
        if df_by_condition:
            df_cond = df_by_condition.get(cond_name, pd.DataFrame())
        else:
            try:
                df_cond = filter_by_condition(df, cond_name)
            except Exception as e:
                raise SystemExit(str(e))

        if df_cond.empty:
            print(f"⚠️  No rows left after condition filter: {cond_name}; skipping")
            continue

        feature_cols_run = _prune_condition_indicator_features(feature_cols, cond_name=cond_name)
        X_all_cond = df_cond[feature_cols_run].apply(pd.to_numeric, errors="coerce")

        group_series_cond = None
        if args.cv == "loso":
            group_series_cond = df_cond[args.group_col].astype(str)

        for target_name, col in resolved_targets.items():
            y_cont = df_cond[col]

            # User-requested rule: no_answer slice does not have trust labels -> skip.
            if str(cond_name).strip().lower() == "no_answer" and str(target_name) in {"trust_info", "trust_sys"}:
                print(f"[skip] condition=no_answer target={target_name}: labels unavailable", flush=True)
                continue

            # quick info
            y_num = pd.to_numeric(y_cont, errors="coerce")
            n_avail = int(y_num.notna().sum())
            print("=" * 70)
            print(f"Target: {target_name} (column='{col}') | available={n_avail}")

            preview = _preview_threshold_splits(
                X=X_all_cond,
                y_cont=y_num,
                cv_mode=args.cv,
                n_splits=int(args.n_splits),
                random_state=int(args.random_state),
                quantile=float(args.quantile),
                threshold_mode=str(args.threshold_mode),
                groups=group_series_cond,
            )
            if preview.get("is_binary"):
                print(
                    f"Target {target_name}: already-binary labels | "
                    f"n={preview['n']} (0={preview['overall_counts_0']}, 1={preview['overall_counts_1']})"
                )
            else:
                print(
                    f"Target {target_name}: quantile={args.quantile}, mode={args.threshold_mode} | "
                    f"threshold_mean={preview['threshold_mean']:.4f}±{preview['threshold_std']:.4f} | "
                    f"overall (0={preview['overall_counts_0']}, 1={preview['overall_counts_1']}); "
                    f"avg test fold counts (0≈{preview['test_counts0_mean']:.1f}, 1={preview['test_counts1_mean']:.1f})"
                )

            if args.preview_only:
                print("Preview-only mode: skipping training for this target.")
                continue

            for model_name, model in models.items():
                try:
                    if args.tuning == "none":
                        metrics = evaluate_binary_classification(
                            X=X_all_cond,
                            y_cont=y_num,
                            model=model,
                            cv_mode=args.cv,
                            n_splits=int(args.n_splits),
                            random_state=int(args.random_state),
                            threshold_quantile=float(args.quantile),
                            threshold_mode=str(args.threshold_mode),
                            groups=group_series_cond,
                        )
                        per_fold = []
                    else:
                        metrics, per_fold = evaluate_binary_classification_nested(
                            X=X_all_cond,
                            y_cont=y_num,
                            model=model,
                            param_grid=param_grids.get(model_name, {}),
                            cv_mode=args.cv,
                            n_splits=int(args.n_splits),
                            random_state=int(args.random_state),
                            threshold_quantile=float(args.quantile),
                            threshold_mode=str(args.threshold_mode),
                            groups=group_series_cond,
                            tuning=str(args.tuning),
                            scoring=str(args.scoring),
                            inner_splits=int(args.inner_splits),
                            n_iter=int(args.n_iter),
                            n_jobs=n_jobs,
                            joblib_backend=joblib_backend,
                            joblib_temp_folder=joblib_temp_folder,
                        )

                    row = {
                        "condition": str(cond_name),
                        "target": target_name,
                        "target_column": col,
                        "model": model_name,
                        "cv": args.cv,
                        "n_splits": int(args.n_splits) if args.cv == "kfold" else None,
                        "tuning": str(args.tuning),
                        "inner_splits": int(args.inner_splits) if args.tuning != "none" else None,
                        # "scoring": str(args.scoring) if args.tuning != "none" else None,
                        "threshold_quantile": float(args.quantile),
                        # "threshold_mode": str(args.threshold_mode),
                        "n_features": int(X_all_cond.shape[1]),
                        **metrics,
                    }
                    rows.append(row)

                    for fr in per_fold:
                        fold_rows_all.append(
                            {
                                "condition": str(cond_name),
                                "target": target_name,
                                "target_column": col,
                                "model": model_name,
                                "cv": args.cv,
                                "tuning": str(args.tuning),
                                "inner_splits": int(args.inner_splits),
                                **fr,
                            }
                        )

                    print(
                        f"  {model_name}: Acc={metrics['accuracy_mean']:.3f}±{metrics['accuracy_std']:.3f} "
                        f"F1={metrics['f1_mean']:.3f}±{metrics['f1_std']:.3f} "
                        f"P={metrics['precision_mean']:.3f} R={metrics['recall_mean']:.3f} "
                        f"AUC={metrics['auc_mean']:.3f} "
                        f"(folds={int(metrics['folds_used'])})"
                    )
                except Exception as e:
                    print(f"  ⚠️  {model_name} failed: {e}")

    if args.preview_only:
        print("Preview-only mode complete. No training results saved.")
        return

    results = pd.DataFrame(rows)
    out_csv = outdir / f"ml_binary_self_reports_metrics{run_suffix}.csv"
    results.to_csv(out_csv, index=False)
    print("\nSaved metrics to:", out_csv)

    if fold_rows_all:
        folds_df = pd.DataFrame(fold_rows_all)
        out_folds = outdir / f"ml_binary_self_reports_folds{run_suffix}.csv"
        folds_df.to_csv(out_folds, index=False)
        print("Saved per-fold details to:", out_folds)

    # Ranking table (per target within this run)
    if not results.empty and "f1_mean" in results.columns:
        rank_cols = [
            "target",
            "model",
            "f1_mean",
            "accuracy_mean",
            "precision_mean",
            "recall_mean",
            "auc_mean",
            "f1_std",
            "auc_std",
        ]
        for c in ["condition", "threshold_quantile", "threshold_mode", "cv", "tuning"]:
            if c in results.columns and c not in rank_cols:
                rank_cols.insert(1, c)
        rank_df = results.copy()
        rank_df["condition"] = str(args.condition)
        rank_df = rank_df[[c for c in rank_cols if c in rank_df.columns]].sort_values(
            ["target", "f1_mean"], ascending=[True, False]
        )
        out_rank = outdir / f"ml_binary_self_reports_ranking{run_suffix}.csv"
        rank_df.to_csv(out_rank, index=False)
        print("Saved ranking table to:", out_rank)

    out_feats = outdir / "ml_binary_self_reports_used_features.json"
    feats_for_file = _union_feature_cols_across_conditions(feature_cols, conditions=conditions_iter)
    out_feats.write_text(json.dumps({"features": feats_for_file}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Saved used feature list to:", out_feats)


def run_ml_prediction(
    *,
    input_path: Optional[Union[str, Path]] = None,
    gaze_csv: Optional[Union[str, Path]] = None,
    responses_csv: Optional[Union[str, Path]] = None,
    demographic_csv: Optional[Union[str, Path]] = None,
    feature_set: Union[str, Sequence[str]] = "gaze",
    features: Optional[Sequence[str]] = None,
    models: Optional[Sequence[str]] = None,
    param_grids: Optional[Dict[str, Dict[str, List[Any]]]] = None,
    targets: Optional[Sequence[str]] = None,
    condition: Union[str, Sequence[str]] = "all",
    pre_review: bool = False,
    preview_only: bool = False,
    cv: str = "kfold",
    n_splits: int = 5,
    group_col: str = "participant_id",
    quantile: float = 0.6,
    threshold_mode: str = "train",
    tuning: str = "none",
    moe_tuning: Optional[str] = None,
    inner_splits: int = 3,
    moe_inner_splits: Optional[int] = None,
    n_iter: int = 25,
    moe_n_iter: Optional[int] = None,
    scoring: str = "accuracy",
    random_state: int = 42,
    n_jobs: int = 1,
    joblib_backend: str = "auto",
    joblib_temp_folder: str = "auto",
    outdir: Union[str, Path] = "analysis_results/ml_prediction",
    save_outputs: bool = True,
    print_progress: bool = True,
    save_partial: bool = True,
    include_condition_as_feature: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Notebook-friendly API.

    Returns (metrics_df, folds_df, feature_cols).
    folds_df is empty if tuning='none'.

        Notes:
        - For condition='moe', you can override tuning behavior with `moe_tuning`
            (and optionally `moe_inner_splits` / `moe_n_iter`) to avoid expensive
            nested hyperparameter search per expert.
    """

    outdir_path = Path(outdir)
    if save_outputs:
        outdir_path.mkdir(parents=True, exist_ok=True)

    if input_path:
        df_raw = load_from_input_table(Path(input_path))
    else:
        if not (gaze_csv and responses_csv):
            raise ValueError("Provide either input_path OR both gaze_csv and responses_csv")
        df_raw, _ = build_merged_dataset(Path(gaze_csv), Path(responses_csv))
        df_raw = add_derived_self_report_columns(df_raw)

    df = df_raw

    # Normalize feature_set(s) (allow running multiple modes in one call).
    if isinstance(feature_set, str):
        feature_sets_to_run = [feature_set]
    else:
        feature_sets_to_run = list(feature_set)
    feature_sets_to_run = [str(fs).strip().lower() for fs in feature_sets_to_run if str(fs).strip()]
    if not feature_sets_to_run:
        feature_sets_to_run = ["gaze"]
    unknown_fs = [fs for fs in feature_sets_to_run if fs not in {"gaze", "demo", "all"}]
    if unknown_fs:
        raise ValueError("feature_set must be one of: gaze/demo/all")

    demo_feature_cols: List[str] = []
    if demographic_csv is not None:
        demo_df, demo_feature_cols = load_demographic_features(Path(demographic_csv))
        if "participant_id" not in df.columns:
            raise ValueError("Merged data missing participant_id; required to merge demographics")
        df = df.copy()
        df["participant_id"] = df["participant_id"].map(_norm_pid)
        # Left join: keep gaze/response rows; demographic features may be missing for some participants.
        df = pd.merge(
            df,
            demo_df,
            on="participant_id",
            how="left",
            validate="m:1",
        )
    condition_feature_cols: List[str] = []
    if bool(include_condition_as_feature):
        df, condition_feature_cols = _add_condition_one_hot_features(df)

    # Allow running multiple conditions in one call.
    if isinstance(condition, str):
        conditions_to_run = [condition]
    else:
        conditions_to_run = list(condition)
    conditions_to_run = [str(c).strip().lower() for c in conditions_to_run if str(c).strip()]

    unknown_c = [c for c in conditions_to_run if c not in set(CONDITION_CHOICES)]
    if unknown_c:
        raise ValueError(f"Unknown condition(s): {unknown_c}. Expected: {list(CONDITION_CHOICES)}")

    resolved_targets: Dict[str, str] = {}
    for spec in TARGETS:
        col = resolve_target_column(df, spec)
        if col is not None:
            resolved_targets[spec.name] = col

    if not resolved_targets:
        raise ValueError("No targets found in the data.")

    if targets:
        unknown = [t for t in targets if t not in resolved_targets]
        if unknown:
            raise ValueError(f"Unknown/unavailable target(s): {unknown}. Available: {sorted(resolved_targets.keys())}")
        resolved_targets = {k: v for k, v in resolved_targets.items() if k in set(targets)}

    target_cols = list(resolved_targets.values())

    def _feature_cols_for_mode(mode: str) -> List[str]:
        m = str(mode).strip().lower()
        if m == "demo":
            if demographic_csv is None:
                raise ValueError("feature_set='demo' requires demographic_csv")
            if not demo_feature_cols:
                raise ValueError("No demographic feature columns detected from demographic_csv")
            return list(demo_feature_cols)

        # gaze or all
        base = infer_feature_columns(df, explicit_features=features, target_cols=target_cols)
        if m == "all" and demographic_csv is not None:
            for c in demo_feature_cols:
                if c not in base:
                    base.append(c)
        return list(base)

    # We'll run one or more feature modes; keep a union for the return value.
    feature_cols_union: List[str] = []

    # Note: condition one-hot columns are appended per feature_set run (after selecting base features).
    # Note: condition suffix is computed per feature_set run (demo may prune conditions).

    backend_resolved = _resolve_joblib_backend(joblib_backend)
    temp_folder_resolved = _resolve_joblib_temp_folder(joblib_temp_folder)
    n_jobs_int = int(n_jobs)
    rf_n_jobs = 1 if (str(tuning) != "none" and n_jobs_int != 1) else n_jobs_int

    models_all = build_models(random_state=int(random_state), rf_n_jobs=rf_n_jobs)
    if models:
        unknown = [m for m in models if m not in models_all]
        if unknown:
            raise ValueError(f"Unknown model(s): {unknown}. Available: {sorted(models_all.keys())}")
        models_use = {k: v for k, v in models_all.items() if k in set(models)}
    else:
        models_use = models_all
    base_grids = build_param_grids()
    param_grids_final = merge_param_grids(base_grids, param_grids)

    rows: List[Dict[str, object]] = []
    fold_rows_all: List[Dict[str, object]] = []

    def _flush_partial_outputs(*, out_metrics_path: Path, out_folds_path: Path, out_rank_path: Path, out_best_path: Path, out_feats_path: Path, features_for_file: Sequence[str]) -> None:
        if not save_outputs or not save_partial:
            return

        pd.DataFrame(rows).to_csv(out_metrics_path, index=False)
        if fold_rows_all:
            pd.DataFrame(fold_rows_all).to_csv(out_folds_path, index=False)

        # Ranking + best (only if proper metric columns exist)
        if rows and isinstance(rows[0], dict) and "f1_mean" in pd.DataFrame(rows).columns:
            mdf = pd.DataFrame(rows)
            if not mdf.empty and "f1_mean" in mdf.columns and "target" in mdf.columns and "model" in mdf.columns:
                try:
                    rank_df = make_model_ranking_table(mdf, score_col="f1_mean", higher_is_better=True)
                    rank_df.to_csv(out_rank_path, index=False)
                    if "condition" in rank_df.columns:
                        group_cols = ["condition", "target"]
                        if "feature_set" in rank_df.columns:
                            group_cols.insert(0, "feature_set")
                        best_df = (
                            rank_df.groupby(group_cols, as_index=False, sort=False)
                            .head(1)
                            .reset_index(drop=True)
                        )
                    else:
                        best_df = rank_df.groupby(["target"], as_index=False, sort=False).head(1).reset_index(drop=True)
                    best_df.to_csv(out_best_path, index=False)
                except Exception:
                    pass

        out_feats_path.write_text(
            json.dumps({"features": list(features_for_file)}, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    if pre_review:
        # Pre-review for each condition.
        pre_rows: List[pd.DataFrame] = []
        # Pre-review uses the first feature_set only.
        feature_cols_pre = _feature_cols_for_mode(feature_sets_to_run[0])
        if bool(include_condition_as_feature) and str(feature_sets_to_run[0]).strip().lower() in {"gaze", "all"}:
            for c in condition_feature_cols:
                if c not in feature_cols_pre:
                    feature_cols_pre.append(c)
        for cond_name in conditions_to_run:
            df_cond = filter_by_condition(df, cond_name)
            if df_cond.empty:
                continue
            feature_cols_run = _prune_condition_indicator_features(feature_cols_pre, cond_name=cond_name)
            X_cond = df_cond[feature_cols_run].apply(pd.to_numeric, errors="coerce")
            group_series_cond = None
            if cv == "loso":
                if group_col not in df_cond.columns:
                    raise ValueError(f"LOSO requires group column '{group_col}'")
                group_series_cond = df_cond[group_col].astype(str)
            split_df = _target_split_preview_table(
                X=X_cond,
                df=df_cond,
                resolved_targets=resolved_targets,
                cv_mode=cv,
                n_splits=int(n_splits),
                random_state=int(random_state),
                quantile=float(quantile),
                threshold_mode=str(threshold_mode),
                groups=group_series_cond,
            )
            split_df.insert(0, "condition", str(cond_name))
            pre_rows.append(split_df)
        out = pd.concat(pre_rows, ignore_index=True) if pre_rows else pd.DataFrame()
        if save_outputs:
            out_path = outdir_path / f"pre_review_target_splits{run_suffix}.csv"
            out.to_csv(out_path, index=False)
        return out, pd.DataFrame(), list(feature_cols_pre)

    # Run training for each feature mode.
    for feature_mode in feature_sets_to_run:
        feature_cols = _feature_cols_for_mode(feature_mode)
        if bool(include_condition_as_feature) and str(feature_mode).strip().lower() in {"gaze", "all"}:
            for c in condition_feature_cols:
                if c not in feature_cols:
                    feature_cols.append(c)

        # For demo-only features, condition slicing is often redundant because features are
        # participant-level and thus identical across condition rows. We keep a single condition
        # run (prefer 'all' if present) to save time unless the user explicitly passed one.
        conditions_to_run_mode = list(conditions_to_run)
        if str(feature_mode).strip().lower() == "demo" and len(conditions_to_run_mode) > 1:
            if "all" in set(conditions_to_run_mode):
                conditions_to_run_mode = ["all"]
            else:
                conditions_to_run_mode = [conditions_to_run_mode[0]]
            if print_progress:
                print(
                    f"[info] feature_set=demo uses participant-level features; collapsing conditions to: {conditions_to_run_mode}",
                    flush=True,
                )

        # If condition is used as a feature and user selected multiple conditions,
        # train a single combined model on the union (condition signal comes from the one-hot columns).
        has_moe = "moe" in set(conditions_to_run_mode)

        # Track features that may actually be used. When multiple conditions are run,
        # condition-indicator columns are only kept for condition='moe'.
        feature_cols_union = _union_feature_cols_across_conditions(
            feature_cols_union + feature_cols,
            conditions=conditions_to_run_mode,
        )

        if bool(include_condition_as_feature) and (len(conditions_to_run_mode) > 1 and "all" not in set(conditions_to_run_mode)) and (not has_moe):
            parts: List[pd.DataFrame] = []
            for c in conditions_to_run_mode:
                parts.append(filter_by_condition(df, c))
            df_sel = pd.concat(parts, axis=0)
            df_sel = df_sel.loc[~df_sel.index.duplicated(keep="first")]
            conditions_iter = ["multi"]
            df_by_condition = {"multi": df_sel}
        else:
            conditions_iter = conditions_to_run_mode
            df_by_condition = {c: (df if c == "moe" else filter_by_condition(df, c)) for c in conditions_iter}

        suffix_condition = conditions_to_run_mode[0] if len(conditions_to_run_mode) == 1 else "multi"
        base_suffix = _make_run_suffix(condition=str(suffix_condition), target_names=list(resolved_targets.keys()))
        run_suffix = base_suffix + f"__feat-{str(feature_mode).strip().lower()}"
        out_metrics_path = outdir_path / f"ml_binary_self_reports_metrics{run_suffix}.csv"
        out_folds_path = outdir_path / f"ml_binary_self_reports_folds{run_suffix}.csv"
        out_rank_path = outdir_path / f"ml_binary_self_reports_ranking{run_suffix}.csv"
        out_best_path = outdir_path / f"ml_binary_self_reports_best{run_suffix}.csv"
        out_feats_path = outdir_path / f"ml_binary_self_reports_used_features{run_suffix}.json"

        for cond_name in conditions_iter:
            df_cond = df_by_condition.get(cond_name, pd.DataFrame())
            if df_cond.empty:
                if print_progress:
                    print(f"[skip] feature_set={feature_mode} condition={cond_name}: no rows", flush=True)
                continue

            feature_cols_run = _prune_condition_indicator_features(feature_cols, cond_name=cond_name)
            X_all = df_cond[feature_cols_run].apply(pd.to_numeric, errors="coerce")

            group_series = None
            if cv == "loso":
                if group_col not in df_cond.columns:
                    raise ValueError(f"LOSO requires group column '{group_col}'")
                group_series = df_cond[group_col].astype(str)

            for target_name, col in resolved_targets.items():
                y_num = pd.to_numeric(df_cond[col], errors="coerce")

                # no_answer slice: trust labels are typically missing -> skip those targets.
                if str(cond_name).strip().lower() == "no_answer" and str(target_name) in {"trust_info", "trust_sys"}:
                    if print_progress:
                        print(f"[skip] condition=no_answer target={target_name}: labels unavailable", flush=True)
                    continue

                if print_progress:
                    n_avail = int(pd.Series(y_num).notna().sum())
                    print(
                        f"[start] feature_set={feature_mode} condition={cond_name} target={target_name} n={n_avail}",
                        flush=True,
                    )

                preview = _preview_threshold_splits(
                    X=X_all,
                    y_cont=y_num,
                    cv_mode=cv,
                    n_splits=int(n_splits),
                    random_state=int(random_state),
                    quantile=float(quantile),
                    threshold_mode=str(threshold_mode),
                    groups=group_series,
                )

                if preview_only:
                    preview_info = {
                        "feature_set": str(feature_mode),
                        "condition": str(cond_name),
                        "target": target_name,
                        "is_binary": preview.get("is_binary", False),
                        "n": preview.get("n"),
                        "threshold_mean": preview.get("threshold_mean"),
                        "threshold_std": preview.get("threshold_std"),
                        "overall_counts_0": preview.get("overall_counts_0"),
                        "overall_counts_1": preview.get("overall_counts_1"),
                        "test_counts0_mean": preview.get("test_counts0_mean"),
                        "test_counts1_mean": preview.get("test_counts1_mean"),
                    }
                    rows.append(preview_info)
                    _flush_partial_outputs(
                        out_metrics_path=out_metrics_path,
                        out_folds_path=out_folds_path,
                        out_rank_path=out_rank_path,
                        out_best_path=out_best_path,
                        out_feats_path=out_feats_path,
                        features_for_file=feature_cols_run,
                    )
                    continue

                for model_name, model in models_use.items():
                    if print_progress:
                        print(
                            f"  [fit] feature_set={feature_mode} condition={cond_name} target={target_name} model={model_name}",
                            flush=True,
                        )

                    is_moe = str(cond_name).strip().lower() == "moe"
                    tuning_eff = str(moe_tuning) if (is_moe and moe_tuning is not None) else str(tuning)
                    inner_splits_eff = int(moe_inner_splits) if (is_moe and moe_inner_splits is not None) else int(inner_splits)
                    n_iter_eff = int(moe_n_iter) if (is_moe and moe_n_iter is not None) else int(n_iter)

                    error_msg: Optional[str] = None
                    try:
                        if is_moe:
                            if "classified_condition" not in df_cond.columns:
                                raise ValueError("condition='moe' requires column 'classified_condition' for routing")
                            cond_series = df_cond["classified_condition"].astype(str)

                            if tuning_eff == "none":
                                metrics = evaluate_binary_classification_moe(
                                    X=X_all,
                                    y_cont=y_num,
                                    condition_series=cond_series,
                                    model=model,
                                    cv_mode=cv,
                                    n_splits=int(n_splits),
                                    random_state=int(random_state),
                                    threshold_quantile=float(quantile),
                                    threshold_mode=str(threshold_mode),
                                    groups=group_series,
                                )
                                per_fold = []
                            else:
                                metrics, per_fold = evaluate_binary_classification_nested_moe(
                                    X=X_all,
                                    y_cont=y_num,
                                    condition_series=cond_series,
                                    model=model,
                                    param_grid=param_grids_final.get(model_name, {}),
                                    cv_mode=cv,
                                    n_splits=int(n_splits),
                                    random_state=int(random_state),
                                    threshold_quantile=float(quantile),
                                    threshold_mode=str(threshold_mode),
                                    groups=group_series,
                                    tuning=str(tuning_eff),
                                    scoring=str(scoring),
                                    inner_splits=int(inner_splits_eff),
                                    n_iter=int(n_iter_eff),
                                    n_jobs=n_jobs_int,
                                    joblib_backend=backend_resolved,
                                    joblib_temp_folder=temp_folder_resolved,
                                )
                        else:
                            if tuning == "none":
                                metrics = evaluate_binary_classification(
                                    X=X_all,
                                    y_cont=y_num,
                                    model=model,
                                    cv_mode=cv,
                                    n_splits=int(n_splits),
                                    random_state=int(random_state),
                                    threshold_quantile=float(quantile),
                                    threshold_mode=str(threshold_mode),
                                    groups=group_series,
                                )
                                per_fold = []
                            else:
                                metrics, per_fold = evaluate_binary_classification_nested(
                                    X=X_all,
                                    y_cont=y_num,
                                    model=model,
                                    param_grid=param_grids_final.get(model_name, {}),
                                    cv_mode=cv,
                                    n_splits=int(n_splits),
                                    random_state=int(random_state),
                                    threshold_quantile=float(quantile),
                                    threshold_mode=str(threshold_mode),
                                    groups=group_series,
                                    tuning=str(tuning),
                                    scoring=str(scoring),
                                    inner_splits=int(inner_splits),
                                    n_iter=int(n_iter),
                                    n_jobs=n_jobs_int,
                                    joblib_backend=backend_resolved,
                                    joblib_temp_folder=temp_folder_resolved,
                                )
                    except Exception as e:
                        error_msg = str(e)
                        metrics = {
                            "accuracy_mean": float("nan"),
                            "accuracy_std": float("nan"),
                            "f1_mean": float("nan"),
                            "f1_std": float("nan"),
                            "precision_mean": float("nan"),
                            "precision_std": float("nan"),
                            "recall_mean": float("nan"),
                            "recall_std": float("nan"),
                            "auc_mean": float("nan"),
                            "auc_std": float("nan"),
                            "folds_used": float("nan"),
                            "threshold_mean": float("nan"),
                        }
                        per_fold = []
                        if print_progress:
                            print(
                                f"  [fail] feature_set={feature_mode} condition={cond_name} target={target_name} model={model_name}: {error_msg}",
                                flush=True,
                            )

                    rows.append(
                        {
                            "feature_set": str(feature_mode),
                            "condition": str(cond_name),
                            "target": target_name,
                            "target_column": col,
                            "model": model_name,
                            "cv": cv,
                            "n_splits": int(n_splits) if cv == "kfold" else None,
                            "tuning": str(tuning_eff) if is_moe else str(tuning),
                            "inner_splits": (int(inner_splits_eff) if tuning_eff != "none" else None) if is_moe else (int(inner_splits) if tuning != "none" else None),
                            "scoring": str(scoring) if ((tuning_eff != "none") if is_moe else (tuning != "none")) else None,
                            "threshold_quantile": float(quantile),
                            "threshold_mode": str(threshold_mode),
                            "n_features": int(X_all.shape[1]),
                            "error": error_msg,
                            **metrics,
                        }
                    )

                    for fr in per_fold:
                        fold_rows_all.append(
                            {
                                "feature_set": str(feature_mode),
                                "condition": str(cond_name),
                                "target": target_name,
                                "target_column": col,
                                "model": model_name,
                                "cv": cv,
                                "tuning": str(tuning_eff) if is_moe else str(tuning),
                                "inner_splits": int(inner_splits_eff) if is_moe else int(inner_splits),
                                **fr,
                            }
                        )

                    if print_progress and error_msg is None:
                        f1v = metrics.get("f1_mean", float("nan"))
                        accv = metrics.get("accuracy_mean", float("nan"))
                        aucv = metrics.get("auc_mean", float("nan"))
                        folds_used = metrics.get("folds_used", float("nan"))
                        print(
                            f"  [done] feature_set={feature_mode} condition={cond_name} target={target_name} model={model_name} "
                            f"F1={f1v:.3f} Acc={accv:.3f} AUC={aucv:.3f} folds={folds_used}",
                            flush=True,
                        )

                    _flush_partial_outputs(
                        out_metrics_path=out_metrics_path,
                        out_folds_path=out_folds_path,
                        out_rank_path=out_rank_path,
                        out_best_path=out_best_path,
                        out_feats_path=out_feats_path,
                        features_for_file=feature_cols_run,
                    )

    metrics_df = pd.DataFrame(rows)
    folds_df = pd.DataFrame(fold_rows_all) if fold_rows_all else pd.DataFrame()

    if save_outputs:
        # Final flush (also ensures ranking/best are up-to-date).
        # Use a stable union in the used-features file when running multiple modes.
        # (Per-mode files already got per-mode features during the loop.)
        if feature_sets_to_run:
            last_mode = str(feature_sets_to_run[-1]).strip().lower()
            run_suffix = base_suffix + f"__feat-{last_mode}"
            out_metrics_path = outdir_path / f"ml_binary_self_reports_metrics{run_suffix}.csv"
            out_folds_path = outdir_path / f"ml_binary_self_reports_folds{run_suffix}.csv"
            out_rank_path = outdir_path / f"ml_binary_self_reports_ranking{run_suffix}.csv"
            out_best_path = outdir_path / f"ml_binary_self_reports_best{run_suffix}.csv"
            out_feats_path = outdir_path / f"ml_binary_self_reports_used_features{run_suffix}.json"
            _flush_partial_outputs(
                out_metrics_path=out_metrics_path,
                out_folds_path=out_folds_path,
                out_rank_path=out_rank_path,
                out_best_path=out_best_path,
                out_feats_path=out_feats_path,
                features_for_file=feature_cols_union,
            )

    # Return the union to stay backward-compatible with the 3-tuple return.
    return metrics_df, folds_df, feature_cols_union


def make_model_ranking_table(
    metrics_df: pd.DataFrame,
    *,
    score_col: str = "f1_mean",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Create a per-target ranking table from run_ml_prediction() output.

    Returns a DataFrame sorted by (target, score_col) so that the best row for each
    target appears first.
    """

    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame()

    if "target" not in metrics_df.columns or "model" not in metrics_df.columns:
        raise ValueError("metrics_df must contain columns: 'target', 'model'")

    if score_col not in metrics_df.columns:
        raise ValueError(f"score_col '{score_col}' not found in metrics_df columns: {list(metrics_df.columns)}")

    asc = [True, not higher_is_better]
    return (
        metrics_df.copy()
        .sort_values(["target", score_col], ascending=asc, na_position="last")
        .reset_index(drop=True)
    )


def make_best_model_table(
    metrics_df: pd.DataFrame,
    *,
    score_col: str = "f1_mean",
    higher_is_better: bool = True,
) -> pd.DataFrame:
    """Pick the single best model row per target from run_ml_prediction() output."""

    ranked = make_model_ranking_table(metrics_df, score_col=score_col, higher_is_better=higher_is_better)
    if ranked.empty:
        return ranked

    return ranked.groupby("target", as_index=False, sort=False).head(1).reset_index(drop=True)


def compute_feature_importance_table(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    *,
    method: str = "auto",
    scoring: str = "f1",
    n_repeats: int = 20,
    random_state: int = 42,
    background_X: Optional[pd.DataFrame] = None,
    shap_background_size: int = 200,
    shap_sample_size: int = 500,
) -> pd.DataFrame:
    """Compute a feature-importance table for a fitted estimator.

        method:
            - auto: native importance if available; else coef_; else permutation
            - native: require feature_importances_ / coef_ (else error)
            - permutation: use sklearn.inspection.permutation_importance
            - shap: mean(|SHAP|) importance (requires optional dependency 'shap')

    Returns a DataFrame with columns: feature, importance, importance_std, method.
    """

    m = str(method).strip().lower()
    if m not in {"auto", "native", "permutation", "shap"}:
        raise ValueError("method must be one of: auto/native/permutation/shap")

    feat_names = list(X.columns)

    # Unwrap pipeline model step if present.
    model = estimator
    if isinstance(estimator, Pipeline) and "model" in estimator.named_steps:
        model = estimator.named_steps["model"]

    if m == "shap":
        try:
            import shap  # type: ignore
        except Exception as e:
            raise ImportError(
                "method='shap' requires the optional package 'shap'. Install with: pip install shap"
            ) from e

        # Use a background dataset (train fold preferred) and a smaller evaluation sample for speed.
        bg_df = X if background_X is None else background_X
        if not isinstance(bg_df, pd.DataFrame):
            bg_df = pd.DataFrame(bg_df, columns=feat_names)

        # Ensure column alignment.
        bg_df = bg_df.reindex(columns=feat_names)
        X_eval = X.reindex(columns=feat_names)

        rs = np.random.RandomState(int(random_state))

        def _subsample(df_in: pd.DataFrame, n_max: int) -> pd.DataFrame:
            n_max = int(n_max)
            if n_max <= 0 or len(df_in) <= n_max:
                return df_in
            # deterministic sampling
            take_idx = rs.choice(df_in.index.to_numpy(), size=n_max, replace=False)
            return df_in.loc[take_idx]

        bg_df = _subsample(bg_df, int(shap_background_size))
        X_eval = _subsample(X_eval, int(shap_sample_size))

        # Transform through preprocessing steps if this is a Pipeline.
        X_bg_np = bg_df
        X_eval_np = X_eval
        if isinstance(estimator, Pipeline) and "model" in estimator.named_steps:
            pre = estimator[:-1]
            if hasattr(pre, "transform"):
                X_bg_np = pre.transform(bg_df)
                X_eval_np = pre.transform(X_eval)
            else:
                X_bg_np = bg_df.to_numpy()
                X_eval_np = X_eval.to_numpy()
        else:
            X_bg_np = bg_df.to_numpy()
            X_eval_np = X_eval.to_numpy()

        # Prefer fast, model-specific explainers; fall back to permutation SHAP if unsupported.
        def _predict_scalar(x_arr: np.ndarray) -> np.ndarray:
            x_arr = np.asarray(x_arr)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_arr)
                proba = np.asarray(proba)
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, 1]
                if proba.ndim == 2 and proba.shape[1] == 1:
                    return proba[:, 0]
            pred = model.predict(x_arr)
            return np.asarray(pred, dtype=float)

        exp = None

        tree_like = hasattr(model, "feature_importances_") or hasattr(model, "tree_") or hasattr(model, "estimators_")
        linear_like = hasattr(model, "coef_")

        if tree_like:
            try:
                explainer = shap.TreeExplainer(model, data=X_bg_np)
                exp = explainer(X_eval_np)
            except Exception:
                exp = None

        if exp is None and linear_like:
            try:
                explainer = shap.LinearExplainer(model, X_bg_np)
                exp = explainer(X_eval_np)
            except Exception:
                exp = None

        if exp is None:
            # Works for any model; can be slower, so we rely on the sampling above.
            explainer = shap.Explainer(_predict_scalar, X_bg_np, algorithm="permutation")
            exp = explainer(X_eval_np)

        vals_raw = getattr(exp, "values", None)
        if vals_raw is None:
            raise ValueError("Unexpected SHAP output: missing 'values'")

        # SHAP can return:
        # - np.ndarray (n, p)
        # - np.ndarray (n, p, n_classes)
        # - np.ndarray (n_classes, n, p)
        # - list of arrays, length=n_classes, each (n, p)
        if isinstance(vals_raw, list):
            if not vals_raw:
                raise ValueError("Unexpected SHAP output: empty values list")
            vals = np.asarray(vals_raw[1] if len(vals_raw) > 1 else vals_raw[0])
        else:
            vals = np.asarray(vals_raw)

        p = len(feat_names)
        if vals.ndim == 2:
            # OK: (n, p)
            if vals.shape[1] != p and vals.shape[0] == p:
                # Some explainers may return (p, n)
                vals = vals.T
        elif vals.ndim == 3:
            # Handle common layouts for binary classification.
            # (n, p, 2)
            if vals.shape[0] != p and vals.shape[1] == p and vals.shape[2] in {1, 2}:
                cls_idx = 1 if vals.shape[2] > 1 else 0
                vals = vals[:, :, cls_idx]
            # (2, n, p)
            elif vals.shape[2] == p and vals.shape[0] in {1, 2}:
                cls_idx = 1 if vals.shape[0] > 1 else 0
                vals = vals[cls_idx, :, :]
            # (n, 2, p)
            elif vals.shape[2] == p and vals.shape[1] in {1, 2}:
                cls_idx = 1 if vals.shape[1] > 1 else 0
                vals = vals[:, cls_idx, :]
            else:
                raise ValueError(f"Unexpected SHAP values shape: {vals.shape} (expected feature dim={p})")

            # After slicing, ensure (n, p)
            if vals.ndim != 2 or vals.shape[1] != p:
                raise ValueError(f"Unexpected SHAP values shape after normalization: {np.asarray(vals).shape} (expected (*, {p}))")
        else:
            raise ValueError(f"Unexpected SHAP values shape: {vals.shape}")

        abs_vals = np.abs(vals)
        importance = abs_vals.mean(axis=0).astype(float)
        importance_std = abs_vals.std(axis=0).astype(float)

        out = pd.DataFrame(
            {
                "feature": feat_names,
                "importance": importance,
                "importance_std": importance_std,
                "method": "shap_mean_abs",
            }
        )
        return out.sort_values("importance", ascending=False, na_position="last").reset_index(drop=True)

    # Native feature_importances_
    if hasattr(model, "feature_importances_"):
        vals = np.asarray(getattr(model, "feature_importances_"), dtype=float)
        if vals.ndim != 1 or vals.shape[0] != len(feat_names):
            raise ValueError("feature_importances_ shape does not match feature count")
        out = pd.DataFrame(
            {
                "feature": feat_names,
                "importance": vals,
                "importance_std": np.nan,
                "method": "native_feature_importances",
            }
        )
        return out.sort_values("importance", ascending=False, na_position="last").reset_index(drop=True)

    # Linear coef_
    if hasattr(model, "coef_"):
        coef = np.asarray(getattr(model, "coef_"), dtype=float)
        if coef.ndim == 2 and coef.shape[0] == 1:
            coef = coef[0]
        if coef.ndim != 1 or coef.shape[0] != len(feat_names):
            raise ValueError("coef_ shape does not match feature count")
        out = pd.DataFrame(
            {
                "feature": feat_names,
                "importance": np.abs(coef),
                "importance_std": np.nan,
                "method": "native_abs_coef",
            }
        )
        return out.sort_values("importance", ascending=False, na_position="last").reset_index(drop=True)

    if m == "native":
        raise ValueError("Native feature importance not available for this model; use method='permutation'.")

    # Permutation importance works for any estimator with predict.
    pi = permutation_importance(
        estimator,
        X,
        y,
        n_repeats=int(n_repeats),
        random_state=int(random_state),
        scoring=str(scoring),
        n_jobs=1,
    )
    out = pd.DataFrame(
        {
            "feature": feat_names,
            "importance": pi.importances_mean.astype(float),
            "importance_std": pi.importances_std.astype(float),
            "method": "permutation",
        }
    )
    return out.sort_values("importance", ascending=False, na_position="last").reset_index(drop=True)


def plot_feature_importance(
    *,
    input_path: Optional[Union[str, Path]] = None,
    gaze_csv: Optional[Union[str, Path]] = None,
    responses_csv: Optional[Union[str, Path]] = None,
    demographic_csv: Optional[Union[str, Path]] = None,
    feature_set: str = "gaze",  # 'gaze' (only gaze) | 'demo' (only demographics) | 'all' (gaze + demographics)
    targets: Sequence[str],
    models: Sequence[str],
    features: Optional[Sequence[str]] = None,
    condition: Union[str, Sequence[str]] = "all",
    quantile: float = 0.5,
    threshold_mode: str = "global",
    cv: Optional[str] = "loso",
    n_splits: int = 5,
    group_col: str = "participant_id",
    method: str = "auto",
    scoring: str = "f1",
    top_k: int = 12,
    error_bars: str = "std",
    error_scale: float = 1.0,
    ci_z: float = 1.96,
    error_cap: Optional[float] = None,
    group_features: bool = False,
    group_normalize: bool = True,
    n_repeats: int = 20,
    random_state: int = 42,
    outdir: Union[str, Path] = "analysis_results/ml_prediction/feature_importance",
    save: bool = True,
    show: bool = True,
    figsize: Tuple[float, float] = (5.0, 2.2),
    pretty_feature_labels: bool = True,
    bar_height: float = 0.08,
    bar_step: float = 0.10,
    xlim_left: float = 0.0,
) -> Dict[Tuple[str, str, str], pd.DataFrame]:
    """Standalone feature-importance visualization (separate from training).

    Fits the requested models on the requested (condition, target) slices and plots
    top-K feature importances.

    Default behavior uses participant-level CV (default: LOSO) and reports mean importance
    across folds with error bars (± std or 95% CI). AOI group is encoded by bar color.
    Set cv=None to compute a single importance table on the full slice (legacy behavior).

    If group_features=True, features are collapsed into coarse groups (e.g., PPT, AI literacy,
    familiarity, saccade/fixation/pupil) and (optionally) normalized within each (condition,target,model)
    slice. This does not affect the default per-feature plot.

    Returns a dict keyed by (condition, target, model) with the importance table.
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    def _feature_to_group_and_aoi(feature_name: str) -> Tuple[str, str]:
        """Return (group_label, aoi_category) for grouped plots.

        Requirements:
        - Demographics (PPT/literacy/numeracy/familiarity) are grouped across their items.
        - Gaze families (fixation/saccade/pupil) are grouped *within AOI* (same AOI only).
        """

        raw = str(feature_name).strip()
        s = raw.lower()
        aoi = _infer_aoi_group(raw)

        # Demographics / questionnaires (from load_demographic_features)
        # Requirement:
        # - All PPT items -> one bar
        # - All AI literacy items -> one bar
        # - All familiarity items -> one bar
        # - All other demo-related features -> one bar
        if s.startswith("demo__") or s.startswith("demo_cat__"):
            if "ppt" in s:
                return ("PPT", "Demographic")
            if "literacy" in s:
                return ("AI literacy", "Demographic")
            if "familiar" in s or "familiarity" in s:
                return ("Familiarity", "Demographic")
            return ("Demographics (other)", "Demographic")

        # Gaze-derived families grouped within the AOI.
        family: Optional[str] = None
        if "saccade" in s:
            family = "Saccade"
        elif "fixation" in s or "fix_count" in s or "fix_dur" in s:
            family = "Fixation"
        elif "pupil" in s:
            family = "Pupil"

        if family is not None:
            # Keep AOI separation: e.g., "Answer - Fixation"
            return (f"{aoi} - {family}", aoi)

        # Not matched by any grouping rule -> keep original feature name.
        return (raw, aoi)

    def _infer_aoi_group(feature_name: str) -> str:
        s = str(feature_name).strip().lower()
        if s.startswith("demo__") or s.startswith("demo_cat__") or ("ai literacy" in s) or ("ppt" in s):
            return "Demographic"
        # Heuristic mapping (robust across naming conventions)
        if ("claim" in s) or ("evidence" in s) or ("claim_evidence" in s) or ("evidence_claim" in s):
            return "Claim/Evidence"
        # Union AOI (Answer + Reasoning) used in the "advice" view.
        # Must be checked before the generic "answer" / "reason" rules.
        if "answer_reasoning" in s:
            return "Advice"
        if "answer" in s:
            return "Answer"
        if ("rationale" in s) or ("reason" in s):
            return "Rationale"
        if "response" in s:
            return "Response"
        return "Other"

    def _format_feature_label(feature_name: str) -> str:
        s = str(feature_name)
        return s.replace("_", " ") if bool(pretty_feature_labels) else s

    # ---- Load data (same logic as run_ml_prediction) ----
    if input_path:
        df = load_from_input_table(Path(input_path))
    else:
        if not (gaze_csv and responses_csv):
            raise ValueError("Provide either input_path OR both gaze_csv and responses_csv")
        df, _ = build_merged_dataset(Path(gaze_csv), Path(responses_csv))
        df = add_derived_self_report_columns(df)

    # Optional: merge demographics (PPT / AI literacy / Qualtrics demographics)
    demo_feature_cols: List[str] = []
    if demographic_csv is not None:
        demo_df, demo_feature_cols = load_demographic_features(Path(demographic_csv))
        if "participant_id" not in df.columns:
            raise ValueError("Merged data missing participant_id; required to merge demographics")
        df = df.copy()
        df["participant_id"] = df["participant_id"].map(_norm_pid)
        df = pd.merge(
            df,
            demo_df,
            on="participant_id",
            how="left",
            validate="m:1",
        )

    # Normalize condition list
    if isinstance(condition, str):
        conditions_to_run = [condition]
    else:
        conditions_to_run = list(condition)
    conditions_to_run = [str(c).strip().lower() for c in conditions_to_run if str(c).strip()]

    unknown_c = [c for c in conditions_to_run if c not in set(CONDITION_CHOICES)]
    if unknown_c:
        raise ValueError(f"Unknown condition(s): {unknown_c}. Expected: {list(CONDITION_CHOICES)}")

    # Resolve target names to columns (allow passing a raw column name too)
    resolved_targets: Dict[str, str] = {}
    for t in targets:
        tname = str(t).strip()
        if not tname:
            continue
        # Known spec name?
        spec_map = {s.name: s for s in TARGETS}
        if tname in spec_map:
            col = resolve_target_column(df, spec_map[tname])
            if col is None:
                raise ValueError(f"Target '{tname}' not found in the data")
            resolved_targets[tname] = col
        else:
            # Treat as raw column
            if tname not in df.columns:
                raise ValueError(f"Target '{tname}' not found (neither known target name nor a column)")
            resolved_targets[tname] = tname

    if not resolved_targets:
        raise ValueError("No targets provided")

    target_cols = list(resolved_targets.values())

    fs = str(feature_set).strip().lower()
    if fs not in {"gaze", "demo", "all"}:
        raise ValueError("feature_set must be one of: gaze/demo/all")

    # Feature matrix
    # - If `features` is provided: treat it as the user-defined *gaze/base* feature subset (alias-aware)
    # - Else: infer gaze features; then optionally append demographic/literacy/PPT features based on feature_set
    if fs == "demo" and features is not None:
        raise ValueError("Do not pass `features` when feature_set='demo' (demo-only mode).")

    if features is not None:
        requested_feature_cols = [str(c) for c in features]
        feature_cols, _feat_map_unused, missing_feats = resolve_feature_columns(df.columns, requested_feature_cols)
        if missing_feats:
            # Provide a very small hint for common AOI prefixes.
            hint = ""
            if any(m.startswith("answer_") for m in missing_feats) and any(c.startswith("answer_aoi_") for c in df.columns):
                hint += " Try 'answer_aoi_*' columns."
            if any(m.startswith("reasoning_") for m in missing_feats) and any(c.startswith("reasoning_aoi_") for c in df.columns):
                hint += " Try 'reasoning_aoi_*' columns."
            raise ValueError(f"Some features are missing from the data: {missing_feats[:10]}.{hint}")
    else:
        feature_cols = infer_feature_columns(df, explicit_features=None, target_cols=target_cols)

    if fs == "demo":
        if demographic_csv is None:
            raise ValueError("feature_set='demo' requires demographic_csv")
        if not demo_feature_cols:
            raise ValueError("No demographic feature columns detected from demographic_csv")
        feature_cols = list(demo_feature_cols)
    elif fs == "all":
        if demographic_csv is not None:
            for c in demo_feature_cols:
                if c not in feature_cols:
                    feature_cols.append(c)

    # Models
    models_all = build_models(random_state=int(random_state), rf_n_jobs=1)
    unknown_m = [m for m in models if m not in models_all]
    if unknown_m:
        raise ValueError(f"Unknown/unavailable model(s): {unknown_m}. Available: {sorted(models_all.keys())}")

    outdir_path = Path(outdir)
    if save:
        outdir_path.mkdir(parents=True, exist_ok=True)

    results: Dict[Tuple[str, str, str], pd.DataFrame] = {}

    # Cleaner base style; we add a subtle x-grid manually.
    plt.style.use("seaborn-v0_8-white")

    err_mode = str(error_bars).strip().lower()
    if err_mode not in {"std", "ci", "ci95", "95ci", "none"}:
        raise ValueError("error_bars must be one of: std/ci95/none")

    # Visual adjustment for error-bar lengths (does not affect ranking/means).
    try:
        error_scale_f = float(error_scale)
    except Exception:
        error_scale_f = 1.0
    if (not np.isfinite(error_scale_f)) or error_scale_f <= 0:
        error_scale_f = 1.0

    try:
        ci_z_f = float(ci_z)
    except Exception:
        ci_z_f = 1.96
    if (not np.isfinite(ci_z_f)) or ci_z_f <= 0:
        ci_z_f = 1.96

    error_cap_f: Optional[float]
    if error_cap is None:
        error_cap_f = None
    else:
        try:
            error_cap_f = float(error_cap)
        except Exception:
            error_cap_f = None
        if error_cap_f is not None and ((not np.isfinite(error_cap_f)) or error_cap_f < 0):
            error_cap_f = None

    cv_mode = None if cv is None else str(cv).strip().lower()
    if cv_mode in {"none", "false", "0", "no"}:
        cv_mode = None

    for cond_name in conditions_to_run:
        df_cond = filter_by_condition(df, cond_name)
        if df_cond.empty:
            continue

        X_all = df_cond[feature_cols].apply(pd.to_numeric, errors="coerce")

        for target_name, col in resolved_targets.items():
            y_cont = pd.to_numeric(df_cond[col], errors="coerce")

            # Binarize (for importance we train on full slice; use global threshold if continuous)
            if is_binary_series(y_cont):
                y_bin = y_cont.dropna().astype(int)
            else:
                if str(threshold_mode).lower() not in {"global", "train"}:
                    raise ValueError("threshold_mode must be 'global' or 'train'")
                thr = float(y_cont.quantile(float(quantile)))
                y_bin = binarize_y(y_cont, thr)

            common = X_all.index.intersection(y_bin.index)
            X_use = X_all.loc[common]
            y_use = y_bin.loc[common]
            if len(X_use) < 10 or len(np.unique(y_use.values)) < 2:
                continue

            for model_name in models:
                # ---- Compute importance ----
                if cv_mode is None:
                    # Legacy: single fit on full slice
                    est = clone(models_all[model_name])
                    est.fit(X_use, y_use)
                    imp_df = compute_feature_importance_table(
                        est,
                        X_use,
                        y_use,
                        method=method,
                        scoring=scoring,
                        n_repeats=int(n_repeats),
                        random_state=int(random_state),
                        background_X=X_use,
                    )
                    imp_df = imp_df.copy()
                    imp_df["aoi"] = imp_df["feature"].map(_infer_aoi_group)
                    imp_df["importance_mean"] = imp_df["importance"].astype(float)
                    imp_df["importance_std_across_folds"] = np.nan
                    imp_df["n_folds"] = 1
                    imp_df["error"] = imp_df.get("importance_std", np.nan)

                    if bool(group_features):
                        tmp = imp_df["feature"].map(_feature_to_group_and_aoi)
                        imp_df["group"] = tmp.map(lambda t: t[0])
                        imp_df["group_aoi"] = tmp.map(lambda t: t[1])
                        # Mean across original features within each group (no fold dimension here)
                        g = (
                            imp_df.groupby(["group", "group_aoi"], as_index=False)
                            .agg(
                                importance_mean=("importance_mean", "mean"),
                                # No fold info in this mode; keep NaN.
                                importance_std_across_folds=("importance_std_across_folds", "first"),
                                n_folds=("n_folds", "first"),
                                method=("method", "first"),
                            )
                            .sort_values("importance_mean", ascending=False, na_position="last")
                            .reset_index(drop=True)
                        )
                        g["aoi"] = g["group_aoi"].astype(str)
                        g["error"] = 0.0
                        agg_df = g.drop(columns=["group_aoi"]).rename(columns={"group": "feature"})
                    else:
                        agg_df = imp_df[[
                            "feature",
                            "aoi",
                            "importance_mean",
                            "importance_std_across_folds",
                            "n_folds",
                            "error",
                            "method",
                        ]].sort_values("importance_mean", ascending=False, na_position="last")
                else:
                    # CV: fold-wise importances -> aggregate mean + std/CI
                    y_num_all = pd.to_numeric(y_cont, errors="coerce")
                    if is_binary_series(y_num_all):
                        y_strat = y_num_all.reindex(df_cond.index).fillna(0).astype(int)
                        thr_global = float("nan")
                    else:
                        thr_global = float(y_num_all.quantile(float(quantile)))
                        y_strat = (
                            (y_num_all.reindex(df_cond.index) >= thr_global)
                            .fillna(False)
                            .astype(int)
                        )

                    splits = list(
                        _cv_splits_by_participant(
                            df=df_cond.loc[common],
                            y_for_strat=y_strat.loc[common],
                            cv=str(cv_mode),
                            n_splits=int(n_splits),
                            group_col=str(group_col),
                            random_state=int(random_state),
                        )
                    )

                    long_rows: List[Dict[str, object]] = []
                    for fold_idx, (tr_idx, te_idx) in enumerate(splits, start=1):
                        X_tr = X_use.iloc[tr_idx]
                        X_te = X_use.iloc[te_idx]
                        y_tr_cont = y_num_all.loc[X_tr.index]
                        y_te_cont = y_num_all.loc[X_te.index]

                        if is_binary_series(y_num_all):
                            y_tr = y_tr_cont.dropna().astype(int)
                            y_te = y_te_cont.dropna().astype(int)
                        else:
                            if str(threshold_mode).lower() == "train":
                                thr = float(pd.to_numeric(y_tr_cont, errors="coerce").quantile(float(quantile)))
                            else:
                                thr = float(thr_global)
                            y_tr = binarize_y(y_tr_cont, thr)
                            y_te = binarize_y(y_te_cont, thr)

                        tr_common = X_tr.index.intersection(y_tr.index)
                        te_common = X_te.index.intersection(y_te.index)
                        X_tr_f, y_tr_f = X_tr.loc[tr_common], y_tr.loc[tr_common]
                        X_te_f, y_te_f = X_te.loc[te_common], y_te.loc[te_common]

                        # Need enough data and at least 2 classes in training fold.
                        if len(X_tr_f) < 5 or len(np.unique(y_tr_f.values)) < 2:
                            continue
                        if len(X_te_f) < 1:
                            continue

                        est = clone(models_all[model_name])
                        est.fit(X_tr_f, y_tr_f)
                        fold_imp = compute_feature_importance_table(
                            est,
                            X_te_f,
                            y_te_f,
                            method=method,
                            scoring=scoring,
                            n_repeats=int(n_repeats),
                            random_state=int(random_state),
                            background_X=X_tr_f,
                        )
                        method_used = str(fold_imp["method"].iloc[0]) if not fold_imp.empty else ""
                        for _, r in fold_imp.iterrows():
                            long_rows.append(
                                {
                                    "fold": int(fold_idx),
                                    "feature": str(r["feature"]),
                                    "importance": float(r["importance"]),
                                    "method": method_used,
                                }
                            )

                    long_df = pd.DataFrame(long_rows)
                    if long_df.empty:
                        continue

                    if bool(group_features):
                        long_df = long_df.copy()
                        tmp = long_df["feature"].map(_feature_to_group_and_aoi)
                        long_df["group"] = tmp.map(lambda t: t[0])
                        long_df["group_aoi"] = tmp.map(lambda t: t[1])

                        # First aggregate within each fold using MEAN across original features in the group,
                        # then compute mean/std across folds for that group.
                        fold_agg = (
                            long_df.groupby(["fold", "group", "group_aoi"], as_index=False)
                            .agg(importance=("importance", "mean"))
                        )
                        agg = (
                            fold_agg.groupby(["group", "group_aoi"], as_index=False)
                            .agg(
                                importance_mean=("importance", "mean"),
                                importance_std_across_folds=("importance", "std"),
                                n_folds=("fold", "nunique"),
                            )
                            .sort_values("importance_mean", ascending=False, na_position="last")
                            .reset_index(drop=True)
                        )
                        agg["aoi"] = agg["group_aoi"].astype(str)
                        agg = agg.drop(columns=["group_aoi"]).rename(columns={"group": "feature"})
                    else:
                        agg = (
                            long_df.groupby(["feature"], as_index=False)
                            .agg(
                                importance_mean=("importance", "mean"),
                                importance_std_across_folds=("importance", "std"),
                                n_folds=("fold", "nunique"),
                            )
                            .sort_values("importance_mean", ascending=False, na_position="last")
                            .reset_index(drop=True)
                        )
                        agg["aoi"] = agg["feature"].map(_infer_aoi_group)

                    # CI half-width for mean (normal approx)
                    agg["importance_ci95"] = ci_z_f * (agg["importance_std_across_folds"] / np.sqrt(agg["n_folds"].clip(lower=1)))
                    if err_mode in {"ci", "ci95", "95ci"}:
                        agg["error"] = agg["importance_ci95"]
                    elif err_mode == "std":
                        agg["error"] = agg["importance_std_across_folds"].fillna(0.0)
                    else:
                        agg["error"] = 0.0

                    # Apply optional visual scaling/capping
                    agg["error"] = agg["error"].astype(float) * float(error_scale_f)
                    if error_cap_f is not None:
                        agg["error"] = np.minimum(agg["error"].astype(float), float(error_cap_f))

                    # Optional normalization (useful for grouped plots)
                    if bool(group_features) and bool(group_normalize):
                        denom = float(np.nansum(agg["importance_mean"].astype(float).values))
                        if np.isfinite(denom) and denom > 0:
                            agg["importance_raw"] = agg["importance_mean"].astype(float)
                            agg["importance_mean"] = agg["importance_mean"].astype(float) / denom
                            agg["error"] = agg["error"].astype(float) / denom
                            agg["normalized"] = True
                        else:
                            agg["importance_raw"] = agg["importance_mean"].astype(float)
                            agg["normalized"] = False

                    # Record method (best-effort; may vary if user uses auto)
                    agg["method"] = str(long_df["method"].dropna().iloc[0]) if long_df["method"].notna().any() else ""
                    agg_df = agg

                results[(cond_name, target_name, model_name)] = agg_df

                # ---- Plot (AOI-colored top features + error bars) ----
                top = agg_df.head(int(top_k)).copy()
                if top.empty:
                    continue

                # Order for barh
                top = top.iloc[::-1].copy()
                top["feature_label"] = top["feature"].map(_format_feature_label)

                # Color palette
                aoi_order = ["Claim/Evidence", "Answer", "Rationale", "Response", "Demographic", "Other"]
                # Harmonious but higher-contrast palette ("premium muted")
                # Goal: clearly distinguish AOIs without neon saturation.
                aoi_colors = {
                    "Claim/Evidence": "#3B5B92",  # deep muted blue
                    "Answer": "#C77C2D",  # refined amber
                    "Rationale": "#2E7D73",  # deep teal-green
                    "Response": "#7A5C8C",  # deep plum
                    "Demographic": "#B45309",  # warm brown/amber
                    "Other": "#6B7280",  # neutral gray
                }

                colors = [aoi_colors.get(a, aoi_colors["Other"]) for a in top["aoi"].tolist()]

                fig, ax = plt.subplots(figsize=figsize)
                # Use compressed y positions to reduce gaps while keeping bars thin.
                step = float(bar_step)
                if not np.isfinite(step) or step <= 0:
                    step = 0.70
                height = float(bar_height)
                if not np.isfinite(height) or height <= 0:
                    height = 0.42
                height = min(height, step * 0.98)

                ys = np.arange(len(top), dtype=float) * step
                ax.barh(
                    ys,
                    top["importance_mean"].astype(float),
                    xerr=top["error"].astype(float) if (err_mode != "none") else None,
                    color=colors,
                    height=height,
                    alpha=0.91,
                    edgecolor="none",
                    error_kw={
                        "ecolor": (0.10, 0.10, 0.10, 0.55),
                        "elinewidth": 0.85,
                        "capsize": 2,
                        "capthick": 0.85,
                    },
                )
                ax.set_yticks(ys)
                ax.set_yticklabels(top["feature_label"].tolist())
                xlab_suffix = ""
                if err_mode == "std":
                    xlab_suffix = " (± std across folds)"
                elif err_mode in {"ci", "ci95", "95ci"}:
                    # Reflect custom z/scale if set
                    if abs(float(ci_z_f) - 1.96) < 1e-9:
                        xlab_suffix = " (± 95% CI)"
                    else:
                        xlab_suffix = f" (± CI, z={ci_z_f:g})"
                if err_mode != "none" and abs(float(error_scale_f) - 1.0) > 1e-9:
                    xlab_suffix += f" ×{error_scale_f:g}"
                if err_mode != "none" and error_cap_f is not None:
                    xlab_suffix += f" (cap={error_cap_f:g})"
                if bool(group_features) and bool(group_normalize):
                    ax.set_xlabel("Normalized importance" + xlab_suffix)
                else:
                    ax.set_xlabel("Mean importance" + xlab_suffix)
                ax.set_ylabel("")

                title_cv = "full-fit" if cv_mode is None else f"cv={cv_mode}" + (f", n_splits={int(n_splits)}" if cv_mode == "kfold" else "")
                ax.set_title(
                    f"Top features (AOI-colored) | condition={cond_name} | target={target_name} | model={model_name}\n{title_cv}",
                    loc="left",
                    fontsize=12,
                )

                # Clean styling
                # Allow focusing on a specific importance range (e.g., start from 0.3)
                try:
                    left = float(xlim_left)
                except Exception:
                    left = 0.0
                ax.set_xlim(left=left)
                ax.tick_params(axis="y", labelsize=8.5)
                ax.tick_params(axis="x", labelsize=9.5)
                for spine in ["top", "right"]:
                    ax.spines[spine].set_visible(False)
                ax.grid(axis="x", linestyle="-", alpha=0.12)
                ax.grid(axis="y", visible=False)
                ax.margins(y=0.01)

                # AOI legend intentionally omitted (keep figure compact/clean)

                # Method note
                method_used = str(top["method"].iloc[0]) if "method" in top.columns and not top.empty else ""
                if method_used:
                    ax.text(
                        1.0,
                        1.02,
                        f"method: {method_used}",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        fontsize=9,
                        color="#444444",
                    )

                fig.tight_layout(pad=0.35)

                if save:
                    safe_cond = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(cond_name))
                    safe_target = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(target_name))
                    safe_model = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(model_name))
                    out_png = (
                        outdir_path
                        / f"feature_importance_aoi__cond-{safe_cond}__target-{safe_target}__model-{safe_model}.png"
                    )
                    fig.savefig(out_png, dpi=220)

                if show:
                    plt.show()
                else:
                    plt.close(fig)

    return results


def _cv_splits_by_participant(
    *,
    df: pd.DataFrame,
    y_for_strat: pd.Series,
    cv: str,
    n_splits: int,
    group_col: str,
    random_state: int,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Return indices into df for CV.

    - loso: LeaveOneGroupOut on participant_id.
    - kfold: KFold on unique participants (avoids subject leakage; not stratified).
    """

    cv_mode = str(cv).strip().lower()
    if cv_mode == "loso":
        if group_col not in df.columns:
            raise ValueError(f"LOSO requires group column '{group_col}'")
        groups = df[group_col].astype(str)
        logo = LeaveOneGroupOut()
        return logo.split(df, y_for_strat, groups=groups)

    if cv_mode == "kfold":
        if group_col not in df.columns:
            # Fall back to row-level stratified kfold if no group column.
            skf = StratifiedKFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
            return skf.split(df, y_for_strat)

        # Participant-level KFold to avoid leakage across conditions.
        pids = df[group_col].astype(str)
        uniq = pd.Series(sorted(pids.unique()))
        kf = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(random_state))
        # Build row indices for each fold based on participant membership.
        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for tr_u, te_u in kf.split(uniq.values):
            train_p = set(uniq.iloc[tr_u].tolist())
            test_p = set(uniq.iloc[te_u].tolist())
            tr_idx = np.where(pids.isin(train_p).values)[0]
            te_idx = np.where(pids.isin(test_p).values)[0]
            splits.append((tr_idx, te_idx))
        return splits

    raise ValueError("cv must be 'loso' or 'kfold'")


def run_within_condition_robustness_experiment(
    *,
    input_path: Optional[Union[str, Path]] = None,
    gaze_csv: Optional[Union[str, Path]] = None,
    responses_csv: Optional[Union[str, Path]] = None,
    targets: Sequence[str],
    models: Sequence[str],
    features: Sequence[str],
    conditions: Sequence[str] = ("all", "correct", "incorrect", "no_reasoning"),
    cv: str = "loso",
    n_splits: int = 5,
    group_col: str = "participant_id",
    quantile: float = 0.6,
    threshold_mode: str = "train",
    random_state: int = 42,
) -> pd.DataFrame:
    """Compute per-fold metrics within each condition (robustness).

    Returns per-fold DataFrame with columns:
      condition, target, model, fold, f1, acc
    """

    # Load
    if input_path:
        df = load_from_input_table(Path(input_path))
    else:
        if not (gaze_csv and responses_csv):
            raise ValueError("Provide either input_path OR both gaze_csv and responses_csv")
        df, _ = build_merged_dataset(Path(gaze_csv), Path(responses_csv))
        df = add_derived_self_report_columns(df)

    # Resolve targets
    spec_map = {s.name: s for s in TARGETS}
    resolved_targets: Dict[str, str] = {}
    for t in targets:
        tname = str(t).strip()
        if tname in spec_map:
            col = resolve_target_column(df, spec_map[tname])
            if col is None:
                raise ValueError(f"Target '{tname}' not found")
            resolved_targets[tname] = col
        else:
            if tname not in df.columns:
                raise ValueError(f"Target '{tname}' not found")
            resolved_targets[tname] = tname

    # Features
    feat_cols = [str(c) for c in features]
    missing_feats = [c for c in feat_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing features: {missing_feats[:10]}")

    # Models
    models_all = build_models(random_state=int(random_state), rf_n_jobs=1)
    unknown_m = [m for m in models if m not in models_all]
    if unknown_m:
        raise ValueError(f"Unknown/unavailable model(s): {unknown_m}. Available: {sorted(models_all.keys())}")

    out_rows: List[Dict[str, object]] = []

    for cond in [str(c).strip().lower() for c in conditions if str(c).strip()]:
        df_c = filter_by_condition(df, cond)
        if df_c.empty:
            continue
        X = df_c[feat_cols].apply(pd.to_numeric, errors="coerce")

        for target_name, col in resolved_targets.items():
            y_cont = pd.to_numeric(df_c[col], errors="coerce")
            # For stratification label only
            if is_binary_series(y_cont):
                y_strat = y_cont.fillna(0).astype(int)
                thr_global = float("nan")
            else:
                thr_global = float(y_cont.quantile(float(quantile)))
                y_strat = binarize_y(y_cont, thr_global)
                # align to df rows index
                y_strat = y_strat.reindex(df_c.index).fillna(0).astype(int)

            splits = list(
                _cv_splits_by_participant(
                    df=df_c,
                    y_for_strat=y_strat,
                    cv=cv,
                    n_splits=int(n_splits),
                    group_col=str(group_col),
                    random_state=int(random_state),
                )
            )

            for fold_idx, (tr_idx, te_idx) in enumerate(splits, start=1):
                X_tr = X.iloc[tr_idx]
                X_te = X.iloc[te_idx]
                y_tr_cont = y_cont.iloc[tr_idx]
                y_te_cont = y_cont.iloc[te_idx]

                if is_binary_series(y_cont):
                    y_tr = y_tr_cont.dropna().astype(int)
                    y_te = y_te_cont.dropna().astype(int)
                    # align
                    tr_common = X_tr.index.intersection(y_tr.index)
                    te_common = X_te.index.intersection(y_te.index)
                    X_tr, y_tr = X_tr.loc[tr_common], y_tr.loc[tr_common]
                    X_te, y_te = X_te.loc[te_common], y_te.loc[te_common]
                else:
                    if str(threshold_mode).lower() == "train":
                        thr = float(pd.to_numeric(y_tr_cont, errors="coerce").quantile(float(quantile)))
                    else:
                        thr = thr_global
                    y_tr = binarize_y(y_tr_cont, thr)
                    y_te = binarize_y(y_te_cont, thr)
                    tr_common = X_tr.index.intersection(y_tr.index)
                    te_common = X_te.index.intersection(y_te.index)
                    X_tr, y_tr = X_tr.loc[tr_common], y_tr.loc[tr_common]
                    X_te, y_te = X_te.loc[te_common], y_te.loc[te_common]

                # With participant-level aggregated rows, LOSO yields 1-row test folds.
                # Keep a small minimum to avoid empty folds, but allow n=1.
                if len(X_tr) < 5 or len(X_te) < 1:
                    continue
                if len(np.unique(y_tr.values)) < 2:
                    continue

                for model_name in models:
                    est = clone(models_all[model_name])
                    est.fit(X_tr, y_tr)
                    y_pred = est.predict(X_te)
                    out_rows.append(
                        {
                            "condition": cond,
                            "target": target_name,
                            "model": model_name,
                            "fold": int(fold_idx),
                            "f1": float(f1_score(y_te, y_pred, zero_division=0)),
                            "acc": float(accuracy_score(y_te, y_pred)),
                        }
                    )

    return pd.DataFrame(out_rows)


def plot_within_condition_robustness(
    folds_df: pd.DataFrame,
    *,
    metric_main: str = "f1",
    metric_aux: str = "acc",
    outpath: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (10.0, 5.2),
) -> "object":
    """Plot condition-wise robustness with 95% CI (main metric) and aux metric."""

    import matplotlib.pyplot as plt

    if folds_df is None or folds_df.empty:
        raise ValueError("folds_df is empty")

    for c in ["condition", "target", "model", metric_main, metric_aux]:
        if c not in folds_df.columns:
            raise ValueError(f"folds_df missing column: {c}")

    df = folds_df.copy()
    order = ["all", "correct", "incorrect", "no_reasoning"]
    conds = [c for c in order if c in set(df["condition"].astype(str))] + [
        c for c in sorted(set(df["condition"].astype(str))) if c not in order
    ]

    # Aggregate mean + 95% CI by (condition, target, model)
    g = df.groupby(["condition", "target", "model"], as_index=False)
    summary = g.agg(
        n=(metric_main, "count"),
        main_mean=(metric_main, "mean"),
        main_std=(metric_main, "std"),
        aux_mean=(metric_aux, "mean"),
        aux_std=(metric_aux, "std"),
    )
    summary["main_se"] = summary["main_std"] / np.sqrt(summary["n"].clip(lower=1))
    summary["main_ci95"] = 1.96 * summary["main_se"]
    summary["aux_se"] = summary["aux_std"] / np.sqrt(summary["n"].clip(lower=1))
    summary["aux_ci95"] = 1.96 * summary["aux_se"]

    targets = list(summary["target"].unique())
    models = list(summary["model"].unique())

    # Facet by target (rows) and metric (main/aux in 2 panels)
    nrows = len(targets)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(figsize[0], max(figsize[1], 2.6 * nrows)), sharex=True)
    if nrows == 1:
        axes = np.array([axes])

    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.get_cmap("tab10")

    for r, tgt in enumerate(targets):
        sub = summary[summary["target"] == tgt]
        for cidx, (metric_label, mean_col, ci_col) in enumerate(
            [("F1 (main)", "main_mean", "main_ci95"), ("Accuracy (aux)", "aux_mean", "aux_ci95")]
        ):
            ax = axes[r, cidx]
            ax.set_title(f"{tgt} | {metric_label}", loc="left", fontsize=12)
            xs = np.arange(len(conds))
            width = 0.75 / max(1, len(models))

            for mi, m in enumerate(models):
                subm = sub[sub["model"] == m]
                # align to conds
                mean_map = dict(zip(subm["condition"].astype(str), subm[mean_col].astype(float)))
                ci_map = dict(zip(subm["condition"].astype(str), subm[ci_col].astype(float)))
                ys = [mean_map.get(cc, np.nan) for cc in conds]
                yerr = [ci_map.get(cc, np.nan) for cc in conds]
                ax.errorbar(
                    xs + (mi - (len(models) - 1) / 2) * width,
                    ys,
                    yerr=yerr,
                    fmt="o",
                    markersize=6,
                    capsize=3,
                    linewidth=1.5,
                    color=colors(mi),
                    label=m if (r == 0 and cidx == 0) else None,
                    alpha=0.95,
                )

            ax.set_xticks(xs)
            ax.set_xticklabels(conds, rotation=0)
            ax.set_ylim(0.0, 1.0)
            ax.grid(axis="y", linestyle="-", alpha=0.25)
            ax.grid(axis="x", visible=False)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    if models:
        axes[0, 0].legend(frameon=False, loc="lower left", ncol=min(3, len(models)))
    fig.tight_layout()

    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(outpath), dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def run_cross_condition_generalization_experiment(
    *,
    input_path: Optional[Union[str, Path]] = None,
    gaze_csv: Optional[Union[str, Path]] = None,
    responses_csv: Optional[Union[str, Path]] = None,
    targets: Sequence[str],
    models: Sequence[str],
    features: Sequence[str],
    conditions: Sequence[str] = ("correct", "incorrect", "no_reasoning"),
    cv: str = "loso",
    n_splits: int = 5,
    group_col: str = "participant_id",
    quantile: float = 0.6,
    threshold_mode: str = "train",
    random_state: int = 42,
) -> pd.DataFrame:
    """Train on condition A, test on condition B (heatmap-ready).

    Returns per-fold rows with columns:
      train_condition, test_condition, target, model, fold, f1, acc
    """

    if input_path:
        df = load_from_input_table(Path(input_path))
    else:
        if not (gaze_csv and responses_csv):
            raise ValueError("Provide either input_path OR both gaze_csv and responses_csv")
        df, _ = build_merged_dataset(Path(gaze_csv), Path(responses_csv))
        df = add_derived_self_report_columns(df)

    spec_map = {s.name: s for s in TARGETS}
    resolved_targets: Dict[str, str] = {}
    for t in targets:
        tname = str(t).strip()
        if tname in spec_map:
            col = resolve_target_column(df, spec_map[tname])
            if col is None:
                raise ValueError(f"Target '{tname}' not found")
            resolved_targets[tname] = col
        else:
            if tname not in df.columns:
                raise ValueError(f"Target '{tname}' not found")
            resolved_targets[tname] = tname

    feat_cols = [str(c) for c in features]
    missing_feats = [c for c in feat_cols if c not in df.columns]
    if missing_feats:
        raise ValueError(f"Missing features: {missing_feats[:10]}")

    models_all = build_models(random_state=int(random_state), rf_n_jobs=1)
    unknown_m = [m for m in models if m not in models_all]
    if unknown_m:
        raise ValueError(f"Unknown/unavailable model(s): {unknown_m}. Available: {sorted(models_all.keys())}")

    conds = [str(c).strip().lower() for c in conditions if str(c).strip()]
    out_rows: List[Dict[str, object]] = []

    # Pre-slice each condition once
    df_by_cond: Dict[str, pd.DataFrame] = {}
    for c in conds:
        dfc = filter_by_condition(df, c)
        if not dfc.empty:
            df_by_cond[c] = dfc

    for train_c in conds:
        if train_c not in df_by_cond:
            continue
        df_train_cond = df_by_cond[train_c]
        X_train_all = df_train_cond[feat_cols].apply(pd.to_numeric, errors="coerce")

        for target_name, col in resolved_targets.items():
            y_train_cont_all = pd.to_numeric(df_train_cond[col], errors="coerce")

            # strat label for splitting (not used in loso)
            if is_binary_series(y_train_cont_all):
                y_strat = y_train_cont_all.reindex(df_train_cond.index).fillna(0).astype(int)
            else:
                thr_tmp = float(y_train_cont_all.quantile(float(quantile)))
                y_strat = binarize_y(y_train_cont_all, thr_tmp).reindex(df_train_cond.index).fillna(0).astype(int)

            splits = list(
                _cv_splits_by_participant(
                    df=df_train_cond,
                    y_for_strat=y_strat,
                    cv=cv,
                    n_splits=int(n_splits),
                    group_col=str(group_col),
                    random_state=int(random_state),
                )
            )

            for fold_idx, (tr_idx, te_idx) in enumerate(splits, start=1):
                # participants in this fold
                if group_col in df_train_cond.columns:
                    pids = df_train_cond[group_col].astype(str)
                    test_pids = set(pids.iloc[te_idx].tolist())
                    train_pids = set(pids.iloc[tr_idx].tolist())
                else:
                    test_pids = set()
                    train_pids = set()

                X_tr = X_train_all.iloc[tr_idx]
                y_tr_cont = y_train_cont_all.iloc[tr_idx]

                # Choose threshold from training fold (recommended)
                if is_binary_series(y_train_cont_all):
                    thr = float("nan")
                    y_tr = y_tr_cont.dropna().astype(int)
                    tr_common = X_tr.index.intersection(y_tr.index)
                    X_tr, y_tr = X_tr.loc[tr_common], y_tr.loc[tr_common]
                else:
                    if str(threshold_mode).lower() == "train":
                        thr = float(pd.to_numeric(y_tr_cont, errors="coerce").quantile(float(quantile)))
                    else:
                        thr = float(y_train_cont_all.quantile(float(quantile)))
                    y_tr = binarize_y(y_tr_cont, thr)
                    tr_common = X_tr.index.intersection(y_tr.index)
                    X_tr, y_tr = X_tr.loc[tr_common], y_tr.loc[tr_common]

                if len(X_tr) < 5 or len(np.unique(y_tr.values)) < 2:
                    continue

                for test_c in conds:
                    if test_c not in df_by_cond:
                        continue
                    df_test_cond = df_by_cond[test_c]

                    # Test participants = fold's held-out participants (strict)
                    if group_col in df_test_cond.columns and test_pids:
                        mask = df_test_cond[group_col].astype(str).isin(test_pids)
                        df_test_fold = df_test_cond[mask]
                    else:
                        # If no group info, fall back to row indices mapping (less strict)
                        df_test_fold = df_test_cond

                    if df_test_fold.empty:
                        continue
                    X_te = df_test_fold[feat_cols].apply(pd.to_numeric, errors="coerce")
                    y_te_cont = pd.to_numeric(df_test_fold[col], errors="coerce")
                    if is_binary_series(y_train_cont_all):
                        y_te = y_te_cont.dropna().astype(int)
                    else:
                        y_te = binarize_y(y_te_cont, thr)
                    te_common = X_te.index.intersection(y_te.index)
                    X_te, y_te = X_te.loc[te_common], y_te.loc[te_common]
                    # With participant-level aggregated rows, LOSO yields 1-row test folds.
                    # Also, held-out folds may contain only one class; metrics still defined.
                    if len(X_te) < 1:
                        continue

                    for model_name in models:
                        est = clone(models_all[model_name])
                        est.fit(X_tr, y_tr)
                        y_pred = est.predict(X_te)
                        out_rows.append(
                            {
                                "train_condition": train_c,
                                "test_condition": test_c,
                                "target": target_name,
                                "model": model_name,
                                "fold": int(fold_idx),
                                "f1": float(f1_score(y_te, y_pred, zero_division=0)),
                                "acc": float(accuracy_score(y_te, y_pred)),
                            }
                        )

    return pd.DataFrame(out_rows)


def plot_cross_condition_heatmap(
    df: pd.DataFrame,
    *,
    target: str,
    model: str,
    metric_main: str = "f1",
    metric_aux: str = "acc",
    outpath: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: Tuple[float, float] = (11.0, 4.8),
) -> "object":
    """Plot train×test heatmaps for cross-condition generalization (F1 main + Acc aux)."""

    import matplotlib.pyplot as plt

    # Seaborn colormaps (e.g., 'mako') are not always registered in matplotlib.
    # Prefer seaborn's cmap object when available; otherwise fall back to a builtin.
    try:
        import seaborn as sns  # type: ignore

        cmap_main = sns.color_palette("mako", as_cmap=True)
    except Exception:
        cmap_main = "viridis"

    if df is None or df.empty:
        raise ValueError("df is empty")

    req = {"train_condition", "test_condition", "target", "model", metric_main, metric_aux}
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    sub = df[(df["target"] == target) & (df["model"] == model)].copy()
    if sub.empty:
        raise ValueError("No rows for the requested target/model")

    # Mean over folds
    piv_main = (
        sub.groupby(["train_condition", "test_condition"], as_index=False)[metric_main]
        .mean()
        .pivot(index="train_condition", columns="test_condition", values=metric_main)
    )
    piv_aux = (
        sub.groupby(["train_condition", "test_condition"], as_index=False)[metric_aux]
        .mean()
        .pivot(index="train_condition", columns="test_condition", values=metric_aux)
    )

    rows = list(piv_main.index)
    cols = list(piv_main.columns)

    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    def _draw(ax, mat: pd.DataFrame, title: str, cmap: str) -> None:
        im = ax.imshow(mat.values, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows)
        for i in range(len(rows)):
            for j in range(len(cols)):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10, color="black")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _draw(axes[0], piv_main.reindex(index=rows, columns=cols), f"Cross-condition | F1 (main)\n{target} | {model}", cmap_main)
    _draw(axes[1], piv_aux.reindex(index=rows, columns=cols), f"Cross-condition | Acc (aux)\n{target} | {model}", "Greys")

    fig.tight_layout()
    if outpath:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(outpath), dpi=220)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


if __name__ == "__main__":
    main()
