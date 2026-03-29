from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from app.config import (
    LEADERBOARD_CANDIDATES,
    MODEL_RESULTS_CANDIDATES,
    OPTIMAL_THRESHOLD_CANDIDATES,
)


def _resolve_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    raise FileNotFoundError(f"No file found in candidates: {paths}")


@lru_cache(maxsize=1)
def load_model_results() -> Tuple[Dict, str]:
    path = _resolve_existing(MODEL_RESULTS_CANDIDATES)
    with open(path) as f:
        data = json.load(f)
    return data, str(path)


@lru_cache(maxsize=1)
def load_optimal_threshold() -> Tuple[Dict, str]:
    path = _resolve_existing(OPTIMAL_THRESHOLD_CANDIDATES)
    with open(path) as f:
        data = json.load(f)
    return data, str(path)


@lru_cache(maxsize=1)
def build_leaderboard() -> Tuple[pd.DataFrame, str]:
    try:
        path = _resolve_existing(LEADERBOARD_CANDIDATES)
        df = pd.read_csv(path)
        source = str(path)
        if "Model" in df.columns:
            return df, source
    except Exception:
        pass

    # Fallback: build from model_results.json
    model_results, src = load_model_results()
    rows = []
    for model, m in model_results.items():
        rows.append(
            {
                "Model": model,
                "ROC-AUC": m.get("ROC-AUC", np.nan),
                "PR-AUC": m.get("PR-AUC", np.nan),
                "Denial_F1": m.get("Denial_F1", np.nan),
                "D_Prec": m.get("Denial_Precision", np.nan),
                "D_Rec": m.get("Denial_Recall", np.nan),
                "Accuracy": m.get("Accuracy", np.nan),
                "Time_s": m.get("Train_Time_s", np.nan),
            }
        )
    df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
    return df, src


def recommend_model(model_results: Dict) -> Dict:
    # Ignore known broken candidates (e.g., predicts no positive class at default threshold)
    candidates = []
    for model, m in model_results.items():
        denial_f1 = float(m.get("Denial_F1", 0.0) or 0.0)
        pr_auc = float(m.get("PR-AUC", 0.0) or 0.0)
        if denial_f1 <= 0.0:
            continue
        candidates.append((model, pr_auc, denial_f1, m))

    if not candidates:
        return {
            "recommended_model": None,
            "reason": "No valid candidate found with Denial_F1 > 0.",
        }

    # Production heuristic: prioritize PR-AUC, tie-break with Denial_F1
    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top_model, top_pr, top_f1, raw = candidates[0]

    return {
        "recommended_model": top_model,
        "pr_auc": top_pr,
        "denial_f1": top_f1,
        "details": {
            "roc_auc": raw.get("ROC-AUC"),
            "accuracy": raw.get("Accuracy"),
            "denial_precision": raw.get("Denial_Precision"),
            "denial_recall": raw.get("Denial_Recall"),
            "train_time_s": raw.get("Train_Time_s"),
        },
        "notes": [
            "Models with Denial_F1=0 are excluded from recommendation.",
            "If regulatory interpretability dominates, consider M2_LogisticRegression.",
        ],
    }


def business_cost_table(model_results: Dict, cost_fp: float = 250.0, cost_fn: float = 2500.0) -> pd.DataFrame:
    rows = []
    for model, m in model_results.items():
        c = m.get("Confusion", {})
        tp = int(c.get("TP", 0))
        fp = int(c.get("FP", 0))
        fn = int(c.get("FN", 0))
        tn = int(c.get("TN", 0))
        total = tp + fp + fn + tn
        if total == 0:
            continue

        expected_loss = fp * cost_fp + fn * cost_fn
        rows.append(
            {
                "Model": model,
                "Expected_Loss": expected_loss,
                "Cost_per_1k_apps": (expected_loss / total) * 1000.0,
                "Alert_Rate": (tp + fp) / total,
                "Miss_Rate": fn / (tp + fn) if (tp + fn) > 0 else 0.0,
                "PR-AUC": m.get("PR-AUC", np.nan),
                "Denial_F1": m.get("Denial_F1", np.nan),
            }
        )

    df = pd.DataFrame(rows).sort_values("Cost_per_1k_apps").reset_index(drop=True)
    return df
