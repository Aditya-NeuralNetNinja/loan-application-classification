from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from app.config import (
    FEATURE_METADATA_CANDIDATES,
    GBT_MODEL_CANDIDATES,
    MAX_UPLOAD_ROWS,
    OPTIMAL_THRESHOLD_CANDIDATES,
    PIPELINE_MODEL_CANDIDATES,
    SCHEMA_PATH,
    SPARK_APP_NAME,
    SPARK_DRIVER_MEMORY,
    SPARK_MASTER,
)

MISSING_VALUE_TOKENS = {"", "exempt", "na", "n/a", "not applicable", "1111"}

# Informative missingness indicators from notebook 3
INFORMATIVE_MISSING_COLS = [
    "interest_rate",
    "combined_loan_to_value_ratio",
    "property_value",
    "income",
    "debt_to_income_ratio",
    "loan_term",
    "intro_rate_period",
    "applicant_credit_score_type",
    "co_applicant_credit_score_type",
]

# Log transforms from notebook 3
LOG_TRANSFORM_COLS = [
    "loan_amount",
    "income",
    "property_value",
    "tract_population",
]


class InferenceError(RuntimeError):
    """Inference-specific runtime error."""


def _resolve_existing(paths: List[Path]) -> Path:
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError(f"No existing path found in candidates: {paths}")


def _normalize_colname(name: str) -> str:
    txt = str(name).strip().lower()
    txt = re.sub(r"[^0-9a-zA-Z]+", "_", txt)
    txt = re.sub(r"_+", "_", txt).strip("_")
    return txt or "col"


def normalize_dataframe_columns(pdf: pd.DataFrame) -> pd.DataFrame:
    seen: Dict[str, int] = {}
    new_cols = []
    for col in pdf.columns:
        base = _normalize_colname(col)
        if base in seen:
            seen[base] += 1
            new_cols.append(f"{base}_{seen[base]}")
        else:
            seen[base] = 0
            new_cols.append(base)

    out = pdf.copy()
    out.columns = new_cols
    return out


@lru_cache(maxsize=1)
def _load_schema_json() -> Dict:
    if not SCHEMA_PATH.exists():
        raise InferenceError(f"Schema file not found: {SCHEMA_PATH}")
    with open(SCHEMA_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_feature_metadata() -> Dict:
    path = _resolve_existing(FEATURE_METADATA_CANDIDATES)
    with open(path) as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _load_optimal_threshold() -> float:
    path = _resolve_existing(OPTIMAL_THRESHOLD_CANDIDATES)
    with open(path) as f:
        obj = json.load(f)
    return float(obj.get("optimal_threshold_f1", 0.5))


@lru_cache(maxsize=1)
def _spark_session():
    try:
        from pyspark.sql import SparkSession
    except Exception as exc:
        raise InferenceError(
            "PySpark is not available. Install deploy dependencies and ensure Java is installed."
        ) from exc

    spark = (
        SparkSession.builder.master(SPARK_MASTER)
        .appName(SPARK_APP_NAME)
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


@lru_cache(maxsize=1)
def _load_models_bundle() -> Dict:
    try:
        from pyspark.ml import PipelineModel
        from pyspark.ml.classification import GBTClassificationModel
    except Exception as exc:
        raise InferenceError("PySpark ML imports failed. Ensure pyspark is installed.") from exc

    pipeline_path = _resolve_existing(PIPELINE_MODEL_CANDIDATES)
    gbt_path = _resolve_existing(GBT_MODEL_CANDIDATES)

    pipeline_model = PipelineModel.load(str(pipeline_path))
    gbt_model = GBTClassificationModel.load(str(gbt_path))

    # Pull required column contracts from saved pipeline stages
    stage0 = pipeline_model.stages[0]  # ImputerModel
    imputer_input_cols = list(stage0.getInputCols())

    string_indexer_inputs: List[str] = []
    for stage in pipeline_model.stages:
        cls_name = stage.__class__.__name__
        if cls_name == "StringIndexerModel" and hasattr(stage, "getInputCol"):
            string_indexer_inputs.append(stage.getInputCol())

    return {
        "pipeline_model": pipeline_model,
        "gbt_model": gbt_model,
        "imputer_input_cols": imputer_input_cols,
        "categorical_input_cols": sorted(set(string_indexer_inputs)),
        "pipeline_path": str(pipeline_path),
        "gbt_path": str(gbt_path),
    }


def _inference_contract() -> Dict:
    schema = _load_schema_json()
    metadata = _load_feature_metadata()
    bundle = _load_models_bundle()

    leakage_cols = set(schema.get("leakage_columns", {}).get("columns", []))
    numeric_continuous = schema.get("expected_dtypes", {}).get("numeric_continuous", [])
    base_numeric = [c for c in numeric_continuous if c not in leakage_cols]

    # Categorical lists used in notebook 3 and reflected in saved pipeline metadata
    onehot_cols = metadata.get("onehot_categoricals", [])
    index_only_cols = metadata.get("index_categoricals", [])

    raw_required = set(base_numeric + onehot_cols + index_only_cols)
    raw_required.update(["hoepa_status", "co_applicant_age", "action_taken", "label"])

    return {
        "base_numeric": base_numeric,
        "onehot_cols": onehot_cols,
        "index_only_cols": index_only_cols,
        "raw_required": sorted(raw_required),
        "imputer_input_cols": bundle["imputer_input_cols"],
        "categorical_input_cols": bundle["categorical_input_cols"],
    }


def get_required_columns() -> Dict[str, List[str]]:
    try:
        c = _inference_contract()
        return {
            "raw_required_minimum": c["raw_required"],
            "numeric_base": c["base_numeric"],
            "categorical_onehot": c["onehot_cols"],
            "categorical_index_only": c["index_only_cols"],
        }
    except Exception:
        # Fallback contract if Spark model artifacts are not yet available
        schema = _load_schema_json()
        metadata = _load_feature_metadata()
        leakage_cols = set(schema.get("leakage_columns", {}).get("columns", []))
        numeric_continuous = schema.get("expected_dtypes", {}).get("numeric_continuous", [])
        base_numeric = [c for c in numeric_continuous if c not in leakage_cols]
        onehot_cols = metadata.get("onehot_categoricals", [])
        index_only_cols = metadata.get("index_categoricals", [])
        raw_required = sorted(set(base_numeric + onehot_cols + index_only_cols))
        return {
            "raw_required_minimum": raw_required,
            "numeric_base": base_numeric,
            "categorical_onehot": onehot_cols,
            "categorical_index_only": index_only_cols,
        }


def _preprocess_for_model(sdf):
    from pyspark.sql import functions as F

    contract = _inference_contract()

    raw_required = contract["raw_required"]
    base_numeric = contract["base_numeric"]
    imputer_input_cols = contract["imputer_input_cols"]
    categorical_input_cols = contract["categorical_input_cols"]

    # Add any missing raw columns
    for col in raw_required:
        if col not in sdf.columns:
            sdf = sdf.withColumn(col, F.lit(None).cast("string"))

    # Normalize known missing tokens -> null
    tokens = list(MISSING_VALUE_TOKENS)
    for col in raw_required:
        sdf = sdf.withColumn(
            col,
            F.when(
                F.lower(F.trim(F.col(col).cast("string"))).isin(tokens),
                F.lit(None),
            ).otherwise(F.col(col)),
        )

    # High-cost flag from hoepa_status (before dropping hoepa_status)
    sdf = sdf.withColumn(
        "high_cost_flag",
        F.when(F.col("hoepa_status").cast("string") == "1", F.lit(1.0)).otherwise(F.lit(0.0)),
    )

    # Cast base numeric fields to double
    for col in base_numeric:
        sdf = sdf.withColumn(col, F.col(col).cast("double"))

    # Missingness indicators
    for col in INFORMATIVE_MISSING_COLS:
        ind = f"{col}_is_missing"
        sdf = sdf.withColumn(ind, F.when(F.col(col).isNull(), F.lit(1.0)).otherwise(F.lit(0.0)))

    # Log transforms
    for col in LOG_TRANSFORM_COLS:
        out_col = f"{col}_log"
        sdf = sdf.withColumn(
            out_col,
            F.when(
                F.col(col).isNotNull() & (F.col(col) >= 0),
                F.log1p(F.col(col)),
            ).otherwise(F.lit(None)),
        )

    # Clip outliers
    sdf = sdf.withColumn(
        "combined_loan_to_value_ratio",
        F.when(F.col("combined_loan_to_value_ratio") > 200, F.lit(200.0))
        .when(F.col("combined_loan_to_value_ratio") < 0, F.lit(None))
        .otherwise(F.col("combined_loan_to_value_ratio")),
    )

    sdf = sdf.withColumn(
        "interest_rate",
        F.when((F.col("interest_rate") > 30) | (F.col("interest_rate") < 0), F.lit(None)).otherwise(
            F.col("interest_rate")
        ),
    )

    # Engineered affordability features
    sdf = sdf.withColumn(
        "loan_to_income_ratio",
        F.when(
            F.col("income").isNotNull() & (F.col("income") > 0),
            F.col("loan_amount") / F.col("income"),
        ).otherwise(F.lit(None)),
    )

    sdf = sdf.withColumn(
        "loan_to_income_ratio",
        F.when(F.col("loan_to_income_ratio") > 50, F.lit(50.0)).otherwise(F.col("loan_to_income_ratio")),
    )

    sdf = sdf.withColumn(
        "loan_to_income_ratio_log",
        F.when(
            F.col("loan_to_income_ratio").isNotNull() & (F.col("loan_to_income_ratio") > 0),
            F.log1p(F.col("loan_to_income_ratio")),
        ).otherwise(F.lit(None)),
    )

    sdf = sdf.withColumn(
        "is_joint_application",
        F.when(
            F.col("co_applicant_age").isNotNull()
            & (~F.col("co_applicant_age").cast("string").isin("9999", "8888")),
            F.lit(1.0),
        ).otherwise(F.lit(0.0)),
    )

    # Ensure imputer contract columns exist and are numeric
    for col in imputer_input_cols:
        if col not in sdf.columns:
            sdf = sdf.withColumn(col, F.lit(None).cast("double"))
        else:
            sdf = sdf.withColumn(col, F.col(col).cast("double"))

    # Ensure categorical contract columns exist and are strings
    for col in categorical_input_cols:
        if col not in sdf.columns:
            sdf = sdf.withColumn(col, F.lit(None).cast("string"))
        else:
            sdf = sdf.withColumn(col, F.col(col).cast("string"))

    return sdf


def _safe_binary_truth(normalized_pdf: pd.DataFrame) -> np.ndarray | None:
    if "label" in normalized_pdf.columns:
        y = pd.to_numeric(normalized_pdf["label"], errors="coerce")
        y = y.where(y.isin([0, 1]))
        return y.to_numpy(dtype=float)

    if "action_taken" in normalized_pdf.columns:
        a = pd.to_numeric(normalized_pdf["action_taken"], errors="coerce")
        y = pd.Series(np.nan, index=a.index, dtype=float)
        y.loc[a == 1] = 0.0
        y.loc[a == 3] = 1.0
        return y.to_numpy(dtype=float)

    return None


def run_gbt_inference_from_pandas(
    upload_pdf: pd.DataFrame,
    threshold: float | None = None,
) -> Tuple[pd.DataFrame, Dict]:
    if upload_pdf is None or upload_pdf.empty:
        raise InferenceError("Uploaded dataset is empty.")

    if len(upload_pdf) > MAX_UPLOAD_ROWS:
        raise InferenceError(
            f"Uploaded dataset has {len(upload_pdf):,} rows. "
            f"Current limit is {MAX_UPLOAD_ROWS:,}. Increase MAX_UPLOAD_ROWS if needed."
        )

    spark = _spark_session()
    bundle = _load_models_bundle()
    pipeline_model = bundle["pipeline_model"]
    gbt_model = bundle["gbt_model"]

    original_pdf = upload_pdf.copy()
    normalized_pdf = normalize_dataframe_columns(upload_pdf)

    # Stable join key from upload order
    normalized_pdf["__row_id"] = np.arange(len(normalized_pdf), dtype=np.int64)
    original_pdf["__row_id"] = np.arange(len(original_pdf), dtype=np.int64)

    sdf = spark.createDataFrame(normalized_pdf)
    sdf = _preprocess_for_model(sdf)

    transformed = pipeline_model.transform(sdf)
    scored = gbt_model.transform(transformed)

    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    t = float(threshold) if threshold is not None else _load_optimal_threshold()

    pred_df = (
        scored.select(
            "__row_id",
            vector_to_array(F.col("probability"))[1].alias("denial_probability"),
        )
        .withColumn("predicted_denial", F.when(F.col("denial_probability") >= F.lit(t), F.lit(1)).otherwise(F.lit(0)))
        .withColumn(
            "risk_band",
            F.when(F.col("denial_probability") >= 0.80, F.lit("Very High"))
            .when(F.col("denial_probability") >= 0.60, F.lit("High"))
            .when(F.col("denial_probability") >= 0.40, F.lit("Medium"))
            .otherwise(F.lit("Low")),
        )
    )

    pred_pdf = pred_df.toPandas().sort_values("__row_id").reset_index(drop=True)

    result_pdf = (
        original_pdf.merge(pred_pdf, on="__row_id", how="left")
        .drop(columns=["__row_id"])  # internal key
        .reset_index(drop=True)
    )

    summary = {
        "rows_scored": int(len(result_pdf)),
        "threshold_used": t,
        "predicted_denials": int((result_pdf["predicted_denial"] == 1).sum()),
        "predicted_denial_rate": float((result_pdf["predicted_denial"] == 1).mean()),
        "avg_denial_probability": float(result_pdf["denial_probability"].mean()),
        "pipeline_model_path": bundle["pipeline_path"],
        "gbt_model_path": bundle["gbt_path"],
    }

    y_true = _safe_binary_truth(normalized_pdf)
    if y_true is not None:
        y_pred = result_pdf["predicted_denial"].to_numpy(dtype=float)
        valid = ~np.isnan(y_true)
        if valid.sum() > 0:
            yt = y_true[valid].astype(int)
            yp = y_pred[valid].astype(int)

            tp = int(((yt == 1) & (yp == 1)).sum())
            fp = int(((yt == 0) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum())
            tn = int(((yt == 0) & (yp == 0)).sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            acc = (tp + tn) / max(tp + fp + fn + tn, 1)

            summary["evaluation"] = {
                "n_with_ground_truth": int(valid.sum()),
                "accuracy": float(acc),
                "denial_precision": float(prec),
                "denial_recall": float(rec),
                "denial_f1": float(f1),
                "confusion": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
            }

    return result_pdf, summary
