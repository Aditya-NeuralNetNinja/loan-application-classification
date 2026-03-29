from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
ASSETS_DIR = APP_DIR / "assets"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SCHEMA_PATH = BASE_DIR / "data" / "schemas" / "hmda_schema.json"

MODEL_RESULTS_CANDIDATES = [
    PROCESSED_DIR / "model_results.json",
    ASSETS_DIR / "model_results.sample.json",
]

OPTIMAL_THRESHOLD_CANDIDATES = [
    PROCESSED_DIR / "optimal_threshold.json",
    ASSETS_DIR / "optimal_threshold.sample.json",
]

LEADERBOARD_CANDIDATES = [
    PROCESSED_DIR / "model_leaderboard.csv",
    ASSETS_DIR / "model_leaderboard.sample.csv",
]

PIPELINE_MODEL_CANDIDATES = [
    PROCESSED_DIR / "pipeline_model",
]

GBT_MODEL_CANDIDATES = [
    PROCESSED_DIR / "models" / "best_gbt",
]

FEATURE_METADATA_CANDIDATES = [
    PROCESSED_DIR / "feature_metadata.json",
]

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
MAX_UPLOAD_ROWS = int(os.getenv("MAX_UPLOAD_ROWS", "200000"))

SPARK_MASTER = os.getenv("SPARK_MASTER", "local[2]")
SPARK_DRIVER_MEMORY = os.getenv("SPARK_DRIVER_MEMORY", "2g")
SPARK_APP_NAME = os.getenv("SPARK_APP_NAME", "hmda_inference_app")
