from __future__ import annotations

import io

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi import File, Form, UploadFile

from app.data_loader import (
    build_leaderboard,
    business_cost_table,
    load_model_results,
    load_optimal_threshold,
    recommend_model,
)
from app.inference import InferenceError, get_required_columns, run_gbt_inference_from_pandas

app = FastAPI(
    title="HMDA Model Deployment API",
    version="1.0.0",
    description="Lightweight API exposing trained model artifacts and deployment recommendations.",
)


@app.get("/")
def root():
    return {
        "service": "hmda-model-api",
        "status": "ok",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    model_results, source = load_model_results()
    return {
        "status": "healthy",
        "models": len(model_results),
        "artifact_source": source,
    }


@app.get("/models")
def models():
    df, source = build_leaderboard()
    return {
        "artifact_source": source,
        "count": int(len(df)),
        "items": df.to_dict(orient="records"),
    }


@app.get("/models/{model_name}")
def model_details(model_name: str):
    model_results, source = load_model_results()
    if model_name not in model_results:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return {
        "artifact_source": source,
        "model": model_name,
        "metrics": model_results[model_name],
    }


@app.get("/recommendation")
def recommendation():
    model_results, source = load_model_results()
    reco = recommend_model(model_results)
    threshold, threshold_source = load_optimal_threshold()
    return {
        "artifact_source": source,
        "threshold_source": threshold_source,
        "recommendation": reco,
        "optimal_threshold": threshold,
    }


@app.get("/business-metrics")
def business_metrics(
    cost_fp: float = Query(default=250.0, gt=0),
    cost_fn: float = Query(default=2500.0, gt=0),
):
    model_results, source = load_model_results()
    df = business_cost_table(model_results, cost_fp=cost_fp, cost_fn=cost_fn)
    return {
        "artifact_source": source,
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "items": df.to_dict(orient="records"),
    }


@app.get("/inference/required-columns")
def inference_required_columns():
    return get_required_columns()


@app.post("/inference/predict-csv")
async def inference_predict_csv(
    file: UploadFile = File(...),
    threshold: float = Form(default=0.62),
):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV upload is supported.")

    try:
        payload = await file.read()
        upload_df = pd.read_csv(io.BytesIO(payload), low_memory=False)
        scored_df, summary = run_gbt_inference_from_pandas(upload_df, threshold=threshold)
    except InferenceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    # Keep response size bounded for API consumers
    max_preview = 500
    preview_df = scored_df.head(max_preview)

    return {
        "summary": summary,
        "total_rows": int(len(scored_df)),
        "preview_rows": int(len(preview_df)),
        "truncated": bool(len(scored_df) > max_preview),
        "predictions_preview": preview_df.to_dict(orient="records"),
    }
