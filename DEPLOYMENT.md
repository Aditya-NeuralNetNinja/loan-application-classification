# Deployment Guide

## Do you need Docker and a web app?

Short answer:
- Docker is **not required** on every free platform.
- A web app/API is **strongly suggested** if you want a shareable deployed demo.

## Recommended free deployment paths

1. Streamlit Community Cloud (no Docker needed): easiest for dashboard demos.
2. Hugging Face Spaces (Streamlit or Docker): good free public hosting.
3. Render free web service (Docker recommended): supports both API and dashboard.
4. Google Cloud Run free tier (Docker required): strong for containerized APIs.

## Included deployment files

- `app/streamlit_app.py`: web dashboard
- `app/api.py`: FastAPI service
- `app/data_loader.py`: shared artifact loading + recommendation logic
- `app/inference.py`: Spark preprocessing + GBT batch inference backend
- `app/entrypoint.sh`: starts API or Streamlit via `APP_MODE`
- `requirements.deploy.txt`: lightweight deploy dependencies
- `Dockerfile`, `.dockerignore`: container deployment
- `Procfile`: non-Docker process entry
- `render.yaml`: Render blueprint
- `railway.json`: Railway Docker deployment config

## Local run

### Streamlit dashboard

```bash
pip install -r requirements.deploy.txt
streamlit run app/streamlit_app.py
```

The web app includes:
- CSV upload
- backend preprocessing (same pipeline contract as Notebook 3)
- inference with saved `best_gbt`
- downloadable scored CSV output

### FastAPI service

```bash
pip install -r requirements.deploy.txt
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

Inference endpoints:
- `GET /inference/required-columns`
- `POST /inference/predict-csv` (multipart CSV upload)

## Docker run

```bash
docker build -t hmda-deploy .
# Dashboard mode

docker run -e APP_MODE=streamlit -e PORT=8000 -p 8000:8000 hmda-deploy
# API mode

docker run -e APP_MODE=api -e PORT=8000 -p 8000:8000 hmda-deploy
```

## Artifact strategy

The model-dashboard pages (leaderboard/recommendation) use this fallback order:

1. `data/processed/*.json|csv` (if present at runtime)
2. `app/assets/*.sample.*` (committed lightweight artifacts)

Batch inference specifically requires real model artifacts:

- `data/processed/pipeline_model/`
- `data/processed/models/best_gbt/`
- `data/processed/feature_metadata.json`
- `data/processed/optimal_threshold.json`
- `data/schemas/hmda_schema.json`

In Docker, these are copied by `Dockerfile` (plus Java runtime for Spark).

For fresh results after retraining, run:

```bash
python scripts/prepare_deploy_artifacts.py
```
