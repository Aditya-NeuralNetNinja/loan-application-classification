#!/usr/bin/env bash
set -euo pipefail

MODE="${APP_MODE:-streamlit}"
PORT="${PORT:-8000}"

if [ "$MODE" = "api" ]; then
  exec uvicorn app.api:app --host 0.0.0.0 --port "$PORT"
else
  exec streamlit run app/streamlit_app.py \
    --server.address 0.0.0.0 \
    --server.port "$PORT" \
    --server.headless true
fi
