FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.deploy.txt /app/requirements.deploy.txt
RUN pip install --no-cache-dir -r /app/requirements.deploy.txt

COPY app /app/app
COPY README.md /app/README.md
COPY DEPLOYMENT.md /app/DEPLOYMENT.md
COPY data/schemas/hmda_schema.json /app/data/schemas/hmda_schema.json
COPY data/processed/feature_metadata.json /app/data/processed/feature_metadata.json
COPY data/processed/optimal_threshold.json /app/data/processed/optimal_threshold.json
COPY data/processed/pipeline_model /app/data/processed/pipeline_model
COPY data/processed/models/best_gbt /app/data/processed/models/best_gbt

EXPOSE 8000

CMD ["/app/app/entrypoint.sh"]
