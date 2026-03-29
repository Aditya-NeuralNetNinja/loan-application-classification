# Loan Application Classification — HMDA 2023 Big Data ML Pipeline

A distributed machine learning pipeline for predicting mortgage loan approval/denial using the **HMDA 2023 Snapshot National Loan-Level Dataset** (10M+ records, 99 features, ~4 GB).

Built with **PySpark MLlib** for scalable processing, a **FastAPI + Streamlit** deployment layer, and **Tableau** for interactive visualization.

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [CFPB HMDA 2023 Snapshot](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2023) |
| **Records** | 10,000,000+ loan applications |
| **Features** | 99 columns (loan, applicant, property, demographics, census) |
| **Size** | ~4 GB (CSV), ~800 MB (Parquet) |
| **Target** | `action_taken`: Originated (1) vs. Denied (3) |

### Download

The dataset is hosted on Hugging Face for easy access:

```python
from huggingface_hub import hf_hub_download

# Download CSV (~4 GB, cached after first download)
csv_path = hf_hub_download(
    repo_id="adi-123/hmda-2023-snapshot",
    filename="hmda_2023.csv",
    repo_type="dataset"
)
```

> **Note:** Replace `adi-123` with the actual Hugging Face username if different.

Alternatively, download directly from the [CFPB website](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2023) and place in `data/raw/`, or use the provided download script:

```bash
bash scripts/download_data.sh
```

---

## Project Structure

```
project/
├── README.md
├── DEPLOYMENT.md                    # Deployment guide (Streamlit, Docker, Railway, Render)
├── requirements.txt                 # Full dev dependencies
├── requirements.deploy.txt          # Lightweight deploy dependencies
├── Dockerfile                       # Container deployment
├── .dockerignore
├── Procfile                         # Heroku/Render process entry
├── render.yaml                      # Render blueprint
├── railway.json                     # Railway deployment config
├── runtime.txt                      # Python version pin (3.11.9)
├── config/
│   ├── spark_config.yaml            # SparkSession configuration
│   └── tableau_config.json          # Tableau export settings
├── data/
│   ├── raw/                         # Original CSV (gitignored)
│   ├── processed/                   # Parquet, model artifacts, JSON results
│   │   ├── hmda_2023.parquet/       # State-partitioned Parquet dataset
│   │   ├── pipeline_model/          # Fitted PySpark PipelineModel (Imputer → Indexers → OHE → Assembler → Scaler)
│   │   ├── models/                  # Trained model artifacts (best_lr, best_rf, best_gbt, best_dt)
│   │   ├── model_results.json       # Final evaluation metrics for all models
│   │   ├── feature_metadata.json    # Feature names, types, pipeline contract
│   │   └── optimal_threshold.json   # F1-optimized classification threshold
│   └── schemas/
│       └── hmda_schema.json         # 99-column schema with validation rules
├── notebooks/
│   ├── 1_data_ingestion.ipynb       # CSV → Parquet, schema validation, null analysis
│   ├── 2_eda_comprehensive.ipynb    # Full EDA: univariate, bivariate, fair lending, leakage audit
│   ├── 3_feature_engineering.ipynb  # Pipeline: Imputer → StringIndexer → OHE → VectorAssembler → Scaler
│   ├── 4a_model_training.ipynb      # 8 models: Baseline, NB, LR, SVM, DT, RF, GBT, MLP
│   ├── 4b_ensembles_evaluation.ipynb # Ensembles, head-to-head comparison, threshold optimization
│   └── 5_model_diagnostics_deepdive.ipynb # Bootstrap CIs, error analysis, scaling experiments, deployment reco
├── scripts/
│   ├── download_data.sh             # Download HMDA LAR/TS/Panel CSVs from CFPB S3
│   └── prepare_deploy_artifacts.py  # Copy processed artifacts into app/assets for deployment
├── app/
│   ├── streamlit_app.py             # Interactive dashboard: leaderboard, threshold policy, CSV inference
│   ├── api.py                       # FastAPI service: /models, /recommend, /inference endpoints
│   ├── data_loader.py               # Shared artifact loading, recommendation logic, business cost table
│   ├── inference.py                 # Spark preprocessing + GBT batch inference backend
│   ├── config.py                    # App configuration
│   ├── entrypoint.sh                # Starts API or Streamlit via APP_MODE env var
│   └── assets/                      # Committed lightweight sample artifacts for demo mode
│       ├── model_results.sample.json
│       ├── model_leaderboard.sample.csv
│       ├── optimal_threshold.sample.json
│       └── demo_inference_100_rows.csv
└── tableau/
    ├── dashboard1.twbx              # Data Quality
    ├── dashboard2.twbx              # Model Performance
    ├── dashboard3.twbx              # Fair Lending
    └── dashboard4.twbx              # Scalability
```

---

## Pipeline Overview

### Notebook 1 — Data Ingestion
- SparkSession with AQE, Arrow optimization, Kryo serialization
- CSV load with corrupt record detection (PERMISSIVE mode)
- Schema validation against `hmda_schema.json`
- Null/missing analysis across all 99 columns (nulls + "Exempt"/"NA" codes)
- Target variable distribution analysis
- CSV → Parquet conversion with `state_code` partitioning (75% compression)

### Notebook 2 — Comprehensive EDA
- Univariate analysis: numeric distributions, skewness, kurtosis, IQR outlier detection
- Univariate analysis: categorical cardinality, value counts, near-zero variance detection
- Bivariate analysis: each feature vs. denial (t-tests, chi-square, Cramer's V)
- Fair lending analysis: denial rates by race, ethnicity, sex
- Pearson correlation matrix with multicollinearity detection
- Missing value pattern analysis (MCAR/MAR/MNAR classification)
- Data leakage audit (empirical verification of 12 leakage columns)

### Notebook 3 — Feature Engineering
- Column name normalization and useless/dangerous column removal
- Target encoding: `action_taken` → binary label (Originated=0, Denied=1)
- HMDA triple-missing normalization ("Exempt", "NA", null → unified handling)
- Numeric transformations: log transforms for right-skewed features
- PySpark ML Pipeline: Imputer → StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler
- Stratified train/test split (80/20) with post-pipeline quality checks

### Notebook 4a — Model Training & Tuning
Eight models trained with class-weight handling and cross-validation:
1. **Majority-Class Baseline** — naive classifier for reference
2. **Naive Bayes** — simplest probabilistic classifier
3. **Logistic Regression** — linear baseline with regParam/elasticNetParam grid
4. **Linear SVM** — maximum margin classification
5. **Decision Tree** — interpretable if-then rules
6. **Random Forest** — variance reduction via bagging
7. **Gradient Boosted Trees** — tabular data champion
8. **Multilayer Perceptron (MLP)** — neural network on tabular data

### Notebook 4b — Ensembles & Evaluation
- Ensemble engineering: model combination strategies
- Comprehensive head-to-head comparison across all models
- F1-optimized threshold selection
- MLflow hyperparameter analysis
- Dataset-specific performance analysis

### Notebook 5 — Model Diagnostics Deep Dive
- Bootstrap confidence intervals (uncertainty quantification)
- Residual and error analysis
- Business-oriented evaluation metrics (cost modeling)
- Strong and weak scaling experiments
- Evidence-based deployment recommendation

---

## ML Approach

| Algorithm | Implementation | Purpose |
|-----------|---------------|---------|
| **Majority-Class Baseline** | PySpark | Naive reference point |
| **Naive Bayes** | PySpark MLlib | Probabilistic baseline |
| **Logistic Regression** | PySpark MLlib | Interpretable linear model |
| **Linear SVM** | PySpark MLlib | Maximum margin classifier |
| **Decision Tree** | PySpark MLlib | Interpretable non-linear model |
| **Random Forest** | PySpark MLlib | Bagged ensemble, feature importance |
| **Gradient Boosted Trees** | PySpark MLlib | Best expected performance |
| **Multilayer Perceptron** | PySpark MLlib | Neural network comparison |
| **Ensembles** | Custom (4b) | Combined model strategies |

**Target:** Binary classification — Originated (0) vs. Denied (1)

**Key Challenges:**
- Class imbalance (~80% originated, ~20% denied)
- Data leakage from 12 post-decision columns (denial_reason, purchaser_type, etc.)
- Fair lending compliance under ECOA/HMDA regulations
- 10M+ row scale requiring distributed processing

---

## Deployment

The project includes a full deployment layer. See [DEPLOYMENT.md](DEPLOYMENT.md) for details.

### Streamlit Dashboard
```bash
pip install -r requirements.deploy.txt
streamlit run app/streamlit_app.py
```
Features: model leaderboard, threshold policy selector, business cost analysis, CSV upload for batch inference.

### FastAPI Service
```bash
uvicorn app.api:app --host 0.0.0.0 --port 8000
```
Endpoints: `/models`, `/recommend`, `/inference/predict-csv`, `/health`

### Docker
```bash
docker build -t hmda-deploy .
docker run -e APP_MODE=streamlit -e PORT=8000 -p 8000:8000 hmda-deploy
```

### Cloud Platforms
- **Render**: `render.yaml` blueprint included
- **Railway**: `railway.json` config included
- **Heroku**: `Procfile` + `runtime.txt` included

---

## Setup

### Prerequisites
- Python 3.10+ (3.11.9 pinned for deployment)
- Java 17 (for PySpark)

### Installation

```bash
# Clone repository
git clone https://github.com/Aditya-NeuralNetNinja/loan-application-classification.git
cd loan-application-classification

# Create virtual environment
python3 -m venv hmda_venv
source hmda_venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Register Jupyter kernel
python -m ipykernel install --user --name=hmda_venv --display-name "HMDA BigData"

# Download dataset (option A: Hugging Face)
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('data/raw', exist_ok=True)
path = hf_hub_download(repo_id='adi-123/hmda-2023-snapshot', filename='hmda_2023.csv', repo_type='dataset')
os.symlink(path, 'data/raw/hmda_2023.csv')
print(f'Dataset ready at data/raw/hmda_2023.csv')
"

# Download dataset (option B: CFPB direct)
bash scripts/download_data.sh

# Launch Jupyter
jupyter notebook
```

Run notebooks in order: **1 → 2 → 3 → 4a → 4b → 5**.

---

## Key Findings (from EDA)

- **Denial rates** vary significantly by demographic group, with minority applicants facing 2-3x higher denial rates
- **Income, DTI ratio, and interest rate** show the strongest statistical differences between denied and originated applications
- **12 columns** confirmed as data leakage sources through empirical fill-rate analysis
- **Class imbalance** of approximately 80/20 (originated/denied) requires stratified sampling and class-weighted models
- **Right-skewed** financial features (loan_amount, income, property_value) need log transformation

---

## Tech Stack

- **Distributed Processing:** PySpark 3.5 (MLlib, SQL, DataFrame API)
- **Data Format:** Apache Parquet with state-level partitioning
- **ML:** PySpark MLlib (8 model types + ensembles)
- **Visualization:** Matplotlib, Seaborn, Tableau
- **Statistical Testing:** SciPy (chi-square, t-tests, Cramer's V, bootstrap CIs)
- **Web App:** Streamlit (dashboard), FastAPI (API)
- **Deployment:** Docker, Render, Railway, Heroku
- **Dataset Hosting:** Hugging Face Datasets

---

## Saved Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| Fitted Pipeline | `data/processed/pipeline_model/` | Imputer → 11 StringIndexers → OHE → Assembler → Scaler |
| Best LR | `data/processed/models/best_lr/` | Logistic Regression model |
| Best RF | `data/processed/models/best_rf/` | Random Forest model |
| Best GBT | `data/processed/models/best_gbt/` | Gradient Boosted Trees model |
| Best DT | `data/processed/models/best_dt/` | Decision Tree model |
| Model Results | `data/processed/model_results.json` | Metrics for all models |
| Feature Metadata | `data/processed/feature_metadata.json` | Pipeline feature contract |
| Optimal Threshold | `data/processed/optimal_threshold.json` | F1-optimized threshold |

---

## Team

| Member | Role |
|--------|------|
| Aditya | ML Pipeline, EDA, Feature Engineering, Deployment |

---

## License

This project uses publicly available HMDA data published by the [CFPB/FFIEC](https://ffiec.cfpb.gov/) under federal open data guidelines.
