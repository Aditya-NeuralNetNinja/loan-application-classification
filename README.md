# Loan Application Classification — HMDA 2023 Big Data ML Pipeline

A distributed machine learning pipeline for predicting mortgage loan approval/denial using the **HMDA 2023 Snapshot National Loan-Level Dataset** (10M+ records, 99 features, ~4 GB).

Built with **PySpark MLlib** for scalable processing and **Tableau** for interactive visualization.

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

Alternatively, download directly from the [CFPB website](https://ffiec.cfpb.gov/data-publication/snapshot-national-loan-level-dataset/2023) and place in `data/raw/`.

---

## Project Structure

```
project/
├── .gitignore
├── README.md
├── requirements.txt
├── config/
│   ├── spark_config.yaml          # SparkSession configuration
│   └── tableau_config.json        # Tableau export settings
├── data/
│   ├── raw/                       # Original CSV (gitignored)
│   ├── processed/                 # Parquet + EDA outputs (gitignored)
│   └── schemas/
│       └── hmda_schema.json       # 99-column schema with validation rules
├── notebooks/
│   ├── 1_data_ingestion.ipynb     # CSV → Parquet, schema validation
│   ├── 2_eda_comprehensive.ipynb  # Full EDA on all 99 features
│   ├── 3_feature_engineering.ipynb # Pipeline: Imputer → Encoder → Scaler
│   └── 4_model_training.ipynb     # LR, RF, GBT, SVM + evaluation
├── scripts/
│   ├── run_pipeline.py            # End-to-end execution script
│   └── performance_profiler.py    # CPU/memory tracking
├── tableau/
│   ├── dashboard1.twbx            # Data Quality
│   ├── dashboard2.twbx            # Model Performance
│   ├── dashboard3.twbx            # Fair Lending
│   └── dashboard4.twbx            # Scalability
└── tests/
    └── test_pipeline.py           # Schema & leakage checks
```

---

## Progress

### Completed

- [x] **Environment Setup** — Java 17, Python venv, PySpark 3.5, Jupyter kernel
- [x] **Schema Definition** — All 99 HMDA columns categorized into 15 groups with dtype classifications, leakage flags, and CFPB data dictionary mapping
- [x] **Notebook 1: Data Ingestion**
  - SparkSession with AQE, Arrow optimization, Kryo serialization
  - CSV load with corrupt record detection (PERMISSIVE mode)
  - Schema validation against `hmda_schema.json`
  - Null/missing analysis across all 99 columns (nulls + "Exempt"/"NA" codes)
  - Target variable distribution analysis
  - CSV → Parquet conversion with `state_code` partitioning (75% compression)
  - Initial profile export for downstream notebooks
- [x] **Notebook 2: Comprehensive EDA** (33 cells)
  - Univariate analysis: numeric distributions, skewness, kurtosis, IQR outlier detection
  - Univariate analysis: categorical cardinality, value counts, near-zero variance detection
  - Bivariate analysis: each feature vs. denial (t-tests, chi-square, Cramer's V)
  - Fair lending analysis: denial rates by race, ethnicity, sex
  - Pearson correlation matrix with multicollinearity detection
  - Missing value pattern analysis (MCAR/MAR/MNAR classification)
  - Data leakage audit (empirical verification of 12 leakage columns)
  - Feature engineering recommendations summary
- [x] **Dataset Upload** — CSV and Parquet hosted on Hugging Face
- [x] **Git Setup** — Repository with `.gitignore` (data files excluded)

### In Progress

- [ ] **Notebook 3: Feature Engineering**
  - Target encoding: `action_taken` → binary label (1→0, 3→1)
  - Missing value imputation (median for numeric, "Exempt" category for categorical)
  - Log transformation for right-skewed features (loan_amount, income, property_value)
  - Domain features: loan-to-income ratio, DTI buckets, high-cost flag, joint application indicator
  - PySpark ML Pipeline: StringIndexer → OneHotEncoder → VectorAssembler → StandardScaler
  - Custom Transformer: LoanRiskScorer
  - Stratified train/test split (80/20)

### Planned

- [ ] **Notebook 4: Model Training & Evaluation**
  - Logistic Regression (CrossValidator with regParam/elasticNetParam grid)
  - Random Forest (numTrees/maxDepth tuning)
  - Gradient Boosted Trees (maxDepth/stepSize optimization)
  - SVM via scikit-learn (1% sample for scalability comparison)
  - Bootstrap confidence intervals (1000 iterations)
  - Feature importance analysis (top 15 features)
  - Scalability analysis: strong scaling (50/100/200/400 partitions)
- [ ] **Tableau Dashboards**
  - Dashboard 1: Data Quality (missing values heatmap, class distribution, state map)
  - Dashboard 2: Model Performance (ROC curves, confusion matrix, metric comparison)
  - Dashboard 3: Fair Lending (denial rates by demographics, geographic approval heatmap)
  - Dashboard 4: Scalability (training time vs. data size, executor slider)
- [ ] **Testing** — Schema validation tests, data leakage checks, pipeline integration tests

---

## ML Approach

| Algorithm | Why | Implementation |
|-----------|-----|----------------|
| **Logistic Regression** | Interpretable baseline, coefficient analysis | PySpark MLlib |
| **Random Forest** | Feature importance, handles non-linearity | PySpark MLlib |
| **Gradient Boosted Trees** | Best predictive performance expected | PySpark MLlib |
| **SVM (Linear)** | scikit-learn comparison on sampled data | scikit-learn |

**Target:** Binary classification — Originated (0) vs. Denied (1)

**Key Challenges:**
- Class imbalance (~80% originated, ~20% denied)
- Data leakage from 12 post-decision columns (denial_reason, purchaser_type, etc.)
- Fair lending compliance under ECOA/HMDA regulations
- 10M+ row scale requiring distributed processing

---

## Setup

### Prerequisites
- Python 3.10+
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

# Download dataset
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('data/raw', exist_ok=True)
path = hf_hub_download(repo_id='adi-123/hmda-2023-snapshot', filename='hmda_2023.csv', repo_type='dataset')
os.symlink(path, 'data/raw/hmda_2023.csv')
print(f'Dataset ready at data/raw/hmda_2023.csv')
"

# Launch Jupyter
jupyter notebook
```

Select the **"HMDA BigData"** kernel and run notebooks in order (1 → 2 → 3 → 4).

---

## Key Findings (Preliminary from EDA)

- **Denial rates** vary significantly by demographic group, with minority applicants facing 2-3x higher denial rates
- **Income, DTI ratio, and interest rate** show the strongest statistical differences between denied and originated applications
- **12 columns** confirmed as data leakage sources through empirical fill-rate analysis
- **Class imbalance** of approximately 80/20 (originated/denied) requires stratified sampling and class-weighted models
- **Right-skewed** financial features (loan_amount, income, property_value) need log transformation

---

## Tech Stack

- **Distributed Processing:** PySpark 3.5 (MLlib, SQL, DataFrame API)
- **Data Format:** Apache Parquet with state-level partitioning
- **ML:** PySpark MLlib + scikit-learn (SVM baseline)
- **Visualization:** Matplotlib, Seaborn, Tableau
- **Statistical Testing:** SciPy (chi-square, t-tests, Cramer's V)
- **Dataset Hosting:** Hugging Face Datasets

---

## Team

| Member | Role |
|--------|------|
| Aditya | ML Pipeline, EDA, Feature Engineering |

---

## License

This project uses publicly available HMDA data published by the [CFPB/FFIEC](https://ffiec.cfpb.gov/) under federal open data guidelines.
