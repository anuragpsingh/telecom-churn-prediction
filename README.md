# Telecom Churn Prediction Framework

A production-grade machine learning framework that predicts which telecom customers are likely to cancel their subscription ("churn"). The framework uses two complementary models — **XGBoost** and a **Tab-Transformer** (neural network) — and combines them into a soft-voting ensemble.

> **GPU Support:** The Tab-Transformer trains on the Apple M5 Pro's Metal GPU (MPS backend). The same code automatically switches to CUDA on NVIDIA GPUs or falls back to CPU.

---

## What is Customer Churn?

In the telecom industry, **churn** means a customer cancelling their subscription and moving to a competitor. Acquiring a new customer costs 5–25× more than retaining an existing one, so predicting churn *before it happens* is extremely valuable.

Given a customer's profile (their contract type, monthly charges, how often they complained, their data usage, etc.) the model outputs:
- A **churn probability** (e.g. 0.82 → 82% likely to churn)
- A **risk label** (`high_risk`, `medium_risk`, `low_risk`)
- Which model drove the decision (XGBoost score vs Transformer score)

---

## Results on 60,000 Synthetic Customers

| Model | AUROC ↑ | AUPRC ↑ | F1 ↑ | Accuracy ↑ |
|---|---|---|---|---|
| **Ensemble (XGB + Transformer)** | **0.9731** | 0.9276 | **0.8406** | **91.8%** |
| TabTransformer (MPS GPU) | 0.9722 | **0.9319** | 0.8125 | 88.4% |
| XGBoost (CPU) | 0.9718 | 0.9313 | 0.8315 | 90.1% |

> **AUROC** (Area Under ROC Curve): 1.0 = perfect, 0.5 = random guessing. 0.97 is excellent.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAW DATA  (60,000 customers)                  │
│  age, tenure, monthly_charges, contract_type, complaints, ...   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                    ┌───────▼────────┐
                    │  PREPROCESSOR  │
                    │                │
                    │ Numerical:     │
                    │  IQR cap       │
                    │  StandardScale │
                    │                │
                    │ Categorical:   │
                    │  OrdinalEncode │
                    └──────┬─────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
     ┌────────▼─────────┐    ┌──────────▼──────────┐
     │    XGBoost       │    │  Tab-Transformer      │
     │  (CPU / hist)    │    │  (MPS GPU on M5 Pro)  │
     │                  │    │                       │
     │  Gradient-boosted│    │  Embeddings           │
     │  decision trees  │    │  + Multi-Head Attn    │
     │                  │    │  + MLP head           │
     └────────┬─────────┘    └──────────┬────────────┘
              │   score_xgb             │  score_transformer
              └────────────┬────────────┘
                           │
                  ┌────────▼─────────┐
                  │  SOFT ENSEMBLE   │
                  │  50% + 50% blend │
                  │  + calibration   │
                  └────────┬─────────┘
                           │
              ┌────────────▼───────────────┐
              │       PREDICTION           │
              │  probability: 0.82         │
              │  label: churned            │
              │  segment: high_risk        │
              └────────────────────────────┘
```

---

## Project Structure

```
churn/
│
├── config.py               ← All settings in one place (hyperparams, paths, device)
├── data_generator.py       ← Creates synthetic telecom customer data
├── preprocessor.py         ← Cleans & scales features for model input
│
├── models/
│   ├── __init__.py         ← Makes `models` importable as a package
│   ├── xgboost_model.py    ← XGBoost classifier wrapper
│   └── transformer_model.py← Tab-Transformer (PyTorch, MPS GPU)
│
├── trainer.py              ← Runs the full training pipeline
├── evaluator.py            ← Computes metrics, plots, reports
├── inference.py            ← Production predictor (single + batch)
├── pipeline.py             ← CLI entry point — run this to train everything
│
├── checkpoints/            ← Saved model files (created automatically)
│   ├── preprocessor.pkl
│   ├── xgboost_churn.json
│   └── transformer_churn_best.pt
│
├── results/                ← Plots & evaluation report (created automatically)
│   ├── xgboost_roc.png
│   ├── ensemble_confusion.png
│   └── evaluation_report.json
│
├── data/                   ← Generated CSV (created automatically)
│   └── telecom_churn.csv
│
└── requirements.txt
```

---

## Quick Start

### 1. Prerequisites

- Python 3.11 or newer (tested on 3.13)
- Apple Silicon Mac **or** any machine with NVIDIA GPU **or** CPU-only

### 2. Install dependencies

```bash
# Clone the repo
git clone https://github.com/anuragpsingh10/telecom-churn-prediction.git
cd telecom-churn-prediction

# Install PyTorch (includes MPS support on macOS arm64 automatically)
pip install torch torchvision

# Install everything else
pip install -r requirements.txt

# macOS only — XGBoost needs OpenMP
brew install libomp
```

### 3. Train the models

```bash
# macOS 26 beta requires OMP_NUM_THREADS=1 (see Troubleshooting section)
OMP_NUM_THREADS=1 python pipeline.py
```

That's it. The pipeline will:
1. Generate 60,000 synthetic customer records
2. Split into train / validation / test sets
3. Train XGBoost
4. Train Tab-Transformer on the GPU
5. Evaluate both models + the ensemble
6. Save all artefacts to `checkpoints/` and `results/`

### 4. Run inference on new customers

```python
from inference import ChurnPredictor

# Load saved models
predictor = ChurnPredictor.from_checkpoints()

# Predict for one customer
result = predictor.predict_single({
    "age": 28,
    "tenure_months": 2,
    "monthly_charges": 95.0,
    "contract_type": "month-to-month",
    "num_complaints_6mo": 3,
    # ... all features
})

print(result.churn_probability)   # e.g. 0.87
print(result.risk_segment)        # "high_risk"
```

---

## CLI Options

```bash
python pipeline.py --help

# Examples:
OMP_NUM_THREADS=1 python pipeline.py                      # default (60k samples, 60 epochs)
OMP_NUM_THREADS=1 python pipeline.py --n-samples 10000    # faster run with less data
OMP_NUM_THREADS=1 python pipeline.py --epochs 10          # fewer training epochs
OMP_NUM_THREADS=1 python pipeline.py --skip-transformer   # XGBoost only (very fast)
OMP_NUM_THREADS=1 python pipeline.py --data data/telecom_churn.csv  # use existing CSV
```

---

## Code Walkthrough (Beginner-Friendly)

### `config.py` — Central Configuration

**Why it exists:** Instead of scattering magic numbers (`64`, `0.05`, `"hist"`) across every file, we put them all here. When you want to tune something, you change it in one place.

```python
# Detects which GPU (or CPU) is available
def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"    # Apple M-series (Metal GPU)
    if torch.cuda.is_available():
        return "cuda"   # NVIDIA GPU
    return "cpu"        # fallback
```

The config is built as **nested dataclasses** — Python objects that hold related settings together:

```python
@dataclass
class XGBoostConfig:
    n_estimators: int  = 1000     # how many trees to build
    learning_rate: float = 0.05   # how fast to learn (smaller = safer)
    max_depth: int = 6            # how deep each tree can grow
    ...
```

`ChurnConfig` is the top-level object that composes all sub-configs:

```python
cfg = ChurnConfig()     # one object to rule them all
cfg.xgboost.max_depth   # access nested settings like this
```

---

### `data_generator.py` — Synthetic Data

**Why synthetic data?** Real telecom data is proprietary and private. Synthetic data lets us build, test, and share the full framework freely, while being realistic enough to validate the approach.

**How the data is generated:**

```python
# Step 1: generate base features using statistical distributions
tenure_months = rng.integers(1, 73, n)        # uniform 1–72 months
monthly_charges = clip(normal(60, 20, n), 18, 120)  # bell curve around $60

# Step 2: compute billing from services subscribed
base += (internet_service == "Fiber_optic") * uniform(50, 80, n)
# Fiber optic adds $50–$80 to the bill

# Step 3: assign churn using domain knowledge
logit += 2.0   # if month-to-month contract → much higher churn risk
logit -= 0.04 * tenure_months   # longer tenure → lower risk
logit += 0.50 * num_complaints  # every complaint raises risk

# sigmoid converts logit to a probability
prob = 1 / (1 + exp(-logit))
churned = (prob >= threshold)   # binary label
```

The final churn rate is calibrated to exactly 26.5% (realistic for telecom).

---

### `preprocessor.py` — Data Cleaning & Scaling

Raw data can't go straight into a model because:
- Features have wildly different scales (age 18–80 vs total_charges 0–8000)
- Outliers (someone with 500 complaints) can dominate learning
- Categorical text ("month-to-month") must become numbers

**The preprocessing pipeline:**

```python
# Outlier Capper: clips extreme values using IQR
# IQR = Q3 - Q1 (the "middle 50%" range)
# Anything beyond Q3 + 3×IQR gets capped
self.lower_ = q1 - factor * iqr
self.upper_ = q3 + factor * iqr
X = np.clip(X, self.lower_, self.upper_)

# StandardScaler: rescales to mean=0, std=1
# Formula: z = (x - mean) / std
# Now every feature contributes equally to the model

# OrdinalEncoder: converts text categories to integers
# "month-to-month" → 0,  "one-year" → 1,  "two-year" → 2
```

We use sklearn's `ColumnTransformer` to apply different transformations to numerical vs categorical columns simultaneously.

```python
ct = ColumnTransformer([
    ("num", numerical_pipeline, NUMERICAL_FEATURES),
    ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
])
```

**Key principle: fit only on training data, transform all splits.**
If we fitted the scaler on test data, we'd be "leaking" test information into training — the model would appear better than it really is.

---

### `models/xgboost_model.py` — XGBoost

**What is XGBoost?**

XGBoost (eXtreme Gradient Boosting) builds an *ensemble of decision trees*, where each new tree corrects the mistakes of the previous ones. It's often called "gradient boosting" because it uses calculus (gradient descent) to improve predictions step by step.

```
Tree 1: predicts churn probability for all customers
         → makes mistakes on some customers
Tree 2: focuses on the customers Tree 1 got wrong
         → makes mistakes on others
Tree 3: focuses on the customers Tree 2 got wrong
         ...
Final: all 1000 trees vote → robust prediction
```

**Key hyperparameters explained:**

```python
n_estimators = 1000       # build up to 1000 trees
learning_rate = 0.05      # each tree contributes only 5% — prevents overfitting
max_depth = 6             # tree can ask at most 6 yes/no questions
subsample = 0.8           # each tree sees a random 80% of data (reduces overfitting)
scale_pos_weight = 2.77   # since only 26.5% churn, give churners 2.77× more weight
early_stopping_rounds = 50 # stop if val score doesn't improve for 50 rounds
```

**GPU note:** XGBoost does not yet support Apple's Metal GPU. `tree_method='hist'` is the fastest CPU algorithm and works well on M-series chips.

---

### `models/transformer_model.py` — Tab-Transformer (MPS GPU)

**What is a Transformer?** Originally designed for natural language (ChatGPT uses transformers), they work by letting every element of the input "pay attention" to every other element. For tabular data, each feature can attend to every other feature, learning interactions automatically.

**Architecture step by step:**

```
Input row: [age=35, tenure=3, contract="month-to-month", ...]

Step 1 — Embed every feature into the same vector space (d_model=64 dims)
  • Numerical: linear projection  35 → [0.2, -0.8, 0.4, ...]  (64 numbers)
  • Categorical: embedding lookup  "month-to-month"=0 → [0.9, 0.1, -0.3, ...] (64 numbers)

Step 2 — Add a CLS token (a learnable "summary" slot)
  [CLS] [age_emb] [tenure_emb] [contract_emb] ...   ← sequence of 34 vectors

Step 3 — Transformer Encoder (3 layers of Multi-Head Attention)
  Every vector attends to every other:
  "contract_type" attends to "monthly_charges", "num_complaints", etc.
  → learns "high charges + month-to-month is especially risky"

Step 4 — Take the CLS token output → MLP classifier
  CLS_out → Linear(64→128) → GELU → Linear(128→64) → Linear(64→1) → sigmoid → probability
```

**Running on the M5 Pro's Metal GPU:**

```python
device = torch.device("mps")          # Metal Performance Shaders
x = x.to(device)                      # move data to GPU
logits = self.net_(x, num_idx, cat_idx)  # computation happens on GPU
proba = torch.sigmoid(logits).cpu()   # move result back to CPU
```

**Why `batch_first=True` in the Transformer?** PyTorch's default shape is `[seq, batch, d_model]`. Setting `batch_first=True` changes it to `[batch, seq, d_model]`, which is more intuitive and works better with MPS.

---

### `trainer.py` — Training Orchestration

The trainer ties everything together:

```python
def run(self, train_df, val_df, test_df):
    # 1. Fit preprocessor on training data ONLY
    self.prep_.fit(train_df)
    X_train = self.prep_.transform(train_df)
    X_val   = self.prep_.transform(val_df)   # uses train's mean/std
    X_test  = self.prep_.transform(test_df)

    # 2. Train XGBoost (1–2 seconds on M5 Pro)
    self.xgb_model_.fit(X_train, y_train, X_val, y_val)

    # 3. Train Transformer (~50 seconds on MPS GPU)
    self.trn_model_.fit(X_train, y_train, X_val, y_val, num_idx, cat_idx)

    # 4. Evaluate ensemble
    ens_score = 0.5 * xgb_score + 0.5 * transformer_score
```

**Early stopping** (in the Transformer): if validation AUROC hasn't improved in 10 consecutive epochs, training stops and the best checkpoint is loaded. This prevents overfitting.

---

### `evaluator.py` — Metrics & Plots

**Metrics explained for beginners:**

| Metric | What it measures | Good value |
|---|---|---|
| **AUROC** | How well the model ranks churners above non-churners. 1.0 = perfect, 0.5 = coin flip | > 0.85 |
| **AUPRC** | Like AUROC but emphasises precision on the positive class (rare churners) | > 0.75 |
| **F1 Score** | Harmonic mean of Precision and Recall. Balances false positives vs false negatives | > 0.75 |
| **Precision** | Of customers we flagged as churners, what % actually churned? | > 0.70 |
| **Recall** | Of customers who actually churned, what % did we catch? | > 0.80 |
| **Brier Score** | Mean squared error of probabilities. Lower = better calibrated | < 0.10 |

**Bootstrap confidence intervals:**

```python
# We can't trust a single AUROC number — what's the uncertainty?
# Resample the test set 1000 times, compute AUROC each time
auroc_scores = [compute_auroc(resample(y_test, y_score)) for _ in range(1000)]
ci_low, ci_high = np.percentile(auroc_scores, [2.5, 97.5])
# Result: "AUROC = 0.9731 (95% CI [0.9703, 0.9759])"
```

**Optimal threshold search:**

The default decision threshold is 0.5 (predict churn if probability ≥ 50%). But this is rarely optimal. The evaluator sweeps all thresholds from 0.05 to 0.95 to find the one that maximises F1:

```python
for threshold in np.linspace(0.05, 0.95, 181):
    y_pred = (y_score >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_threshold = threshold
# Result: "Optimal threshold: 0.360 → F1=0.8440"
```

---

### `inference.py` — Production Predictor

The `ChurnPredictor` class is designed for deployment:

```python
# Load once at application startup (lazy loading)
predictor = ChurnPredictor.from_checkpoints()
predictor.warmup()   # triggers GPU kernel compilation

# Single prediction (real-time API call, ~1ms)
result = predictor.predict_single(customer_dict)
# Returns: ChurnPrediction(probability=0.87, risk="high_risk", latency_ms=0.9)

# Batch prediction (nightly batch job)
results_df = predictor.predict_batch(customers_df)
# Adds columns: churn_probability, churn_label, risk_segment
```

The **risk segments** let business teams prioritise interventions:
- `high_risk` (≥ 65%) → immediate retention call / special offer
- `medium_risk` (35–65%) → targeted email campaign
- `low_risk` (< 35%) → standard engagement

---

## Telecom Features Explained

| Feature | Type | Why it matters |
|---|---|---|
| `tenure_months` | Numerical | Longer customers are more loyal |
| `contract_type` | Categorical | Month-to-month → easiest to leave |
| `monthly_charges` | Numerical | High bills drive customers away |
| `num_complaints_6mo` | Numerical | Strongest single predictor of churn |
| `avg_call_drop_rate` | Numerical | Poor network = frustration = churn |
| `feature_adoption_score` | Numerical | More services used → harder to leave |
| `internet_service` | Categorical | Fiber optic users have higher expectations |
| `payment_method` | Categorical | Electronic check correlates with churn |
| `app_logins_monthly` | Numerical | Engaged users are less likely to leave |

---

## Troubleshooting

### XGBoost segfault on macOS 26 beta

**Symptom:** Python crashes with exit code 139 right after "Training XGBoost".

**Cause:** On macOS 26 (beta), OpenMP auto-detects CPU core count and crashes when spawning threads without `OMP_NUM_THREADS` set.

**Fix:** Always prefix the command:
```bash
OMP_NUM_THREADS=1 python pipeline.py
```

### MPS device not detected

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# Should print: True
```
If it prints `False`, your PyTorch may be the Linux/CUDA wheel. Reinstall with the standard macOS wheel:
```bash
pip uninstall torch torchvision
pip install torch torchvision
```

### Out of memory on MPS

Reduce `batch_size` in `config.py`:
```python
TransformerConfig(batch_size=128)   # default is 512
```

---

## Key ML Concepts (Beginner Glossary)

| Term | Meaning |
|---|---|
| **Training set** | Data the model learns from |
| **Validation set** | Data used during training to check if the model is overfitting |
| **Test set** | Data held out until the very end — the final honest evaluation |
| **Overfitting** | Model memorises training data but fails on new data |
| **Gradient Descent** | An optimisation algorithm that iteratively reduces prediction error |
| **Embedding** | A learned dense vector representation for a category |
| **Attention** | A mechanism that lets each token/feature look at all others |
| **Early Stopping** | Stop training when validation score stops improving |
| **Ensemble** | Combining multiple models (often better than any single model) |
| **Calibration** | Adjusting probabilities so that "80% confident" actually means 80% of the time correct |
| **AUROC** | A threshold-free measure of ranking quality |
| **MPS** | Apple's Metal Performance Shaders — the GPU compute API on M-series chips |

---

## License

MIT — free to use, modify, and distribute.
