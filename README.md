# Telecom Mobile Churn Prediction Framework

A production-grade machine learning framework that predicts which **mobile subscribers** are likely to cancel their subscription ("churn"). The framework uses two complementary models — **XGBoost** and a **Tab-Transformer** (neural network) — and combines them into a calibrated soft-voting ensemble.

> **GPU Support:** The Tab-Transformer trains on the Apple M5 Pro's Metal GPU (MPS). The same code auto-switches to CUDA on NVIDIA GPUs, or falls back to CPU.

---

## What is Mobile Churn?

In the mobile industry, **churn** means a subscriber cancelling their SIM, porting to a competitor, or simply stopping top-ups on a prepaid account. Retaining an existing subscriber costs far less than acquiring a new one, so predicting churn *before it happens* is extremely valuable for CRM teams.

Given a subscriber's profile — their plan type, monthly cost, network signal quality (SINR/RSRP), call drop rate, top-up behaviour, billing history, and complaint count — the model outputs:

- A **churn probability** (e.g. `0.87` → 87% likely to churn)
- A **risk label** (`high_risk`, `medium_risk`, `low_risk`)
- Individual scores from XGBoost and the Transformer for explainability

---

## Results on 60,000 Synthetic Mobile Subscribers

| Model | AUROC ↑ | AUPRC ↑ | F1 ↑ | Accuracy ↑ |
|---|---|---|---|---|
| **Ensemble (XGB + Transformer)** | **0.9465** | 0.8676 | **0.7834** | **88.0%** |
| XGBoost (CPU) | 0.9462 | **0.8784** | 0.7822 | 86.4% |
| TabTransformer (MPS GPU) | 0.9427 | 0.8702 | 0.7769 | 86.2% |

> Optimal decision threshold: **0.325** (instead of default 0.5) → F1 = **0.790**

> **AUROC** = 1.0 means perfect ranking; 0.5 = random. 0.94+ is production-grade.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│              RAW MOBILE SUBSCRIBER DATA  (60,000 records)            │
│  SINR, RSRP, Call Success Rate, Bearer Establishment Rate,           │
│  Plan Type (prepaid/postpaid), Monthly Cost (EUR), Top-ups,          │
│  Daily Data Usage (MB), Complaints, Contract Duration, ...           │
└────────────────────────────┬─────────────────────────────────────────┘
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
                            │  32 scaled features
               ┌────────────┴────────────┐
               │                         │
      ┌─────────▼──────────┐   ┌──────────▼──────────┐
      │    XGBoost         │   │   Tab-Transformer     │
      │  (CPU / hist)      │   │   (MPS GPU, M5 Pro)   │
      │                    │   │                       │
      │  Gradient-boosted  │   │  Per-feature embeds   │
      │  decision trees    │   │  + Multi-Head Attn    │
      │  1000 trees max    │   │  + CLS token + MLP    │
      │  early stopping    │   │  early stopping       │
      └─────────┬──────────┘   └──────────┬────────────┘
                │  score_xgb              │  score_transformer
                └────────────┬────────────┘
                             │  50% / 50% blend
                    ┌────────▼──────────┐
                    │   SOFT ENSEMBLE   │
                    │ + isotonic calib  │
                    └────────┬──────────┘
                             │
               ┌─────────────▼──────────────┐
               │         PREDICTION          │
               │  probability:  0.87         │
               │  label:        churned      │
               │  segment:      high_risk    │
               └─────────────────────────────┘
```

---

## Project Structure

```
churn/
│
├── config.py                ← All settings: feature schema, hyperparams, device detection
├── data_generator.py        ← Synthetic mobile subscriber data (prepaid + postpaid)
├── preprocessor.py          ← IQR outlier capping + StandardScaler + OrdinalEncoder
│
├── models/
│   ├── __init__.py
│   ├── xgboost_model.py     ← XGBoost wrapper (early stopping, feature importance)
│   └── transformer_model.py ← Tab-Transformer (PyTorch, MPS GPU)
│
├── trainer.py               ← Orchestrates both models + ensemble + calibration
├── evaluator.py             ← AUROC/AUPRC/F1 + bootstrap CIs + 4 plots per model
├── inference.py             ← Production predictor: single subscriber or batch CSV
├── pipeline.py              ← CLI entry point — run this to train everything
│
├── checkpoints/             ← Auto-created: preprocessor.pkl, model .json/.pt files
├── results/                 ← Auto-created: ROC, PR, confusion, calibration plots
├── data/                    ← Auto-created: mobile_churn.csv
└── requirements.txt
```

---

## Feature Schema (32 Features)

### Numerical Features (23)

#### Demographics
| Feature | Description |
|---|---|
| `age` | Subscriber age in years (16–77) |

#### Account & Tenure
| Feature | Description |
|---|---|
| `tenure_months` | Months since SIM activation |
| `monthly_plan_cost_eur` | Base plan price in EUR (5–20 EUR) |
| `total_spend_6mo_eur` | Actual total spend over last 6 months (EUR) |

#### Network Quality KPIs (LTE / 5G)
| Feature | Range | Interpretation |
|---|---|---|
| `avg_sinr_db` | −10 to +32 dB | **SINR** (Signal-to-Interference-plus-Noise Ratio). > 10 dB = good; < 0 dB = very poor |
| `avg_rsrp_dbm` | −120 to −65 dBm | **RSRP** (Reference Signal Received Power). > −80 dBm = excellent; < −100 dBm = poor |
| `call_success_rate_pct` | 88–99.9 % | % of voice calls successfully connected. < 97 % triggers frustration |
| `bearer_establishment_rate_pct` | 88–99.9 % | % of data bearer sessions successfully set up. < 97 % is problematic |
| `avg_call_drop_rate_pct` | 0.05–8 % | % of active calls that drop mid-conversation. > 2 % is a strong churn signal |
| `avg_data_speed_mbps` | 0.1–350 Mbps | Average download speed (3G: 0.5–7 / 4G: 10–80 / 5G: 50–300) |

#### Usage
| Feature | Description |
|---|---|
| `avg_monthly_data_gb` | Monthly mobile data consumption (GB) |
| `avg_daily_data_usage_mb` | Daily data consumption in MB (derived: monthly GB × 1024 / 30) |
| `avg_daily_call_minutes` | Average call duration per day (minutes) |
| `avg_monthly_sms` | Monthly SMS count |
| `roaming_sessions_6mo` | Number of roaming sessions in last 6 months |
| `intl_call_minutes_monthly` | Monthly international call minutes |

#### Billing
| Feature | Prepaid | Postpaid |
|---|---|---|
| `num_topups_6mo` | ✅ Low count = churn risk | 0 (n/a) |
| `avg_topup_amount_eur` | ✅ Average recharge value | 0 (n/a) |
| `num_late_payments_6mo` | 0 (n/a) | ✅ Financial stress signal |
| `data_overage_charges_eur` | ✅ Unexpected bill shock | ✅ Unexpected bill shock |

#### Customer Service
| Feature | Description |
|---|---|
| `num_complaints_6mo` | Number of formal complaints in last 6 months |
| `num_support_calls_6mo` | Number of calls to customer support |
| `days_since_last_complaint` | Days since most recent complaint (180 if none) |

---

### Categorical Features (9)

| Feature | Values | Notes |
|---|---|---|
| `plan_type` | `prepaid` / `postpaid` | 40% / 60% split |
| `plan_tier` | `basic` / `standard` / `premium` / `unlimited` | Maps to 5–9 / 9–14 / 14–18 / 17–20 EUR |
| `contract_duration` | `prepaid` / `monthly` / `12_month` / `24_month` | Monthly = highest churn risk |
| `payment_method` | `direct_debit` / `credit_card` / `mobile_wallet` / `cash` | |
| `network_generation` | `3G` / `4G` / `5G` | Affects speed, SINR baseline |
| `device_type` | `smartphone` / `feature_phone` / `tablet` / `mifi` | |
| `gender` | `Male` / `Female` | |
| `is_roaming_enabled` | `yes` / `no` | |
| `has_handset_subsidy` | `yes` / `no` | Subsidised phone → locked in → lower churn |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+ (tested on 3.13)
- Apple Silicon Mac (M1–M5) **or** NVIDIA GPU **or** CPU-only

### 2. Install

```bash
git clone https://github.com/anuragpsingh/telecom-churn-prediction.git
cd telecom-churn-prediction

# PyTorch — MPS support included in the standard macOS arm64 wheel
pip install torch torchvision

# Remaining dependencies
pip install -r requirements.txt

# macOS: XGBoost requires OpenMP
brew install libomp
```

### 3. Train

```bash
# OMP_NUM_THREADS=1 is required on macOS 26 beta (see Troubleshooting)
OMP_NUM_THREADS=1 python pipeline.py
```

The pipeline will:
1. Generate 60,000 synthetic mobile subscriber records
2. Stratified train / val / test split (75 / 10 / 15 %)
3. Fit preprocessor on training data only
4. Train XGBoost (~1 second on M5 Pro)
5. Train Tab-Transformer on MPS GPU (~35 seconds)
6. Evaluate both + ensemble, save plots and JSON report
7. Save all artefacts to `checkpoints/` and `results/`

### 4. CLI Options

```bash
OMP_NUM_THREADS=1 python pipeline.py                          # default: 60k samples, 60 epochs
OMP_NUM_THREADS=1 python pipeline.py --n-samples 10000        # faster test run
OMP_NUM_THREADS=1 python pipeline.py --epochs 10              # fewer epochs
OMP_NUM_THREADS=1 python pipeline.py --skip-transformer       # XGBoost only (very fast)
OMP_NUM_THREADS=1 python pipeline.py --data data/mobile_churn.csv  # reuse existing data
```

### 5. Run Inference

```python
from inference import ChurnPredictor

predictor = ChurnPredictor.from_checkpoints()

# High-risk prepaid subscriber with poor signal and low top-ups
result = predictor.predict_single({
    "age": 24,
    "tenure_months": 3.0,
    "monthly_plan_cost_eur": 8.5,
    "total_spend_6mo_eur": 22.0,
    "avg_sinr_db": -2.5,
    "avg_rsrp_dbm": -107.0,
    "call_success_rate_pct": 93.5,
    "bearer_establishment_rate_pct": 94.1,
    "avg_call_drop_rate_pct": 5.2,
    "avg_data_speed_mbps": 2.1,
    "avg_monthly_data_gb": 1.2,
    "avg_daily_data_usage_mb": 41.0,
    "avg_daily_call_minutes": 18.0,
    "avg_monthly_sms": 45,
    "roaming_sessions_6mo": 0,
    "intl_call_minutes_monthly": 0.0,
    "num_topups_6mo": 1,
    "avg_topup_amount_eur": 7.0,
    "num_late_payments_6mo": 0,
    "data_overage_charges_eur": 0.0,
    "num_complaints_6mo": 4,
    "num_support_calls_6mo": 5,
    "days_since_last_complaint": 8.0,
    "plan_type": "prepaid",
    "plan_tier": "basic",
    "contract_duration": "prepaid",
    "payment_method": "cash",
    "network_generation": "4G",
    "device_type": "smartphone",
    "gender": "Male",
    "is_roaming_enabled": "no",
    "has_handset_subsidy": "no",
}, subscriber_id="SUB-0042")

print(result.churn_probability)   # e.g. 0.87
print(result.risk_segment)        # "high_risk"
print(result.latency_ms)          # e.g. 1.2 ms

# Batch prediction from CSV
results_df = predictor.predict_batch_from_file(
    "data/mobile_churn.csv",
    output_path="results/predictions.csv",
)
```

**Risk segments and recommended actions:**
| Segment | Threshold | CRM Action |
|---|---|---|
| `high_risk` | ≥ 65% | Immediate outbound call + retention offer |
| `medium_risk` | 35–65% | Targeted SMS / email campaign |
| `low_risk` | < 35% | Standard engagement / upsell |

---

## Code Walkthrough (Beginner-Friendly)

### `config.py` — Central Configuration

Every number, path, and setting lives here. No magic numbers scattered across files.

```python
# Automatically detects the best available compute device
def _detect_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"    # Apple M-series (Metal GPU) ← M5 Pro uses this
    if torch.cuda.is_available():
        return "cuda"   # NVIDIA GPU
    return "cpu"        # fallback
```

Settings are grouped into **dataclasses** — Python objects that hold related values:

```python
@dataclass
class XGBoostConfig:
    n_estimators: int   = 1000   # build up to 1000 trees
    learning_rate: float = 0.05  # small = careful learning
    max_depth: int      = 6      # max depth of each decision tree
    scale_pos_weight: float = 2.57  # up-weights churners (only 28% of data)
```

The plan price ranges are also centralised:
```python
PLAN_PRICE_RANGES = {
    "basic":     (5.0,  9.0),    # EUR
    "standard":  (9.0,  14.0),
    "premium":   (14.0, 18.0),
    "unlimited": (17.0, 20.0),
}
```

---

### `data_generator.py` — Synthetic Mobile Data

Real subscriber data is private. Synthetic data lets us build and share the full framework freely while being statistically realistic.

The generator runs in 6 sequential stages:

```
Stage 1 — Demographics      (age, gender)
Stage 2 — Plan & Contract   (plan type, tier, cost, contract, payment method)
Stage 3 — Network Quality   (SINR, RSRP, Call Success Rate, Bearer Rate, Drop Rate)
Stage 4 — Usage             (data GB, daily MB, calls, SMS, roaming)
Stage 5 — Billing           (top-ups, late payments, overage charges)
Stage 6 — Customer Service  (complaints, support calls)
Stage 7 — Churn Assignment  (domain-driven logit model)
```

**How SINR and RSRP are generated:**

```python
# SINR baseline depends on network generation
sinr_base = {"5G": 15.0, "4G": 11.0, "3G": 6.0}[network_generation]
avg_sinr_db = clip(normal(sinr_base, std=7.0), low=-10, high=32)

# RSRP correlates with SINR — better signal = higher received power
rsrp_base = -85.0 + 0.8 * (sinr_db - 10.0)
avg_rsrp_dbm = clip(normal(rsrp_base, std=8.0), low=-120, high=-65)
```

**How Call Success Rate is derived from SINR:**

```python
# Poor SINR (< 10 dB) degrades call success rate
csr_base = 98.5 - 0.15 * max(0, 10.0 - sinr_db)
call_success_rate_pct = clip(normal(csr_base, std=0.8), 88, 99.9)
```

**How churn is assigned (domain-driven logit model):**

```python
logit = 0.0

# Contract risk (month-to-month = easiest to leave)
logit += {"monthly": 1.8, "prepaid": 0.8, "12_month": 0.0, "24_month": -1.2}[contract]

# Longer tenure → more loyal
logit -= 0.025 * tenure_months

# Network frustration
logit += 0.12 * max(0, 5.0 - sinr_db)          # poor SINR
logit += 0.18 * max(0, 97.0 - call_success_rate_pct)  # dropped calls
logit += 0.20 * max(0, drop_rate_pct - 2.0)     # high call drop rate

# Prepaid lapse signal: not topping up is the clearest sign of leaving
logit += 0.35 * max(0, 4.0 - num_topups_6mo)    # prepaid only

# Complaints are the strongest individual predictor
logit += 0.55 * num_complaints_6mo

# Loyalty anchors (reduce churn)
logit -= 0.60 * has_handset_subsidy   # locked in by phone repayment
logit -= 0.25 * (plan == "unlimited")  # all-in customers rarely leave

# Convert logit to probability
prob = 1 / (1 + exp(-logit))
churned = (prob >= threshold_for_28_percent_churn_rate)
```

---

### `preprocessor.py` — Data Cleaning & Scaling

Raw features can't go straight into models because:
- Different scales: `avg_sinr_db` ranges −10 to +32 while `total_spend_6mo_eur` can be 0–200
- Outliers in `num_complaints_6mo` can dominate the model
- Categorical text (`"month-to-month"`) must become numbers

**Pipeline steps:**

```python
# Step 1 — Outlier Capper (IQR method)
# IQR = Q3 − Q1 (the middle 50% of values)
lower_fence = Q1 − 3 × IQR
upper_fence = Q3 + 3 × IQR
X = clip(X, lower_fence, upper_fence)   # cap extremes

# Step 2 — StandardScaler
# Makes every feature have mean=0 and std=1
# Formula: z = (x − mean) / std
# Now SINR and monthly_cost contribute equally to learning

# Step 3 — OrdinalEncoder (categorical only)
# Converts text to integer codes:
# "prepaid" → 0,  "postpaid" → 1
# "basic"   → 0,  "standard" → 1,  "premium" → 2,  "unlimited" → 3
```

**Critical rule: fit only on training data.**
The scaler learns mean/std from the training set only. Applying it to val/test ensures no information leaks:

```python
prep.fit(train_df)        # learn statistics from train only
X_train = prep.transform(train_df)
X_val   = prep.transform(val_df)   # uses train's mean/std
X_test  = prep.transform(test_df)  # uses train's mean/std
```

---

### `models/xgboost_model.py` — XGBoost

**What is XGBoost?**
It builds a sequence of decision trees where each new tree corrects the mistakes of the previous ones — like a team where each new member focuses on what the rest got wrong.

```
Tree 1 → predicts for all 45,000 subscribers → makes mistakes
Tree 2 → focuses on subscribers Tree 1 got wrong → makes mistakes
Tree 3 → focuses on Tree 2's mistakes
...
Tree 190 → early stopping triggered (val AUPRC stopped improving)

Final: all 190 trees vote → robust churn probability
```

**Key parameters:**

```python
scale_pos_weight = 2.57   # (1 - 0.28) / 0.28
# Without this, the model ignores churners (only 28% of data)
# With it, each churner counts 2.57× as much as a non-churner

early_stopping_rounds = 50
# If the validation AUPRC doesn't improve for 50 consecutive trees, stop.
# Prevents memorising training data.

tree_method = "hist"
# Best CPU algorithm for large datasets on Apple M-series
# Bins continuous values into histograms for fast splitting
```

---

### `models/transformer_model.py` — Tab-Transformer (MPS GPU)

**What is a Transformer?**
Originally invented for language models (GPT, BERT), transformers use "attention" to let every input token look at every other token. For a table row, each feature can attend to every other feature — automatically learning that "low SINR + high complaint count" is a stronger churn signal than either alone.

**Architecture step by step:**

```
Input row (32 features after preprocessing):

Step 1 — Project every feature to the same dimension (d_model=64)
  Numerical features → Linear layer:
    avg_sinr_db = -2.5  →  [-0.3, 0.8, 0.1, ..., 0.5]  (64 numbers)

  Categorical features → Embedding lookup table:
    plan_type = "prepaid" → index 0 → [0.9, -0.2, ..., 0.4]  (64 numbers)

Step 2 — Prepend a CLS token (a learnable "summary" vector)
  Sequence: [CLS] [sinr_emb] [rsrp_emb] [csr_emb] [plan_emb] ...
  Shape: [batch_size, 33, 64]  (1 CLS + 32 features, each 64-dim)

Step 3 — Transformer Encoder (3 layers of Multi-Head Attention)
  Every feature attends to every other feature:
  "avg_sinr_db" can attend to "num_complaints_6mo", "plan_type", etc.
  The model learns: "poor signal AND many complaints → very high churn risk"

Step 4 — Take the CLS token's output vector
  CLS aggregates global context from all feature interactions
  Shape: [batch_size, 64]

Step 5 — MLP classifier head
  Linear(64 → 128) → GELU → Dropout → Linear(128 → 64) → Linear(64 → 1)
  → sigmoid → churn probability ∈ [0, 1]
```

**Running on the Apple M5 Pro Metal GPU:**

```python
device = torch.device("mps")        # Metal Performance Shaders
x = x.to(device)                    # move data to GPU memory
out = self.transformer(x)           # attention computed on GPU
proba = torch.sigmoid(out).cpu()    # result back to CPU
```

> XGBoost has no Apple Metal backend. It uses `tree_method='hist'` with a single OpenMP thread for stability on macOS 26 beta.

---

### `trainer.py` — Training Orchestration

```python
def run(self, train_df, val_df, test_df):

    # 1. Fit preprocessor on train only (prevents data leakage)
    prep.fit(train_df)
    X_train, X_val, X_test = prep.transform(each split)

    # 2. XGBoost — ~1.3 seconds on M5 Pro
    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
    # Early stopping: stops at ~190 trees when val AUPRC plateaus

    # 3. TabTransformer — ~35 seconds on MPS GPU
    transformer.fit(X_train, y_train, X_val, y_val)
    # Early stopping: stops at epoch ~19 when val AUROC plateaus

    # 4. Ensemble (soft voting)
    ens_score = 0.5 × xgb_score + 0.5 × transformer_score

    # 5. Isotonic calibration
    # Adjusts probabilities so "80% predicted" ≈ "80% actual"
```

---

### `evaluator.py` — Metrics & Plots

**Every metric explained:**

| Metric | Formula | What it means |
|---|---|---|
| **AUROC** | Area under ROC curve | How well the model *ranks* churners above non-churners. 1.0 = perfect |
| **AUPRC** | Area under Precision-Recall curve | Better than AUROC when classes are imbalanced (28% churn) |
| **F1** | 2 × P × R / (P + R) | Balances precision and recall — the practical performance metric |
| **Precision** | TP / (TP + FP) | "Of subscribers flagged as churners, how many actually churned?" |
| **Recall** | TP / (TP + FN) | "Of subscribers who actually churned, how many did we catch?" |
| **Brier** | mean((prob − label)²) | Probability calibration quality. Lower is better |

**Bootstrap confidence intervals (why they matter):**

```python
# A single AUROC number on a test set has sampling uncertainty.
# Bootstrap: resample the test set 1000 times with replacement
for _ in range(1000):
    idx = random_sample_with_replacement(len(y_test))
    auroc_samples.append(roc_auc_score(y_test[idx], y_score[idx]))

ci_low, ci_high = percentile(auroc_samples, [2.5, 97.5])
# "AUROC = 0.9465  (95% CI [0.9419, 0.9506])"
# The true model AUROC on the population lies in this range with 95% confidence
```

**Optimal threshold search:**

The default 0.50 threshold is arbitrary. For this dataset, 0.325 gives the best F1:

```python
for threshold in linspace(0.05, 0.95, 181):
    f1 = f1_score(y_true, (y_score >= threshold))
# Best found: threshold=0.325 → F1=0.790
```

Plots saved to `results/` for every model:
- `*_roc.png`          — ROC curve
- `*_pr.png`           — Precision-Recall curve
- `*_confusion.png`    — Confusion matrix
- `*_calibration.png`  — Predicted probability vs actual fraction
- `xgboost_feature_importance.png` — Top-20 features by gain

---

### `inference.py` — Production Predictor

Designed for deployment — load once, predict fast:

```python
# At application startup (models load from disk once)
predictor = ChurnPredictor.from_checkpoints()
predictor.warmup()     # compile MPS/CUDA kernels on a dummy batch

# Real-time API call (~1 ms per subscriber)
result = predictor.predict_single(subscriber_dict)
# ChurnPrediction(probability=0.87, risk="high_risk", latency_ms=1.1)

# Nightly batch job over millions of rows
results_df = predictor.predict_batch(subscribers_df)
# Returns DataFrame with: churn_probability, churn_label, risk_segment,
#                          xgb_score, transformer_score
```

---

## Network KPIs Explained (for Beginners)

### SINR — Signal-to-Interference-plus-Noise Ratio

Think of a conversation in a noisy café. SINR measures how loud the useful signal is compared to the background noise and interference from other calls.

| SINR (dB) | Quality | User experience |
|---|---|---|
| > 20 dB | Excellent | HD voice, fast 5G data |
| 10–20 dB | Good | Clear calls, smooth streaming |
| 0–10 dB | Fair | Occasional glitches |
| < 0 dB | Poor | Dropped calls, slow data |

### RSRP — Reference Signal Received Power

RSRP measures the actual power level of the LTE/5G reference signal arriving at the phone. It indicates how far the phone is from the cell tower and how much the signal has been attenuated.

| RSRP (dBm) | Quality |
|---|---|
| > −80 | Excellent (close to tower) |
| −80 to −90 | Good |
| −90 to −100 | Fair |
| −100 to −110 | Poor |
| < −110 | Very poor (likely to drop calls) |

### Call Success Rate & Bearer Establishment Rate

- **Call Success Rate**: percentage of outgoing/incoming call attempts that connect successfully. A rate below 97% means roughly 1 in 33 calls fails to connect.
- **Bearer Establishment Rate**: percentage of data sessions (e.g. opening an app, loading a webpage) that successfully set up a data bearer. Below 97% means noticeable app failures.

Both metrics are computed from the network's operations data and strongly predict subscriber frustration and churn.

---

## Troubleshooting

### XGBoost segfault on macOS 26 beta

```bash
# Always set this before running on macOS 26 beta:
OMP_NUM_THREADS=1 python pipeline.py
```

**Cause:** OpenMP auto-detects all CPU cores on macOS 26 and crashes during thread pool initialisation in `libxgboost.dylib`. Capping at 1 thread avoids this.

### MPS not detected

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
# Must print: True
```

If `False`, you may have the Linux/CUDA PyTorch wheel. Fix:
```bash
pip uninstall torch torchvision
pip install torch torchvision     # macOS arm64 wheel includes MPS
```

### Out of memory on MPS

Reduce `batch_size` in `config.py`:
```python
TransformerConfig(batch_size=128)   # default is 512
```

---

## Beginner ML Glossary

| Term | Plain-English meaning |
|---|---|
| **Training set** | The data the model learns from |
| **Validation set** | Data checked during training to detect overfitting |
| **Test set** | Held out until the very end — the honest final report card |
| **Overfitting** | Model memorises training examples; fails on new ones |
| **Logit** | The raw score before sigmoid; can be any real number |
| **Sigmoid** | Squashes any number into [0, 1] → churn probability |
| **Gradient descent** | Iteratively adjust weights to reduce prediction error |
| **Embedding** | A dense vector representing a category (learned during training) |
| **Attention** | Mechanism that lets each feature "look at" all others |
| **CLS token** | Learnable summary vector prepended to the input sequence |
| **Early stopping** | Stop training when validation score stops improving |
| **Ensemble** | Combine multiple models — usually better than any single model |
| **Calibration** | Adjusting probabilities so 80% confident really means 80% correct |
| **AUROC** | Threshold-free ranking quality metric (1.0 = perfect) |
| **Bootstrap CI** | Estimate uncertainty by resampling the test set 1000 times |
| **MPS** | Metal Performance Shaders — Apple Silicon GPU compute API |
| **SINR** | Signal quality ratio: how strong your signal is vs background noise |
| **RSRP** | Raw received power of the cell tower signal at the device |

---

## License

MIT — free to use, modify, and distribute.
