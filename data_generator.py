"""
data_generator.py — Synthetic telecom customer data generator.

Produces realistic churn-correlated data by modelling domain knowledge:
  - Month-to-month contracts → higher churn
  - High charges + low tenure → higher churn
  - Multiple complaints / poor network → higher churn
  - Long-term customers + premium bundles → lower churn
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ChurnConfig, cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bernoulli(p, rng: np.random.Generator) -> np.ndarray:
    return np.asarray(rng.binomial(1, p), dtype=int)


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

class TelecomDataGenerator:
    """Generate synthetic telecom customer records with realistic churn signal."""

    def __init__(self, config: ChurnConfig = cfg):
        self.cfg = config
        self.n   = config.data.n_samples
        self.rng = np.random.default_rng(config.data.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Return a DataFrame of n_samples customer records."""
        logger.info("Generating %d synthetic telecom records …", self.n)
        df = self._build_base_features()
        df = self._derive_billing_features(df)
        df = self._assign_churn(df)
        df = self._post_process(df)
        logger.info(
            "Generated %d records | churn rate = %.2f%%",
            len(df), df[self.cfg.target_column].mean() * 100,
        )
        return df

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified train / val / test split."""
        d = self.cfg.data
        X, y = df.drop(columns=[self.cfg.target_column]), df[self.cfg.target_column]

        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=d.test_size,
            stratify=y, random_state=d.random_seed,
        )
        val_frac = d.val_size / (1.0 - d.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_frac,
            stratify=y_trainval, random_state=d.random_seed,
        )

        train_df = pd.concat([X_train, y_train], axis=1)
        val_df   = pd.concat([X_val,   y_val],   axis=1)
        test_df  = pd.concat([X_test,  y_test],  axis=1)

        logger.info(
            "Split → train=%d | val=%d | test=%d",
            len(train_df), len(val_df), len(test_df),
        )
        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Internal builders
    # ------------------------------------------------------------------

    def _build_base_features(self) -> pd.DataFrame:
        n, rng = self.n, self.rng

        # ---- Demographics ------------------------------------------------
        age             = rng.integers(18, 80, n).astype(float)
        gender          = rng.choice(["Male", "Female"], n)
        senior_citizen  = (age >= 65).astype(int)
        has_partner     = _bernoulli(0.48, rng)
        has_dependents  = _bernoulli(np.where(has_partner == 1, 0.40, 0.15), rng)

        # ---- Account / Contract ------------------------------------------
        tenure_months   = rng.integers(1, 73, n).astype(float)
        contract_type   = rng.choice(
            ["month-to-month", "one-year", "two-year"],
            p=[0.55, 0.25, 0.20], size=n,
        )

        payment_method  = rng.choice(
            ["electronic_check", "mailed_check", "bank_transfer", "credit_card"],
            p=[0.35, 0.23, 0.22, 0.20], size=n,
        )
        paperless_billing = _bernoulli(0.60, rng)

        # ---- Phone / Internet services -----------------------------------
        phone_service   = _bernoulli(0.90, rng)
        multiple_lines  = np.where(
            phone_service == 0, "no_phone_service",
            np.where(_bernoulli(0.42, rng) == 1, "Yes", "No")
        )

        internet_service = rng.choice(
            ["DSL", "Fiber_optic", "No"],
            p=[0.34, 0.44, 0.22], size=n,
        )
        has_internet = (internet_service != "No").astype(int)

        def _internet_addon(p_yes: float) -> np.ndarray:
            base = _bernoulli(p_yes * has_internet, rng)
            return np.where(has_internet == 0, "no_internet_service",
                            np.where(base == 1, "Yes", "No"))

        online_security  = _internet_addon(0.29)
        online_backup    = _internet_addon(0.34)
        device_protection= _internet_addon(0.34)
        tech_support     = _internet_addon(0.29)
        streaming_tv     = _internet_addon(0.39)
        streaming_movies = _internet_addon(0.39)

        # ---- Usage metrics -----------------------------------------------
        avg_daily_call_minutes = np.clip(
            rng.normal(45, 20, n), 0, 200
        )
        avg_monthly_data_gb = np.where(
            has_internet == 1,
            np.clip(rng.normal(15, 8, n), 0.5, 60),
            0.0,
        )
        avg_monthly_sms     = np.clip(rng.normal(120, 60, n), 0, 500).astype(int)
        roaming_usage_min   = np.clip(
            rng.exponential(10, n), 0, 300
        )

        # ---- Network quality ---------------------------------------------
        # Fiber optic customers may experience more variability
        base_drop = np.where(internet_service == "Fiber_optic", 0.04, 0.02)
        avg_call_drop_rate  = np.clip(
            rng.normal(base_drop, 0.015, n), 0.001, 0.30
        )
        avg_data_speed_mbps = np.where(
            internet_service == "Fiber_optic",
            np.clip(rng.normal(250, 50, n), 50, 500),
            np.where(
                internet_service == "DSL",
                np.clip(rng.normal(25, 8, n), 5, 80),
                0.0,
            ),
        )
        network_outage_hours_6mo = np.clip(
            rng.exponential(2, n), 0, 48
        )

        # ---- Customer service --------------------------------------------
        # More complaints for high charges, poor network, fiber optic
        complaint_base = (
            0.05
            + 0.10 * (internet_service == "Fiber_optic").astype(float)
            + 0.15 * (avg_call_drop_rate > 0.06).astype(float)
        )
        num_complaints_6mo   = rng.poisson(complaint_base * 3, n).astype(int)
        num_support_calls_6mo= rng.poisson(0.5 + num_complaints_6mo * 0.5, n).astype(int)
        days_since_last_complaint = np.where(
            num_complaints_6mo > 0,
            rng.integers(1, 180, n).astype(float),
            180.0,
        )

        # ---- Engagement --------------------------------------------------
        app_logins_monthly  = np.clip(
            rng.normal(8, 5, n), 0, 40
        ).astype(int)
        # Count of premium features used (0–6 range)
        feature_adoption_score = (
            (multiple_lines  == "Yes").astype(int)
            + (online_security   == "Yes").astype(int)
            + (online_backup     == "Yes").astype(int)
            + (device_protection == "Yes").astype(int)
            + (streaming_tv      == "Yes").astype(int)
            + (streaming_movies  == "Yes").astype(int)
        ).astype(float)

        return pd.DataFrame({
            "age": age, "gender": gender,
            "senior_citizen": senior_citizen, "has_partner": has_partner,
            "has_dependents": has_dependents,
            "tenure_months": tenure_months, "contract_type": contract_type,
            "payment_method": payment_method, "paperless_billing": paperless_billing,
            "phone_service": phone_service, "multiple_lines": multiple_lines,
            "internet_service": internet_service,
            "online_security": online_security, "online_backup": online_backup,
            "device_protection": device_protection, "tech_support": tech_support,
            "streaming_tv": streaming_tv, "streaming_movies": streaming_movies,
            "avg_daily_call_minutes": avg_daily_call_minutes,
            "avg_monthly_data_gb": avg_monthly_data_gb,
            "avg_monthly_sms": avg_monthly_sms,
            "roaming_usage_min": roaming_usage_min,
            "avg_call_drop_rate": avg_call_drop_rate,
            "avg_data_speed_mbps": avg_data_speed_mbps,
            "network_outage_hours_6mo": network_outage_hours_6mo,
            "num_complaints_6mo": num_complaints_6mo,
            "num_support_calls_6mo": num_support_calls_6mo,
            "days_since_last_complaint": days_since_last_complaint,
            "app_logins_monthly": app_logins_monthly,
            "feature_adoption_score": feature_adoption_score,
        })

    def _derive_billing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute monthly_charges, total_charges, avg_monthly_charges."""
        rng = self.rng

        base  = 20.0
        base += (df["phone_service"] == 1) * rng.uniform(10, 25, len(df))
        base += (df["multiple_lines"] == "Yes") * rng.uniform(10, 20, len(df))
        base += (df["internet_service"] == "DSL") * rng.uniform(20, 35, len(df))
        base += (df["internet_service"] == "Fiber_optic") * rng.uniform(50, 80, len(df))
        base += (df["online_security"]   == "Yes") * rng.uniform(5, 15, len(df))
        base += (df["online_backup"]     == "Yes") * rng.uniform(5, 15, len(df))
        base += (df["device_protection"] == "Yes") * rng.uniform(5, 15, len(df))
        base += (df["tech_support"]      == "Yes") * rng.uniform(5, 15, len(df))
        base += (df["streaming_tv"]      == "Yes") * rng.uniform(5, 15, len(df))
        base += (df["streaming_movies"]  == "Yes") * rng.uniform(5, 15, len(df))
        base += rng.normal(0, 3, len(df))          # small noise

        monthly_charges = np.clip(base, 18, 120)

        # Slight discount for long-term customers
        loyalty_factor  = 1.0 - 0.001 * df["tenure_months"].values
        total_charges   = (
            monthly_charges * df["tenure_months"].values * loyalty_factor
            + rng.normal(0, 20, len(df))
        ).clip(0)
        avg_monthly     = total_charges / df["tenure_months"].values.clip(1)

        df["monthly_charges"]     = monthly_charges.round(2)
        df["total_charges"]       = total_charges.round(2)
        df["avg_monthly_charges"] = avg_monthly.round(2)
        return df

    def _assign_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a logistic churn probability from domain-driven risk factors,
        then calibrate the mean to the target churn rate.
        """
        # Logit contributions (positive = higher churn risk)
        logit = np.zeros(self.n)

        # Contract risk
        logit += np.where(df["contract_type"] == "month-to-month",  2.0,
                 np.where(df["contract_type"] == "one-year",         0.0, -1.5))

        # Tenure (longer = lower risk)
        logit -= 0.04 * df["tenure_months"].values

        # Charges (higher bills → more likely to leave)
        logit += 0.015 * (df["monthly_charges"].values - 60)

        # Complaints / support
        logit += 0.50 * df["num_complaints_6mo"].values
        logit += 0.25 * df["num_support_calls_6mo"].values

        # Network quality
        logit += 8.0  * (df["avg_call_drop_rate"].values - 0.03)
        logit += 0.04 * df["network_outage_hours_6mo"].values

        # Payment method (electronic check → risk)
        logit += (df["payment_method"] == "electronic_check").astype(float) * 0.50

        # Engagement (more features used → lower churn)
        logit -= 0.20 * df["feature_adoption_score"].values

        # App logins (engaged users stay)
        logit -= 0.04 * df["app_logins_monthly"].values

        # Senior citizens slightly higher churn
        logit += 0.30 * df["senior_citizen"].values

        # Paperless billing (digital-savvy, slight negative for churn)
        logit -= 0.20 * df["paperless_billing"].values

        # Fiber optic — higher expectations, higher churn when issues occur
        logit += 0.40 * (df["internet_service"] == "Fiber_optic").astype(float)

        # Jitter
        logit += self.rng.normal(0, 0.5, self.n)

        # Sigmoid → probability
        prob = 1.0 / (1.0 + np.exp(-logit))

        # Calibrate to target churn rate via threshold shift
        target = self.cfg.data.churn_rate
        threshold = np.percentile(prob, 100 * (1 - target))
        churned = (prob >= threshold).astype(int)

        df[self.cfg.target_column] = churned
        df["churn_probability"]    = prob.round(4)   # latent score (drop before modelling)
        return df

    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Type cleanup and column ordering."""
        # Cast binary flags stored as int arrays
        for col in ["senior_citizen", "has_partner", "has_dependents",
                    "phone_service", "paperless_billing"]:
            df[col] = df[col].astype(int)

        # Drop the latent probability (ground-truth, not available in prod)
        df = df.drop(columns=["churn_probability"])

        all_cols = (
            self.cfg.numerical_features
            + self.cfg.categorical_features
            + [self.cfg.target_column]
        )
        return df[all_cols].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, df: pd.DataFrame, path: str = "data/telecom_churn.csv") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %d rows → %s", len(df), path)

    @staticmethod
    def load(path: str = "data/telecom_churn.csv") -> pd.DataFrame:
        df = pd.read_csv(path)
        logger.info("Loaded %d rows from %s", len(df), path)
        return df


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    gen = TelecomDataGenerator()
    df  = gen.generate()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    print(df.dtypes)
    gen.save(df)
