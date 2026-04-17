"""
data_generator.py — Synthetic European mobile subscriber data generator.

Generates realistic mobile churn data covering:
  - Plan types  : prepaid / postpaid
  - Plan tiers  : basic (5–9 EUR) / standard (9–14 EUR) /
                  premium (14–18 EUR) / unlimited (17–20 EUR)
  - Network KPIs: SINR (dB), RSRP (dBm), Call Success Rate,
                  Bearer Establishment Rate, Call Drop Rate
  - Billing     : top-ups, billing rate, overage charges, late payments
  - Usage       : data (GB), calls (min), SMS, roaming, international
  - CX          : complaints, support calls

Churn is modelled from domain-driven risk factors:
  prepaid   → lapse when top-up frequency drops and network is poor
  postpaid  → leave when charges are high, contract is monthly, or quality is poor
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import ChurnConfig, cfg, PLAN_PRICE_RANGES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _bern(p, rng: np.random.Generator) -> np.ndarray:
    """Bernoulli draw that handles both scalar and array probabilities."""
    return np.asarray(rng.binomial(1, np.clip(p, 0, 1)), dtype=int)


def _normal_clip(mu, sigma, lo, hi, n, rng):
    return np.clip(rng.normal(mu, sigma, n), lo, hi)


def _choice(options, p, n, rng):
    return rng.choice(options, p=np.array(p) / sum(p), size=n)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class MobileDataGenerator:
    """
    Generates synthetic mobile subscriber records aligned with a
    European MVNO / MNO dataset.

    Key domain decisions
    --------------------
    • Monthly plan cost: 5–20 EUR (split across 4 tiers).
    • SINR range: –8 to +32 dB  (log-normal mix of good/poor coverage areas).
    • RSRP range: –120 to –65 dBm  (log-normal distribution).
    • Call Success Rate: 94–99.8 %.
    • Bearer Establishment Rate: 94–99.9 %.
    • Top-up logic: prepaid only; low top-up frequency → strong churn signal.
    • Overage charges: apply only when actual usage exceeds plan bundle.
    """

    def __init__(self, config: ChurnConfig = cfg):
        self.cfg = config
        self.n   = config.data.n_samples
        self.rng = np.random.default_rng(config.data.random_seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        logger.info("Generating %d synthetic mobile subscriber records …", self.n)
        df = self._demographics()
        df = self._plan_and_contract(df)
        df = self._network_quality(df)
        df = self._usage(df)
        df = self._billing(df)
        df = self._customer_service(df)
        df = self._assign_churn(df)
        df = self._finalise(df)
        logger.info(
            "Done — %d records | churn = %.2f%%",
            len(df), df[self.cfg.target_column].mean() * 100,
        )
        return df

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        d = self.cfg.data
        X = df.drop(columns=[self.cfg.target_column])
        y = df[self.cfg.target_column]

        X_tv, X_te, y_tv, y_te = train_test_split(
            X, y, test_size=d.test_size, stratify=y, random_state=d.random_seed
        )
        val_frac = d.val_size / (1.0 - d.test_size)
        X_tr, X_va, y_tr, y_va = train_test_split(
            X_tv, y_tv, test_size=val_frac, stratify=y_tv, random_state=d.random_seed
        )
        train = pd.concat([X_tr, y_tr], axis=1)
        val   = pd.concat([X_va, y_va], axis=1)
        test  = pd.concat([X_te, y_te], axis=1)
        logger.info("Split → train=%d | val=%d | test=%d", len(train), len(val), len(test))
        return train, val, test

    def save(self, df: pd.DataFrame, path: str = "data/mobile_churn.csv") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Saved %d rows → %s", len(df), path)

    @staticmethod
    def load(path: str = "data/mobile_churn.csv") -> pd.DataFrame:
        df = pd.read_csv(path)
        logger.info("Loaded %d rows from %s", len(df), path)
        return df

    # ------------------------------------------------------------------
    # Stage 1 — Demographics
    # ------------------------------------------------------------------

    def _demographics(self) -> pd.DataFrame:
        n, rng = self.n, self.rng
        age    = rng.integers(16, 78, n).astype(float)
        gender = _choice(["Male", "Female"], [50, 50], n, rng)
        return pd.DataFrame({"age": age, "gender": gender})

    # ------------------------------------------------------------------
    # Stage 2 — Plan & Contract
    # ------------------------------------------------------------------

    def _plan_and_contract(self, df: pd.DataFrame) -> pd.DataFrame:
        n, rng = self.n, self.rng

        # Plan type: ~40% prepaid (common in Europe), 60% postpaid
        plan_type = _choice(["prepaid", "postpaid"], [40, 60], n, rng)

        # Plan tier: cheaper tiers more common
        plan_tier = _choice(
            ["basic", "standard", "premium", "unlimited"],
            [35, 35, 20, 10], n, rng,
        )

        # Monthly plan base cost from tier price range
        monthly_plan_cost_eur = np.array([
            rng.uniform(*PLAN_PRICE_RANGES[t])
            for t in plan_tier
        ]).round(2)

        # Contract duration
        # Prepaid → always "prepaid" duration type
        # Postpaid → monthly (higher churn risk) or 12/24 month
        contract_duration = np.where(
            plan_type == "prepaid",
            "prepaid",
            _choice(["monthly", "12_month", "24_month"], [45, 35, 20], n, rng),
        )

        # Payment method
        # Prepaid: mostly cash/mobile wallet; postpaid: direct debit / credit card
        pay_prepaid  = _choice(["cash", "mobile_wallet", "credit_card", "direct_debit"], [45, 35, 15, 5],  n, rng)
        pay_postpaid = _choice(["direct_debit", "credit_card", "mobile_wallet", "cash"], [55, 30, 10, 5], n, rng)
        payment_method = np.where(plan_type == "prepaid", pay_prepaid, pay_postpaid)

        # Network generation
        network_generation = _choice(["5G", "4G", "3G"], [25, 65, 10], n, rng)

        # Device type
        device_type = _choice(
            ["smartphone", "feature_phone", "tablet", "mifi"],
            [78, 12, 7, 3], n, rng,
        )

        # Roaming enabled (more likely for postpaid / premium)
        p_roam = (
            0.50 * (plan_type == "postpaid").astype(float)
            + 0.15 * (plan_type == "prepaid").astype(float)
            + 0.10 * (plan_tier == "premium").astype(float)
            + 0.15 * (plan_tier == "unlimited").astype(float)
        )
        is_roaming_enabled = np.where(_bern(p_roam, rng) == 1, "yes", "no")

        # Handset subsidy (postpaid + 12/24 month contracts)
        p_subsidy = (
            0.50 * (contract_duration == "24_month").astype(float)
            + 0.35 * (contract_duration == "12_month").astype(float)
            + 0.05
        )
        has_handset_subsidy = np.where(_bern(p_subsidy, rng) == 1, "yes", "no")

        # Tenure (months since activation)
        # Postpaid with subsidy → tends to be newer (locked-in cycle)
        tenure_months = np.clip(
            rng.exponential(24, n) + 1, 1, 120
        ).astype(float).round(0)

        df["plan_type"]           = plan_type
        df["plan_tier"]           = plan_tier
        df["monthly_plan_cost_eur"] = monthly_plan_cost_eur
        df["contract_duration"]   = contract_duration
        df["payment_method"]      = payment_method
        df["network_generation"]  = network_generation
        df["device_type"]         = device_type
        df["is_roaming_enabled"]  = is_roaming_enabled
        df["has_handset_subsidy"] = has_handset_subsidy
        df["tenure_months"]       = tenure_months
        return df

    # ------------------------------------------------------------------
    # Stage 3 — Network Quality KPIs
    # ------------------------------------------------------------------

    def _network_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        n, rng = self.n, self.rng

        # --- SINR (Signal-to-Interference-plus-Noise Ratio, dB) -----------
        # 5G users experience slightly better SINR on average
        sinr_base = np.where(
            df["network_generation"] == "5G",  15.0,
            np.where(df["network_generation"] == "4G", 11.0, 6.0)
        )
        avg_sinr_db = np.clip(
            rng.normal(sinr_base, 7.0, n), -10.0, 32.0
        ).round(2)

        # --- RSRP (Reference Signal Received Power, dBm) ------------------
        # Better SINR correlates with better RSRP
        # RSRP ≈ -85 + some signal strength variation
        rsrp_base = -85.0 + 0.8 * (avg_sinr_db - 10.0)   # rough correlation
        avg_rsrp_dbm = np.clip(
            rng.normal(rsrp_base, 8.0, n), -120.0, -65.0
        ).round(2)

        # --- Call Success Rate (%) ----------------------------------------
        # Poor SINR → lower call success rate
        # Good threshold: SINR > 5 dB → ~98–99.5 %
        # Poor:           SINR < 0 dB → ~94–97 %
        csr_base = 98.5 - 0.15 * np.clip(10.0 - avg_sinr_db, 0, 15)
        call_success_rate_pct = np.clip(
            rng.normal(csr_base, 0.8, n), 88.0, 99.9
        ).round(2)

        # --- Bearer Establishment Rate (%) --------------------------------
        # Data bearer setup success — typically slightly higher than call success
        ber_base = 98.8 - 0.10 * np.clip(10.0 - avg_sinr_db, 0, 15)
        bearer_establishment_rate_pct = np.clip(
            rng.normal(ber_base, 0.7, n), 88.0, 99.9
        ).round(2)

        # --- Call Drop Rate (%) -------------------------------------------
        # Inversely correlated with call success rate
        avg_call_drop_rate_pct = np.clip(
            (100.0 - call_success_rate_pct) * rng.uniform(0.3, 0.6, n),
            0.05, 8.0
        ).round(3)

        # --- Average data speed (Mbps) ------------------------------------
        speed_map = {"5G": (50, 300), "4G": (10, 80), "3G": (0.5, 7)}
        avg_data_speed_mbps = np.array([
            rng.uniform(*speed_map[g])
            for g in df["network_generation"]
        ]).round(1)
        # Scale down in poor SINR areas
        avg_data_speed_mbps *= np.clip(0.5 + 0.05 * avg_sinr_db, 0.3, 1.0)
        avg_data_speed_mbps = np.clip(avg_data_speed_mbps, 0.1, 350).round(1)

        df["avg_sinr_db"]                   = avg_sinr_db
        df["avg_rsrp_dbm"]                  = avg_rsrp_dbm
        df["call_success_rate_pct"]         = call_success_rate_pct
        df["bearer_establishment_rate_pct"] = bearer_establishment_rate_pct
        df["avg_call_drop_rate_pct"]        = avg_call_drop_rate_pct
        df["avg_data_speed_mbps"]           = avg_data_speed_mbps
        return df

    # ------------------------------------------------------------------
    # Stage 4 — Usage
    # ------------------------------------------------------------------

    def _usage(self, df: pd.DataFrame) -> pd.DataFrame:
        n, rng = self.n, self.rng

        is_smartphone = (df["device_type"] == "smartphone").astype(float)
        is_unlimited  = (df["plan_tier"]   == "unlimited").astype(float)
        is_premium    = (df["plan_tier"]   == "premium").astype(float)

        # Data usage (GB/month) — unlimited users consume more
        data_base = 3.0 + 10.0 * is_unlimited + 5.0 * is_premium
        avg_monthly_data_gb = np.clip(
            rng.lognormal(np.log(data_base + 1), 0.8, n), 0.1, 80.0
        ).round(2)

        # Daily call minutes
        avg_daily_call_minutes = np.clip(
            rng.lognormal(np.log(20), 0.7, n), 0.5, 180.0
        ).round(1)

        # Monthly SMS (declining with smartphone data users)
        sms_base = np.where(is_smartphone == 1, 30, 80)
        avg_monthly_sms = np.clip(
            rng.lognormal(np.log(sms_base), 0.9, n), 0, 500
        ).astype(int)

        # Roaming sessions (6 months)
        p_roam = (df["is_roaming_enabled"] == "yes").astype(float)
        roaming_sessions_6mo = np.where(
            p_roam == 1,
            rng.poisson(2.5, n),
            rng.poisson(0.1, n),
        ).astype(int)

        # International call minutes per month
        intl_call_minutes_monthly = np.clip(
            rng.exponential(5, n), 0, 120
        ).round(1)

        # Daily data usage in MB (derived from monthly GB, with daily variance)
        avg_daily_data_usage_mb = np.clip(
            (avg_monthly_data_gb * 1024 / 30.0) * rng.uniform(0.7, 1.3, n),
            0.5, 3000.0
        ).round(1)

        df["avg_monthly_data_gb"]      = avg_monthly_data_gb
        df["avg_daily_data_usage_mb"]  = avg_daily_data_usage_mb
        df["avg_daily_call_minutes"]   = avg_daily_call_minutes
        df["avg_monthly_sms"]        = avg_monthly_sms
        df["roaming_sessions_6mo"]   = roaming_sessions_6mo
        df["intl_call_minutes_monthly"] = intl_call_minutes_monthly
        return df

    # ------------------------------------------------------------------
    # Stage 5 — Billing
    # ------------------------------------------------------------------

    def _billing(self, df: pd.DataFrame) -> pd.DataFrame:
        n, rng = self.n, self.rng
        is_prepaid  = (df["plan_type"] == "prepaid").astype(float)
        is_postpaid = 1.0 - is_prepaid

        # --- Top-ups (prepaid only) ---------------------------------------
        # Low-tenure or unhappy prepaid users top up less often → churn risk
        avg_topup_frequency = np.clip(
            rng.normal(4.5, 1.5, n), 0.5, 10
        )
        num_topups_6mo = np.where(
            is_prepaid == 1,
            np.round(rng.normal(avg_topup_frequency, 1.0, n)).clip(0, 18).astype(int),
            0,
        ).astype(int)

        # Average top-up amount (EUR)
        # Tends to match plan tier cost roughly
        topup_base = df["monthly_plan_cost_eur"].values * 0.9
        avg_topup_amount_eur = np.where(
            is_prepaid == 1,
            np.clip(rng.normal(topup_base, 2.0, n), 3.0, 30.0).round(2),
            0.0,
        )

        # --- Late payments (postpaid only) --------------------------------
        # Higher monthly cost + lower tenure → more late payments
        p_late = np.clip(
            0.03
            + 0.005 * (df["monthly_plan_cost_eur"].values - 10)
            - 0.003 * df["tenure_months"].values,
            0.01, 0.35,
        )
        num_late_payments_6mo = np.where(
            is_postpaid == 1,
            rng.binomial(6, p_late, n),
            0,
        ).astype(int)

        # --- Data overage charges (EUR) -----------------------------------
        # Non-unlimited users who exceed their bundle get charged extra
        is_limited_plan = (df["plan_tier"] != "unlimited").astype(float)
        overage_prob = 0.15 * is_limited_plan * is_postpaid
        data_overage_charges_eur = np.where(
            _bern(overage_prob, rng) == 1,
            np.clip(rng.exponential(3.5, n), 0.5, 20.0).round(2),
            0.0,
        )

        # --- Total spend last 6 months (EUR) ------------------------------
        total_spend_6mo_eur = np.where(
            is_prepaid == 1,
            (num_topups_6mo * avg_topup_amount_eur
             + data_overage_charges_eur * 6).round(2),
            (df["monthly_plan_cost_eur"].values * 6
             + data_overage_charges_eur * 6
             + num_late_payments_6mo * rng.uniform(1.5, 5.0, n)).round(2),
        )

        df["num_topups_6mo"]           = num_topups_6mo
        df["avg_topup_amount_eur"]     = avg_topup_amount_eur
        df["num_late_payments_6mo"]    = num_late_payments_6mo
        df["data_overage_charges_eur"] = data_overage_charges_eur
        df["total_spend_6mo_eur"]      = total_spend_6mo_eur
        return df

    # ------------------------------------------------------------------
    # Stage 6 — Customer Service
    # ------------------------------------------------------------------

    def _customer_service(self, df: pd.DataFrame) -> pd.DataFrame:
        n, rng = self.n, self.rng

        # Complaint rate driven by poor network quality and billing issues
        poor_sinr   = (df["avg_sinr_db"].values < 5).astype(float)
        poor_rsrp   = (df["avg_rsrp_dbm"].values < -100).astype(float)
        poor_csr    = (df["call_success_rate_pct"].values < 96).astype(float)
        poor_ber    = (df["bearer_establishment_rate_pct"].values < 96).astype(float)
        overcharged = (df["data_overage_charges_eur"].values > 0).astype(float)
        late_payer  = (df["num_late_payments_6mo"].values > 0).astype(float)

        complaint_rate = (
            0.05
            + 0.20 * poor_sinr
            + 0.15 * poor_rsrp
            + 0.25 * poor_csr
            + 0.20 * poor_ber
            + 0.15 * overcharged
            + 0.10 * late_payer
        ).clip(0.02, 0.85)

        num_complaints_6mo    = rng.poisson(complaint_rate * 3, n).astype(int)
        num_support_calls_6mo = (
            num_complaints_6mo
            + rng.poisson(0.3 + 0.4 * num_complaints_6mo, n)
        ).astype(int)

        days_since_last_complaint = np.where(
            num_complaints_6mo > 0,
            rng.integers(1, 181, n).astype(float),
            180.0,
        )

        df["num_complaints_6mo"]         = num_complaints_6mo
        df["num_support_calls_6mo"]      = num_support_calls_6mo
        df["days_since_last_complaint"]  = days_since_last_complaint
        return df

    # ------------------------------------------------------------------
    # Stage 7 — Churn assignment
    # ------------------------------------------------------------------

    def _assign_churn(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a domain-driven logit for churn probability.

        Positive logit  → higher churn risk
        Negative logit  → lower churn risk
        """
        n   = self.n
        rng = self.rng

        logit = np.zeros(n)

        # ---- Plan type & contract ----------------------------------------
        logit += np.where(df["plan_type"] == "prepaid", 0.5, 0.0)
        logit += np.where(df["contract_duration"] == "monthly",  1.8,
                 np.where(df["contract_duration"] == "prepaid",  0.8,
                 np.where(df["contract_duration"] == "12_month", 0.0, -1.2)))

        # ---- Tenure (longer = more loyal) --------------------------------
        logit -= 0.025 * df["tenure_months"].values

        # ---- Monthly cost (higher cost = more willing to switch) ---------
        logit += 0.08 * (df["monthly_plan_cost_eur"].values - 12.0)

        # ---- Network quality KPIs ----------------------------------------
        # SINR below 5 dB is noticeably poor
        logit += 0.12 * np.clip(5.0  - df["avg_sinr_db"].values,        0, 15)
        # RSRP below -95 dBm is weak signal
        logit += 0.08 * np.clip(-95.0 - df["avg_rsrp_dbm"].values,      -20, 0) * (-1)
        logit += 0.08 * np.clip(-95.0 - df["avg_rsrp_dbm"].values,       0, 25)
        # Call success rate below 97 % is frustrating
        logit += 0.18 * np.clip(97.0 - df["call_success_rate_pct"].values, 0, 10)
        # Bearer establishment below 97 %
        logit += 0.15 * np.clip(97.0 - df["bearer_establishment_rate_pct"].values, 0, 10)
        # Call drop rate above 2 % is frustrating
        logit += 0.20 * np.clip(df["avg_call_drop_rate_pct"].values - 2.0, 0, 6)

        # ---- Billing signals ---------------------------------------------
        # Prepaid users who barely top up are about to lapse
        logit += np.where(
            df["plan_type"] == "prepaid",
            np.clip(4.0 - df["num_topups_6mo"].values, 0, 4) * 0.35,
            0.0,
        )
        # Postpaid late payments signal financial stress / dissatisfaction
        logit += 0.30 * df["num_late_payments_6mo"].values
        # Unexpected overage charges are a churn trigger
        logit += 0.40 * (df["data_overage_charges_eur"].values > 0).astype(float)
        logit += 0.03 * df["data_overage_charges_eur"].values

        # ---- Complaints & support ----------------------------------------
        logit += 0.55 * df["num_complaints_6mo"].values
        logit += 0.20 * df["num_support_calls_6mo"].values

        # ---- Loyalty drivers (reduce churn) ------------------------------
        logit -= 0.60 * (df["has_handset_subsidy"] == "yes").astype(float)
        logit -= 0.30 * (df["contract_duration"] == "24_month").astype(float)
        logit -= 0.25 * (df["plan_tier"] == "unlimited").astype(float)

        # ---- Jitter -------------------------------------------------------
        logit += rng.normal(0, 0.6, n)

        # Sigmoid → probability, calibrate to target churn rate
        prob      = 1.0 / (1.0 + np.exp(-logit))
        threshold = np.percentile(prob, 100 * (1 - self.cfg.data.churn_rate))
        churned   = (prob >= threshold).astype(int)

        df[self.cfg.target_column] = churned
        df["_churn_prob_latent"]   = prob.round(4)   # internal; dropped before modelling
        return df

    # ------------------------------------------------------------------
    # Stage 8 — Finalise
    # ------------------------------------------------------------------

    def _finalise(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=["_churn_prob_latent"])

        all_cols = (
            self.cfg.numerical_features
            + self.cfg.categorical_features
            + [self.cfg.target_column]
        )
        return df[all_cols].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    gen = MobileDataGenerator()
    df  = gen.generate()

    print(f"\nShape         : {df.shape}")
    print(f"Churn rate    : {df['churned'].mean():.2%}")
    print(f"\nPlan type dist:\n{df['plan_type'].value_counts()}")
    print(f"\nPlan tier dist:\n{df['plan_tier'].value_counts()}")
    print(f"\nNetwork gen   :\n{df['network_generation'].value_counts()}")
    print(f"\nSINR stats (dB):\n{df['avg_sinr_db'].describe().round(2)}")
    print(f"\nRSRP stats (dBm):\n{df['avg_rsrp_dbm'].describe().round(2)}")
    print(f"\nCall Success Rate (%):\n{df['call_success_rate_pct'].describe().round(2)}")
    print(f"\nBearer Estab. Rate (%):\n{df['bearer_establishment_rate_pct'].describe().round(2)}")
    print(f"\nMonthly plan cost (EUR):\n{df['monthly_plan_cost_eur'].describe().round(2)}")
    print(f"\nTop-ups 6mo (prepaid only):\n{df[df['plan_type']=='prepaid']['num_topups_6mo'].describe().round(2)}")

    gen.save(df)
