from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


ALLOWED_CONTRACTS = {"month-to-month", "one_year", "two_year"}
ALLOWED_INTERNET = {"dsl", "fiber", "none"}
ALLOWED_PAYMENT = {"electronic_check", "mailed_check", "bank_transfer", "credit_card"}


@dataclass(frozen=True)
class FeatureConfig:
    numeric_cols: tuple[str, ...] = ("tenure_months", "monthly_charges", "total_charges")
    categorical_cols: tuple[str, ...] = ("contract_type", "internet_service", "payment_method")
    bool_cols: tuple[str, ...] = ("paperless_billing",)


FEATURE_CONFIG = FeatureConfig()


def clean_and_validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate raw input data so it matches
    what the ML model expects.
    """
    df = df.copy()

    # ---- numeric features ----
    for col in FEATURE_CONFIG.numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
        df[col] = df[col].clip(lower=0)

    # ---- categorical features ----
    df["contract_type"] = (
        df["contract_type"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
    )

    df["internet_service"] = (
        df["internet_service"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
    )

    df["payment_method"] = (
        df["payment_method"]
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace("-", "_")
        .str.replace(" ", "_")
    )

    # ---- handle unknown categories ----
    df.loc[~df["contract_type"].isin(ALLOWED_CONTRACTS), "contract_type"] = "month-to-month"
    df.loc[~df["internet_service"].isin(ALLOWED_INTERNET), "internet_service"] = "none"
    df.loc[~df["payment_method"].isin(ALLOWED_PAYMENT), "payment_method"] = "electronic_check"

    # ---- boolean ----
    df["paperless_billing"] = df["paperless_billing"].astype(bool)

    return df
