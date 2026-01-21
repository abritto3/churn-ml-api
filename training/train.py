from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import dump

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from training.features import FEATURE_CONFIG, clean_and_validate

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"


def make_synthetic_dataset(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic dataset so the repo runs without private data.
    You can later swap this for a real dataset.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 72, size=n)
    monthly = rng.normal(70, 25, size=n).clip(15, 150)
    total = (tenure * monthly + rng.normal(0, 200, size=n)).clip(0, None)

    contract = rng.choice(["month-to-month", "one_year", "two_year"], size=n, p=[0.6, 0.25, 0.15])
    internet = rng.choice(["dsl", "fiber", "none"], size=n, p=[0.35, 0.5, 0.15])
    payment = rng.choice(
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"], size=n, p=[0.45, 0.2, 0.2, 0.15]
    )
    paperless = rng.choice([True, False], size=n, p=[0.6, 0.4])

    df = pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "contract_type": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": paperless,
        }
    )

    # "true" churn probability (synthetic but realistic-ish)
    # higher churn for month-to-month, fiber, electronic check, low tenure, high monthly charges
    logits = (
        -1.2
        + 0.9 * (df["contract_type"] == "month-to-month").astype(int)
        + 0.4 * (df["internet_service"] == "fiber").astype(int)
        + 0.35 * (df["payment_method"] == "electronic_check").astype(int)
        - 0.02 * df["tenure_months"]
        + 0.01 * (df["monthly_charges"] - 60)
        + 0.25 * df["paperless_billing"].astype(int)
    )
    probs = 1 / (1 + np.exp(-logits))
    churn = rng.binomial(1, probs)

    df["churned"] = churn
    return df


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(FEATURE_CONFIG.numeric_cols)),
            ("cat", categorical_transformer, list(FEATURE_CONFIG.categorical_cols)),
            ("bool", "passthrough", list(FEATURE_CONFIG.bool_cols)),
        ]
    )

    model = LogisticRegression(max_iter=1000)

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = make_synthetic_dataset()
    df = clean_and_validate(df)

    X = df.drop(columns=["churned"])
    y = df["churned"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    # quick metric for README bragging rights
    pred_proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pred_proba)
    print(f"Trained model ROC-AUC: {auc:.3f}")

    dump(pipe, MODEL_PATH)
    print(f"Saved model → {MODEL_PATH}")


if __name__ == "__main__":
    main()
