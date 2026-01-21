from __future__ import annotations

from pathlib import Path
import pandas as pd
from joblib import load

from training.features import clean_and_validate

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT / "models" / "model.joblib"


class ModelNotReady(Exception):
    pass


class ChurnModel:
    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self._pipe = None

    def load(self) -> None:
        if not self.model_path.exists():
            raise ModelNotReady(f"Model file not found at {self.model_path}. Train first.")
        self._pipe = load(self.model_path)

    def predict_proba(self, row: dict) -> float:
        if self._pipe is None:
            self.load()

        df = pd.DataFrame([row])
        df = clean_and_validate(df)
        proba = float(self._pipe.predict_proba(df)[:, 1][0])
        return max(0.0, min(1.0, proba))
