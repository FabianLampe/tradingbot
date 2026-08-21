"""XGBoost predictor for forward-return classification.

Why XGBoost as the first model:
  - Strong on tabular data with mixed scales (technical + sentiment + macro).
  - Handles missing values natively.
  - Trains in seconds on 1M rows — fast iteration cycle.
  - SHAP works out of the box, which is critical for Phase 3 post-mortems.

The model is trained **symmetrically** (multi-class: down/flat/up) so the
same predictor can recommend both long and short positions. The runtime
layer decides whether to act on short signals (initially: log only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

from config import MODELS_DIR


# Map fwd_class {-1, 0, 1} <-> XGBoost classes {0, 1, 2}
_TO_XGB = {-1: 0, 0: 1, 1: 2}
_FROM_XGB = {v: k for k, v in _TO_XGB.items()}


@dataclass
class Predictor:
    feature_cols: list[str]
    horizon_days: int = 5
    params: dict = field(default_factory=lambda: {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "tree_method": "hist",
        "device": "cuda",        # use GPU; falls back to CPU if not available
        "verbosity": 0,
    })
    n_estimators: int = 500
    early_stopping_rounds: int = 30
    booster: xgb.Booster | None = None
    # How the features were scaled when this model was trained (see
    # build_dataset.feature_transform_of). Raw levels and per-date ranks share
    # the same column names, so this is the only thing separating them.
    feature_transform: str = "raw"

    # ---------- training ----------

    def train(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame | None = None,
    ) -> None:
        X_train, y_train = self._xy(df_train)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_cols)
        evals = [(dtrain, "train")]
        if df_val is not None and not df_val.empty:
            X_val, y_val = self._xy(df_val)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_cols)
            evals.append((dval, "val"))

        self.booster = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds if df_val is not None else None,
            verbose_eval=False,
        )

    def update(self, df_new: pd.DataFrame, n_rounds: int = 50) -> None:
        """Warm-start update for daily incremental retraining (Phase 5)."""
        if self.booster is None:
            raise RuntimeError("No base model yet. Call .train() first.")
        X, y = self._xy(df_new)
        d = xgb.DMatrix(X, label=y, feature_names=self.feature_cols)
        self.booster = xgb.train(
            self.params,
            d,
            num_boost_round=n_rounds,
            xgb_model=self.booster,
            verbose_eval=False,
        )

    # ---------- inference ----------

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if self.booster is None:
            raise RuntimeError("Model not trained.")
        X = df[self.feature_cols].values
        d = xgb.DMatrix(X, feature_names=self.feature_cols)
        return self.booster.predict(d)

    def predict_score(self, df: pd.DataFrame) -> np.ndarray:
        """Single signed score in [-1, 1]: P(up) - P(down).
        Use this for Top-N ranking."""
        proba = self.predict_proba(df)
        return proba[:, _TO_XGB[1]] - proba[:, _TO_XGB[-1]]

    def predict_class(self, df: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(df)
        idx = proba.argmax(axis=1)
        return np.array([_FROM_XGB[i] for i in idx])

    # ---------- evaluation ----------

    def evaluate(self, df: pd.DataFrame) -> dict:
        y_true = df["fwd_class"].values
        y_pred = self.predict_class(df)
        return {
            "report": classification_report(y_true, y_pred, output_dict=True),
            "confusion": confusion_matrix(y_true, y_pred, labels=[-1, 0, 1]).tolist(),
            "n": len(df),
        }

    # ---------- persistence ----------

    def save(self, name: str = "predictor") -> Path:
        if self.booster is None:
            raise RuntimeError("Nothing to save.")
        path = MODELS_DIR / f"{name}.json"
        self.booster.save_model(str(path))
        meta = {
            "feature_cols": self.feature_cols,
            "horizon_days": self.horizon_days,
            "params": self.params,
            "feature_transform": self.feature_transform,
        }
        meta_path = MODELS_DIR / f"{name}.meta.json"
        import json
        meta_path.write_text(json.dumps(meta, indent=2))
        return path

    @classmethod
    def load(cls, name: str = "predictor") -> "Predictor":
        import json
        meta_path = MODELS_DIR / f"{name}.meta.json"
        meta = json.loads(meta_path.read_text())
        p = cls(feature_cols=meta["feature_cols"], horizon_days=meta["horizon_days"],
                params=meta["params"],
                # Models saved before feature scaling was tracked were raw-level.
                feature_transform=meta.get("feature_transform", "raw"))
        p.booster = xgb.Booster()
        p.booster.load_model(str(MODELS_DIR / f"{name}.json"))
        return p

    # ---------- internals ----------

    def _xy(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = df[self.feature_cols].values
        y = df["fwd_class"].map(_TO_XGB).values
        return X, y
