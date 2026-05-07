"""SHAP attribution for the XGBoost predictor.

For every prediction, SHAP tells us *which feature pushed the score
which way*. We use this for two things:

  1. The Streamlit dashboard shows per-recommendation feature breakdown
     ("AAPL +0.65: sentiment +0.30, RSI +0.20, VIX -0.10, ...").
  2. The LLM post-mortem layer (`postmortem.py`) gets the SHAP values
     for losing trades and writes a natural-language analysis.

Why TreeExplainer specifically: it's exact (not approximate) for
gradient-boosted trees and runs in milliseconds per prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from src.model.predictor import Predictor


class ShapExplainer:
    def __init__(self, predictor: Predictor):
        if predictor.booster is None:
            raise RuntimeError("Predictor has no trained booster.")
        self.predictor = predictor
        self.explainer = shap.TreeExplainer(predictor.booster)

    def shap_values(self, df: pd.DataFrame) -> np.ndarray:
        """Returns SHAP values shape (n_rows, n_classes, n_features)."""
        X = df[self.predictor.feature_cols].values
        vals = self.explainer.shap_values(X)
        # multi-class returns list of arrays — stack into (n, n_classes, n_features)
        if isinstance(vals, list):
            return np.stack(vals, axis=1)
        return vals

    def signed_attribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """SHAP attribution to the *signed score* (P(up) - P(down)).

        Returns long-format: (row_idx, symbol, feature, shap_value).
        Sorted descending by absolute value within each row.
        """
        X = df[self.predictor.feature_cols].values
        vals = self.shap_values(df)        # (n, 3, n_features)
        # XGBoost classes order: 0=down, 1=flat, 2=up  (see predictor._TO_XGB)
        signed = vals[:, 2, :] - vals[:, 0, :]   # contribution to P(up)-P(down)
        rows = []
        for i, sym in enumerate(df["symbol"].values):
            for j, feat in enumerate(self.predictor.feature_cols):
                rows.append({
                    "row_idx": i,
                    "symbol": sym,
                    "feature": feat,
                    "value": X[i, j],
                    "shap": float(signed[i, j]),
                })
        return pd.DataFrame(rows)

    def top_features_for_row(
        self,
        df: pd.DataFrame,
        row_idx: int,
        top_n: int = 5,
    ) -> pd.DataFrame:
        """Top-N features (by |shap|) driving the score of one row.
        This is what gets stored in the trade journal."""
        attribution = self.signed_attribution(df.iloc[[row_idx]].reset_index(drop=True))
        attribution["abs_shap"] = attribution["shap"].abs()
        return attribution.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap")
