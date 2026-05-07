"""News sentiment scoring with FinBERT.

We use **ProsusAI/finbert** — a BERT model fine-tuned on the Financial
PhraseBank dataset. It outputs three classes: positive / neutral / negative.
We expose two surfaces:

  - `FinBERTScorer.score(texts)` -> per-text probabilities + a signed score.
  - `aggregate_daily(news_df)`   -> per-(symbol, date) features ready to
                                    join onto the price panel.

The signed score is `P(positive) - P(negative)`, range [-1, 1]. This
collapses three probabilities into one number that downstream models
can consume directly.

Performance notes for your hardware (4× 32GB GPU):
  - FinBERT base is ~440 MB; batch_size=64 fits easily on one GPU.
  - For ~1M news headlines, scoring runs in ~5–10 minutes on a single
    A100/V100. No need for multi-GPU at this scale.
  - Set `device="cuda"` in production; "cpu" works for testing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]  # ProsusAI/finbert label order


@dataclass
class SentimentResult:
    text: str
    p_positive: float
    p_negative: float
    p_neutral: float
    score: float  # P(pos) - P(neg)


class FinBERTScorer:
    """Wraps a HuggingFace pipeline with a clean batch API."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Map model's id2label to canonical order [positive, negative, neutral]
        id2label = self.model.config.id2label
        self._idx = {LABELS[i]: next(k for k, v in id2label.items()
                                      if v.lower() == LABELS[i]) for i in range(3)}

    @torch.no_grad()
    def score(
        self,
        texts: Iterable[str],
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Score a list of strings. Returns DataFrame with columns:
        text, p_positive, p_negative, p_neutral, score.
        """
        texts = [str(t) if t is not None else "" for t in texts]
        n = len(texts)
        out = np.zeros((n, 3), dtype=np.float32)

        rng = range(0, n, batch_size)
        if show_progress:
            rng = tqdm(rng, desc=f"FinBERT ({self.device})", total=(n + batch_size - 1) // batch_size)

        for i in rng:
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out[i : i + len(batch)] = probs

        p_pos = out[:, self._idx["positive"]]
        p_neg = out[:, self._idx["negative"]]
        p_neu = out[:, self._idx["neutral"]]
        return pd.DataFrame({
            "text": texts,
            "p_positive": p_pos,
            "p_negative": p_neg,
            "p_neutral": p_neu,
            "score": p_pos - p_neg,
        })


def score_news_dataframe(
    news_df: pd.DataFrame,
    scorer: FinBERTScorer | None = None,
    text_columns: tuple[str, ...] = ("headline", "summary"),
    batch_size: int = 64,
) -> pd.DataFrame:
    """Score a Finnhub-style news DataFrame in place.

    Concatenates the chosen text columns (headline + summary by default)
    before scoring — gives FinBERT more context than headline alone.
    """
    scorer = scorer or FinBERTScorer()
    texts = (
        news_df[list(text_columns)]
        .fillna("")
        .agg(" — ".join, axis=1)
        .tolist()
    )
    scored = scorer.score(texts, batch_size=batch_size)
    out = news_df.reset_index(drop=True).copy()
    for col in ("p_positive", "p_negative", "p_neutral", "score"):
        out[col] = scored[col].values
    return out


def aggregate_daily(
    scored_news: pd.DataFrame,
    symbol_col: str = "symbol",
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Aggregate per-article sentiment to per-(symbol, date).

    Returns columns:
        n_articles, mean_score, mean_pos, mean_neg, max_neg, std_score
    These five capture both *direction* and *intensity/disagreement* of
    the day's news flow.
    """
    df = scored_news.copy()
    df["date"] = pd.to_datetime(df[datetime_col]).dt.tz_convert(None).dt.normalize()

    grouped = df.groupby([symbol_col, "date"]).agg(
        n_articles=("score", "size"),
        mean_score=("score", "mean"),
        mean_pos=("p_positive", "mean"),
        mean_neg=("p_negative", "mean"),
        max_neg=("p_negative", "max"),
        std_score=("score", "std"),
    ).reset_index()
    grouped["std_score"] = grouped["std_score"].fillna(0.0)
    return grouped
