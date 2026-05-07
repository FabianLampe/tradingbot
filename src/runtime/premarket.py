"""Pre-market module: react to overnight news.

US market closes 22:00 dt. time, opens 15:30 next day. Anything that
breaks between those two times — earnings releases, geopolitical events,
European session moves — is "overnight news" the model should weigh.

This module:
  1. Pulls news with `published >= last_market_close`.
  2. Scores them with FinBERT (Phase 2a).
  3. Aggregates per ticker.
  4. Boosts/Damps the predictor's score using overnight sentiment.
  5. Emits the pre-market recommendation list.

Designed to run at ~14:30 dt. time (1 hour before open). Evening
end-of-day run at 22:30 produces the *base* recommendation; this run
*adjusts* it for overnight events.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.data import news as news_mod
from src.features import sentiment as sent_mod
from src.model.predictor import Predictor

ET = ZoneInfo("America/New_York")
DE = ZoneInfo("Europe/Berlin")


def last_us_market_close(now: datetime | None = None) -> datetime:
    """Return the most recent US market close (16:00 ET) before `now`."""
    now = (now or datetime.now(tz=DE)).astimezone(ET)
    candidate = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now < candidate or now.weekday() >= 5:
        # walk back to last weekday close
        candidate -= timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate -= timedelta(days=1)
    return candidate


def fetch_overnight_news(
    symbols: list[str],
    since: datetime | None = None,
) -> pd.DataFrame:
    """Pull news per symbol since last market close. Returns long DataFrame."""
    since = since or last_us_market_close()
    days_back = max(1, (datetime.now(tz=DE).astimezone(ET) - since).days + 1)
    bulk = news_mod.fetch_bulk_company_news(symbols, days_back=days_back)
    rows = []
    for sym, df in bulk.items():
        if df.empty:
            continue
        df = df[df["datetime"] >= pd.Timestamp(since)]
        if df.empty:
            continue
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def overnight_sentiment_features(
    overnight_news: pd.DataFrame,
    scorer: sent_mod.FinBERTScorer | None = None,
) -> pd.DataFrame:
    """Score + aggregate overnight news per symbol.

    Returns columns: symbol, n_articles_overnight, sent_overnight,
                     sent_overnight_max_neg, has_breaking_neg
    """
    if overnight_news.empty:
        return pd.DataFrame(columns=[
            "symbol", "n_articles_overnight", "sent_overnight",
            "sent_overnight_max_neg", "has_breaking_neg",
        ])

    scorer = scorer or sent_mod.FinBERTScorer()
    scored = sent_mod.score_news_dataframe(overnight_news, scorer=scorer)

    grouped = scored.groupby("symbol").agg(
        n_articles_overnight=("score", "size"),
        sent_overnight=("score", "mean"),
        sent_overnight_max_neg=("p_negative", "max"),
    ).reset_index()
    # "Breaking negative" = at least one article with P(neg) > 0.85
    grouped["has_breaking_neg"] = (grouped["sent_overnight_max_neg"] > 0.85).astype(int)
    return grouped


def adjust_predictions(
    base_recommendations: pd.DataFrame,
    overnight_features: pd.DataFrame,
    boost_strength: float = 0.3,
) -> pd.DataFrame:
    """Combine base score with overnight sentiment.

    `base_recommendations` must have columns [symbol, score].
    Returns a copy with extra columns and a new `adjusted_score`.

    Heuristic (deliberately simple — Phase 5 we let the model learn this):
        adjusted = base_score + boost_strength * overnight_sentiment
    Plus: if `has_breaking_neg`, cap long signals at 0 (don't recommend
    going long into a fresh disaster headline).
    """
    out = base_recommendations.merge(overnight_features, on="symbol", how="left")
    out[["n_articles_overnight", "sent_overnight",
         "sent_overnight_max_neg", "has_breaking_neg"]] = (
        out[["n_articles_overnight", "sent_overnight",
             "sent_overnight_max_neg", "has_breaking_neg"]].fillna(0.0)
    )
    out["adjusted_score"] = out["score"] + boost_strength * out["sent_overnight"]
    # Veto rule: never recommend long after breaking-negative news
    veto = (out["has_breaking_neg"] == 1) & (out["adjusted_score"] > 0)
    out.loc[veto, "adjusted_score"] = 0.0
    out["vetoed"] = veto.astype(int)
    return out.sort_values("adjusted_score", ascending=False)


def run_premarket(
    symbols: list[str],
    today_features: pd.DataFrame,
    predictor: Predictor,
    scorer: sent_mod.FinBERTScorer | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Full pre-market pipeline: returns the adjusted Top-N recommendation."""
    today_features = today_features.copy()
    today_features["score"] = predictor.predict_score(today_features)
    base = today_features[["symbol", "score"]]

    overnight_news = fetch_overnight_news(symbols)
    overnight_feat = overnight_sentiment_features(overnight_news, scorer=scorer)

    adjusted = adjust_predictions(base, overnight_feat)
    return adjusted.head(top_n)
