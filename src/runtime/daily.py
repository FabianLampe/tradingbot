"""Daily orchestrator — wires all phases into one pipeline.

Designed to run twice per trading day:
  - 22:30 dt. time (after US close): full daily run — refresh data,
    score news, retrain (warm-start), generate next-day recommendations.
  - 14:30 dt. time (before US open): pre-market adjustment using
    overnight news.

Each step is wrapped so a single failure does not kill the whole run.
Errors are logged; the pipeline continues with whatever data it has.
"""
from __future__ import annotations

import logging
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd

import config
from src.data import macro as macro_mod
from src.data import news as news_mod
from src.data import prices as prices_mod
from src.data import rss_news, social, universe
from src.features import build_dataset, sentiment
from src.model import journal as journal_mod
from src.model.journal import PredictionRecord
from src.model.predictor import Predictor
from src.runtime import premarket as pre_mod
from src.storage import db

log = logging.getLogger("trading_bot.daily")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# ------------------ pipeline steps (each wrapped, failure-tolerant) -------

def _safe(step_name: str):
    """Decorator: log + swallow exceptions so one step's failure
    doesn't kill the daily run."""
    def deco(fn):
        def wrapper(*a, **kw):
            log.info(">>> %s", step_name)
            try:
                result = fn(*a, **kw)
                log.info("<<< %s OK", step_name)
                return result
            except Exception as e:  # noqa: BLE001
                log.error("<<< %s FAILED: %s\n%s", step_name, e, traceback.format_exc())
                return None
        return wrapper
    return deco


@_safe("refresh prices")
def step_refresh_prices(symbols: list[str]) -> int:
    """Re-download last 30 days for every symbol (overwrites/appends file)."""
    from datetime import timedelta
    start = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
    cached = prices_mod.download_and_cache(symbols, start=start)
    return len(cached)


@_safe("refresh macro")
def step_refresh_macro() -> int:
    df = macro_mod.fetch_panel(start="2005-01-01")
    df.to_parquet(config.MACRO_DIR / "fred_panel.parquet")
    return len(df)


@_safe("refresh company news + score")
def step_refresh_news(symbols: list[str], scorer: sentiment.FinBERTScorer) -> int:
    bulk = news_mod.fetch_bulk_company_news(symbols, days_back=2)
    year = datetime.utcnow().year
    n_articles = 0
    for sym, df in bulk.items():
        if df.empty:
            continue
        scored = sentiment.score_news_dataframe(df, scorer=scorer)
        # merge with existing year-file (idempotent on news_id)
        existing = pd.DataFrame()
        path = db.news_path(sym, year)
        if path.exists():
            existing = pd.read_parquet(path)
        combined = pd.concat([existing, scored], ignore_index=True)
        if "news_id" in combined.columns:
            combined = combined.drop_duplicates(subset=["news_id"], keep="last")
        db.write_news(sym, year, combined)
        n_articles += len(scored)
    return n_articles


@_safe("refresh RSS market news")
def step_refresh_rss() -> int:
    df = rss_news.fetch_all_whitelisted()
    if df.empty:
        return 0
    out_path = config.NEWS_DIR / "_rss_market.parquet"
    df.to_parquet(out_path, index=False)
    return len(df)


@_safe("refresh Reddit")
def step_refresh_reddit() -> int:
    df = social.fetch_whitelisted(time_filter="day", limit_per_sub=100)
    if df.empty:
        return 0
    out_path = config.NEWS_DIR / "_reddit.parquet"
    df.to_parquet(out_path, index=False)
    return len(df)


@_safe("warm-start retrain")
def step_retrain(
    symbols: list[str],
    horizon_days: int = 5,
    force_full: bool = False,
) -> Predictor:
    """Daily warm-start update, OR full retrain if forced (drift / weekly job)."""
    dataset = build_dataset.build_dataset(symbols, horizon_days=horizon_days)
    if dataset.empty:
        raise RuntimeError("Empty dataset — check that prices + features built correctly.")

    feature_cols = build_dataset.feature_columns(dataset)
    transform = build_dataset.feature_transform_of(dataset)

    do_full = force_full
    if not do_full:
        try:
            predictor = Predictor.load("predictor")
            # Feature schema changed -> must full-retrain
            if set(predictor.feature_cols) != set(feature_cols):
                log.info("Feature schema changed — forcing full retrain.")
                do_full = True
            # Same column names, different scaling — a warm start here would
            # feed per-date ranks to a model trained on raw levels.
            elif predictor.feature_transform != transform:
                log.info("Feature scaling changed (%s -> %s) — forcing full retrain.",
                         predictor.feature_transform, transform)
                do_full = True
        except FileNotFoundError:
            do_full = True

    if do_full:
        predictor = Predictor(feature_cols=feature_cols, horizon_days=horizon_days,
                              feature_transform=transform)
        cutoff = dataset["date"].quantile(0.9)
        train = dataset[dataset["date"] < cutoff]
        val = dataset[dataset["date"] >= cutoff]
        predictor.train(train, df_val=val)
        log.info("Full retrain on %d rows (val %d)", len(train), len(val))
    else:
        recent = dataset[dataset["date"] >= dataset["date"].max() - pd.Timedelta(days=30)]
        predictor.update(recent, n_rounds=50)
        log.info("Warm-start update on %d rows", len(recent))
    predictor.save("predictor")
    return predictor


@_safe("drift check")
def step_drift_check() -> "object | None":
    """Returns a DriftReport (or None if drift module not available)."""
    from src.model import drift
    rep = drift.assess_drift()
    log.info("Drift: %s", rep.reason)
    return rep


@_safe("generate recommendations")
def step_generate_recommendations(
    predictor: Predictor,
    symbols: list[str],
    top_n_long: int = 10,
    top_n_short: int = 5,
) -> pd.DataFrame:
    """Build today's feature row per symbol, predict, save Top-N."""
    dataset = build_dataset.build_dataset(symbols, horizon_days=predictor.horizon_days)
    if dataset.empty:
        raise RuntimeError("No dataset rows for recommendations.")

    today = dataset.sort_values("date").groupby("symbol").tail(1)
    today = today.dropna(subset=predictor.feature_cols)
    today["score"] = predictor.predict_score(today)
    today = today.sort_values("score", ascending=False)

    longs = today.head(top_n_long).assign(action="long")
    shorts = today.tail(top_n_short).assign(action="short")
    rec = pd.concat([longs, shorts])[["date", "symbol", "score", "action"]]
    rec.to_parquet(config.JOURNAL_DIR / "today_recommendations.parquet", index=False)

    # log to trade journal
    asof = str(today["date"].max().date()) if hasattr(today["date"].iloc[0], "date") else str(today["date"].max())
    macro_snapshot = _last_macro_row()
    for _, row in rec.iterrows():
        prediction = PredictionRecord(
            asof_date=asof,
            symbol=row["symbol"],
            score=float(row["score"]),
            action=row["action"],
            horizon_days=predictor.horizon_days,
            top_features={},   # filled by SHAP step below
            sentiment_inputs={},
            macro_snapshot=macro_snapshot,
        )
        journal_mod.log_prediction(prediction)
    return rec


@_safe("evaluate due predictions")
def step_evaluate_outcomes() -> int:
    """Find predictions whose horizon has elapsed and book the actual return."""
    pending = journal_mod.predictions_pending_outcome(horizon_days=5)
    n = 0
    for _, p in pending.iterrows():
        try:
            prices = db.read_prices(p["symbol"])
        except FileNotFoundError:
            continue
        asof = pd.Timestamp(p["asof_date"])
        try:
            entry = prices.loc[prices.index >= asof].iloc[0]["adj_close"]
            exit_ = prices.loc[prices.index >= asof + pd.tseries.offsets.BDay(p["horizon_days"])].iloc[0]["adj_close"]
        except (IndexError, KeyError):
            continue
        realised = float((exit_ - entry) / entry)
        if p["action"] == "short":
            realised = -realised
        success = realised > 0
        journal_mod.log_outcome(p["id"], realised_return=realised, success=success)
        n += 1
    return n


def _last_macro_row() -> dict:
    p = config.MACRO_DIR / "fred_panel.parquet"
    if not p.exists():
        return {}
    df = pd.read_parquet(p)
    if df.empty:
        return {}
    last = df.iloc[-1]
    return {k: float(v) for k, v in last.items() if pd.notna(v)}


# ------------------------- entry points ---------------------------------

def run_daily_eod(universe_size: str = "etfs", force_full_retrain: bool = False) -> dict:
    """End-of-Day pipeline (~22:30 dt. time).

    universe_size:
      'etfs'   - just the ~30 ETFs (fast, for testing)
      'sp500'  - all S&P 500 stocks (~10x slower)
      'all'    - ETFs + S&P 500 stocks
    force_full_retrain:
      True  - always do a from-scratch retrain (used by weekly Sunday job)
      False - warm-start unless drift detected or feature schema changed
    """
    log.info("=== DAILY END-OF-DAY RUN ===  (universe=%s, full_retrain=%s)",
             universe_size, force_full_retrain)

    if universe_size == "etfs":
        symbols = universe.etf_symbols()
    elif universe_size == "sp500":
        symbols = universe.symbols()
    else:
        symbols = universe.all_symbols()
    log.info("Universe size: %d symbols", len(symbols))

    n_evaluated = step_evaluate_outcomes()

    # Drift check — if drift detected, force full retrain
    drift_rep = step_drift_check()
    drift_triggered = bool(drift_rep and getattr(drift_rep, "drift_detected", False))
    if drift_triggered:
        log.warning("Drift detected — forcing full retrain. Reason: %s", drift_rep.reason)
        force_full_retrain = True

    n_prices = step_refresh_prices(symbols)
    n_macro = step_refresh_macro()
    n_rss = step_refresh_rss()
    n_reddit = step_refresh_reddit()

    # FinBERT scorer is heavy — load once, reuse
    scorer = sentiment.FinBERTScorer()
    n_news = step_refresh_news(symbols, scorer=scorer)

    predictor = step_retrain(symbols, force_full=force_full_retrain)
    rec = step_generate_recommendations(predictor, symbols) if predictor is not None else None

    summary = {
        "evaluated_outcomes": n_evaluated,
        "drift_detected": drift_triggered,
        "drift_reason": getattr(drift_rep, "reason", None) if drift_rep else None,
        "full_retrain_done": force_full_retrain,
        "prices_refreshed": n_prices,
        "macro_rows": n_macro,
        "rss_articles": n_rss,
        "reddit_posts": n_reddit,
        "news_articles_scored": n_news,
        "recommendations": len(rec) if rec is not None else 0,
    }
    log.info("=== DAILY EOD COMPLETE ===  %s", summary)
    return summary


def run_premarket(universe_size: str = "etfs") -> dict:
    """Pre-market pipeline (~14:30 dt. time)."""
    log.info("=== PRE-MARKET RUN ===  (universe=%s)", universe_size)
    if universe_size == "etfs":
        symbols = universe.etf_symbols()
    elif universe_size == "sp500":
        symbols = universe.symbols()
    else:
        symbols = universe.all_symbols()

    try:
        predictor = Predictor.load("predictor")
    except FileNotFoundError:
        log.error("No trained predictor — run EOD first.")
        return {"error": "no predictor"}

    dataset = build_dataset.build_dataset(symbols, horizon_days=predictor.horizon_days)
    today = dataset.sort_values("date").groupby("symbol").tail(1).dropna(subset=predictor.feature_cols)
    scorer = sentiment.FinBERTScorer()
    rec = pre_mod.run_premarket(symbols, today, predictor, scorer=scorer, top_n=10)
    rec.to_parquet(config.JOURNAL_DIR / "today_recommendations.parquet", index=False)
    log.info("Pre-market recommendations: %d rows", len(rec))
    return {"recommendations": len(rec)}
