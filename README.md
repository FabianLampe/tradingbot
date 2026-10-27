# tradingbot

News-driven recommendation system for ETFs and stocks. It combines price data, news sentiment and macro indicators, trains a gradient-boosting model, backtests it walk-forward and reviews its own losing trades. By default it only produces recommendations and does not place orders.

> This is a learning and analysis project, not financial advice. Trading carries a real risk of loss. Use paper trading before putting any money at risk.

## Features

- Data ingestion: prices (yfinance), news (Finnhub, RSS, Reddit), macro series (FRED), SEC filings (EDGAR)
- Features: returns, correlations, technical indicators, FinBERT sentiment
- Model: XGBoost predictor with walk-forward backtesting and a trade journal (SQLite)
- Explanations: SHAP attribution and LLM post-mortems of losing trades, with a Chroma vector store for filings
- Interfaces: Streamlit dashboard and Telegram bot
- Scheduler for the daily and pre-market pipelines, deployable with Docker Compose

## Setup

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env             # add API keys, see DEPLOY.md
jupyter lab
```

Run the notebooks `01` to `04` in order to download data, build features and train the first model.

## Usage

| Command | Purpose |
|---|---|
| `python scripts/run_daily.py --universe etfs` | end-of-day pipeline: data, training, recommendations |
| `python scripts/run_premarket.py` | adjust for overnight news |
| `python scripts/run_backtest.py --universe etfs` | walk-forward backtest |
| `python scripts/run_postmortems.py --limit 10` | LLM analysis of the last 10 losing trades |
| `python scripts/ingest_edgar.py --universe sp500` | embed SEC filings for retrieval |
| `streamlit run app/dashboard.py` | dashboard on port 8501 |
| `python -m app.telegram_bot` | Telegram bot listener |

## Project layout

```
src/
  data/          ingestion
  features/      returns, technical indicators, sentiment
  model/         predictor, backtest, trade journal
  runtime/       daily and pre-market orchestration
  explain/       SHAP attribution, LLM post-mortems
  knowledge/     SEC EDGAR and Chroma retrieval
app/             Streamlit dashboard, Telegram bot
scripts/         pipeline entry points, scheduler
config/          curated source whitelist
notebooks/       setup, data, features, model
```

## Configuration

| Setting | Location |
|---|---|
| API keys | `.env` (not committed) |
| Source whitelist | `config/whitelist.yaml` |
| Model hyperparameters | `Predictor.params` in `src/model/predictor.py` |
| LLM backend (hosted or local) | `LLM_BACKEND` in `.env` |

## Deployment

See [DEPLOY.md](DEPLOY.md) for the Docker Compose setup with dashboard, Telegram bot and scheduler.

## Notes

The bot is a recommendation system. Live trading would require a broker integration that this project does not include. Tax rules for short-term gains and derivatives are described in [DEPLOY.md](DEPLOY.md).
