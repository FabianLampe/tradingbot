# Trading Bot — News-driven, Multi-Source, Self-Reflective

Ein news-getriebener Trading-Bot der Empfehlungen für ETFs/Aktien gibt
und aus eigenen Fehltrades lernt. Default: nur Empfehlungen, kein
Auto-Trading.

> **WARNUNG:** Trading birgt reales Verlustrisiko. Dieser Bot ist ein
> Lern- und Analyse-Tool, kein Geldautomat. Nutze Paper-Trading bevor
> du auch nur einen Euro live einsetzt.

---

## Schnellstart (lokal)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env             # Keys eintragen (siehe DEPLOY.md)
python -m ipykernel install --user --name=trading-bot --display-name "Python (trading-bot)"
jupyter lab
```

Dann die Notebooks `01 → 02 → 03 → 04` der Reihe nach ausführen.

---

## Architektur

```
data/                       # Parquet-Cache, Modelle, Journal — niemals committen
├── prices/                 # OHLCV pro Ticker
├── news/                   # Finnhub-News + RSS + Reddit
├── macro/                  # FRED-Panel
├── edgar/                  # SEC-Filings (Phase 4)
├── knowledge/              # Chroma Vector-DB (Phase 4)
├── models/                 # Trainierte Predictoren
└── journal/                # Trade Journal (SQLite) + Backtest-Output

src/
├── data/                   # Ingestion: yfinance, Finnhub, FRED, Reddit, RSS
│   ├── universe.py         # S&P-500-Liste (Wikipedia, gecacht) + ETF-Universum
│   ├── prices.py           # OHLCV via yfinance (Cache wird gemerged, nie ersetzt)
│   ├── macro.py            # FRED-Panel (mit Publikations-Lag gegen Leakage)
│   ├── intraday.py         # Stundenbalken (yfinance, ~2 Jahre Historie)
│   ├── news.py             # Company-News (Finnhub, Fallback Alpha Vantage)
│   ├── rss_news.py         # Markt-News aus den Whitelist-RSS-Feeds
│   └── social.py           # Reddit (kuratierte Subreddits)
├── features/               # Returns, Korrelationen, TA, FinBERT-Sentiment
├── model/                  # XGBoost-Predictor, Walk-Forward-Backtest, Trade Journal
│   └── costs.py            # Spread/Slippage/Gebühren — in Euro, nicht in bps
├── execution/
│   └── paper.py            # Paper-Broker: echte Kurse, echte Kosten, kein Geld
├── runtime/                # Daily/Pre-Market Orchestrator
│   └── events.py           # Ereignisgetriebene Signale statt festem Takt
├── explain/                # SHAP-Attribution + LLM-Postmortem
└── knowledge/              # SEC EDGAR + Chroma RAG

app/
├── dashboard.py            # Streamlit Dashboard
└── telegram_bot.py         # Telegram Bot

scripts/
├── run_daily.py            # End-of-Day Pipeline
├── run_premarket.py        # Pre-Market Adjustment
├── run_backtest.py         # Walk-Forward Backtest
├── run_postmortems.py      # LLM-Analyse Fehltrades
├── ingest_news.py          # News-Backfill in den Parquet-Cache
├── ingest_edgar.py         # SEC EDGAR -> RAG
└── scheduler.py            # APScheduler-Daemon

config/
└── whitelist.yaml          # kuratierte Quellen (Twitter/Reddit/News)
```

---

## CLI-Cheatsheet

| Kommando | Was es tut |
|---|---|
| `python scripts/run_daily.py --universe etfs` | EOD-Pipeline (Daten, Training, Empfehlungen) |
| `python scripts/run_premarket.py` | Overnight-News-Anpassung |
| `python scripts/run_backtest.py --universe etfs` | Walk-Forward-Backtest (inkl. Alpha/IR gegen S&P 500) |
| `python scripts/run_backtest.py --initial 1200 --cost-preset neobroker_1eur` | Backtest mit realistischen Kosten für dein echtes Kapital |
| `python scripts/run_backtest.py --no-cross-sectional` | Backtest auf Feature-Rohwerten statt Tagesrängen |
| `python scripts/run_paper.py --refresh --replay 45` | Ereignisgetriebenes Paper-Trading auf Stundenbalken |
| `python scripts/run_paper.py --report` | Stand des Paper-Depots |
| `python scripts/run_postmortems.py --limit 10` | LLM analysiert die letzten 10 Fehltrades |
| `python scripts/ingest_news.py --symbols AAPL,MSFT --days-back 90` | News-Backfill pro Ticker |
| `python scripts/ingest_news.py --rss --reddit` | Markt-RSS + Reddit einmalig ziehen |
| `python scripts/ingest_edgar.py --universe sp500` | SEC-Filings in RAG einbetten |
| `streamlit run app/dashboard.py` | Dashboard auf :8501 |
| `python -m app.telegram_bot` | Telegram-Bot Listener |
| `python -m app.telegram_bot push` | Einmaliger Push der Empfehlungen |

---

## Server-Deployment

Siehe **[DEPLOY.md](DEPLOY.md)** — Docker-Compose-Setup mit drei Services
(Dashboard, Telegram, Scheduler) für deinen GPU-Server.

---

## Konfiguration anpassen

| Was | Wo |
|---|---|
| API-Keys | `.env` |
| Quellen-Whitelist (Twitter/Reddit/News) | `config/whitelist.yaml` |
| RSS-Feed-URLs pro Outlet | `config/whitelist.yaml` `rss_feeds:` |
| Trading-Parameter (Top-N, Kosten, Hebel) | `scripts/run_*.py` Args |
| Modell-Hyperparameter | `src/model/predictor.py` `Predictor.params` |
| Pipeline-Reihenfolge | `src/runtime/daily.py` |
| LLM-Backend (Claude vs lokal) | `.env` `LLM_BACKEND` |

---

## Sicherheit & Recht

- `.env` ist gitignored — niemals committen.
- DE-Steuerrecht: Kurzfrist-Gewinne unterliegen 25% + Soli. **Cap auf
  Termingeschäft-Verluste 20.000 EUR/Jahr** — siehe DEPLOY.md.
- Bot ist Empfehlungssystem. Live-Trading wäre eigene Entscheidung
  und braucht zusätzliche Implementierung der Broker-Anbindung.
