# Deployment — Trading Bot auf deinem GPU-Server

Vollständige Anleitung für den Server (4× 32 GB GPU, 1 TB RAM).

---

## Voraussetzungen auf dem Server

```bash
# Linux mit NVIDIA-Treibern
nvidia-smi                    # muss alle 4 GPUs zeigen

# Docker + NVIDIA Container Toolkit
sudo apt-get install docker.io docker-compose
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

---

## Erstes Deployment

```bash
# 1. Code auf den Server kopieren
rsync -avz --exclude data --exclude .venv --exclude .git \
    "c:/Users/fabia/Tranding bot/" user@server:/srv/trading-bot/
ssh user@server
cd /srv/trading-bot

# 2. .env anlegen
cp .env.example .env
nano .env                     # Keys eintragen (siehe unten)

# 3. Image bauen (~5–10 Min beim ersten Mal)
docker compose build

# 4. Erst-Daten ziehen (Notebooks 01 + 02 + 03 laufen lassen)
# Variante A: lokal vorbereiten und per rsync hochladen
# Variante B: direkt auf dem Server in einem Container
docker compose run --rm dashboard python -c "
import sys; sys.path.insert(0, '.')
from src.data import universe, prices, macro
prices.download_and_cache(universe.etf_symbols())   # ETFs first (klein, schnell)
df = macro.fetch_panel(start='2005-01-01')
df.to_parquet('data/macro/fred_panel.parquet')
print('Daten gezogen.')
"

# 5. Erstes Modell trainieren + Backtest
docker compose run --rm scheduler python scripts/run_backtest.py --universe etfs

# 6. Alle Services starten
docker compose up -d

# 7. Logs anschauen
docker compose logs -f scheduler
```

---

## Welche API-Keys du wirklich brauchst

| Service | Pflicht? | Kosten | Wofür |
|---|---|---|---|
| `FINNHUB_API_KEY` | empfohlen | gratis | Company-News |
| `FRED_API_KEY` | empfohlen | gratis | Makrodaten (VIX, Fed-Funds, …) |
| `REDDIT_CLIENT_ID/SECRET` | optional | gratis | Reddit-Whitelist (Phase 3c) |
| `ANTHROPIC_API_KEY` | optional | ~$0.01/Postmortem | LLM-Fehleranalyse |
| `TELEGRAM_BOT_TOKEN/CHAT_ID` | optional | gratis | Push-Empfehlungen |

Ohne Telegram/LLM/Reddit läuft der Bot, du verlierst nur diese Features.

---

## Services & Ports

Nach `docker compose up -d`:

| Container | Was | Wo erreichbar |
|---|---|---|
| `tb-dashboard` | Streamlit UI | http://server:8501 |
| `tb-telegram` | Telegram Bot Listener | Telegram-App |
| `tb-scheduler` | Daily 22:30 + Pre-Market 14:30 | nur Logs |

**Dashboard absichern:** Port 8501 nicht direkt ins Internet öffnen. Stattdessen:
```bash
# SSH-Tunnel von deinem Laptop
ssh -L 8501:localhost:8501 user@server
# dann lokal http://localhost:8501 aufrufen
```
Oder einen Reverse-Proxy mit HTTPS + Basic-Auth davorschalten (caddy / nginx).

---

## Daten initial vollständig laden

ETF-Universum reicht zum Testen. Für Produktiveinsatz S&P 500 + News:

```bash
# Im scheduler-Container starten (hat GPUs für FinBERT)
docker compose exec scheduler bash

# Innerhalb des Containers:
python scripts/run_daily.py --universe sp500    # 1× komplett ziehen
python scripts/ingest_edgar.py --universe sp500 --limit 2 --email DEINEMAIL@example.com
exit
```

EDGAR-Ingest läuft ~2 Stunden für alle 500 Tickers (SEC erlaubt nur 1 req/s).

---

## Monitoring

```bash
docker compose logs -f                      # alle Services
docker compose logs -f scheduler            # nur scheduler
docker compose ps                           # Container-Status
docker stats                                # CPU/RAM/GPU live

# Disk
du -sh /srv/trading-bot/data/*
# typische Größen nach 1 Monat:
#   data/prices/    150 MB
#   data/news/      50 MB
#   data/macro/     <1 MB
#   data/edgar/     5 GB    (komprimiert ~1 GB)
#   data/knowledge/ 500 MB  (Chroma vector store)
#   data/journal/   ~10 MB  (SQLite, wächst langsam)
```

---

## Updates ausrollen

```bash
cd /srv/trading-bot
git pull   # falls du es per Git deployst — sonst rsync wie oben
docker compose build
docker compose up -d --force-recreate
```

Daten und Modelle überleben Restarts (durch das Volume-Mount auf `./data`).

---

## Backup

Wichtig sind nur diese drei Verzeichnisse:
```bash
tar czf trading-bot-backup-$(date +%F).tar.gz \
    /srv/trading-bot/data/journal \
    /srv/trading-bot/data/models \
    /srv/trading-bot/.env
```
Alles andere kann jederzeit neu heruntergeladen werden.

---

## Troubleshooting

| Symptom | Wahrscheinliche Ursache | Fix |
|---|---|---|
| Dashboard zeigt „Noch keine Empfehlungen" | Daily-Run noch nie gelaufen | `docker compose exec scheduler python scripts/run_daily.py` |
| FinBERT nutzt CPU statt GPU | NVIDIA-Toolkit fehlt im Container | `nvidia-smi` im Container testen, ggf. Toolkit neu installieren |
| Telegram Bot reagiert nicht | falsche `TELEGRAM_CHAT_ID` | nochmal `getUpdates`-URL aufrufen |
| Backtest sehr langsam | volle S&P 500 statt ETFs | `--universe etfs` für Tests |
| EDGAR-Ingest bricht ab | SEC blockiert (User-Agent) | `--email DEINE_ECHTE@example.com` setzen |
| OOM bei FinBERT | Batch-Size zu hoch | `FinBERTScorer(batch_size=32)` |

---

## Live-Modus aktivieren (NACH 6 Monaten Paper-Trading!)

Der Bot ist standardmäßig nur **Empfehlungs-System**. Um echte Orders auszulösen:
1. Broker-Anbindung implementieren (z.B. `ib_insync` für Interactive Brokers)
2. Position-Sizing-Modul mit deinem realen Kapital ergänzen
3. Stop-Loss-Logik in den Broker übergeben

Diese Schritte sind **bewusst nicht** automatisch enthalten — Live-Trading
braucht eine bewusste Einzelentscheidung.
