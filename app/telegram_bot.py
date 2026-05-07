"""Telegram Bot — Push & Query (Deutsch).

Funktionen:
  - Pusht tägliche Empfehlungen automatisch um 22:30 (nach Marktschluss US)
    und um 14:30 (Pre-Market mit Overnight-News-Adjustment).
  - Antwortet auf Befehle:
        /heute        - heutige Empfehlungen
        /journal      - letzte Trades aus dem Journal
        /performance  - Win-Rate, Sharpe etc.
        /status       - Bot-Status
        /hilfe        - Befehlsübersicht

Setup:
  1. Auf Telegram @BotFather schreiben → /newbot → Token speichern
  2. Bot anschreiben (irgendeine Nachricht senden)
  3. https://api.telegram.org/bot<TOKEN>/getUpdates aufrufen → chat_id finden
  4. In .env: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

Start:
    python -m app.telegram_bot           # Bot läuft als Listener
    python -m app.telegram_bot push      # Einmaliger Push der heutigen Empfehlungen
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

import config
from src.model import journal as journal_mod


# ----------------------------- Helpers ---------------------------------

def _today_recommendations() -> pd.DataFrame | None:
    p = config.JOURNAL_DIR / "today_recommendations.parquet"
    return pd.read_parquet(p) if p.exists() else None


def _format_recommendations_de(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Keine Empfehlungen vorhanden. Lass den Daily-Lauf einmal durch."
    long = df[df["action"] == "long"].sort_values("score", ascending=False).head(10)
    short = df[df["action"] == "short"].sort_values("score").head(5)
    parts = ["📈 *Heutige Empfehlungen*\n"]
    if not long.empty:
        parts.append("*🟢 Long Top 10:*")
        for _, r in long.iterrows():
            parts.append(f"  • `{r['symbol']:6}` Score {r['score']:+.2f}")
    if not short.empty:
        parts.append("\n*🔴 Short Top 5* _(nur Logging, nicht ausführen)_:")
        for _, r in short.iterrows():
            parts.append(f"  • `{r['symbol']:6}` Score {r['score']:+.2f}")
    parts.append("\n_Quality > Quantity. Risiko 2% pro Position. Stop-Loss aktiv._")
    return "\n".join(parts)


def _format_performance_de() -> str:
    s = journal_mod.journal_summary()
    return (
        "*Performance-Übersicht*\n\n"
        f"Predictions gesamt: *{s['n_predictions']}*\n"
        f"Davon evaluiert:    *{s['n_evaluated']}*\n"
        f"Win-Rate:           *{s['win_rate']*100:.1f}%*\n"
        f"Ø Realised Return:  *{s['mean_realised_return']*100:+.2f}%*"
    )


def _format_journal_de(days: int = 7) -> str:
    rec = journal_mod.recent_predictions(days=days)
    if rec.empty:
        return f"Keine Trades in den letzten {days} Tagen."
    out = [f"*Letzte {days} Tage — {len(rec)} Predictions*\n"]
    for _, r in rec.head(15).iterrows():
        out.append(f"`{r['asof_date']}` {r['action']:5} {r['symbol']:6} Score {r['score']:+.2f}")
    if len(rec) > 15:
        out.append(f"\n_… und {len(rec)-15} weitere._")
    return "\n".join(out)


# ----------------------------- Bot Setup -------------------------------

async def _send(text: str, chat_id: str | None = None):
    from telegram import Bot   # type: ignore
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        raise RuntimeError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required in .env")
    bot = Bot(token=token)
    await bot.send_message(chat_id=chat_id, text=text, parse_mode="Markdown")


def push_today():
    """One-shot push of today's recommendations. Used by daily orchestrator."""
    df = _today_recommendations()
    text = _format_recommendations_de(df)
    asyncio.run(_send(text))


def run_listener():
    """Long-running command listener."""
    from telegram import Update  # type: ignore
    from telegram.ext import (   # type: ignore
        Application, CommandHandler, ContextTypes,
    )

    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN required in .env")

    HELP = (
        "*Trading Bot — Befehle*\n\n"
        "/heute — heutige Empfehlungen\n"
        "/journal — letzte 7 Tage Trades\n"
        "/performance — Win-Rate & Returns\n"
        "/status — Bot-Status\n"
        "/hilfe — diese Übersicht"
    )

    async def heute(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        df = _today_recommendations()
        await update.message.reply_text(_format_recommendations_de(df), parse_mode="Markdown")

    async def journal(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(_format_journal_de(7), parse_mode="Markdown")

    async def performance(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(_format_performance_de(), parse_mode="Markdown")

    async def status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        n_models = len(list(config.MODELS_DIR.glob("*.json")))
        n_prices = len(list(config.PRICES_DIR.glob("*.parquet")))
        await update.message.reply_text(
            f"*Status*\nModelle: {n_models}\nKurs-Files: {n_prices}\nDB: `{journal_mod.DEFAULT_DB.name}`",
            parse_mode="Markdown",
        )

    async def hilfe(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text(HELP, parse_mode="Markdown")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("heute", heute))
    app.add_handler(CommandHandler("journal", journal))
    app.add_handler(CommandHandler("performance", performance))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler(["hilfe", "help", "start"], hilfe))
    print("Telegram bot listener läuft. Strg+C zum Beenden.")
    app.run_polling()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "push":
        push_today()
    else:
        run_listener()
