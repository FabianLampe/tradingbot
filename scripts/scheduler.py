"""APScheduler daemon that fires the daily and pre-market pipelines on a cron.

Runs forever. Used inside the `scheduler` docker-compose service.

Schedule (Europe/Berlin):
    14:30 Mo–Fr  -> pre-market run
    22:30 Mo–Fr  -> end-of-day run

Override via env vars TB_UNIVERSE (etfs|sp500|all).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from apscheduler.schedulers.blocking import BlockingScheduler   # type: ignore
from apscheduler.triggers.cron import CronTrigger              # type: ignore

from src.runtime import daily

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("trading_bot.scheduler")

UNIVERSE = os.getenv("TB_UNIVERSE", "etfs")


def job_eod():
    log.info("Cron: EOD run (warm-start)")
    daily.run_daily_eod(universe_size=UNIVERSE, force_full_retrain=False)
    try:
        from app.telegram_bot import push_today
        push_today()
    except Exception as e:  # noqa: BLE001
        log.warning("telegram push failed: %s", e)


def job_premarket():
    log.info("Cron: Pre-market run")
    daily.run_premarket(universe_size=UNIVERSE)
    try:
        from app.telegram_bot import push_today
        push_today()
    except Exception as e:  # noqa: BLE001
        log.warning("telegram push failed: %s", e)


def job_weekly_full_retrain():
    """Sunday 03:00 — full retrain on all data, fresh model, fresh hyperparam fit."""
    log.info("Cron: WEEKLY full retrain")
    daily.run_daily_eod(universe_size=UNIVERSE, force_full_retrain=True)


def job_postmortems():
    """Sunday 04:00 — let the LLM analyse the past week's losing trades."""
    log.info("Cron: Weekly post-mortems")
    import subprocess, sys
    subprocess.run([sys.executable, "scripts/run_postmortems.py", "--limit", "20"])


def main():
    scheduler = BlockingScheduler(timezone="Europe/Berlin")
    scheduler.add_job(
        job_premarket,
        CronTrigger(day_of_week="mon-fri", hour=14, minute=30),
        id="premarket",
    )
    scheduler.add_job(
        job_eod,
        CronTrigger(day_of_week="mon-fri", hour=22, minute=30),
        id="eod",
    )
    scheduler.add_job(
        job_weekly_full_retrain,
        CronTrigger(day_of_week="sun", hour=3, minute=0),
        id="weekly_full_retrain",
    )
    scheduler.add_job(
        job_postmortems,
        CronTrigger(day_of_week="sun", hour=4, minute=0),
        id="postmortems",
    )
    log.info("Scheduler started (universe=%s). Jobs:", UNIVERSE)
    for j in scheduler.get_jobs():
        log.info("  - %s: next run %s", j.id, j.next_run_time)
    scheduler.start()


if __name__ == "__main__":
    main()
