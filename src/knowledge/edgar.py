"""SEC EDGAR ingestion -> chunked documents for the RAG knowledge base.

Pulls the most recent 10-K (annual), 10-Q (quarterly), and 8-K (event)
filings for each ticker in the universe.

Why these three:
  - **10-K**: full annual report — risk factors, business overview
  - **10-Q**: quarterly numbers + management discussion
  - **8-K**: material events between regular filings — the most
    *recent* signal of company-specific risk (lawsuits, exec changes,
    M&A, etc.)

We use the `sec-edgar-downloader` package which handles SEC's strict
1-second rate limit and User-Agent requirement.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from tqdm.auto import tqdm

from config import DATA_DIR
from src.knowledge.rag import Document, add_documents, chunk_text

EDGAR_DIR = DATA_DIR / "edgar"
EDGAR_DIR.mkdir(parents=True, exist_ok=True)


def _downloader(user_agent_email: str = "research@example.com"):
    from sec_edgar_downloader import Downloader  # type: ignore
    return Downloader("trading-bot-research", user_agent_email, str(EDGAR_DIR))


def download_filings(
    symbol: str,
    forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
    limit_per_form: int = 4,
    user_agent_email: str = "research@example.com",
) -> dict[str, list[Path]]:
    """Download recent filings for one ticker. Returns {form: [paths]}."""
    dl = _downloader(user_agent_email)
    out: dict[str, list[Path]] = {f: [] for f in forms}
    for form in forms:
        try:
            dl.get(form, symbol, limit=limit_per_form, download_details=False)
        except Exception as e:  # noqa: BLE001
            print(f"[{symbol} {form}] {e}")
            continue
        # downloader saves to: EDGAR_DIR/sec-edgar-filings/{symbol}/{form}/.../*.txt
        base = EDGAR_DIR / "sec-edgar-filings" / symbol / form
        if base.exists():
            out[form] = sorted(base.rglob("*.txt"))
    return out


_HTML_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def _clean_filing_text(raw: str) -> str:
    """Strip HTML, collapse whitespace. Filings are noisy; this is enough
    to make them embeddable but not enough for legal analysis."""
    text = _HTML_TAG.sub(" ", raw)
    text = _WS.sub(" ", text)
    return text.strip()


def _filing_to_documents(
    filing_path: Path,
    symbol: str,
    form: str,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[Document]:
    raw = filing_path.read_text(encoding="utf-8", errors="ignore")
    cleaned = _clean_filing_text(raw)
    if len(cleaned) < 500:
        return []
    # use last-modified as proxy for filing date
    filing_date = datetime.fromtimestamp(filing_path.stat().st_mtime, tz=timezone.utc)
    docs = []
    for i, chunk in enumerate(chunk_text(cleaned, chunk_size=chunk_size, overlap=overlap)):
        docs.append(Document(
            id=f"sec/{symbol}/{form}/{filing_path.parent.name}/chunk{i}",
            text=chunk,
            metadata={
                "source": "sec_edgar",
                "ticker": symbol,
                "doc_type": form,
                "date": filing_date.isoformat(),
                "filing_dir": filing_path.parent.name,
            },
        ))
    return docs


def ingest_universe(
    symbols: Iterable[str],
    forms: tuple[str, ...] = ("10-K", "10-Q", "8-K"),
    limit_per_form: int = 2,
    user_agent_email: str = "research@example.com",
) -> int:
    """End-to-end: download + chunk + embed + store. Returns docs added."""
    total = 0
    for sym in tqdm(list(symbols), desc="EDGAR ingest"):
        paths_by_form = download_filings(
            sym, forms=forms, limit_per_form=limit_per_form,
            user_agent_email=user_agent_email,
        )
        all_docs: list[Document] = []
        for form, paths in paths_by_form.items():
            for p in paths:
                all_docs.extend(_filing_to_documents(p, sym, form))
        if all_docs:
            total += add_documents(all_docs)
    return total
