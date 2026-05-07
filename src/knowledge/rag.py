"""RAG (Retrieval-Augmented Generation) knowledge base.

A persistent Chroma collection holds:
  - SEC filings (10-K / 10-Q / 8-K) per S&P 500 ticker
  - Fed FOMC statements + press conference transcripts
  - Investopedia term definitions (concepts the post-mortem LLM may reference)
  - arXiv q-fin abstracts (recent quant research)

For any post-mortem or recommendation, we retrieve top-k relevant chunks
and pass them as context to the LLM. This grounds the LLM in real
financial knowledge instead of letting it hallucinate.

We use **Chroma** because:
  - Pure Python, no separate server.
  - Persists to disk, easy to back up.
  - Built-in embedding via sentence-transformers.

Embedding model default: `BAAI/bge-small-en-v1.5` — small (133MB), fast,
good retrieval quality. Upgrade to bge-large-en for better quality at
3x the storage.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from config import DATA_DIR

CHROMA_DIR = DATA_DIR / "knowledge"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_EMBEDDING = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "trading_knowledge"


@dataclass
class Document:
    id: str                  # globally unique
    text: str
    metadata: dict           # {source, ticker?, doc_type, date, url}


def _client():
    import chromadb  # type: ignore
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _collection(embedding_model: str = DEFAULT_EMBEDDING):
    from chromadb.utils import embedding_functions  # type: ignore
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=embedding_model
    )
    client = _client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(docs: Iterable[Document], batch_size: int = 100) -> int:
    """Add documents to the knowledge base. Returns count added."""
    coll = _collection()
    docs = list(docs)
    n = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i : i + batch_size]
        coll.upsert(
            ids=[d.id for d in batch],
            documents=[d.text for d in batch],
            metadatas=[d.metadata for d in batch],
        )
        n += len(batch)
    return n


def query(
    text: str,
    top_k: int = 5,
    where: dict | None = None,
) -> list[dict]:
    """Semantic search. `where` filters on metadata, e.g. {"doc_type": "10-K"}."""
    coll = _collection()
    res = coll.query(
        query_texts=[text],
        n_results=top_k,
        where=where,
    )
    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
        })
    return out


def stats() -> dict:
    coll = _collection()
    return {
        "n_documents": coll.count(),
        "path": str(CHROMA_DIR),
        "embedding_model": DEFAULT_EMBEDDING,
    }


# ---------- chunking helper for long documents (10-K, transcripts) ----------

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Naive char-based chunking. Good enough for SEC filings & transcripts."""
    text = text.replace("\r\n", "\n")
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i : i + chunk_size])
        i += chunk_size - overlap
    return chunks
