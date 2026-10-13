"""
Reranker - a second, more careful relevance-scoring pass on top of the
initial vector search.

WHY THIS EXISTS (the concept to remember):

- Your vector database uses a "bi-encoder": it embeds the query and embeds
  each document SEPARATELY, then compares the two vectors with cosine
  similarity. Fast (embeddings can be pre-computed and stored), but a bit
  approximate - the model never actually looks at the query and the
  document side-by-side.

- A "cross-encoder" reranker instead feeds the query AND one candidate
  document into the model TOGETHER, in a single pass, and outputs one
  relevance score for that specific pair. This is much more accurate
  because the model can directly compare them - but it's too slow to run
  over an entire database, since every document needs its own fresh pass.

THE PATTERN: use the fast bi-encoder to narrow thousands of documents down
to a small shortlist (this project already does that via vector_store.py),
then use the slower, more accurate cross-encoder to re-rank just that
shortlist. This two-stage "retrieve then rerank" pattern is standard in
production RAG systems (e.g. Cohere Rerank, Elasticsearch ELSER work the
same way).
"""

import math

from sentence_transformers import CrossEncoder

# A small, fast cross-encoder trained on MS MARCO, a widely used
# passage-ranking benchmark - a common, well-regarded choice for this task.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Loaded on first use, not at import - the first load downloads the model,
# and a failed download (e.g. offline) shouldn't stop the agent from starting.
_reranker = None


def _get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def _to_probability(raw_score: float) -> float:
    """Cross-encoders output raw, unbounded scores (logits), not 0-1
    probabilities. Squash it into a 0-1 range with a sigmoid, purely so the
    final output is easier to read - doesn't change the ranking order."""
    return 1 / (1 + math.exp(-raw_score))


def rerank(query: str, candidates: list[tuple], top_n: int) -> list[tuple]:
    """
    candidates: list of (Document, embedding_score) pairs from vector search
                (the bi-encoder's shortlist).
    Returns the top_n (Document, rerank_score) pairs, re-sorted by the
    cross-encoder's relevance score - which can, and often does, reorder
    the list compared to the original embedding-based ranking.
    """
    if not candidates:
        return []

    documents = [doc for doc, _ in candidates]
    pairs = [(query, doc.page_content) for doc in documents]

    raw_scores = _get_reranker().predict(pairs)  # one score per (query, document) pair
    scored = [(doc, _to_probability(score)) for doc, score in zip(documents, raw_scores)]

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]