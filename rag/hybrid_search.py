"""
Hybrid search: merges dense (embedding) and sparse (BM25 keyword) search
results using Reciprocal Rank Fusion (RRF), producing one combined ranking.

WHY NOT JUST AVERAGE THE SCORES?
Cosine similarity (dense) lives on a 0-1 scale. BM25 scores (sparse) are
unbounded and often range 0-20+. Averaging or adding these directly would
be meaningless - a BM25 score of 8 isn't "8x more relevant" than a cosine
score of 1.0, they're just not comparable numbers.

RRF sidesteps this entirely by ignoring raw scores and using each result's
RANK (1st, 2nd, 3rd...) in its own list instead:

    RRF_score(chunk) = sum over each list it appears in of  1 / (k + rank)

A chunk that appears reasonably high in BOTH lists ends up ranked above a
chunk that's #1 in only one list and missing from the other - which is
usually exactly what you want from a hybrid system: reward agreement
between both search methods, don't just trust one blindly.
"""

from langchain_core.documents import Document

from rag.vector_store import query_existing
from rag.keyword_store import keyword_search

RRF_K = 60  # standard smoothing constant from the original RRF paper


def hybrid_search(query: str, k: int) -> list[tuple[Document, float]]:
    dense_results = query_existing(query, k=k)     # [(Document, cosine_score), ...]
    sparse_results = keyword_search(query, k=k)     # [(chunk_dict, bm25_score), ...]

    rrf_scores: dict[str, float] = {}
    chunk_lookup: dict[str, Document] = {}

    for rank, (doc, _) in enumerate(dense_results):
        chunk_id = doc.metadata.get("chunk_id")
        if not chunk_id:
            continue
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (RRF_K + rank + 1)
        chunk_lookup[chunk_id] = doc

    for rank, (chunk, _) in enumerate(sparse_results):
        chunk_id = chunk["chunk_id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (RRF_K + rank + 1)
        if chunk_id not in chunk_lookup:
            # found by keyword search but wasn't in the dense shortlist -
            # rebuild it as a Document so downstream code stays consistent
            chunk_lookup[chunk_id] = Document(
                page_content=chunk["text"],
                metadata={
                    "chunk_id": chunk_id,
                    "pmid": chunk["pmid"],
                    "title": chunk["title"],
                    "url": chunk["url"],
                },
            )

    ranked_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
    return [(chunk_lookup[cid], rrf_scores[cid]) for cid in ranked_ids[:k]]