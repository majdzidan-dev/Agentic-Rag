"""
The 5 retrieval configurations being compared. Every function has the same
signature - (query, k) -> ranked list of PMIDs - so run_eval.py can treat
them interchangeably.

IMPORTANT: none of these fetch new papers from PubMed. They only search
whatever is ALREADY indexed (set up ahead of time by run_eval.py via
index_specific_papers()). This is what makes the comparison fair - every
variant sees the exact same corpus, so differences in the scores reflect
differences in SEARCH QUALITY, not differences in what's available to find.
"""

from rag.vector_store import query_existing
from rag.keyword_store import keyword_search
from rag.hybrid_search import hybrid_search
from rag.reranker import rerank
from rag.retrieval_grader import grade_sufficiency, rewrite_query
from tools.pubmed_rag import RERANK_CANDIDATE_K, MAX_RETRIEVAL_ATTEMPTS


def _pmids_from_documents(pairs: list[tuple]) -> list[str]:
    """pairs: list of (Document, score). Returns just the PMIDs, in order,
    keeping duplicates (same paper can supply multiple chunks)."""
    return [doc.metadata["pmid"] for doc, _ in pairs]


def _pmids_from_dicts(pairs: list[tuple]) -> list[str]:
    """Same idea, for BM25 results which return plain dicts, not Documents."""
    return [chunk["pmid"] for chunk, _ in pairs]


def dense_only(query: str, k: int) -> list[str]:
    results = query_existing(query, k=k)
    return _pmids_from_documents(results)


def bm25_only(query: str, k: int) -> list[str]:
    results = keyword_search(query, k=k)
    return _pmids_from_dicts(results)


def hybrid(query: str, k: int) -> list[str]:
    results = hybrid_search(query, k=k)
    return _pmids_from_documents(results)


def hybrid_plus_rerank(query: str, k: int) -> list[str]:
    candidates = hybrid_search(query, k=RERANK_CANDIDATE_K)
    reranked = rerank(query, candidates, top_n=k) if candidates else []
    return _pmids_from_documents(reranked)


def hybrid_rerank_rewrite(query: str, k: int) -> list[str]:
    """The full agentic loop - but searching the FIXED eval corpus only,
    never fetching new papers (unlike the production pubmed_search tool)."""
    current_query = query
    top_results = []

    for attempt in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
        candidates = hybrid_search(current_query, k=RERANK_CANDIDATE_K)
        top_results = rerank(current_query, candidates, top_n=k) if candidates else []

        excerpts_text = (
            "\n\n".join(f"{doc.metadata['title']}: {doc.page_content}" for doc, _ in top_results)
            if top_results else "(no passages retrieved)"
        )

        grade = grade_sufficiency(query, excerpts_text)  # grade against ORIGINAL query
        if grade["sufficient"] or attempt == MAX_RETRIEVAL_ATTEMPTS:
            break

        current_query = rewrite_query(current_query, grade["reason"])

    return _pmids_from_documents(top_results)


# Registry used by run_eval.py to loop over all variants generically
VARIANTS = {
    "1. Dense retrieval only": dense_only,
    "2. BM25 only": bm25_only,
    "3. Hybrid retrieval": hybrid,
    "4. Hybrid + reranker": hybrid_plus_rerank,
    "5. Hybrid + reranker + query rewriting": hybrid_rerank_rewrite,
}