"""
Standard information-retrieval metrics, computed against ground-truth
relevant PMIDs for a question.

All 4 functions take the SAME two inputs:
- ranked_pmids: a list of PMIDs in the order they were retrieved (rank 1
  first). Duplicates can occur since multiple chunks may come from the
  same paper - that's expected and handled.
- relevant_pmids: the set of PMIDs a human (or verified source) has
  confirmed are actually relevant to the question.
"""

import math


def recall_at_k(ranked_pmids: list[str], relevant_pmids: set[str], k: int) -> float:
    """What fraction of ALL relevant papers were found somewhere in the
    top-k retrieved results? (0-1, higher is better)"""
    if not relevant_pmids:
        return 0.0
    top_k_unique = set(ranked_pmids[:k])
    found = top_k_unique & relevant_pmids
    return len(found) / len(relevant_pmids)


def mean_reciprocal_rank(ranked_pmids: list[str], relevant_pmids: set[str]) -> float:
    """1 / (rank of the FIRST relevant result). 1.0 if the very first
    result is relevant, 0.5 if the second result is the first relevant
    one, 0 if nothing relevant appears at all. Rewards getting *something*
    good near the top, regardless of how many relevant items exist total."""
    for rank, pmid in enumerate(ranked_pmids, start=1):
        if pmid in relevant_pmids:
            return 1 / rank
    return 0.0


def ndcg_at_k(ranked_pmids: list[str], relevant_pmids: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain. Like recall, but POSITION-
    SENSITIVE: a relevant result at rank 1 counts for more than the same
    result at rank 5. Uses binary relevance (1 if relevant, 0 if not).
    Returns 0-1, where 1.0 means the top-k results are perfectly ordered
    (all relevant items first)."""
    if not relevant_pmids:
        return 0.0

    def dcg(pmids: list[str]) -> float:
        return sum(
            (1 if pmid in relevant_pmids else 0) / math.log2(rank + 1)
            for rank, pmid in enumerate(pmids[:k], start=1)
        )

    actual_dcg = dcg(ranked_pmids)

    # Ideal ordering: every relevant item packed at the top, up to k of them
    ideal_ranking = ["relevant"] * min(len(relevant_pmids), k)
    ideal_dcg = sum(1 / math.log2(rank + 1) for rank, _ in enumerate(ideal_ranking, start=1))

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def evaluate_ranking(ranked_pmids: list[str], relevant_pmids: set[str]) -> dict:
    """Convenience wrapper - computes all 4 metrics at once for one question."""
    return {
        "recall@5": recall_at_k(ranked_pmids, relevant_pmids, k=5),
        "recall@10": recall_at_k(ranked_pmids, relevant_pmids, k=10),
        "mrr": mean_reciprocal_rank(ranked_pmids, relevant_pmids),
        "ndcg@5": ndcg_at_k(ranked_pmids, relevant_pmids, k=5),
    }