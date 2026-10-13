"""
Keyword (sparse) search using BM25 - the classic term-matching algorithm
search engines used for decades before embeddings existed.

This is the "sparse" half of hybrid search. Dense (embedding) search is
great at matching MEANING even when the wording differs, but it can blur
together things that share a topic while missing exact terms that matter a
lot in science/medicine - a specific drug name, gene, or acronym (e.g.
"GLP-1" or "BRCA1"). BM25 catches those exact-term matches that embeddings
sometimes smooth over.

This mirrors the same chunks stored in the vector database (matched by
chunk_id), just indexed a different way.
"""

import json
import os
import re

from rank_bm25 import BM25Okapi

CORPUS_PATH = "./bm25_corpus.json"


def use_corpus(path: str, reset: bool = False) -> None:
    """Point every function in this module at a different corpus file.
    Used by the evaluation harness to search an isolated eval corpus
    instead of the agent's live one. reset=True empties it first."""
    global CORPUS_PATH
    CORPUS_PATH = path
    if reset and os.path.exists(CORPUS_PATH):
        os.remove(CORPUS_PATH)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace/punctuation tokenizer - lowercase words only.
    BM25 just needs consistent tokens, not linguistic sophistication."""
    return re.findall(r"\w+", text.lower())


def _load_corpus() -> dict:
    """chunk_id -> chunk dict. Returns {} if nothing stored yet."""
    if not os.path.exists(CORPUS_PATH):
        return {}
    with open(CORPUS_PATH, "r") as f:
        return json.load(f)


def _save_corpus(corpus: dict) -> None:
    with open(CORPUS_PATH, "w") as f:
        json.dump(corpus, f)


def add_chunks(chunks: list[dict]) -> None:
    """Add new chunks to the keyword-searchable corpus.
    Deduped by chunk_id - same ID scheme the vector store uses."""
    if not chunks:
        return

    corpus = _load_corpus()
    for chunk in chunks:
        corpus[chunk["chunk_id"]] = chunk
    _save_corpus(corpus)


def stored_pmids() -> set[str]:
    """PMIDs of every paper that already has chunks in the corpus."""
    return {chunk["pmid"] for chunk in _load_corpus().values()}


def keyword_search(query: str, k: int) -> list[tuple[dict, float]]:
    """Returns up to k (chunk_dict, bm25_score) pairs, ranked by keyword
    overlap with the query. Score is unbounded (not 0-1) - only relative
    order matters here, which is why hybrid search fuses by RANK, not score."""
    corpus = _load_corpus()
    if not corpus:
        return []

    chunk_ids = list(corpus.keys())
    tokenized_corpus = [_tokenize(corpus[cid]["text"]) for cid in chunk_ids]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)

    return [(corpus[cid], score) for cid, score in ranked[:k] if score > 0]