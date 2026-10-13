"""
PubMed + RAG tool - hybrid search (dense + BM25) + reranking.

Pipeline when this tool is called:
1. CHECK DATABASE - see how many already-stored chunks are relevant to the
   question (decision metric, dense-only for simplicity/speed).
2. DECIDE - based on that, decide how many NEW papers (if any) to pull from
   PubMed.
3. FETCH + CHUNK (only if step 2 decided we need to) - query PubMed's
   esearch/efetch APIs, split abstracts into passages, each tagged with a
   unique chunk_id.
4. STORE - save any new chunks into BOTH the vector database (dense) and
   the BM25 index (sparse), keyed by the same chunk_id.
5. HYBRID RETRIEVE - run dense search and BM25 search in parallel, merge
   their rankings with Reciprocal Rank Fusion into one shortlist.
6. RERANK - cross-encoder reranks that shortlist for final precise ordering.
7. RETURN - hand the top excerpts back to the agent, with citations.
"""

import xml.etree.ElementTree as ET

import requests
from langchain.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import NCBI_API_KEY, NCBI_EMAIL
from rag.vector_store import query_existing, add_chunks as add_to_vector_store
from rag.keyword_store import add_chunks as add_to_keyword_store, stored_pmids
from rag.hybrid_search import hybrid_search
from rag.reranker import rerank
from rag.retrieval_grader import grade_sufficiency, rewrite_query

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# ============================================================
# THE DECISION METRIC - tune these numbers to change behavior
# ============================================================
RELEVANCE_THRESHOLD = 0.75   # how similar (0-1) a stored chunk must be to count as "relevant"
DESIRED_CHUNKS = 4           # how many relevant chunks we ideally want to answer with
FRESH_FETCH_PAPERS = 5       # papers to pull when the database has NOTHING relevant yet
TOPUP_FETCH_PAPERS = 2       # papers to pull when the database has SOME, but not enough
RERANK_CANDIDATE_K = 12      # wider shortlist pulled via hybrid search, before reranking
MAX_RETRIEVAL_ATTEMPTS = 3   # agentic loop cap - stops infinite retry on a bad query
NEW_PAPER_SEARCH_WINDOW = 20 # how many top PubMed hits to scan for papers not yet stored


def _fetch_pubmed_ids(query: str, max_results: int) -> list[str]:
    """SEARCH - ask PubMed for the IDs of matching papers."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json().get("esearchresult", {}).get("idlist", [])


def _fetch_new_pubmed_ids(query: str, count: int) -> list[str]:
    """Like _fetch_pubmed_ids, but skips papers already in the database.
    PubMed's top hits for a query are usually the ones stored last time,
    so asking for just the top `count` would re-fetch them and add nothing."""
    already_stored = stored_pmids()
    candidates = _fetch_pubmed_ids(query, max_results=NEW_PAPER_SEARCH_WINDOW)
    return [pmid for pmid in candidates if pmid not in already_stored][:count]


def _element_text(element) -> str:
    """Full text of an XML element, including text inside nested tags like
    <i>, <sub> or <sup> (common in gene names and formulas). element.text
    alone stops at the first nested tag and silently truncates."""
    return "".join(element.itertext()).strip()


def _fetch_pubmed_details(pmids: list[str]) -> list[dict]:
    """FETCH - get title + abstract for each PMID."""
    if not pmids:
        return []

    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "email": NCBI_EMAIL,
    }
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY

    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=15)
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    papers = []
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        title_el = article.find(".//ArticleTitle")
        abstract_parts = article.findall(".//AbstractText")

        pmid = pmid_el.text if pmid_el is not None else "unknown"
        title = (_element_text(title_el) if title_el is not None else "") or "No title"

        # Structured abstracts split into labelled sections (BACKGROUND,
        # METHODS, RESULTS...) - keep the label so the section is clear.
        sections = []
        for part in abstract_parts:
            text = _element_text(part)
            if not text:
                continue
            label = part.get("Label")
            sections.append(f"{label}: {text}" if label else text)
        abstract = " ".join(sections) or "No abstract available."

        papers.append({
            "pmid": pmid,
            "title": title,
            "abstract": abstract,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return papers


def _chunk_papers(papers: list[dict]) -> list[dict]:
    """CHUNK - split each abstract into smaller passages, each with a
    unique chunk_id (pmid + its index within that paper). This ID is what
    lets the vector store and BM25 index refer to "the same chunk"."""
    chunks = []
    for paper in papers:
        pieces = _splitter.split_text(paper["abstract"])
        for i, piece in enumerate(pieces):
            chunks.append({
                "chunk_id": f"{paper['pmid']}-{i}",
                "text": piece,
                "pmid": paper["pmid"],
                "title": paper["title"],
                "url": paper["url"],
            })
    return chunks


def _decide_fetch_count(existing_results: list) -> int:
    """THE METRIC - decide how many new papers to fetch, based on how many
    already-stored chunks are relevant enough to trust. Uses dense search
    only here - fast, and good enough for this yes/no decision; the more
    expensive hybrid + rerank pipeline is reserved for building the actual
    answer below."""
    relevant_count = sum(
        1 for _, score in existing_results if score >= RELEVANCE_THRESHOLD
    )

    if relevant_count >= DESIRED_CHUNKS:
        return 0  # database already covers this well - skip PubMed entirely
    elif relevant_count > 0:
        return TOPUP_FETCH_PAPERS  # partial coverage - just fill the gap
    else:
        return FRESH_FETCH_PAPERS  # nothing useful stored yet - full fetch


def index_specific_papers(pmids: list[str]) -> None:
    """Fetch and index a known, explicit list of PMIDs (no PubMed search
    involved) into both the vector store and BM25 index. Used by the
    evaluation harness to build a fixed, reproducible corpus from
    ground-truth papers, independent of the live fetch-decision logic
    pubmed_search() uses during normal operation."""
    papers = _fetch_pubmed_details(pmids)
    chunks = _chunk_papers(papers)
    add_to_vector_store(chunks)
    add_to_keyword_store(chunks)


@tool
def pubmed_search(query: str) -> str:
    """Search PubMed for peer-reviewed scientific and medical research papers,
    then retrieve the most relevant passages using hybrid search (dense +
    keyword), reranking, and an agentic retry loop that rewrites the search
    query if the first attempt doesn't find good results. Use this for ANY
    question involving biology, medicine, health, clinical research, drugs,
    diseases, or other scientific topics - not general web search."""
    # Any uncaught exception inside a tool would crash the whole agent, so
    # failures are returned to the agent as a message instead. Only the
    # error TYPE is reported - full messages from requests include the
    # request URL, which can contain the NCBI API key.
    try:
        return _run_pubmed_search(query)
    except Exception as e:
        print(f"[pubmed_search] failed: {type(e).__name__}")
        return (
            f"pubmed_search failed ({type(e).__name__}), so no PubMed results "
            "are available for this question right now."
        )


def _run_pubmed_search(query: str) -> str:
    current_query = query
    top_results = []

    for attempt in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
        # STAGE 1: narrow dense check, decide whether PubMed needs to be hit at all
        existing_results = query_existing(current_query, k=DESIRED_CHUNKS + 2)
        fetch_count = _decide_fetch_count(existing_results)

        if fetch_count > 0:
            try:
                pmids = _fetch_new_pubmed_ids(current_query, count=fetch_count)
                papers = _fetch_pubmed_details(pmids)
            except (requests.RequestException, ET.ParseError) as e:
                # PubMed unreachable - still answer from what's stored locally
                print(f"[pubmed_search] PubMed fetch failed ({type(e).__name__}), using local index only")
                papers = []
            new_chunks = _chunk_papers(papers)
            add_to_vector_store(new_chunks)   # dense index
            add_to_keyword_store(new_chunks)  # sparse (BM25) index

        # STAGE 2: hybrid search - dense + BM25, merged via Reciprocal Rank Fusion
        candidates = hybrid_search(current_query, k=RERANK_CANDIDATE_K)

        # STAGE 3: cross-encoder reranking - precise final ordering
        top_results = rerank(current_query, candidates, top_n=DESIRED_CHUNKS) if candidates else []

        excerpts_text = (
            "\n\n".join(f"{doc.metadata['title']}: {doc.page_content}" for doc, _ in top_results)
            if top_results else "(no passages retrieved)"
        )

        # STAGE 4 (AGENTIC CHECK): grade against the ORIGINAL question, not
        # the possibly-rewritten current_query - what matters is whether the
        # user's actual question got answered.
        grade = grade_sufficiency(query, excerpts_text)
        print(
            f"[pubmed_search] attempt {attempt} (query: '{current_query}') -> "
            f"{'sufficient' if grade['sufficient'] else 'insufficient'}: {grade['reason']}"
        )

        if grade["sufficient"] or attempt == MAX_RETRIEVAL_ATTEMPTS:
            break

        # STAGE 5: rewrite the query and loop again
        current_query = rewrite_query(current_query, grade["reason"])

    if not top_results:
        return (
            "No relevant PubMed papers could be found for this query, even "
            "after retrying with alternate search terms."
        )

    formatted = [
        f"Source: {doc.metadata['title']} (PMID: {doc.metadata['pmid']}, "
        f"{doc.metadata['url']}) [relevance: {score:.2f}]\nExcerpt: {doc.page_content}"
        for doc, score in top_results
    ]
    return "\n\n".join(formatted)