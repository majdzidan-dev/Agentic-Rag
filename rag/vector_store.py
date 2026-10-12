"""
Persistent vector database for PubMed paper chunks.

Without this, every question re-fetches and re-embeds papers from scratch -
wasteful if you ask something similar twice. With this, embedded chunks are
saved to disk (./chroma_db) and reused across runs, so the agent only goes
back to PubMed when what's already stored isn't good enough.
"""

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from config import GOOGLE_API_KEY

PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "pubmed_papers"

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

_vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=PERSIST_DIR,
)


def query_existing(query: str, k: int = 6) -> list[tuple[Document, float]]:
    """Search what's already stored locally.

    Returns (document, relevance_score) pairs, where relevance_score is
    0-1 and higher means more relevant to the query.
    """
    return _vector_store.similarity_search_with_relevance_scores(query, k=k)


def add_chunks(chunks: list[dict]) -> None:
    """Embed and permanently store new chunks.

    Uses each chunk's "chunk_id" (assigned in pubmed_rag.py) as its database
    ID, and also stores it in metadata - this is what lets the keyword
    (BM25) index and this vector index agree on "this is the same chunk"
    when their results get merged in hybrid search.
    """
    if not chunks:
        return

    documents = [
        Document(
            page_content=chunk["text"],
            metadata={
                "chunk_id": chunk["chunk_id"],
                "pmid": chunk["pmid"],
                "title": chunk["title"],
                "url": chunk["url"],
            },
        )
        for chunk in chunks
    ]
    ids = [chunk["chunk_id"] for chunk in chunks]

    _vector_store.add_documents(documents, ids=ids)