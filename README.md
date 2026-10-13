# AgenticRag

A command-line research assistant built with LangChain and Google Gemini. It answers questions using three tools: a calculator, web search (SerpAPI), and a PubMed retrieval-augmented generation (RAG) pipeline that grounds scientific answers in peer-reviewed abstracts and cites them by PMID.

## How it works

```
main.py                      chat loop (in-memory conversation history)
└─ agent.py                  Gemini agent + system prompt + tools
   ├─ tools/calculator.py    safe math via numexpr
   ├─ tools/web_search.py    SerpAPI web search
   └─ tools/pubmed_rag.py    PubMed RAG pipeline
      ├─ rag/vector_store.py     Chroma vector DB (./chroma_db), Gemini embeddings
      ├─ rag/keyword_store.py    BM25 keyword index (./bm25_corpus.json)
      ├─ rag/hybrid_search.py    merges both rankings with Reciprocal Rank Fusion
      ├─ rag/reranker.py         cross-encoder reranking
      └─ rag/retrieval_grader.py LLM judges the results and rewrites the query if needed
```

When `pubmed_search` is called, it:

1. Checks how many relevant chunks are already stored locally, and fetches only as many new papers from PubMed as needed, skipping papers it already has.
2. Splits the abstracts into chunks and stores them in both the vector DB and the BM25 index.
3. Runs hybrid search (dense + BM25), then reranks the shortlist with a cross-encoder.
4. Asks an LLM whether the excerpts answer the question. If they don't, it rewrites the query and retries (at most 3 attempts).
5. Returns the top excerpts with title, PMID and link.

The knowledge base grows over time: each question can add new papers to the local stores.

## Setup

Requires Python 3.13 (the version the pinned dependencies are tested with).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root. It is git-ignored, so never commit it:

| Variable          | Required | Purpose                                                         |
|-------------------|----------|-----------------------------------------------------------------|
| `GOOGLE_API_KEY`  | yes      | Gemini chat model and embeddings                                |
| `SERPAPI_API_KEY` | for web search | Web search tool; the agent still starts without it        |
| `NCBI_EMAIL`      | recommended | NCBI asks PubMed API users to identify themselves with an email |
| `NCBI_API_KEY`    | optional | Higher PubMed rate limit                                        |

On first use, the reranker downloads a small model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) from Hugging Face.

## Usage

```bash
python main.py
```

Type `quit` or `exit` to stop. Conversation memory lasts only while the program is running.

## Evaluation

`eval/` compares five retrieval setups (dense only, BM25 only, hybrid, hybrid + reranker, hybrid + reranker + query rewriting) on a hand-labelled dataset, using Recall@5, Recall@10, MRR and NDCG@5.

```bash
python -m eval.label_helper   # add questions and mark relevant PubMed papers by hand
python -m eval.run_eval       # run all setups, print a table, save eval/results.json
```

The eval builds its own isolated corpus: a separate Chroma collection plus `eval/bm25_eval_corpus.json`, both emptied on each run. This means results don't depend on what the agent has stored during normal use, and running the eval doesn't change the agent's data. The current dataset is small, so treat its numbers as a smoke test rather than a benchmark.

## Tests

```bash
python -m unittest discover -s tests -t .
```

The unit tests cover the metrics, rank fusion, PubMed parsing and chunking, the fetch decision, and tool error handling. They run offline and don't need API keys.

## Docker

```bash
docker compose build
docker compose run --rm agent                         # chat
docker compose run --rm agent python -m eval.run_eval # evaluation
```

API keys are passed in from `.env` when the container runs and are never copied into the image. `chroma_db/` and `bm25_corpus.json` are mounted as volumes so stored papers survive container restarts. Create `bm25_corpus.json` before the first run, otherwise Docker creates a directory with that name.
