"""
Evaluation harness. Run with:  python -m eval.run_eval

1. Loads eval/dataset.json (ground-truth questions + relevant PMIDs).
2. SETUP - indexes every relevant PMID across the whole dataset into a
   SEPARATE, freshly emptied eval vector collection + BM25 file, so all 5
   variants search the exact same fixed corpus (see retrieval_variants.py
   for why this matters), unaffected by whatever the agent has stored
   during normal use.
3. For each variant, for each question, runs retrieval and scores it
   against the ground truth using the metrics in eval/metrics.py.
4. Prints a results table, and saves it to eval/results.json.
"""

import json
import time

from eval.metrics import evaluate_ranking
from eval.retrieval_variants import VARIANTS
from rag import keyword_store, vector_store
from tools.pubmed_rag import index_specific_papers

DATASET_PATH = "eval/dataset.json"
RESULTS_PATH = "eval/results.json"
TOP_K = 10  # widest ranked list any metric here needs (Recall@10)

# Isolated stores for the eval corpus - never the agent's live ones
EVAL_COLLECTION_NAME = "pubmed_papers_eval"
EVAL_BM25_CORPUS_PATH = "./eval/bm25_eval_corpus.json"


def load_dataset() -> list[dict]:
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


def setup_corpus(dataset: list[dict]) -> None:
    """Index every unique ground-truth PMID across the whole dataset once,
    up front - so every variant is tested against identical, complete
    coverage rather than whatever a live search happens to surface.
    Both eval stores are emptied first so the corpus is exactly the
    ground-truth papers on every run."""
    vector_store.use_collection(EVAL_COLLECTION_NAME, reset=True)
    keyword_store.use_corpus(EVAL_BM25_CORPUS_PATH, reset=True)

    all_pmids = sorted({pmid for item in dataset for pmid in item["relevant_pmids"]})
    print(f"Indexing {len(all_pmids)} known ground-truth papers into the corpus...")
    index_specific_papers(all_pmids)
    print("Corpus ready.\n")


def run() -> None:
    dataset = load_dataset()
    setup_corpus(dataset)

    # variant_name -> list of per-question metric dicts
    all_results: dict[str, list[dict]] = {name: [] for name in VARIANTS}

    for item in dataset:
        question = item["question"]
        relevant = set(item["relevant_pmids"])
        print(f"Question: {question}")

        for variant_name, variant_fn in VARIANTS.items():
            start = time.time()
            ranked_pmids = variant_fn(question, k=TOP_K)
            elapsed = time.time() - start

            scores = evaluate_ranking(ranked_pmids, relevant)
            scores["latency_sec"] = round(elapsed, 2)
            all_results[variant_name].append(scores)

            print(f"  {variant_name:42s} recall@5={scores['recall@5']:.2f}  "
                  f"recall@10={scores['recall@10']:.2f}  mrr={scores['mrr']:.2f}  "
                  f"ndcg@5={scores['ndcg@5']:.2f}  ({elapsed:.1f}s)")
        print()

    print_summary_table(all_results, num_questions=len(dataset))
    save_results(all_results)


def print_summary_table(all_results: dict[str, list[dict]], num_questions: int) -> None:
    print("=" * 100)
    print(f"SUMMARY (averaged over {num_questions} questions)")
    print("=" * 100)
    header = f"{'Variant':42s} {'Recall@5':>10s} {'Recall@10':>10s} {'MRR':>8s} {'NDCG@5':>8s} {'Avg sec':>9s}"
    print(header)
    print("-" * len(header))

    for variant_name, results in all_results.items():
        n = len(results)
        avg_recall5 = sum(r["recall@5"] for r in results) / n
        avg_recall10 = sum(r["recall@10"] for r in results) / n
        avg_mrr = sum(r["mrr"] for r in results) / n
        avg_ndcg5 = sum(r["ndcg@5"] for r in results) / n
        avg_latency = sum(r["latency_sec"] for r in results) / n

        print(f"{variant_name:42s} {avg_recall5:10.3f} {avg_recall10:10.3f} "
              f"{avg_mrr:8.3f} {avg_ndcg5:8.3f} {avg_latency:9.2f}")


def save_results(all_results: dict[str, list[dict]]) -> None:
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull per-question results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run()