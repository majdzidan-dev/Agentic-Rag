"""
Interactive tool for building your ground-truth dataset up to 50-100
questions. Run with:  python -m eval.label_helper

For each question you type, it searches real PubMed and shows you each
candidate paper's title + abstract snippet. You just mark yes/no - the
tool builds the ground-truth JSON entry for you.

Why not automate this with an LLM instead of asking you? Because that
would be circular: using the same kind of model you're trying to evaluate
to decide what "correct" looks like defeats the purpose of an independent
benchmark. A humans-label-it eval is slower, but it's the only kind that
actually means something.
"""

import json
import os

from tools.pubmed_rag import _fetch_pubmed_ids, _fetch_pubmed_details  # noqa: reuse internal helpers

DATASET_PATH = "eval/dataset.json"


def load_dataset() -> list[dict]:
    if not os.path.exists(DATASET_PATH):
        return []
    with open(DATASET_PATH, "r") as f:
        return json.load(f)


def save_dataset(dataset: list[dict]) -> None:
    with open(DATASET_PATH, "w") as f:
        json.dump(dataset, f, indent=2)


def label_one_question(question_id: str) -> dict:
    question = input("\nEnter the scientific question: ").strip()

    pmids = _fetch_pubmed_ids(question, max_results=8)
    papers = _fetch_pubmed_details(pmids)

    if not papers:
        print("No PubMed results found for this question - try rephrasing it.")
        return None

    relevant_pmids = []
    print(f"\nFound {len(papers)} candidate papers. Mark each as relevant or not.\n")

    for paper in papers:
        print("-" * 80)
        print(f"PMID: {paper['pmid']}")
        print(f"Title: {paper['title']}")
        print(f"Abstract: {paper['abstract'][:400]}...")
        answer = input("Relevant to the question? (y/n/skip all remaining): ").strip().lower()

        if answer == "skip":
            break
        if answer == "y":
            relevant_pmids.append(paper["pmid"])

    if not relevant_pmids:
        print("\nNo papers were marked relevant - this question won't be saved.")
        return None

    notes = input("\nOptional note about why these are relevant (or press Enter to skip): ").strip()

    return {
        "id": question_id,
        "question": question,
        "relevant_pmids": relevant_pmids,
        "notes": notes,
    }


def main():
    dataset = load_dataset()
    print(f"Current dataset has {len(dataset)} questions.")

    while True:
        next_id = f"q{len(dataset) + 1}"
        entry = label_one_question(next_id)

        if entry:
            dataset.append(entry)
            save_dataset(dataset)
            print(f"\nSaved as '{next_id}'. Dataset now has {len(dataset)} questions.")

        again = input("\nAdd another question? (y/n): ").strip().lower()
        if again != "y":
            break

    print(f"\nDone. Final dataset size: {len(dataset)} questions, saved to {DATASET_PATH}.")


if __name__ == "__main__":
    main()