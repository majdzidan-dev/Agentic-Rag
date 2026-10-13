import unittest
from unittest import mock

from tests import _stubs

_stubs.install()

from langchain_core.documents import Document  # noqa: E402

from rag import hybrid_search as hs  # noqa: E402


def _doc(chunk_id: str) -> Document:
    return Document(page_content=f"text {chunk_id}", metadata={"chunk_id": chunk_id})


def _chunk(chunk_id: str) -> dict:
    return {"chunk_id": chunk_id, "text": f"text {chunk_id}", "pmid": chunk_id.split("-")[0],
            "title": "title", "url": "url"}


class HybridSearchTest(unittest.TestCase):
    def run_search(self, dense_ids, sparse_ids, k=10):
        dense = [(_doc(cid), 0.9) for cid in dense_ids]
        sparse = [(_chunk(cid), 5.0) for cid in sparse_ids]
        with mock.patch.object(hs, "query_existing", return_value=dense), \
             mock.patch.object(hs, "keyword_search", return_value=sparse):
            return hs.hybrid_search("query", k=k)

    def test_chunk_found_by_both_ranks_above_single_list_winners(self):
        results = self.run_search(dense_ids=["1-0", "2-0"], sparse_ids=["3-0", "2-0"])
        self.assertEqual(results[0][0].metadata["chunk_id"], "2-0")

    def test_rrf_score_formula(self):
        results = self.run_search(dense_ids=["1-0"], sparse_ids=["1-0"])
        self.assertAlmostEqual(results[0][1], 2 / (hs.RRF_K + 1))

    def test_keyword_only_hit_is_rebuilt_as_document(self):
        results = self.run_search(dense_ids=[], sparse_ids=["7-0"])
        doc, _ = results[0]
        self.assertIsInstance(doc, Document)
        self.assertEqual(doc.metadata["pmid"], "7")

    def test_respects_k(self):
        results = self.run_search(dense_ids=["1-0", "2-0", "3-0"], sparse_ids=["4-0"], k=2)
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
