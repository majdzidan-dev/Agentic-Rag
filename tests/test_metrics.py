import unittest

from eval.metrics import mean_reciprocal_rank, ndcg_at_k, recall_at_k


class RecallTest(unittest.TestCase):
    def test_counts_relevant_found_in_top_k(self):
        self.assertEqual(recall_at_k(["a", "b", "c"], {"a", "c", "z"}, k=2), 1 / 3)

    def test_no_relevant_items_scores_zero(self):
        self.assertEqual(recall_at_k(["a"], set(), k=5), 0.0)


class MrrTest(unittest.TestCase):
    def test_uses_first_relevant_rank(self):
        self.assertEqual(mean_reciprocal_rank(["x", "a", "b"], {"a", "b"}), 0.5)

    def test_nothing_relevant_scores_zero(self):
        self.assertEqual(mean_reciprocal_rank(["x", "y"], {"a"}), 0.0)


class NdcgTest(unittest.TestCase):
    def test_perfect_ranking_scores_one(self):
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "x"], {"a", "b"}, k=5), 1.0)

    def test_duplicate_chunks_of_one_paper_never_exceed_one(self):
        # Several chunks from the same relevant paper used to push NDCG above 1
        ranked = ["a", "a", "a", "x", "y"]
        self.assertAlmostEqual(ndcg_at_k(ranked, {"a"}, k=5), 1.0)

    def test_lower_position_scores_less(self):
        self.assertLess(ndcg_at_k(["x", "a"], {"a"}, k=5), ndcg_at_k(["a", "x"], {"a"}, k=5))

    def test_duplicates_dont_push_later_papers_out_of_top_k(self):
        # "b" is the 2nd distinct paper, so it counts within k=2
        self.assertAlmostEqual(ndcg_at_k(["a", "a", "b"], {"a", "b"}, k=2), 1.0)


if __name__ == "__main__":
    unittest.main()
