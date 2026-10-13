import unittest
from unittest import mock

import requests

from tests import _stubs

_stubs.install()

from tools import pubmed_rag  # noqa: E402

SAMPLE_XML = b"""<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>123</PMID>
      <Article>
        <ArticleTitle>Role of <i>BRCA1</i> in DNA repair</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Levels of CO<sub>2</sub> were measured.</AbstractText>
          <AbstractText Label="RESULTS">Repair improved.</AbstractText>
        </Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"""


def _response(content=b"", json_data=None):
    resp = mock.MagicMock()
    resp.content = content
    resp.json.return_value = json_data
    return resp


class ParseDetailsTest(unittest.TestCase):
    def setUp(self):
        with mock.patch.object(pubmed_rag.requests, "get", return_value=_response(SAMPLE_XML)):
            self.paper = pubmed_rag._fetch_pubmed_details(["123"])[0]

    def test_title_keeps_text_inside_nested_tags(self):
        self.assertEqual(self.paper["title"], "Role of BRCA1 in DNA repair")

    def test_abstract_keeps_nested_text_and_section_labels(self):
        self.assertEqual(
            self.paper["abstract"],
            "BACKGROUND: Levels of CO2 were measured. RESULTS: Repair improved.",
        )


class ChunkPapersTest(unittest.TestCase):
    def test_chunk_ids_are_pmid_plus_index(self):
        paper = {"pmid": "9", "title": "t", "url": "u", "abstract": "word " * 400}
        chunks = pubmed_rag._chunk_papers([paper])
        self.assertGreater(len(chunks), 1)
        self.assertEqual([c["chunk_id"] for c in chunks], [f"9-{i}" for i in range(len(chunks))])
        self.assertTrue(all(len(c["text"]) <= 800 for c in chunks))


class FetchNewIdsTest(unittest.TestCase):
    def test_skips_papers_already_stored(self):
        search = _response(json_data={"esearchresult": {"idlist": ["1", "2", "3", "4"]}})
        with mock.patch.object(pubmed_rag.requests, "get", return_value=search), \
             mock.patch.object(pubmed_rag, "stored_pmids", return_value={"1", "2"}):
            self.assertEqual(pubmed_rag._fetch_new_pubmed_ids("q", count=2), ["3", "4"])


class DecideFetchCountTest(unittest.TestCase):
    def results(self, *scores):
        return [(None, s) for s in scores]

    def test_enough_relevant_chunks_skips_pubmed(self):
        self.assertEqual(pubmed_rag._decide_fetch_count(self.results(0.9, 0.9, 0.8, 0.8)), 0)

    def test_partial_coverage_tops_up(self):
        self.assertEqual(pubmed_rag._decide_fetch_count(self.results(0.9, 0.1)),
                         pubmed_rag.TOPUP_FETCH_PAPERS)

    def test_nothing_relevant_does_full_fetch(self):
        self.assertEqual(pubmed_rag._decide_fetch_count(self.results(0.1)),
                         pubmed_rag.FRESH_FETCH_PAPERS)


class ToolErrorHandlingTest(unittest.TestCase):
    def test_failure_returns_message_without_secret_details(self):
        error = requests.HTTPError("400 for url: https://eutils.example/esearch?api_key=SECRET")
        with mock.patch.object(pubmed_rag, "_run_pubmed_search", side_effect=error):
            result = pubmed_rag.pubmed_search.invoke({"query": "q"})
        self.assertIn("HTTPError", result)
        self.assertNotIn("SECRET", result)


if __name__ == "__main__":
    unittest.main()
