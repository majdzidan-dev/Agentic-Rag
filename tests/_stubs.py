"""
Replaces the modules that connect to external services at import time
(Gemini embeddings/LLM, Chroma, the cross-encoder download) with empty
stand-ins, so the pure logic around them can be unit-tested offline and
without API keys. Tests then patch in whatever behaviour they need.
"""

import sys
import types
from unittest import mock

_STUBBED = {
    "rag.vector_store": ["query_existing", "add_chunks"],
    "rag.keyword_store": ["keyword_search", "add_chunks", "stored_pmids"],
    "rag.reranker": ["rerank"],
    "rag.retrieval_grader": ["grade_sufficiency", "rewrite_query"],
}


def install() -> None:
    for module_name, attributes in _STUBBED.items():
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)
        for attribute in attributes:
            setattr(module, attribute, mock.MagicMock(name=f"{module_name}.{attribute}"))
        sys.modules[module_name] = module
