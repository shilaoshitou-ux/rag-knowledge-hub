"""No-network tests: optional rerank cannot make usable Hybrid retrieval fail."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.modules.repository import Repository
from src.modules.retrieval import HybridRetriever


class RerankFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Repository(Path(self.tmp.name) / "test.sqlite3")
        self.owner = self.repo.register("reader", "password-for-tests")
        kb_id = self.repo.create_base(self.owner, "books", "enterprise", "", "fake",
                                      "company", "https://example.invalid/v1")
        self.kb = self.repo.get_base(self.owner, kb_id)
        doc, _ = self.repo.create_document(self.owner, kb_id, "guide.txt", "digest", 10,
                                           str(Path(self.tmp.name) / "guide.txt"))
        self.repo.job_status(self.owner, doc["id"], "processing")
        self.repo.complete_document(self.owner, doc["id"], [
            {"id": "a", "content": "renewal deadline is May 5", "location": "1"},
            {"id": "b", "content": "holiday schedule is August 8", "location": "2"},
        ])
        self.doc_id = doc["id"]

    def test_rerank_invalid_order_or_timeout_returns_hybrid_and_records_fallback(self):
        store = MagicMock()
        store.similarity_search.return_value = [Document(page_content="renewal deadline is May 5",
            metadata={"chunk_id": "a", "document_id": self.doc_id})]
        with patch("src.modules.retrieval.vector_store", return_value=store), \
                patch("src.modules.retrieval.llm_rerank", side_effect=[ValueError("bad order"), TimeoutError()]):
            hybrid = HybridRetriever(self.repo, self.owner, self.kb, "key", "model", "hybrid")
            expected = [d.metadata["chunk_id"] for d in hybrid.similarity_search("renewal deadline", 2)]
            for _ in range(2):
                rerank = HybridRetriever(self.repo, self.owner, self.kb, "key", "model", "rerank")
                self.assertEqual([d.metadata["chunk_id"] for d in rerank.similarity_search("renewal deadline", 2)], expected)
                self.assertTrue(rerank.last_metrics["rerank_fallback"])
                self.assertFalse(rerank.last_metrics["reranked"])
                self.assertGreaterEqual(rerank.last_metrics["retrieval_ms"], 0)

    def test_embedding_failure_still_propagates(self):
        store = MagicMock()
        store.similarity_search.side_effect = RuntimeError("embedding unavailable")
        with patch("src.modules.retrieval.vector_store", return_value=store):
            retriever = HybridRetriever(self.repo, self.owner, self.kb, "key", "model", "rerank")
            with self.assertRaisesRegex(RuntimeError, "embedding unavailable"):
                retriever.similarity_search("question", 2)


if __name__ == "__main__":
    unittest.main()
