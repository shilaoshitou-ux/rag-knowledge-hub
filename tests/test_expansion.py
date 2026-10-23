"""No-network tests of paper scoping, version races and actual evaluation metrics."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.modules import evaluation, managed_index
from src.modules.paper_analysis import run_comparison, synthesize_comparison
from src.modules.repository import Repository


class ExpansionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Repository(Path(self.tmp.name) / "test.db")
        self.owner = self.repo.register("demo_user", "long-test-password")
        self.other = self.repo.register("another_user", "long-test-password")
        kb_id = self.repo.create_base(self.owner, "研究", "paper", "", "fake", "company", "https://example.invalid/v1")
        self.kb = self.repo.get_base(self.owner, kb_id)

    def create_doc(self, name, digest):
        return self.repo.create_document(self.owner, self.kb["id"], name, digest, 10,
                                         str(Path(self.tmp.name) / name))[0]

    def test_late_older_version_does_not_replace_new_success(self):
        first = self.create_doc("a.txt", "first")
        second = self.create_doc("a.txt", "second")
        self.repo.job_status(self.owner, second["id"], "processing")
        self.assertEqual(self.repo.complete_document(self.owner, second["id"], [
            {"id": "new", "location": "1", "content": "new"}]), [])
        self.repo.job_status(self.owner, first["id"], "processing")
        self.assertIsNone(self.repo.complete_document(self.owner, first["id"], [
            {"id": "old", "location": "1", "content": "old"}]))
        self.assertEqual([c["id"] for c in self.repo.list_chunks(self.owner, self.kb["id"])], ["new"])
        self.assertEqual(self.repo.stats(self.owner)["documents"], 1)

    def test_eval_parsing_scoped_and_metrics_from_executed_ranks(self):
        doc = self.create_doc("a.txt", "one")
        self.repo.job_status(self.owner, doc["id"], "processing")
        self.repo.complete_document(self.owner, doc["id"], [{"id": "chunk", "location": "1", "content": "text"}])
        cases = evaluation.parse_cases(json.dumps([{"question": "test?", "relevant_documents": ["a.txt"]}]).encode(),
                                       self.repo, self.owner, self.kb["id"])
        self.assertEqual(cases[0].relevant_ids, {doc["id"]})
        with self.assertRaises(PermissionError):
            evaluation.parse_cases(json.dumps([{'question': 'x', 'relevant_documents': ['a.txt']}]).encode(), self.repo, self.other, self.kb['id'])
        irrelevant = Document(page_content="x", metadata={"document_id": "unknown"})
        relevant = Document(page_content="y", metadata={"document_id": doc["id"]})
        with patch.object(evaluation, "HybridRetriever") as retriever:
            retriever.return_value.similarity_search.side_effect = [
                [irrelevant], [irrelevant, relevant], [relevant]
            ]
            rows = evaluation.evaluate(self.repo, self.owner, self.kb, cases, "fake-key", "fake-model", 2)
        summary = evaluation.summarize(rows)
        self.assertEqual([row["MRR"] for row in summary], [0, .5, 1])
        self.assertEqual([row["Recall@K"] for row in summary], [0, 1, 1])
        self.assertTrue(all(row["参考答案包含率"] is None for row in summary))

    def test_transient_partial_vector_write_is_cleaned_and_can_retry(self):
        doc = self.create_doc("a.txt", "one")
        Path(doc["storage_path"]).write_text("a short paper", encoding="utf-8")
        store = MagicMock()
        failure = RuntimeError("private response omitted")
        failure.status_code = 503
        store.add_documents.side_effect = failure
        with patch.object(managed_index, "vector_store", return_value=store):
            retryable = managed_index._process(self.repo.path, self.owner, self.kb["id"], doc["id"], "test-key")
        self.assertTrue(retryable)
        self.assertEqual(self.repo.get_document(self.owner, doc["id"])["status"], "failed")
        self.assertNotIn("private response omitted", self.repo.get_document(self.owner, doc["id"])["error"])
        self.assertEqual(store.delete.call_count, 1)

    def test_semantic_judge_consumes_actual_model_boolean(self):
        with patch.object(evaluation, "ChatOpenAI") as model:
            model.return_value.invoke.return_value.content = '{"correct":true}'
            self.assertEqual(evaluation.judge_answer("question", "expected", "answer",
                                                     "fake-key", "fake-model", "https://example.invalid/v1"), 1)
            model.return_value.invoke.return_value.content = "not-json"
            with self.assertRaises(ValueError):
                evaluation.judge_answer("question", "expected", "answer", "fake-key",
                                        "fake-model", "https://example.invalid/v1")

    def test_comparison_citations_are_paper_scoped(self):
        reports = [
            {"document": {"name": "one.pdf"}, "answer": "方法一 [1]", "sources": [{"name": "one.pdf"}]},
            {"document": {"name": "two.pdf"}, "answer": "方法二 [1]", "sources": [{"name": "two.pdf"}]},
        ]
        with patch("src.modules.paper_analysis.ChatOpenAI") as model:
            model.return_value.invoke.return_value.content = "| 方法 | 一 [A:1] | 二 [B:1] |"
            self.assertIn("[B:1]", synthesize_comparison(reports, "fake-key", "fake-model", "https://example.invalid/v1"))
            model.return_value.invoke.return_value.content = "错误 [B:2]"
            with self.assertRaises(ValueError):
                synthesize_comparison(reports, "fake-key", "fake-model", "https://example.invalid/v1")

    def test_paper_comparison_rejects_foreign_doc_without_model_call(self):
        with self.assertRaises(ValueError):
            run_comparison(self.repo, self.owner, self.kb, ["same", "same"], "key", "model")
        with self.assertRaises(PermissionError):
            run_comparison(self.repo, self.owner, self.kb, ["foreign", "another"], "key", "model")


if __name__ == "__main__":
    unittest.main()
