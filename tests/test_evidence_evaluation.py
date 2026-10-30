"""Offline checks for active, owner-scoped evidence labels and chunk-level metrics."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.documents import Document

from src.modules import evaluation
from src.modules.repository import Repository


class EvidenceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Repository(Path(self.tmp.name) / "test.db")
        self.owner = self.repo.register("eva", "long-test-password")
        self.other = self.repo.register("other", "long-test-password")
        self.kb_id = self.repo.create_base(self.owner, "research", "paper", "", "fake",
                                           "company", "https://example.invalid/v1")
        self.kb = self.repo.get_base(self.owner, self.kb_id)
        self._add_doc(self.owner, self.kb_id, "a.txt", "a", ["a1", "a2"])
        self._add_doc(self.owner, self.kb_id, "b.txt", "b", ["b1"])

    def _add_doc(self, owner, kb_id, filename, digest, ids):
        doc, _ = self.repo.create_document(owner, kb_id, filename, digest, 100,
                                           str(Path(self.tmp.name) / filename))
        self.repo.job_status(owner, doc["id"], "processing")
        self.repo.complete_document(owner, doc["id"], [
            {"id": cid, "location": str(i), "content": f"evidence {cid}"}
            for i, cid in enumerate(ids, 1)
        ])
        return doc

    def _parse(self, entry):
        return evaluation.parse_cases(json.dumps([entry]).encode("utf-8"),
                                      self.repo, self.owner, self.kb_id)[0]

    def test_question_set_preflight_explains_missing_and_pending_files(self):
        data = json.dumps([
            {"question": "a?", "relevant_documents": ["a.txt", "waiting.md"]},
            {"question": "b?", "relevant_documents": ["absent.md"]},
        ]).encode("utf-8")
        documents = self.repo.list_documents(self.owner, self.kb_id)
        documents.append({"name": "waiting.md", "status": "waiting"})
        count, unavailable = evaluation.check_question_documents(data, documents)
        self.assertEqual(count, 2)
        self.assertEqual(unavailable, [("absent.md", "未上传"), ("waiting.md", "waiting")])
        documents.append({"name": "waiting.md", "status": "success"})
        self.assertEqual(evaluation.check_question_documents(data, documents)[1],
                         [("absent.md", "未上传")])
        with self.assertRaises(ValueError):
            evaluation.check_question_documents(b"not json", documents)

    def test_legacy_questions_and_manifest_are_owner_scoped(self):
        case = self._parse({"question": "What?", "relevant_documents": ["a.txt"]})
        self.assertFalse(case.relevant_chunk_ids)
        with patch.object(evaluation, "HybridRetriever") as retriever:
            retriever.return_value.similarity_search.return_value = []
            legacy_rows = evaluation.evaluate(self.repo, self.owner, self.kb, [case],
                                              "offline-key", "offline-model")
        self.assertTrue(all(row["evidence_recall_at_k"] is None for row in legacy_rows))
        self.assertTrue(all(row["证据 Recall@K"] is None
                            for row in evaluation.summarize(legacy_rows)))
        self.assertEqual({row["chunk_id"] for row in
                          evaluation.chunk_manifest(self.repo, self.owner, self.kb_id)},
                         {"a1", "a2", "b1"})
        with self.assertRaises(PermissionError):
            evaluation.chunk_manifest(self.repo, self.other, self.kb_id)

    def test_labels_must_be_current_and_from_labelled_documents(self):
        valid = {"question": "What?", "relevant_documents": ["a.txt"],
                 "relevant_chunks": ["a2"]}
        self.assertEqual(self._parse(valid).relevant_chunk_ids, {"a2"})
        for ids in ([], ["a2", "a2"], ["missing"], ["b1"]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                self._parse({**valid, "relevant_chunks": ids})
        with self.assertRaises(ValueError):
            self._parse({**valid, "relevant_chunks": "a2"})
        foreign_kb = self.repo.create_base(self.other, "foreign", "paper", "", "fake",
                                           "company", "https://example.invalid/v1")
        self._add_doc(self.other, foreign_kb, "other.txt", "foreign", ["other1"])
        with self.assertRaises(ValueError):
            self._parse({**valid, "relevant_chunks": ["other1"]})
        a_id = next(doc["id"] for doc in self.repo.list_documents(self.owner, self.kb_id)
                    if doc["name"] == "a.txt")
        self.repo.replace_chunk(self.owner, self.kb_id, a_id, "a2", "a3", "new text", "2")
        with self.assertRaises(ValueError):
            self._parse(valid)  # a2 is obsolete after a chunk edit
        self.assertEqual(self._parse({**valid, "relevant_chunks": ["a3"]}).relevant_chunk_ids,
                         {"a3"})
        self.assertNotIn("a2", {row["chunk_id"] for row in
                                  evaluation.chunk_manifest(self.repo, self.owner, self.kb_id)})
        self.repo.remove_chunk(self.owner, self.kb_id, a_id, "a3")
        with self.assertRaises(ValueError):
            self._parse({**valid, "relevant_chunks": ["a3"]})

    def test_evidence_ranks_distinguish_same_document_chunks(self):
        case = self._parse({"question": "specific detail?", "relevant_documents": ["a.txt"],
                            "relevant_chunks": ["a2"]})
        a_id = next(doc["id"] for doc in self.repo.list_documents(self.owner, self.kb_id)
                    if doc["name"] == "a.txt")
        def hit(chunk_id):
            return Document(page_content="some content",
                            metadata={"document_id": a_id, "chunk_id": chunk_id})
        with patch.object(evaluation, "HybridRetriever") as retriever:
            retriever.return_value.similarity_search.side_effect = [
                [hit("a1")], [hit("a1"), hit("a2")], [hit("a2")]
            ]
            rows = evaluation.evaluate(self.repo, self.owner, self.kb, [case],
                                       "offline-key", "offline-model", k=2)
        self.assertEqual([row["recall_at_k"] for row in rows], [1, 1, 1])
        self.assertEqual([row["evidence_recall_at_k"] for row in rows], [0, 1, 1])
        self.assertEqual([row["evidence_reciprocal_rank"] for row in rows], [0, .5, 1])
        self.assertEqual([row["retrieved_chunks"] for row in rows],
                         [["a1"], ["a1", "a2"], ["a2"]])
        self.assertTrue(all(row["elapsed_ms"] >= row["retrieval_ms"] >= 0 for row in rows))
        summaries = evaluation.summarize(rows)
        self.assertEqual([row["证据 MRR"] for row in summaries], [0, .5, 1])
        self.assertEqual([row["证据 Recall@K"] for row in summaries], [0, 1, 1])
        self.assertEqual([row["证据标注题数"] for row in summaries], [1, 1, 1])
        self.assertTrue(all(row["参考答案包含率"] is None for row in summaries))


if __name__ == "__main__":
    unittest.main()
