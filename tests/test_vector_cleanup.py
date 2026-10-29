"""Offline audit/outbox tests: stale vectors are retried without deleting active IDs."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.modules import managed_index
from src.modules.repository import Repository
from src.modules.vector_cleanup import audit_vectors, repair_missing, retry_cleanup


class FakeStore:
    def __init__(self, ids):
        self.ids = list(ids)
        self.fail_on = set()
        self.deleted = []

    def get(self, *, include, limit, offset):
        return {"ids": self.ids[offset:offset + limit]}

    def add_documents(self, docs, *, ids):
        self.ids.extend(ids)
        self.last_docs = docs

    def delete(self, *, ids):
        for cid in ids:
            if cid in self.fail_on:
                raise TimeoutError("temporary local index failure")
            self.deleted.append(cid)
            if cid in self.ids:
                self.ids.remove(cid)


class VectorCleanupTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Repository(Path(self.temp.name) / "database.sqlite3")
        self.owner = self.repo.register("writer", "strong-password-123")
        self.other = self.repo.register("other", "strong-password-123")
        kb_id = self.repo.create_base(self.owner, "docs", "enterprise", "", "fake", "company",
                                      "https://example.invalid/v1")
        self.kb = self.repo.get_base(self.owner, kb_id)
        doc, _ = self.repo.create_document(self.owner, kb_id, "doc.txt", "one", 10,
                                           str(Path(self.temp.name) / "doc.txt"))
        self.repo.job_status(self.owner, doc["id"], "processing")
        self.repo.complete_document(self.owner, doc["id"], [
            {"id": "a", "location": "1", "content": "first"},
            {"id": "b", "location": "2", "content": "second"},
        ])
        self.doc_id = doc["id"]

    def test_audit_reports_missing_and_queues_orphan_without_deleting(self):
        store = FakeStore(["a", "orphan"])
        with patch("src.modules.vector_cleanup.vector_store", return_value=store):
            result = audit_vectors(self.repo, self.owner, self.kb, "key", page_size=1)
        self.assertEqual(result["stale_queued"], 1)
        self.assertEqual(result["missing_active"], 1)
        self.assertEqual(result["missing_ids"], ["b"])
        self.assertEqual(store.deleted, [])
        self.assertEqual([job["chunk_id"] for job in
                          Repository(self.repo.path).list_vector_cleanup(self.owner, self.kb["id"])],
                         ["orphan"])
        with self.assertRaises(PermissionError):
            audit_vectors(self.repo, self.other, self.kb, "key")

    def test_missing_active_vector_can_be_repaired_from_authoritative_chunk(self):
        store = FakeStore(["a"])
        with patch("src.modules.vector_cleanup.vector_store", return_value=store):
            outcome = repair_missing(self.repo, self.owner, self.kb, "key", limit=1)
            self.assertEqual(outcome, {"missing": 1, "repaired": 1, "remaining_estimate": 0})
            self.assertEqual(store.last_docs[0].metadata["chunk_id"], "b")
            self.assertEqual(store.last_docs[0].page_content, "second")
            self.assertEqual(audit_vectors(self.repo, self.owner, self.kb, "key")["missing_active"], 0)
        with self.assertRaises(PermissionError):
            repair_missing(self.repo, self.other, self.kb, "key")

    def test_retry_survives_restart_and_skips_active_id(self):
        self.repo.enqueue_vector_cleanup(self.owner, self.kb["id"], ["a", "orphan"])
        store = FakeStore(["a", "b", "orphan"])
        store.fail_on.add("orphan")
        with patch("src.modules.vector_cleanup.vector_store", return_value=store):
            first = retry_cleanup(self.repo, self.owner, self.kb, "key")
            self.assertEqual((first["failed"], first["active_skipped"]), (1, 1))
            self.assertNotIn("a", store.deleted)
            queued = Repository(self.repo.path).list_vector_cleanup(self.owner, self.kb["id"])
            self.assertEqual([(j["chunk_id"], j["attempts"]) for j in queued], [("orphan", 1)])
            store.fail_on.clear()
            second = retry_cleanup(Repository(self.repo.path), self.owner, self.kb, "key")
        self.assertEqual(second["deleted"], 1)
        self.assertEqual(store.deleted, ["orphan"])
        self.assertFalse(self.repo.list_vector_cleanup(self.owner, self.kb["id"]))

    def test_document_delete_failure_unpublishes_then_records_retry(self):
        store = FakeStore(["a", "b"])
        store.fail_on.add("a")
        with patch("src.modules.managed_index.vector_store", return_value=store):
            managed_index.delete_document(self.repo, self.owner, self.kb, self.doc_id, "key")
        self.assertFalse(self.repo.get_document(self.owner, self.doc_id)["active"])
        self.assertEqual(self.repo.list_chunks(self.owner, self.kb["id"]), [])
        self.assertEqual({j["chunk_id"] for j in self.repo.list_vector_cleanup(self.owner, self.kb["id"])},
                         {"a", "b"})

    def test_do_not_audit_while_ingestion_is_running(self):
        self.repo.create_document(self.owner, self.kb["id"], "later.txt", "second", 10,
                                  str(Path(self.temp.name) / "later.txt"))
        with self.assertRaisesRegex(ValueError, "建索引"):
            audit_vectors(self.repo, self.owner, self.kb, "key")
        with self.assertRaisesRegex(ValueError, "建索引"):
            retry_cleanup(self.repo, self.owner, self.kb, "key")


if __name__ == "__main__":
    unittest.main()
