"""Single-chunk knowledge-base revisions without remote embedding requests."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.documents import Document
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modules.chunk_edits import REVISION_SUFFIX, delete_chunk, update_chunk
from modules.managed_ui import ModelSettings
from modules.repository import Repository
from modules.retrieval import HybridRetriever
from modules.workflow import _source_records
from modules.citations import original_page


class MemoryVectorStore:
    def __init__(self):
        self.items = {}
        self.added = []
        self.deleted = []
        self.fail_add = False
        self.fail_delete = False

    def add_documents(self, docs, ids):
        self.added.extend(ids)
        for doc, key in zip(docs, ids):
            self.items[key] = doc
        if self.fail_add:
            raise RuntimeError("private gateway response")

    def delete(self, ids):
        self.deleted.extend(ids)
        if self.fail_delete:
            raise RuntimeError("cleanup temporarily unavailable")
        for key in ids:
            self.items.pop(key, None)

    def similarity_search(self, query, k, filter):
        allowed = set(filter["document_id"]["$in"])
        return [doc for doc in self.items.values()
                if doc.metadata["document_id"] in allowed][:k]


class ChunkEditTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Repository(Path(self.temp.name) / "data.sqlite3")
        self.owner = self.repo.register("chunk_owner", "secure-password-123")
        self.other = self.repo.register("chunk_other", "secure-password-123")
        kb_id = self.repo.create_base(self.owner, "资料库", "enterprise", "", "fake", "company",
                                      "https://example.invalid/v1")
        self.kb = self.repo.get_base(self.owner, kb_id)
        self.file = Path(self.temp.name) / "original.txt"
        self.file.write_text("原始文件不应被修改", encoding="utf-8")
        self.doc = self.repo.create_document(self.owner, kb_id, "original.txt", "hash", 100, str(self.file))[0]
        self.repo.job_status(self.owner, self.doc["id"], "processing")
        self.repo.complete_document(self.owner, self.doc["id"], [
            {"id": "first", "location": "第 1 页", "content": "旧规则：十天"},
            {"id": "second", "location": "第 2 页", "content": "其他规则：远程办公"},
        ])
        self.store = MemoryVectorStore()
        for key, text, location in [("first", "旧规则：十天", "第 1 页"),
                                    ("second", "其他规则：远程办公", "第 2 页")]:
            self.store.items[key] = Document(page_content=text, metadata={
                "chunk_id": key, "source": "original.txt", "document_id": self.doc["id"],
                "location": location, "version": 1,
            })
        for target in ("modules.chunk_edits.vector_store", "modules.retrieval.vector_store"):
            mocked = patch(target, return_value=self.store)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_one_chunk_updated_and_marked_not_original(self):
        new_id, pending = update_chunk(self.repo, self.owner, self.kb["id"], self.doc["id"],
                                       "first", "新规则：十五天", "key")
        self.assertFalse(pending)
        self.assertEqual(self.store.added, [new_id])
        self.assertEqual(self.store.deleted, ["first"])
        rows = self.repo.list_chunks(self.owner, self.kb["id"])
        self.assertEqual({row["id"] for row in rows}, {new_id, "second"})
        self.assertEqual(self.repo.get_chunk(self.owner, self.kb["id"], new_id)["content"], "新规则：十五天")
        self.assertTrue(self.repo.get_chunk(self.owner, self.kb["id"], new_id)["location"].endswith(REVISION_SUFFIX))
        self.assertEqual(self.repo.get_document(self.owner, self.doc["id"])["chunk_count"], 2)
        self.assertEqual(self.file.read_text(encoding="utf-8"), "原始文件不应被修改")
        found = HybridRetriever(self.repo, self.owner, self.kb, "key", "model", "vector").similarity_search("规则", 4)
        self.assertEqual({doc.metadata["chunk_id"] for doc in found}, {new_id, "second"})
        self.assertTrue(_source_records([self.store.items[new_id]])[0]["edited"])
        self.assertIsNone(original_page(self.repo, self.owner, self.kb["id"],
                                        _source_records([self.store.items[new_id]])[0]))

    def test_reject_foreign_owner_and_stale_chunk(self):
        with self.assertRaises(PermissionError):
            update_chunk(self.repo, self.other, self.kb["id"], self.doc["id"], "first", "越权", "key")
        new_id, _ = update_chunk(self.repo, self.owner, self.kb["id"], self.doc["id"],
                                 "first", "新规则：十五天", "key")
        with self.assertRaises(ValueError):
            update_chunk(self.repo, self.owner, self.kb["id"], self.doc["id"], "first", "并发覆盖", "key")
        self.assertIsNotNone(self.repo.get_chunk(self.owner, self.kb["id"], new_id))

    def test_failed_embedding_rolls_back_and_retry_can_update(self):
        self.store.fail_add = True
        with self.assertRaisesRegex(RuntimeError, "向量化失败") as caught:
            update_chunk(self.repo, self.owner, self.kb["id"], self.doc["id"], "first", "新内容", "key")
        self.assertNotIn("private gateway response", str(caught.exception))
        self.assertIsNotNone(self.repo.get_chunk(self.owner, self.kb["id"], "first"))
        self.assertEqual(set(self.store.items), {"first", "second"})
        self.store.fail_add = False
        update_chunk(self.repo, self.owner, self.kb["id"], self.doc["id"], "first", "新内容", "key")

    def test_failed_vector_cleanup_is_filtered_from_search(self):
        self.store.fail_delete = True
        new_id, pending = update_chunk(self.repo, self.owner, self.kb["id"], self.doc["id"],
                                       "first", "新规则：十五天", "key")
        self.assertTrue(pending)
        self.assertIn("first", self.store.items)
        found = HybridRetriever(self.repo, self.owner, self.kb, "key", "model", "vector").similarity_search("规则", 4)
        self.assertNotIn("first", [doc.metadata["chunk_id"] for doc in found])
        self.assertIn(new_id, [doc.metadata["chunk_id"] for doc in found])

    def test_delete_single_chunk_but_not_last(self):
        pending = delete_chunk(self.repo, self.owner, self.kb["id"], self.doc["id"], "first", "key")
        self.assertFalse(pending)
        self.assertEqual([row["id"] for row in self.repo.list_chunks(self.owner, self.kb["id"])], ["second"])
        self.assertEqual(self.repo.get_document(self.owner, self.doc["id"])["chunk_count"], 1)
        with self.assertRaisesRegex(ValueError, "最后一个片段"):
            delete_chunk(self.repo, self.owner, self.kb["id"], self.doc["id"], "second", "key")
        self.assertEqual(self.file.read_text(encoding="utf-8"), "原始文件不应被修改")

    def test_db_conflict_does_not_publish_new_vector(self):
        with patch.object(self.repo, "replace_chunk", side_effect=ValueError("concurrent update")):
            with self.assertRaisesRegex(ValueError, "concurrent update"):
                update_chunk(self.repo, self.owner, self.kb["id"], self.doc["id"],
                             "first", "新规则：十五天", "key")
        self.assertEqual(set(self.store.items), {"first", "second"})
        self.assertIsNotNone(self.repo.get_chunk(self.owner, self.kb["id"], "first"))

    def test_deleted_vector_left_by_gateway_stays_inaccessible(self):
        self.store.fail_delete = True
        self.assertTrue(delete_chunk(self.repo, self.owner, self.kb["id"], self.doc["id"], "first", "key"))
        found = HybridRetriever(self.repo, self.owner, self.kb, "key", "model", "vector").similarity_search("规则", 4)
        self.assertEqual([doc.metadata["chunk_id"] for doc in found], ["second"])

    def test_real_chroma_only_reembeds_selected_chunk(self):
        import gc
        from chromadb.api.client import SharedSystemClient
        from langchain_core.embeddings import FakeEmbeddings
        from modules import managed_index

        self.addCleanup(lambda: (SharedSystemClient.clear_system_cache(), gc.collect()))
        with patch.object(managed_index, "DATA_DIR", Path(self.temp.name)), \
             patch.object(managed_index, "make_embeddings", return_value=FakeEmbeddings(size=8)), \
             patch("modules.chunk_edits.vector_store", side_effect=managed_index.vector_store):
            real = managed_index.vector_store(self.kb, "key")
            real.add_documents(list(self.store.items.values()), ids=["first", "second"])
            new_id, pending = update_chunk(self.repo, self.owner, self.kb["id"], self.doc["id"],
                                           "first", "局部修订后的文本", "key")
            self.assertFalse(pending)
            self.assertEqual(real.get(ids=["first"])["ids"], [])
            self.assertEqual(real.get(ids=["second"])["ids"], ["second"])
            self.assertEqual(real.get(ids=[new_id])["documents"], ["局部修订后的文本"])

    def test_ui_can_delete_selected_chunk(self):
        settings = ModelSettings(self.kb["base_url"], "key", "model", "fake", "company")
        patches = [
            patch("modules.repository.DB_PATH", self.repo.path),
            patch("modules.auth_ui.require_user", return_value=self.repo.get_user(self.owner)),
            patch("modules.managed_ui.render_model_settings", return_value=settings),
        ]
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)
        app = AppTest.from_file(str(ROOT / "src" / "Home.py"), default_timeout=20).run()
        app.session_state["rag_managed_kb"] = self.kb["id"]
        app = app.switch_page("app_pages/knowledge.py").run()
        app.get_by_key("rag_chunk_confirm_first").set_value(True).run()
        app.get_by_key("rag_chunk_delete_first").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual([row["id"] for row in self.repo.list_chunks(self.owner, self.kb["id"])], ["second"])

    def test_ui_can_edit_selected_chunk(self):
        settings = ModelSettings(self.kb["base_url"], "key", "model", "fake", "company")
        patches = [
            patch("modules.repository.DB_PATH", self.repo.path),
            patch("modules.auth_ui.require_user", return_value=self.repo.get_user(self.owner)),
            patch("modules.managed_ui.render_model_settings", return_value=settings),
        ]
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)
        app = AppTest.from_file(str(ROOT / "src" / "Home.py"), default_timeout=20).run()
        app.session_state["rag_managed_kb"] = self.kb["id"]
        app = app.switch_page("app_pages/knowledge.py").run()
        self.assertEqual(len(app.exception), 0)
        self.assertIsNotNone(app.get_by_key(f"rag_chunk_text_first"))
        app.get_by_key("rag_chunk_text_first").set_value("界面改成十五天")
        app.get_by_key("rag_chunk_edit_form_first").get("form_submit_button")[0].click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(any(row["content"] == "界面改成十五天" for row in self.repo.list_chunks(self.owner, self.kb["id"])))


if __name__ == "__main__":
    unittest.main()
