"""Consent, owner scope, provenance and real vector-index lifecycle for personal memory."""
import json
import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from langchain_core.embeddings import FakeEmbeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_chroma import Chroma

from src.modules import long_term_memory as memory
from src.modules.chatbot import Chatbot
from src.modules.repository import Repository
from src.modules.workflow import stream_answer_turn


class PersonalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Repository(Path(self.temp.name) / "metadata.sqlite3")
        self.owner = self.repo.register("owner", "examplepassword1")
        self.other = self.repo.register("someoneelse", "examplepassword2")
        self.kb = self.repo.create_base(self.owner, "KB", "enterprise", "", "fake", "company",
                                        "https://example.invalid/v1")
        self.other_kb = self.repo.create_base(self.owner, "KB 2", "enterprise", "", "fake", "company",
                                              "https://example.invalid/v1")
        self.sid = self.repo.create_session(self.owner, self.kb)
        self.base = self.repo.get_base(self.owner, self.kb)
        # Real Chroma search/add/delete, but ephemeral so Windows can close temp files.
        stores = {}
        def in_memory_store(kb, api_key):
            return stores.setdefault(kb["id"], Chroma(collection_name=f"test_{kb['id']}",
                embedding_function=FakeEmbeddings(size=32)))
        self.index_patch = patch.object(memory, "memory_store", side_effect=in_memory_store)
        self.index_patch.start()
        self.addCleanup(self.index_patch.stop)

    def test_safe_migration_default_off_and_owner_scoped(self):
        db = sqlite3.connect(self.repo.path)
        db.execute("PRAGMA user_version=2")
        db.commit()
        db.close()
        self.repo = Repository(self.repo.path)
        self.assertIsNotNone(self.repo.get_session(self.owner, self.sid))
        self.assertFalse(self.repo.memory_enabled(self.owner, self.kb))
        with self.assertRaises(PermissionError):
            self.repo.memory_enabled(self.other, self.kb)
        with self.assertRaises(PermissionError):
            self.repo.reserve_memory(self.owner, self.kb, self.sid, "goal", "我计划面试研发岗位。", 1.0)
        self.repo.set_memory_enabled(self.owner, self.kb, True)
        self.assertTrue(Repository(self.repo.path).memory_enabled(self.owner, self.kb))
        with self.assertRaises(PermissionError):
            self.repo.set_memory_enabled(self.other, self.kb, True)
        with self.assertRaises(PermissionError):
            self.repo.reserve_memory(self.owner, self.other_kb, self.sid, "goal", "我计划面试研发岗位。", 1.0)

    def test_auto_extract_requires_exact_user_quote_importance_and_no_secrets(self):
        question = "我希望秋招找 Python 后端开发岗位，请记住。"
        with patch.object(memory, "ChatOpenAI") as model:
            model.return_value.invoke.return_value.content = json.dumps({"items": [
                {"category": "goal", "quote": "我希望秋招找 Python 后端开发岗位", "importance": .94},
                {"category": "fact", "quote": "我已经在企业工作了十年", "importance": 1.0},
            ]})
            found = memory.candidate_quotes(question, "fake", "key", self.base["base_url"])
            self.assertEqual(found, [("我希望秋招找 Python 后端开发岗位", "goal", .94)])
            model.return_value.invoke.return_value.content = json.dumps({"items": [
                {"category": "goal", "quote": "我希望秋招找 Python 后端开发岗位", "importance": .4}]})
            self.assertEqual(memory.candidate_quotes(question, "fake", "key", self.base["base_url"]), [])
            self.assertEqual(memory.candidate_quotes("我的 API Key 是 test-secret", "fake", "key", self.base["base_url"]), [])
            self.assertEqual(memory.candidate_quotes("论文里的方法有什么优势？", "fake", "key", self.base["base_url"]), [])
            self.assertEqual(memory.candidate_quotes("帮我总结这篇论文的方法", "fake", "key", self.base["base_url"]), [])
            self.assertEqual(model.return_value.invoke.call_count, 2)

    def test_index_recall_delete_isolated_between_sessions_and_bases(self):
        self.repo.set_memory_enabled(self.owner, self.kb, True)
        quote = "我希望秋招找 Python 后端开发岗位"
        self.assertTrue(memory.add_memory(self.repo, self.owner, self.base, self.sid,
                                          quote, "goal", .95, "key"))
        self.assertFalse(memory.add_memory(self.repo, self.owner, self.base, self.sid,
                                           quote, "goal", .95, "key"))
        second_sid = self.repo.create_session(self.owner, self.kb)
        self.assertNotEqual(second_sid, self.sid)
        self.assertEqual(memory.recall_memories(self.repo, self.owner, self.base,
                                                "你还记得我的秋招目标吗？", "key"), [quote])
        self.assertEqual(memory.recall_memories(self.repo, self.owner, self.base,
                                                "论文的目标是什么？", "key"), [])
        self.assertEqual(memory.recall_memories(self.repo, self.owner, self.repo.get_base(self.owner, self.other_kb),
                                                "我的目标是什么？", "key"), [])
        with self.assertRaises(PermissionError):
            memory.recall_memories(self.repo, self.other, self.base, "我的目标是什么？", "key")
        self.repo.delete_session(self.owner, self.sid)
        self.assertEqual(len(self.repo.list_memories(self.owner, self.kb)), 1)
        mem_id = self.repo.list_memories(self.owner, self.kb)[0]["id"]
        with self.assertRaises(PermissionError):
            memory.delete_memory(self.repo, self.other, self.base, mem_id, "key")
        memory.delete_memory(self.repo, self.owner, self.base, mem_id, "key")
        self.assertEqual(self.repo.list_memories(self.owner, self.kb), [])
        self.assertEqual(memory.recall_memories(self.repo, self.owner, self.base,
                                                "你还记得我的秋招目标吗？", "key"), [])

    def test_recall_service_error_does_not_drop_completed_answer(self):
        self.repo.set_memory_enabled(self.owner, self.kb, True)
        with patch("src.modules.workflow.recall_memories", side_effect=RuntimeError("service down")), \
             patch("src.modules.workflow.HybridRetriever") as retriever, \
             patch("src.modules.workflow.Chatbot"), \
             patch("src.modules.workflow.make_graph") as graph, \
             patch("src.modules.workflow.remember_user_turn", return_value=0):
            retriever.return_value.last_metrics = {}
            graph.return_value.stream.return_value = [
                ("updates", {"retrieve": {"documents": []}}),
                ("updates", {"answer": {"answer": "无可引用资料。"}}),
            ]
            metrics = {}
            chunks = list(stream_answer_turn(self.repo, self.owner, self.base, self.sid,
                                             "我的目标是什么？", "key", "fake", metrics=metrics))
        self.assertEqual(chunks, ["无可引用资料。"])
        self.assertTrue(metrics["memory_recall_error"])
        self.assertEqual(len(self.repo.list_messages(self.owner, self.sid)), 2)

    def test_disabled_memory_does_not_recall_and_invalid_embedding_rolls_back(self):
        self.repo.set_memory_enabled(self.owner, self.kb, True)
        with patch.object(memory, "memory_store", side_effect=RuntimeError("embedding failed")):
            with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                memory.add_memory(self.repo, self.owner, self.base, self.sid,
                                  "我希望长期从事后端开发", "goal", 1.0, "key")
        self.assertEqual(self.repo.list_memories(self.owner, self.kb), [])
        quote = "我希望长期从事后端开发"
        memory.add_memory(self.repo, self.owner, self.base, self.sid, quote, "goal", 1.0, "key")
        self.repo.set_memory_enabled(self.owner, self.kb, False)
        self.assertEqual(memory.recall_memories(self.repo, self.owner, self.base, "我以前说过什么？", "key"), [])
        self.assertEqual(len(self.repo.list_memories(self.owner, self.kb)), 1)

    def test_completed_turn_learns_and_new_session_can_recall_without_documents(self):
        self.repo.set_memory_enabled(self.owner, self.kb, True)
        statement = "我希望秋招找 Python 后端开发岗位"
        fresh = self.repo.create_session(self.owner, self.kb)
        fake_retriever = MagicMock()
        fake_retriever.similarity_search.return_value = []
        fake_retriever.last_metrics = {}
        with patch("src.modules.workflow.HybridRetriever", return_value=fake_retriever), \
             patch("src.modules.chatbot.ChatOpenAI", return_value=FakeListChatModel(
                 responses=["根据你之前保存的记忆，你希望秋招找 Python 后端开发岗位。"])), \
             patch("src.modules.workflow.remember_user_turn", wraps=memory.remember_user_turn) as remember, \
             patch.object(memory, "candidate_quotes", side_effect=[[(statement, "goal", .95)], []]):
            first_metrics = {}
            first = "我希望秋招找 Python 后端开发岗位，请记住。"
            self.assertIn("没有检索到相关文档", "".join(stream_answer_turn(
                self.repo, self.owner, self.base, self.sid, first,
                "key", "fake", metrics=first_metrics)))
            self.assertEqual(first_metrics["memories_saved"], 1)
            second = "你还记得我的目标吗？"
            answer = "".join(stream_answer_turn(self.repo, self.owner, self.base,
                                                  fresh, second, "key", "fake"))
            self.assertIn("后端开发岗位", answer)
            self.assertEqual(remember.call_count, 2)
        self.assertEqual(len(self.repo.list_memories(self.owner, self.kb)), 1)
        self.assertEqual(self.repo.list_messages(self.owner, fresh)[-1]["sources"], [])

    def test_deleting_knowledge_base_removes_memory_index_and_consent(self):
        from src.modules.managed_index import delete_base
        self.repo.set_memory_enabled(self.owner, self.kb, True)
        memory.add_memory(self.repo, self.owner, self.base, self.sid,
                          "我希望长期从事后端开发", "goal", 1.0, "key")
        with patch("src.modules.managed_index.vector_store") as document_index:
            delete_base(self.repo, self.owner, self.base, "key")
            document_index.return_value.delete_collection.assert_called_once()
        self.assertIsNone(self.repo.get_base(self.owner, self.kb))
        with self.repo.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM personal_memories WHERE kb_id=?", (self.kb,)).fetchone()[0], 0)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM memory_settings WHERE kb_id=?", (self.kb,)).fetchone()[0], 0)

    def test_memory_only_question_can_answer_without_document_sources(self):
        with patch("src.modules.chatbot.ChatOpenAI"):
            bot = Chatbot("fake", MagicMock(), "key", base_url=self.base["base_url"])
        bot.llm = FakeListChatModel(responses=["根据你保存的记录：你希望从事后端开发。"])
        no_evidence = bot.answer("这篇论文的实验结果是什么？", [], [], memories=["我希望从事后端开发"])
        self.assertEqual(no_evidence.sources, [])
        self.assertIn("没有检索到相关文档", no_evidence.answer)
        result = bot.answer("你还记得我的目标吗？", [], [], memories=["我希望从事后端开发"])
        self.assertEqual(result.sources, [])
        self.assertIn("根据你保存的记录", result.answer)
        from src.modules.chatbot import ANSWER_PROMPT
        system_text = ANSWER_PROMPT.format_messages(context="", summary="", chat_history=[],
            question="你还记得我的目标吗？", memory_context="我希望从事后端开发")[0].content
        self.assertIn("我希望从事后端开发", system_text)
        self.assertIn("不是文档证据", system_text)


if __name__ == "__main__":
    unittest.main()
