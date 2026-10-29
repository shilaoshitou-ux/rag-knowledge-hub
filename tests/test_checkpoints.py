"""Durable checkpoint resume and exactly-once metadata commit without network calls."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.modules.chatbot import ChatResult
from src.modules.repository import Repository
from src.modules.workflow import (abandon_turn, checkpoint_path, checkpoint_stage,
                                  stream_answer_turn)
from langgraph.checkpoint.sqlite import SqliteSaver


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db_path = Path(self.tmp.name) / "knowledge.sqlite3"
        self.repo = Repository(self.db_path)
        self.owner = self.repo.register("reader", "long-test-password")
        self.other = self.repo.register("stranger", "long-test-password")
        kb_id = self.repo.create_base(self.owner, "papers", "paper", "", "fake",
                                      "company", "https://example.invalid/v1")
        self.kb = self.repo.get_base(self.owner, kb_id)
        self.sid = self.repo.create_session(self.owner, kb_id)
        doc, _ = self.repo.create_document(self.owner, kb_id, "paper.txt", "digest", 100,
                                           str(Path(self.tmp.name) / "paper.txt"))
        self.repo.job_status(self.owner, doc["id"], "processing")
        self.repo.complete_document(self.owner, doc["id"], [
            {"id": "c1", "location": "1", "content": "valid evidence"},
            {"id": "c2", "location": "2", "content": "other evidence"},
        ])
        self.doc_id = doc["id"]

    def begin(self, question="What does it say?", *, regenerate=False):
        return self.repo.begin_turn(self.owner, self.sid, question, "hybrid", 4,
                                    8, "offline-model", regenerate=regenerate)

    def stream(self, run_id, repo=None, question="What does it say?", regenerate=False):
        return stream_answer_turn(repo or self.repo, self.owner, self.kb, self.sid,
                                  question, "offline-key", "offline-model",
                                  regenerate=regenerate, run_id=run_id)

    def test_crash_after_answer_checkpoint_resumes_without_reexecuting_nodes(self):
        run = self.begin()
        evidence = Document(page_content="valid evidence", metadata={"chunk_id": "c1", "source": "paper.txt"})
        with patch("src.modules.workflow.HybridRetriever") as retriever, \
                patch("src.modules.workflow.Chatbot") as bot:
            bot.return_value.rewrite_question.return_value = "What does it say?"
            bot.return_value.retrieve.return_value = [evidence]
            bot.return_value.answer.return_value = ChatResult("The answer", [evidence])
            iterator = self.stream(run["id"])
            self.assertEqual(next(iterator), "The answer")
            iterator.close()  # Crash/stop after the graph is complete, before metadata commit.
            self.assertEqual(self.repo.list_messages(self.owner, self.sid), [])
            self.assertEqual(checkpoint_stage(self.repo, self.owner, self.sid),
                             "回答已完成，等待写入会话")
            previous = (bot.return_value.rewrite_question.call_count,
                        bot.return_value.retrieve.call_count, bot.return_value.answer.call_count)
            reloaded = Repository(self.db_path)
            self.assertEqual(list(self.stream(run["id"], repo=reloaded)), ["The answer"])
            self.assertEqual((bot.return_value.rewrite_question.call_count,
                              bot.return_value.retrieve.call_count, bot.return_value.answer.call_count), previous)
        self.assertIsNone(self.repo.pending_turn(self.owner, self.sid))
        self.assertEqual([m["role"] for m in self.repo.list_messages(self.owner, self.sid)],
                         ["user", "assistant"])
        with self.assertRaises(ValueError):
            self.repo.complete_pending_turn(self.owner, self.sid, run["id"], "duplicate", [])
        self.assertEqual(len(self.repo.list_messages(self.owner, self.sid)), 2)
        with SqliteSaver.from_conn_string(checkpoint_path(self.repo)) as saver:
            self.assertIsNone(saver.get({"configurable": {"thread_id": run["id"]}}))

    def test_stopped_answer_resumes_without_redoing_retrieval(self):
        run = self.begin()
        evidence = Document(page_content="valid evidence", metadata={
            "chunk_id": "c1", "source": "paper.txt", "document_id": self.doc_id})
        with patch("src.modules.workflow.HybridRetriever") as retriever:
            retriever.return_value.similarity_search.return_value = [evidence]
            with patch("src.modules.chatbot.ChatOpenAI",
                       return_value=FakeListChatModel(responses=["Partial answer"])):
                iterator = self.stream(run["id"])
                self.assertTrue(next(iterator))
                iterator.close()
            self.assertEqual(checkpoint_stage(self.repo, self.owner, self.sid),
                             "已完成检索；继续时恢复回答节点")
            self.assertEqual(self.repo.list_messages(self.owner, self.sid), [])
            with patch("src.modules.chatbot.ChatOpenAI",
                       return_value=FakeListChatModel(responses=["Fresh final answer"])):
                self.assertEqual("".join(self.stream(run["id"])), "Partial answer")
            self.assertEqual(retriever.return_value.similarity_search.call_count, 1)
        self.assertEqual(self.repo.list_messages(self.owner, self.sid)[-1]["content"],
                         "Partial answer")

    def test_stale_evidence_blocks_resume_and_discard_cleans_checkpoint(self):
        run = self.begin()
        evidence = Document(page_content="valid evidence", metadata={
            "chunk_id": "c1", "source": "paper.txt", "document_id": self.doc_id})
        with patch("src.modules.workflow.HybridRetriever") as retriever, \
                patch("src.modules.chatbot.ChatOpenAI",
                      return_value=FakeListChatModel(responses=["answer", "answer again"])):
            retriever.return_value.similarity_search.return_value = [evidence]
            iterator = self.stream(run["id"])
            next(iterator)
            iterator.close()
            self.repo.replace_chunk(self.owner, self.kb["id"], self.doc_id, "c1",
                                    "new", "revised evidence", "1")
            with self.assertRaisesRegex(ValueError, "知识库证据"):
                list(self.stream(run["id"]))
        with self.assertRaises(PermissionError):
            checkpoint_stage(self.repo, self.other, self.sid)
        with self.assertRaises(PermissionError):
            abandon_turn(self.repo, self.other, self.sid, run["id"])
        abandon_turn(self.repo, self.owner, self.sid, run["id"])
        self.assertIsNone(self.repo.pending_turn(self.owner, self.sid))
        with SqliteSaver.from_conn_string(checkpoint_path(self.repo)) as saver:
            self.assertIsNone(saver.get({"configurable": {"thread_id": run["id"]}}))

    def test_failed_answer_resumes_from_answer_node(self):
        run = self.begin()
        evidence = Document(page_content="valid evidence", metadata={
            "chunk_id": "c1", "source": "paper.txt", "document_id": self.doc_id})
        with patch("src.modules.workflow.HybridRetriever"), \
                patch("src.modules.workflow.Chatbot") as bot:
            bot.return_value.rewrite_question.return_value = "What does it say?"
            bot.return_value.retrieve.return_value = [evidence]
            bot.return_value.answer.side_effect = [RuntimeError("gateway failed"),
                                                   ChatResult("Recovered answer", [evidence])]
            with self.assertRaisesRegex(RuntimeError, "gateway failed"):
                list(self.stream(run["id"]))
            self.assertEqual(self.repo.list_messages(self.owner, self.sid), [])
            self.assertEqual(checkpoint_stage(self.repo, self.owner, self.sid),
                             "已完成检索；继续时恢复回答节点")
            self.assertEqual(list(self.stream(run["id"])), ["Recovered answer"])
            self.assertEqual(bot.return_value.rewrite_question.call_count, 1)
            self.assertEqual(bot.return_value.retrieve.call_count, 1)
            self.assertEqual(bot.return_value.answer.call_count, 2)

    def test_deleting_session_erases_unfinished_checkpoint(self):
        run = self.begin()
        evidence = Document(page_content="valid evidence", metadata={
            "chunk_id": "c1", "source": "paper.txt", "document_id": self.doc_id})
        with patch("src.modules.workflow.HybridRetriever"), \
                patch("src.modules.workflow.Chatbot") as bot:
            bot.return_value.rewrite_question.return_value = "What does it say?"
            bot.return_value.retrieve.return_value = [evidence]
            bot.return_value.answer.return_value = ChatResult("Answer", [evidence])
            iterator = self.stream(run["id"])
            next(iterator)
            iterator.close()
        self.repo.delete_session(self.owner, self.sid)
        with SqliteSaver.from_conn_string(checkpoint_path(self.repo)) as saver:
            self.assertIsNone(saver.get({"configurable": {"thread_id": run["id"]}}))

    def test_abandon_before_first_node_cleans_pending(self):
        run = self.begin()
        self.assertEqual(checkpoint_stage(self.repo, self.owner, self.sid),
                         "尚无已完成节点；将重新开始")
        abandon_turn(self.repo, self.owner, self.sid, run["id"])
        self.assertIsNone(self.repo.pending_turn(self.owner, self.sid))
        self.assertTrue(self.begin())

    def test_parallel_resume_is_rejected_while_stream_is_open(self):
        run = self.begin()
        evidence = Document(page_content="valid evidence", metadata={
            "chunk_id": "c1", "source": "paper.txt", "document_id": self.doc_id})
        with patch("src.modules.workflow.HybridRetriever"), \
                patch("src.modules.workflow.Chatbot") as bot:
            bot.return_value.rewrite_question.return_value = "What does it say?"
            bot.return_value.retrieve.return_value = [evidence]
            bot.return_value.answer.return_value = ChatResult("Answer", [evidence])
            first = self.stream(run["id"])
            self.assertEqual(next(first), "Answer")
            with self.assertRaisesRegex(ValueError, "正在执行"):
                list(self.stream(run["id"]))
            first.close()
        abandon_turn(self.repo, self.owner, self.sid, run["id"])

    def test_chat_page_recovers_pending_task_after_fresh_session(self):
        import sys
        from streamlit.testing.v1 import AppTest
        root = Path(__file__).resolve().parents[1]
        if str(root / "src") not in sys.path:
            sys.path.insert(0, str(root / "src"))
        from modules.managed_ui import ModelSettings
        run = self.begin()
        settings = ModelSettings(self.kb["base_url"], "offline-key", "offline-model",
                                 "fake", "company")
        with patch("modules.repository.DB_PATH", self.db_path), \
                patch("modules.auth_ui.require_user", return_value=self.repo.get_user(self.owner)), \
                patch("modules.managed_ui.render_model_settings", return_value=settings):
            page = str(root / "src" / "Home.py")
            fresh = AppTest.from_file(page, default_timeout=20).run()
            fresh = fresh.switch_page("app_pages/chat.py").run()
            self.assertEqual(len(fresh.exception), 0)
            self.assertTrue(any(b.label == "从检查点继续" for b in fresh.button))
            self.assertTrue(any(b.label == "放弃本轮任务" for b in fresh.button))
            self.assertTrue(fresh.get_by_key("rag_managed_question").disabled)
            for button in fresh.button:
                if button.label == "放弃本轮任务":
                    button.click().run()
                    break
            self.assertEqual(len(fresh.exception), 0)
        self.assertIsNone(self.repo.pending_turn(self.owner, self.sid))

    def test_single_pending_run_and_regenerate_are_idempotent(self):
        run = self.begin()
        with self.assertRaises(ValueError):
            self.begin("a different question")
        with self.assertRaises(PermissionError):
            self.repo.pending_turn(self.other, self.sid)
        self.repo.complete_pending_turn(self.owner, self.sid, run["id"], "old answer", [])
        regen = self.begin(regenerate=True)
        self.repo.complete_pending_turn(self.owner, self.sid, regen["id"], "new answer", [])
        messages = self.repo.list_messages(self.owner, self.sid)
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[-1]["content"], "new answer")


if __name__ == "__main__":
    unittest.main()
