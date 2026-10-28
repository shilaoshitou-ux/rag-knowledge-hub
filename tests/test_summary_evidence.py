"""Regression tests for evidence-rich summaries without inventing document facts."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.modules.chatbot import ANSWER_PROMPT, Chatbot, ChatResult, overview_search_queries
from src.modules.retrieval import HybridRetriever
from src.modules.workflow import answer_turn, stream_answer_turn


QUESTION = "我在示例项目做的问题你给我总结一下"
BASE = "https://example.invalid/v1"


def excerpt(number, text):
    return Document(page_content=text, metadata={
        "chunk_id": f"doc-{number}", "document_id": "doc", "version": 2,
        "source": "work.md", "location": f"段落 {number}",
    })


class SummaryEvidenceTests(unittest.TestCase):
    def test_summary_intent_is_bounded_and_specific_question_stays_specific(self):
        queries = overview_search_queries(QUESTION)
        self.assertEqual(len(queries), 3)
        self.assertTrue(all("示例项目" in query for query in queries))
        self.assertTrue(any("排查过程" in query for query in queries))
        for question in ("公司的年假有几天？", "Transformer 的缺点是什么？", "总结一下这个项目的缺点"):
            self.assertEqual(overview_search_queries(question), ())

    def test_multi_angle_retrieval_keeps_original_hits_then_completes_details(self):
        snippets = [excerpt(i, f"示例项目 实际资料第 {i} 段") for i in range(12)]
        vectors = MagicMock()
        def ranked(query, k):
            if query == QUESTION or query.startswith("公司的"):
                return snippets[:k]
            if "背景" in query:
                return [snippets[0], snippets[8], snippets[1]]
            if "具体工作" in query:
                return [snippets[0], snippets[9], snippets[2]]
            return [snippets[0], snippets[10], snippets[11]]
        vectors.similarity_search.side_effect = ranked
        with patch("src.modules.chatbot.ChatOpenAI"):
            bot = Chatbot("fake", vectors, "key", top_k=10, base_url=BASE)
        found = bot.retrieve(QUESTION, overview_search_queries(QUESTION))
        self.assertEqual(len(found), 10)
        self.assertEqual(len({doc.metadata["chunk_id"] for doc in found}), 10)
        self.assertIn("doc-8", [doc.metadata["chunk_id"] for doc in found])
        self.assertIn("doc-9", [doc.metadata["chunk_id"] for doc in found])
        self.assertEqual(vectors.similarity_search.call_args_list[0].kwargs["k"], 10)
        self.assertEqual(bot.retrieve("公司的年假有几天？"), snippets[:10])
        self.assertEqual(vectors.similarity_search.call_args.kwargs["k"], 10)

    def test_chat_turn_wires_expansion_and_keeps_explicit_paper_queries(self):
        repo = MagicMock()
        repo.get_session.return_value = {"kb_id": "kb", "summary": "", "summary_upto": 0}
        repo.list_messages.return_value = []
        repo.memory_enabled.return_value = False
        kb = {"id": "kb", "kind": "enterprise", "base_url": BASE}
        with patch("src.modules.workflow.HybridRetriever") as retriever, patch("src.modules.workflow.Chatbot") as bot:
            bot.return_value.rewrite_question.side_effect = lambda question, *_: question
            bot.return_value.retrieve.return_value = [excerpt(1, "原文：实际排查步骤")]
            bot.return_value.answer.return_value = ChatResult("有证据的答案 [1]", [])
            result, _ = answer_turn(repo, "owner", kb, "sid", QUESTION, "key", "model", top_k=6)
            self.assertEqual(result.sources[0].metadata["version"], 2)
            self.assertEqual(bot.call_args.args[3], 10)
            self.assertEqual(len(bot.return_value.retrieve.call_args.args[1]), 3)
            repo.append_turn.assert_called_once()
            self.assertEqual(repo.append_turn.call_args.args[4][0]["chunk_id"], "doc-1")
            bot.reset_mock()
            answer_turn(repo, "owner", kb, "sid", "这篇论文主要讲了什么？", "key", "model",
                        top_k=8, search_queries=("original method",))
            self.assertEqual(bot.call_args.args[3], 8)
            self.assertEqual(bot.return_value.retrieve.call_args.args[1], ("original method",))

    def test_streaming_chat_turn_expands_summaries_but_not_fact_questions(self):
        repo = MagicMock()
        repo.get_session.return_value = {"kb_id": "kb", "summary": "", "summary_upto": 0}
        repo.list_messages.return_value = []
        repo.memory_enabled.return_value = False
        kb = {"id": "kb", "kind": "enterprise", "base_url": BASE}
        documents = [excerpt(1, "合成证据：这里有步骤")]
        updates = [("updates", {"retrieve": {"documents": documents}}),
                   ("updates", {"answer": {"answer": "测试返回 [1]"}})]
        with patch("src.modules.workflow.HybridRetriever"), patch("src.modules.workflow.Chatbot") as bot, \
             patch("src.modules.workflow.make_graph") as graph:
            graph.return_value.stream.return_value = updates
            self.assertEqual(list(stream_answer_turn(repo, "owner", kb, "sid", QUESTION,
                                                      "key", "model", top_k=6)), ["测试返回 [1]"])
            self.assertEqual(bot.call_args.args[3], 10)
            self.assertEqual(len(graph.return_value.stream.call_args.args[0]["search_queries"]), 3)
            self.assertEqual(repo.append_turn.call_args.args[4][0]["chunk_id"], "doc-1")
            bot.reset_mock()
            list(stream_answer_turn(repo, "owner", kb, "sid", "公司的年假有几天？",
                                    "key", "model", top_k=6))
            self.assertEqual(bot.call_args.args[3], 6)
            self.assertEqual(graph.return_value.stream.call_args.args[0]["search_queries"], ())

    def test_enterprise_hybrid_uses_original_owner_scoped_chunks_and_larger_candidate_pool(self):
        rows = [{"id": f"doc-{i}", "document_id": "doc", "version": 2,
                 "source": "work.md", "location": f"段落 {i}",
                 "content": f"示例项目 工作解决方案结果第 {i} 条"} for i in range(25)]
        repo = MagicMock()
        repo.get_base.return_value = {"id": "kb"}
        repo.list_chunks.return_value = rows
        store = MagicMock()
        store.similarity_search.return_value = [excerpt(i, rows[i]["content"]) for i in range(25)]
        kb = {"id": "kb", "kind": "enterprise", "base_url": BASE}
        with patch("src.modules.retrieval.vector_store", return_value=store):
            result = HybridRetriever(repo, "owner", kb, "key", "model").similarity_search(QUESTION, k=10)
        self.assertEqual(len(result), 10)
        self.assertEqual(len({doc.metadata["chunk_id"] for doc in result}), 10)
        self.assertTrue(all(doc.metadata["document_id"] == "doc" and doc.metadata["version"] == 2 for doc in result))
        repo.list_chunks.assert_called_once_with("owner", "kb", None)
        self.assertEqual(store.similarity_search.call_args.kwargs["filter"], {"document_id": {"$in": ["doc"]}})

    def test_prompt_requests_detail_but_not_unfounded_claims(self):
        prompt = ANSWER_PROMPT.format_messages(summary="", chat_history=[], question=QUESTION,
                                                context="[1] work.md（段落 1）\n只有章节标题")
        system = prompt[0].content
        self.assertIn("在证据足够时尽量全面", system)
        self.assertIn("不要从标题推断原因", system)
        self.assertIn("文档未提及的关键维度", system)


if __name__ == "__main__":
    unittest.main()
