"""Regression tests for evidence-first English-paper retrieval from Chinese questions."""

import unittest
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from src.modules.chatbot import Chatbot
from src.modules.paper_evidence import keyword_query, paper_intents, section_evidence
from src.modules.retrieval import HybridRetriever


QUESTION = "这篇论文主要讲了什么，用了什么方法？"


def example_chunks(document_id="doc-1"):
    rows = []
    for i in range(24):
        text = f"Ordinary background note {i}, unrelated to the author's method."
        if i == 0:
            text = "Abstract. This paper proposes a new adaptive scheduler for distributed tasks."
        if i == 5:
            text = "The main contri- butions of this paper: a new scheduling model."
        if i == 11:
            text = "4.3 Our proposed method: reward function design, Actor and Critic policy training."
        if i == 14:
            text = "Algorithm 1: Deployment policy. Require: task state. Ensure: final policy. 1: Update Actor."
        if i == 18:
            text = "Experimental results show this method outperforms the baseline in a simulation."
        if i == 22:
            text = "References. [1] Author, Proposed method and Algorithm 1, 2024."
        rows.append({"id": f"{document_id}-{i}", "document_id": document_id,
                     "source": f"{document_id}.pdf", "location": f"第 {i+1} 页", "version": 1,
                     "content": text})
    return rows


class PaperEvidenceTests(unittest.TestCase):
    def test_chinese_overview_reserves_original_abstract_contributions_method_algorithm_results(self):
        rows = example_chunks()
        found = section_evidence(QUESTION, rows, 6)
        self.assertEqual([doc.metadata["chunk_id"] for doc in found],
                         ["doc-1-0", "doc-1-5", "doc-1-11", "doc-1-14", "doc-1-18"])
        self.assertTrue(all(doc.page_content == rows[int(doc.metadata["chunk_id"].split("-")[-1])]["content"]
                            for doc in found))
        self.assertIn("algorithm", keyword_query(QUESTION))
        self.assertEqual(paper_intents("公司的年假有几天？"), ())
        self.assertEqual(section_evidence("公司的年假有几天？", rows, 6), [])

    def test_no_cross_document_section_completion_and_no_reference_fill(self):
        self.assertEqual(section_evidence(QUESTION, example_chunks("a") + example_chunks("b"), 6), [])
        self.assertNotIn("doc-1-22", [doc.metadata["chunk_id"]
                                       for doc in section_evidence(QUESTION, example_chunks(), 8)])

    def test_hybrid_rescues_methods_even_when_semantic_returns_references(self):
        rows = example_chunks()
        repo = MagicMock()
        repo.get_base.return_value = {"id": "kb"}
        repo.list_chunks.return_value = rows
        kb = {"id": "kb", "kind": "paper", "base_url": "https://example.invalid/v1"}
        bad = [Document(page_content=row["content"], metadata={
            "chunk_id": row["id"], "document_id": row["document_id"],
            "source": row["source"], "location": row["location"], "version": 1,
        }) for row in (rows[22], rows[23])]
        store = MagicMock()
        store.similarity_search.return_value = bad
        with patch("src.modules.retrieval.vector_store", return_value=store):
            hybrid = HybridRetriever(repo, "owner", kb, "fake", "fake")
            found = hybrid.similarity_search(QUESTION, k=6)
            self.assertEqual([doc.metadata["chunk_id"] for doc in found[:5]],
                             ["doc-1-0", "doc-1-5", "doc-1-11", "doc-1-14", "doc-1-18"])
            self.assertEqual(len(found), 6)
            vector = HybridRetriever(repo, "owner", kb, "fake", "fake", mode="vector")
            self.assertEqual([doc.metadata["chunk_id"] for doc in vector.similarity_search(QUESTION, k=6)],
                             ["doc-1-22", "doc-1-23"])
            scoped = HybridRetriever(repo, "owner", kb, "fake", "fake", document_ids=["doc-1"])
            scoped.similarity_search(QUESTION, k=6)
            self.assertEqual(store.similarity_search.call_args.kwargs["filter"],
                             {"document_id": {"$in": ["doc-1"]}})

    def test_paper_task_retrieval_always_keeps_original_question(self):
        vectors = MagicMock()
        vectors.similarity_search.side_effect = lambda query, k: [Document(
            page_content=query, metadata={"chunk_id": query, "source": "p.pdf"})]
        with patch("src.modules.chatbot.ChatOpenAI"):
            bot = Chatbot("fake", vectors, "fake", top_k=4, base_url="https://example.invalid/v1")
        results = bot.retrieve("original method question", ("abstract summary", "algorithm details"))
        self.assertEqual([doc.page_content for doc in results],
                         ["original method question", "abstract summary", "algorithm details"])
        self.assertEqual(vectors.similarity_search.call_args_list[0].args[0], "original method question")


if __name__ == "__main__":
    unittest.main()