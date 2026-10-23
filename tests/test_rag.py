import unittest
import gc
import json

import httpx
import sys
from io import BytesIO
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.embeddings import FakeEmbeddings
from pypdf import PdfWriter

from src.modules.chatbot import Chatbot
from src.modules.embedder import (
    MAX_FILE_MB, ONLINE_BATCH_SIZE, EmbeddingBatchError, UploadedDocument,
    build_index, index_id, make_embeddings, parse_document,
)
from src.modules.model_config import (
    CHAT_MODELS, COMPANY_BASE_URL, COMPANY_EMBEDDING_MODELS, LOCAL_OLLAMA_BASE_URL,
    OPENAI_BASE_URL, DEFAULT_EMBEDDING_MODEL, DEFAULT_SERVICE, OPENAI_SERVICE, CUSTOM_SERVICE,
    clean_model_id, safe_error, service_base_url, validate_base_url,
)


class ParserTests(unittest.TestCase):
    def test_txt_utf8_and_chinese_fallback(self):
        docs = parse_document(UploadedDocument(r"folder\说明.txt", "测试内容".encode("gb18030")))
        self.assertEqual(docs[0].page_content, "测试内容")
        self.assertEqual(docs[0].metadata["source"], "说明.txt")

    def test_csv_has_row_references(self):
        docs = parse_document(UploadedDocument("people.csv", b"name,age\nAda,31\nBob,28\n"))
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0].metadata["location"], "第 2 行")
        self.assertIn("name: Ada", docs[0].page_content)

    def test_upload_limit_is_100mb_and_validated_by_parser(self):
        self.assertEqual(MAX_FILE_MB, 100)
        with patch("src.modules.embedder.MAX_FILE_BYTES", 10):
            with self.assertRaisesRegex(ValueError, "100 MB"):
                parse_document(UploadedDocument("large.txt", b"x" * 11))

    def test_online_embedding_payload_is_text_with_float_encoding(self):
        observed = {}

        def handler(request):
            payload = json.loads(request.content)
            observed.update(payload)
            return httpx.Response(200, json={
                "data": [{"object": "embedding", "index": i, "embedding": [0.1, 0.2, 0.3]}
                         for i in range(len(payload["input"]))],
                "model": "text-embedding-3-small", "object": "list",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            })

        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            from langchain_openai import OpenAIEmbeddings as RealEmbeddings
            with patch("src.modules.embedder.OpenAIEmbeddings",
                       side_effect=lambda **kw: RealEmbeddings(http_client=client, **kw)):
                embedding = make_embeddings("company", "text-embedding-3-small",
                                            "fake-test-key", "https://example.invalid/v1")
                self.assertEqual(len(embedding.embed_documents(["harmless sample"])[0]), 3)
        self.assertEqual(observed["input"], ["harmless sample"])
        self.assertEqual(observed["encoding_format"], "float")

    def test_scanned_pdf_explains_ocr(self):
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        buffer = BytesIO()
        writer.write(buffer)
        with self.assertRaisesRegex(ValueError, "OCR"):
            parse_document(UploadedDocument("blank.pdf", buffer.getvalue()))

    def test_index_key_changes_when_same_name_content_changes(self):
        first = index_id([UploadedDocument("a.txt", b"old")])
        second = index_id([UploadedDocument("a.txt", b"new")])
        self.assertNotEqual(first, second)
        self.assertEqual(first, index_id([UploadedDocument("a.txt", b"old")]))
        self.assertNotEqual(first, index_id([UploadedDocument("a.txt", b"old")], base_url="https://company.example/v1"))

    @patch("src.modules.embedder.OpenAIEmbeddings")
    @patch("src.modules.embedder.Chroma")
    def test_cache_does_not_embed_again(self, chroma, embeddings):
        store = chroma.return_value
        store.get.side_effect = lambda ids, include: {"ids": ids}
        with TemporaryDirectory() as directory:
            result = build_index([UploadedDocument("a.txt", b"Hello world")], "test-key", directory, embedding_model="company-embed", base_url=COMPANY_BASE_URL)
        embeddings.assert_called_once_with(model="company-embed", api_key="test-key",
                                           base_url=COMPANY_BASE_URL, chunk_size=ONLINE_BATCH_SIZE, check_embedding_ctx_length=False,
                                           model_kwargs={"encoding_format": "float"})
        self.assertTrue(result.cached)
        self.assertEqual(result.chunk_count, 1)
        store.add_documents.assert_not_called()

    @patch("src.modules.embedder.OpenAIEmbeddings")
    @patch("src.modules.embedder.Chroma")
    def test_ollama_uses_loopback_without_company_embedding_key(self, chroma, company_embeddings):
        local_embeddings = MagicMock()
        chroma.return_value.get.side_effect = lambda ids, include: {"ids": ids}
        with patch.dict(sys.modules, {"langchain_ollama": MagicMock(OllamaEmbeddings=local_embeddings)}):
            with TemporaryDirectory() as directory:
                result = build_index(
                    [UploadedDocument("a.txt", b"Hello world")], "", directory,
                    embedding_model="bge-m3", embedding_provider="ollama",
                )
        local_embeddings.assert_called_once_with(model="bge-m3", base_url=LOCAL_OLLAMA_BASE_URL)
        company_embeddings.assert_not_called()
        self.assertTrue(result.cached)
        self.assertIs(chroma.call_args.kwargs["embedding_function"], local_embeddings.return_value)

    def test_different_embedding_providers_have_different_collections(self):
        files = [UploadedDocument("a.txt", b"Hello world")]
        self.assertNotEqual(index_id(files, "bge-m3", embedding_provider="company"),
                            index_id(files, "bge-m3", embedding_provider="ollama"))
        self.assertNotEqual(index_id(files, "bge-m3", embedding_provider="ollama"),
                            index_id(files, "nomic-embed-text", embedding_provider="ollama"))

    @patch("src.modules.embedder.OpenAIEmbeddings")
    @patch("src.modules.embedder.Chroma")
    def test_large_document_is_submitted_in_small_online_batches(self, chroma, embeddings):
        store = chroma.return_value
        store.get.side_effect = lambda ids, include: {"ids": []}
        document = UploadedDocument("long.txt", ("alpha beta gamma. " * 1500).encode())
        with TemporaryDirectory() as directory:
            result = build_index([document], "test-key", directory)
        self.assertGreater(result.chunk_count, ONLINE_BATCH_SIZE)
        self.assertGreater(store.add_documents.call_count, 1)
        self.assertTrue(all(len(call.kwargs["ids"]) <= ONLINE_BATCH_SIZE
                            for call in store.add_documents.call_args_list))
        embeddings.assert_called_once_with(model=DEFAULT_EMBEDDING_MODEL, api_key="test-key",
                                           base_url=COMPANY_BASE_URL, chunk_size=ONLINE_BATCH_SIZE, check_embedding_ctx_length=False,
                                           model_kwargs={"encoding_format": "float"})

    @patch("src.modules.embedder.OpenAIEmbeddings")
    @patch("src.modules.embedder.Chroma")
    def test_gateway_failure_shows_status_and_progress_not_document(self, chroma, embeddings):
        class GatewayError(Exception):
            status_code = 503

        store = chroma.return_value
        store.get.side_effect = lambda ids, include: {"ids": []}
        store.add_documents.side_effect = GatewayError("secret document body from gateway")
        with TemporaryDirectory() as directory, self.assertRaises(EmbeddingBatchError) as captured:
            build_index([UploadedDocument("notes.txt", b"secret document body from gateway")],
                        "test-key", directory)
        self.assertEqual(captured.exception.status_code, 503)
        self.assertIn("HTTP 503", str(captured.exception))
        self.assertNotIn("secret document body", str(captured.exception))

    @patch("src.modules.embedder.OpenAIEmbeddings")
    def test_real_chroma_persists_without_reembedding(self, embeddings):
        embeddings.return_value = FakeEmbeddings(size=8)
        files = [UploadedDocument("notes.txt", b"LangChain and Chroma are used for retrieval.")]
        with TemporaryDirectory() as directory:
            first = build_index(files, "test-key", directory)
            self.assertFalse(first.cached)
            self.assertEqual(first.vectorstore.get(include=["metadatas"])["metadatas"][0]["source"], "notes.txt")
            second = build_index(files, "test-key", directory)
            self.assertTrue(second.cached)
            self.assertEqual(len(second.vectorstore.get(include=[])["ids"]), 1)
            del first, second
            from chromadb.api.client import SharedSystemClient
            SharedSystemClient.clear_system_cache()
            gc.collect()

    def test_duplicate_filename_rejected(self):
        with self.assertRaisesRegex(ValueError, "重名"):
            build_index([UploadedDocument("same.txt", b"a"), UploadedDocument("same.txt", b"b")], "key")


class ChatTests(unittest.TestCase):
    @patch("src.modules.chatbot.ChatOpenAI")
    def test_rewrites_followup_then_retrieves_and_cites(self, model):
        model.return_value = FakeListChatModel(responses=["阿达的年龄", "阿达 31 岁 [1]"])
        store = MagicMock()
        store.similarity_search.return_value = [Document(
            page_content="name: Ada, age: 31", metadata={"source": "people.csv", "location": "第 2 行"}
        )]
        bot = Chatbot("gpt-6-sol", store, "test-key", 3, base_url=COMPANY_BASE_URL)
        self.assertEqual(model.call_args.kwargs["base_url"], COMPANY_BASE_URL)
        self.assertNotIn("temperature", model.call_args.kwargs)
        result = bot.conversational_chat("她几岁？", [("阿达是谁？", "阿达是工程师")])
        store.similarity_search.assert_called_once_with("阿达的年龄", k=3)
        self.assertEqual(result.answer, "阿达 31 岁 [1]")
        self.assertEqual(result.sources[0].metadata["source"], "people.csv")

    @patch("src.modules.chatbot.ChatOpenAI")
    def test_no_documents_no_answer_model_call(self, model):
        store = MagicMock()
        store.similarity_search.return_value = []
        result = Chatbot("test-model", store, "test-key").conversational_chat("问题")
        self.assertIn("没有检索到", result.answer)
        model.return_value.invoke.assert_not_called()


    @patch("src.modules.chatbot.ChatOpenAI")
    def test_reading_task_diversifies_evidence_without_duplicates(self, model):
        from src.modules.reading_tasks import READING_TASKS

        model.return_value = FakeListChatModel(responses=["Summary [1] [2]"])
        documents = [
            Document(page_content=f"Evidence {i}", metadata={"source": "paper.pdf", "location": f"page {i}"})
            for i in range(4)
        ]
        store = MagicMock()
        store.similarity_search.side_effect = [
            [documents[0], documents[3]],
            [documents[0], documents[1]],
            [documents[2]],
            [documents[0]],
        ]
        task = READING_TASKS[0]
        result = Chatbot("gpt-6-sol", store, "test-key", 4).conversational_chat(
            task.question, search_queries=task.search_queries,
        )
        self.assertEqual(store.similarity_search.call_count, 4)
        self.assertEqual(store.similarity_search.call_args_list[0].args[0], task.question)
        self.assertEqual([doc.page_content for doc in result.sources],
                         ["Evidence 0", "Evidence 2", "Evidence 3", "Evidence 1"])
        self.assertEqual([call.kwargs["k"] for call in store.similarity_search.call_args_list], [2, 2, 2, 2])
        self.assertEqual(result.answer, "Summary [1] [2]")
        self.assertEqual(len(READING_TASKS), 6)
        self.assertEqual(len({task.slug for task in READING_TASKS}), 6)



class ModelConfigTests(unittest.TestCase):
    def test_public_choices_and_custom_model(self):
        self.assertIn("gpt-6-sol", CHAT_MODELS)
        self.assertIn("gpt-5.6-terra", CHAT_MODELS)
        self.assertEqual(COMPANY_EMBEDDING_MODELS, ("text-embedding-3-small", "text-embedding-3-large", "text-embedding-v4"))
        self.assertEqual(clean_model_id(" company-gpt-5x "), "company-gpt-5x")
        with self.assertRaises(ValueError):
            clean_model_id("bad model")

    def test_gateway_requires_https_and_no_credentials_in_url(self):
        self.assertEqual(validate_base_url("https://company.example/v1/"), "https://company.example/v1")
        self.assertEqual(validate_base_url("http://localhost:8000/v1"), "http://localhost:8000/v1")
        for value in ("http://company.example/v1", "https://key@company.example/v1", "https://company.example/v1?key=secret"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_base_url(value)

    def test_selectable_services_and_no_implicit_remote_key_reuse(self):
        self.assertEqual(service_base_url(DEFAULT_SERVICE), COMPANY_BASE_URL)
        self.assertEqual(service_base_url(OPENAI_SERVICE), OPENAI_BASE_URL)
        self.assertEqual(service_base_url(CUSTOM_SERVICE, "https://proxy.example/v1/"), "https://proxy.example/v1")
        with self.assertRaises(ValueError):
            service_base_url(CUSTOM_SERVICE, "http://remote.example/v1")
        with self.assertRaises(ValueError):
            service_base_url("invalid")

    @patch("src.modules.chatbot.ChatOpenAI")
    @patch("src.modules.embedder.OpenAIEmbeddings")
    @patch("src.modules.embedder.Chroma")
    def test_personal_endpoint_used_by_chat_and_embedding(self, chroma, embeddings, model):
        store = chroma.return_value
        store.get.side_effect = lambda ids, include: {"ids": ids}
        with TemporaryDirectory() as directory:
            build_index([UploadedDocument("a.txt", b"hello")], "personal-key", directory,
                        embedding_model="text-embedding-3-small", base_url=OPENAI_BASE_URL)
        embeddings.assert_called_once_with(model="text-embedding-3-small", api_key="personal-key",
                                           base_url=OPENAI_BASE_URL, chunk_size=ONLINE_BATCH_SIZE, check_embedding_ctx_length=False,
                                           model_kwargs={"encoding_format": "float"})
        Chatbot("personal-model", store, "personal-key", base_url=OPENAI_BASE_URL)
        self.assertEqual(model.call_args.kwargs["base_url"], OPENAI_BASE_URL)
        self.assertEqual(model.call_args.kwargs["api_key"], "personal-key")

    def test_invalid_remote_endpoints_rejected_by_clients(self):
        for url in ("http://remote.example/v1", "https://key@remote.example/v1", "https://remote.example/v1?token=secret"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                Chatbot("model", object(), "test-key", base_url=url)
            with self.subTest(url=url), self.assertRaises(ValueError):
                build_index([UploadedDocument("a.txt", b"hello")], "test-key", base_url=url)

    def test_error_redacts_key(self):
        self.assertNotIn("company-secret", safe_error(ValueError("error company-secret"), "company-secret"))


if __name__ == "__main__":
    unittest.main()
