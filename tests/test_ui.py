"""Smoke-test both document embedding routes without making model requests."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from src.modules.model_config import OPENAI_BASE_URL
from src.modules.reading_tasks import READING_TASKS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class DocumentPageTests(unittest.TestCase):
    def setUp(self):
        # Never read a developer's actual keys or local credentials during UI tests.
        self.env_patch = patch.dict(os.environ, {"RAG_DEFAULT_API_KEY": "", "OPENAI_API_KEY": ""})
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        self.store_patch = patch("modules.credential_ui.CredentialStore")
        self.store = self.store_patch.start().return_value
        self.addCleanup(self.store_patch.stop)
        self.store.load.return_value = None
        self.store.has.return_value = False

    def test_company_and_local_embedding_choices_render(self):
        page = ROOT / "src" / "pages" / "1_📄文档问答.py"
        app = AppTest.from_file(str(page), default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[-1].options[:3], [
            "text-embedding-3-small", "text-embedding-3-large", "text-embedding-v4",
        ])
        self.assertEqual(app.selectbox[-1].value, "text-embedding-v4")
        app.selectbox[-1].select("text-embedding-3-small").run()
        self.assertTrue(any("text-embedding-v4" in item.value for item in app.warning))
        app.selectbox[0].select("OpenAI").run()
        self.assertEqual(app.selectbox[-1].value, "text-embedding-3-small")
        app.selectbox[0].select(app.selectbox[0].options[0]).run()

        app.segmented_control[0].set_value("本地 Ollama").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[-1].options[:2], ["bge-m3", "nomic-embed-text"])

    def test_reading_tasks_visible_but_disabled_before_indexing(self):
        page = ROOT / "src" / "pages" / "1_\U0001f4c4\u6587\u6863\u95ee\u7b54.py"
        app = AppTest.from_file(str(page), default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        for task in READING_TASKS:
            buttons = [button for button in app.button if button.label == task.title]
            self.assertEqual(len(buttons), 1)
            self.assertTrue(buttons[0].disabled)
        self.assertEqual(len(app.chat_input), 0)

    def test_research_task_runs_only_after_index_and_shows_sources(self):
        from src.modules.chatbot import ChatResult
        from src.modules.embedder import UploadedDocument, index_id
        from langchain_core.documents import Document
        from src.modules.model_config import COMPANY_BASE_URL

        page = ROOT / "src" / "pages" / "1_📄文档问答.py"
        file = SimpleNamespace(name="paper.txt", getvalue=lambda: b"research evidence")
        indexed = SimpleNamespace(
            collection_id=index_id([UploadedDocument("paper.txt", b"research evidence")]),
            file_count=1, chunk_count=1, vectorstore=object(), cached=False,
        )
        citation = Document(page_content="research evidence", metadata={"source": "paper.txt", "location": "page 1"})
        self.store.load.return_value = "fake-stored-key"
        self.store.has.return_value = True
        with patch("streamlit.file_uploader", return_value=[file]), \
                patch("modules.embedder.build_index", return_value=indexed) as build, \
                patch("modules.chatbot.Chatbot") as chatbot:
            chatbot.return_value.conversational_chat.return_value = ChatResult("Grounded answer [1]", [citation])
            app = AppTest.from_file(str(page), default_timeout=20).run()
            self.assertEqual(app.text_input[0].value, "")
            app.get_by_key("rag_build").click().run()
            self.assertEqual(len(app.exception), 0)
            build.assert_called_once()
            self.assertEqual(build.call_args.args[1], "fake-stored-key")
            self.assertEqual(build.call_args.kwargs["base_url"], COMPANY_BASE_URL)
            task = READING_TASKS[0]
            button = next(button for button in app.button if button.label == task.title)
            self.assertFalse(button.disabled)
            button.click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(chatbot.return_value.conversational_chat.call_args.kwargs["search_queries"],
                             task.search_queries)
            self.assertEqual(chatbot.return_value.conversational_chat.call_args.args[1], [])
            self.assertTrue(any("Grounded answer" in item.value for item in app.markdown))

    def test_changing_service_does_not_carry_over_default_key(self):
        page = ROOT / "src" / "pages" / "1_📄文档问答.py"
        app = AppTest.from_file(str(page), default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        app.text_input[0].set_value("fake-default-key").run()
        app.selectbox[0].select("OpenAI").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.text_input[0].value, "")
        self.assertIn(OPENAI_BASE_URL, " ".join(item.value for item in app.caption))
        self.assertEqual(app.selectbox[1].options, ["自定义（填写模型 ID）"])
        app.selectbox[0].select("自定义接口").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.text_input[0].value, "")
        self.assertEqual(app.text_input[1].value, "")

    def test_custom_url_change_does_not_reuse_key(self):
        page = ROOT / "src" / "pages" / "1_📄文档问答.py"
        app = AppTest.from_file(str(page), default_timeout=20).run()
        app.selectbox[0].select("自定义接口").run()
        app.text_input[0].set_value("https://one.example/v1").run()
        app.text_input[1].set_value("fake-one-key").run()
        self.assertEqual(app.text_input[1].value, "fake-one-key")
        app.text_input[0].set_value("https://two.example/v1").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.text_input[1].value, "")

    def test_explicit_save_survives_new_browser_session_and_delete(self):
        saved = {}
        self.store.load.side_effect = lambda scope: saved.get(scope)
        self.store.has.side_effect = lambda scope: scope in saved
        self.store.save.side_effect = lambda scope, key: saved.__setitem__(scope, key)
        self.store.delete.side_effect = lambda scope: saved.pop(scope, None)
        page = ROOT / "src" / "pages" / "1_📄文档问答.py"
        app = AppTest.from_file(str(page), default_timeout=20).run()
        app.text_input[0].set_value("fake-test-key").run()
        next(button for button in app.button if button.label == "保存到本机").click().run()
        self.assertEqual(len(saved), 1)
        self.assertEqual(len(app.exception), 0)

        fresh = AppTest.from_file(str(page), default_timeout=20).run()
        self.assertEqual(fresh.text_input[0].value, "")
        self.assertNotIn("fake-test-key", " ".join(item.value for item in fresh.caption))
        self.assertEqual(len(saved), 1)
        next(button for button in fresh.button if button.label == "删除已保存").click().run()
        again = AppTest.from_file(str(page), default_timeout=20).run()
        self.assertEqual(again.text_input[0].value, "")
        self.assertEqual(len(again.exception), 0)


if __name__ == "__main__":
    unittest.main()
