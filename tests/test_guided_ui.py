"""UI journey regressions use a temporary SQLite database and no gateway calls."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modules.managed_ui import ModelSettings
from modules.repository import Repository


class GuidedPagesTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Repository(Path(self.temp.name) / "ui.sqlite3")
        self.owner = self.repo.register("tester", "strong-test-password")
        self.settings = ModelSettings("https://example.invalid/v1", "fake-key", "fake-chat", "fake-embed", "company")
        for item in (
            patch("modules.repository.DB_PATH", self.repo.path),
            patch("modules.auth_ui.require_user", return_value=self.repo.get_user(self.owner)),
            patch("modules.managed_ui.render_model_settings", return_value=self.settings),
        ):
            item.start()
            self.addCleanup(item.stop)

    def page(self, name):
        app = AppTest.from_file(str(ROOT / "src" / "Home.py"), default_timeout=20).run()
        app = app.switch_page(f"app_pages/{name}.py").run()
        self.assertEqual(len(app.exception), 0)
        return app

    def ready_document(self, kb):
        doc, _ = self.repo.create_document(self.owner, kb, "synthetic.txt", "hash", 10, "synthetic")
        self.repo.complete_document(self.owner, doc["id"], [{"id": "synthetic-chunk", "location": "段落 1", "content": "test"}])

    def test_dashboard_guides_from_account_state(self):
        app = self.page("dashboard")
        self.assertEqual(app.get_by_key("rag_next_action").label, "创建知识库")
        kb = self.repo.create_base(self.owner, "论文", "paper", "", "fake-embed", "company",
                                   self.settings.base_url)
        app.run()
        self.assertEqual(app.get_by_key("rag_next_action").label, "去上传资料")
        app.get_by_key("rag_next_action").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state["rag_managed_kb"], kb)
        self.assertIsNotNone(app.get_by_key(f"enqueue_{kb}"))
        self.ready_document(kb)
        app = self.page("dashboard")
        app.run()
        self.assertEqual(app.get_by_key("rag_next_action").label, "开始问答")
        app.get_by_key("rag_next_action").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertIsNotNone(app.get_by_key("managed_new_session"))

    def test_chat_and_upload_empty_states_are_actionable(self):
        kb = self.repo.create_base(self.owner, "论文", "paper", "", "fake-embed", "company",
                                   self.settings.base_url)
        chat = self.page("chat")
        self.assertTrue(chat.get_by_key("managed_new_session").disabled)
        self.assertTrue(any("先准备一份可检索资料" in item.value for item in chat.subheader))
        knowledge = self.page("knowledge")
        knowledge.get_by_key(f"manage_{kb}").click().run()
        self.assertEqual(len(knowledge.exception), 0)
        self.assertIsNotNone(knowledge.get_by_key(f"enqueue_{kb}"))
        self.assertTrue(any("从资料到答案" in item.value for item in knowledge.caption))
        self.ready_document(kb)
        chat = self.page("chat")
        self.assertFalse(chat.get_by_key("managed_new_session").disabled)
        self.assertFalse(any("开始一段新对话" == item.label for item in chat.button))
        self.assertTrue(any("资料已就绪" in item.value for item in chat.subheader))


if __name__ == "__main__":
    unittest.main()
