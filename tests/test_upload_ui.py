"""Upload UI regression: submission feedback must remain visible after clicking."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modules.managed_ui import ModelSettings
from modules.repository import Repository


class UploadPageTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Repository(Path(self.temp.name) / "ui.sqlite3")
        self.owner = self.repo.register("tester", "strong-test-password")
        self.user = self.repo.get_user(self.owner)
        self.settings = ModelSettings("https://example.invalid/v1", "test-key", "test-chat", "test-embed", "company")

    def app(self, *, selected=(), upload=None):
        patches = [
            patch("modules.repository.DB_PATH", self.repo.path),
            patch("modules.auth_ui.require_user", return_value=self.user),
            patch("modules.managed_ui.render_model_settings", return_value=self.settings),
            patch("streamlit.file_uploader", return_value=list(selected)),
        ]
        if upload is not None:
            patches.append(patch("modules.managed_index.queue_upload", side_effect=upload))
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)
        at = AppTest.from_file(str(ROOT / "src" / "Home.py"), default_timeout=20).run()
        return at.switch_page("app_pages/knowledge.py").run()

    def test_new_base_opens_upload_workspace(self):
        at = self.app()
        at.text_input[0].set_value("我的论文")
        at.selectbox[0].select("学术论文库")
        at.form_submit_button[0].click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(at.session_state["rag_managed_kb"])
        self.assertTrue(any("现在点击下方" in item.value for item in at.success))
        self.assertTrue(any("仅选中文件不会调用模型" in item.value for item in at.caption))
        self.assertFalse(at.get_by_key(f"enqueue_{at.session_state['rag_managed_kb']}").disabled)

    def test_submit_without_file_reports_missing_upload(self):
        kb_id = self.repo.create_base(self.owner, "我的论文", "paper", "", "test-embed", "company",
                                      self.settings.base_url)
        at = self.app()
        at.session_state["rag_managed_kb"] = kb_id
        at.run()
        at.get_by_key(f"enqueue_{kb_id}").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("服务端没有收到文件" in item.value for item in at.error))
        self.assertFalse(self.repo.list_documents(self.owner, kb_id))
    def test_upload_success_remains_visible(self):
        kb_id = self.repo.create_base(self.owner, "我的论文", "paper", "", "test-embed", "company",
                                      self.settings.base_url)
        sample = SimpleNamespace(name="paper.pdf", getvalue=lambda: b"%PDF-test")
        doc = {"name": "paper.pdf"}
        at = self.app(selected=[sample], upload=lambda *args: (doc, True))
        at.session_state["rag_managed_kb"] = kb_id
        at.run()
        at.get_by_key(f"enqueue_{kb_id}").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("已进入处理队列" in item.value for item in at.success))

    def test_upload_error_remains_visible(self):
        kb_id = self.repo.create_base(self.owner, "我的论文", "paper", "", "test-embed", "company",
                                      self.settings.base_url)
        sample = SimpleNamespace(name="paper.pdf", getvalue=lambda: b"%PDF-test")
        at = self.app(selected=[sample], upload=lambda *args: (_ for _ in ()).throw(ValueError("在线向量化需要密钥")))
        at.session_state["rag_managed_kb"] = kb_id
        at.run()
        at.get_by_key(f"enqueue_{kb_id}").click().run()
        self.assertEqual(len(at.exception), 0)
        self.assertTrue(any("在线向量化需要密钥" in item.value for item in at.error))


if __name__ == "__main__":
    unittest.main()