"""The opt-in memory toggle is backed by SQLite, not Streamlit session_state."""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modules.managed_ui import ModelSettings
from modules.repository import Repository


class MemoryPageTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Repository(Path(self.temp.name) / "ui.sqlite3")
        self.owner = self.repo.register("tester", "example-password")
        self.kb_id = self.repo.create_base(self.owner, "测试库", "enterprise", "", "fake", "company",
                                           "https://example.invalid/v1")
        self.repo.create_session(self.owner, self.kb_id)
        self.settings = ModelSettings("https://example.invalid/v1", "fake", "fake", "fake", "company")
        self.patches = [
            patch("modules.repository.DB_PATH", self.repo.path),
            patch("modules.auth_ui.require_user", return_value=self.repo.get_user(self.owner)),
            patch("modules.managed_ui.render_model_settings", return_value=self.settings),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_opt_in_and_opt_out_persist_across_reload(self):
        page = ROOT / "src" / "Home.py"
        app = AppTest.from_file(str(page), default_timeout=20).run().switch_page("app_pages/chat.py").run()
        self.assertEqual(len(app.exception), 0)
        toggle_key = f"rag_memory_on_{self.owner}_{self.kb_id}"
        self.assertFalse(self.repo.memory_enabled(self.owner, self.kb_id))
        app.get_by_key(toggle_key).set_value(True).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(Repository(self.repo.path).memory_enabled(self.owner, self.kb_id))
        # A fresh browser session reads consent from the database.
        fresh = AppTest.from_file(str(page), default_timeout=20).run().switch_page("app_pages/chat.py").run()
        self.assertEqual(len(fresh.exception), 0)
        self.assertTrue(fresh.get_by_key(toggle_key).value)
        fresh.get_by_key(toggle_key).set_value(False).run()
        self.assertFalse(self.repo.memory_enabled(self.owner, self.kb_id))


    def test_manual_add_review_and_confirmed_delete(self):
        self.repo.set_memory_enabled(self.owner, self.kb_id, True)
        index = MagicMock()
        with patch("modules.long_term_memory.memory_store", return_value=index):
            app = AppTest.from_file(str(ROOT / "src" / "Home.py"), default_timeout=20).run()
            app = app.switch_page("app_pages/chat.py").run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(app.get_by_key(f"rag_memory_on_{self.owner}_{self.kb_id}").value)
            app.get_by_key(f"rag_memory_input_{self.kb_id}").set_value("我希望长期从事 Python 后端开发")
            app.get_by_key(f"rag_memory_submit_{self.kb_id}").click().run()
            self.assertEqual(len(app.exception), 0)
            items = self.repo.list_memories(self.owner, self.kb_id)
            self.assertEqual(len(items), 1, f"errors: {[x.value for x in app.error]}; warnings: {[x.value for x in app.warning]}")
            self.assertEqual(items[0]["quote"], "我希望长期从事 Python 后端开发")
            self.assertTrue(items[0]["indexed"])
            memory_id = items[0]["id"]
            app.get_by_key(f"rag_memory_confirm_{self.kb_id}_{memory_id}").set_value(True).run()
            app.get_by_key(f"rag_memory_delete_{self.kb_id}").click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertEqual(self.repo.list_memories(self.owner, self.kb_id), [])
            index.delete.assert_called_once_with(ids=[memory_id])


if __name__ == "__main__":
    unittest.main()
