"""Password success enters the workspace in one submission, even before cookie exchange."""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modules.auth_ui import _exchange_html
from modules.repository import Repository


class LoginFlowTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.repo = Repository(Path(temp.name) / "app.sqlite3")
        self.owner = self.repo.register("first_click", "strong-password-123")
        mocked = patch("modules.repository.DB_PATH", self.repo.path)
        mocked.start()
        self.addCleanup(mocked.stop)

    def login(self, remember=True):
        app = AppTest.from_file(str(ROOT / "src" / "Home.py"), default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        app.get_by_key("rag_auth_name").set_value("first_click")
        app.get_by_key("rag_auth_password").set_value("strong-password-123")
        if not remember:
            app.get_by_key("rag_remember").set_value(False)
        app.get("form_submit_button")[0].click().run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_remembered_login_enters_workspace_after_one_click(self):
        app = self.login()
        self.assertEqual(app.session_state["rag_current_user"], self.owner)
        self.assertIsNotNone(app.get_by_key("rag_next_action"))
        self.assertFalse(any(item.value == "登录工作区" for item in app.title))
        with self.repo.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM auth_tickets").fetchone()[0], 1)
        self.assertIn("本次会话已登录", _exchange_html("test-ticket"))

    def test_session_only_login_enters_workspace_after_one_click(self):
        app = self.login(remember=False)
        self.assertEqual(app.session_state["rag_current_user"], self.owner)
        self.assertFalse(any(item.value == "登录工作区" for item in app.title))
        with self.repo.connection() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM auth_tickets").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
