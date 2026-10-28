"""Style profile sidebar persists consent and supports review without model calls."""
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
from modules.style_profile import (
    approved_style_instructions, get_style_profiles, list_style_proposals,
    observe_style_turn, set_style_enabled,
)


class StylePageTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Repository(Path(self.temp.name) / "ui.sqlite3")
        self.owner = self.repo.register("tester", "example-password")
        self.kb = self.repo.create_base(self.owner, "论文", "paper", "", "fake", "company",
                                        "https://example.invalid/v1")
        self.sid = self.repo.create_session(self.owner, self.kb)
        settings = ModelSettings("https://example.invalid/v1", "", "fake", "fake", "company")
        for item in (
            patch("modules.repository.DB_PATH", self.repo.path),
            patch("modules.auth_ui.require_user", return_value=self.repo.get_user(self.owner)),
            patch("modules.managed_ui.render_model_settings", return_value=settings),
        ):
            item.start()
            self.addCleanup(item.stop)

    def page(self):
        app = AppTest.from_file(str(ROOT / "src" / "Home.py"), default_timeout=20).run()
        app = app.switch_page("app_pages/chat.py").run()
        self.assertEqual(len(app.exception), 0)
        return app

    def test_toggle_proposal_review_remove_and_clear(self):
        app = self.page()
        toggle = f"rag_style_on_{self.owner}_{self.kb}"
        self.assertFalse(app.get_by_key(toggle).value)
        app.get_by_key(toggle).set_value(True).run()
        self.assertEqual(len(app.exception), 0)
        self.assertTrue(self.page().get_by_key(toggle).value)
        for feedback in ("回答太短了", "再详细一点", "内容太少"):
            mid = self.repo.append_turn(self.owner, self.sid, feedback, "已完成", [])
            observe_style_turn(self.repo, self.owner, self.kb, self.sid, mid)
        candidate = list_style_proposals(self.repo, self.owner, self.kb)[0]
        app = self.page()
        self.assertEqual(approved_style_instructions(self.repo, self.owner, self.kb), "")
        app.get_by_key(f"rag_style_accept_{candidate['id']}").click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(get_style_profiles(self.repo, self.owner, self.kb)[0]["value"], "detailed")
        app.get_by_key(f"rag_style_remove_{self.kb}_detail").click().run()
        self.assertEqual(get_style_profiles(self.repo, self.owner, self.kb), [])
        app.get_by_key(toggle).set_value(False).run()
        self.assertFalse(self.page().get_by_key(toggle).value)


if __name__ == "__main__":
    unittest.main()
