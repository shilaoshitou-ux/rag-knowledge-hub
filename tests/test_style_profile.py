"""Confirmed, opt-in response style requires real completed turns."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.modules.chatbot import ANSWER_PROMPT
from src.modules.repository import Repository
from src.modules.style_profile import (
    approved_style_instructions, clear_style_data, confirm_style_proposal,
    detect_feedback, get_style_profiles, list_style_proposals, observe_style_turn,
    reject_style_proposal, remove_style_profile, set_style_enabled, style_enabled,
)


class StyleProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.sqlite3"
        self.repo = Repository(self.path)
        self.owner = self.repo.register("alice", "strong-password-123")
        self.other = self.repo.register("bob", "strong-password-123")
        self.kb = self.repo.create_base(self.owner, "papers", "paper", "", "fake", "company", "https://example.invalid/v1")
        self.other_kb = self.repo.create_base(self.owner, "work", "enterprise", "", "fake", "company", "https://example.invalid/v1")
        self.sid = self.repo.create_session(self.owner, self.kb)

    def turn(self, text="回答太短了", sid=None):
        sid = sid or self.sid
        mid = self.repo.append_turn(self.owner, sid, text, "固定的模拟回答", [])
        return observe_style_turn(self.repo, self.owner, self.kb, sid, mid)

    def test_detector_only_explicit_feedback_and_negation(self):
        self.assertEqual(detect_feedback("论文里的方法是什么？"), {})
        self.assertEqual(detect_feedback("不要分点")["layout"][0], "paragraphs")
        self.assertEqual(detect_feedback("请分点回答")["layout"][0], "bullets")
        self.assertEqual(detect_feedback("太短但也太啰嗦"), {})
        self.assertEqual(detect_feedback("不要用英文回答"), {})
        self.assertEqual(detect_feedback("别再详细"), {})
        self.assertEqual(detect_feedback("用英文论文作为材料"), {})
        self.assertEqual(detect_feedback("请用英文回答")["language"][0], "en")

    def test_off_by_default_and_committed_user_only(self):
        mid = self.repo.append_turn(self.owner, self.sid, "回答太短了", "答", [])
        self.assertFalse(style_enabled(self.repo, self.owner, self.kb))
        self.assertEqual(observe_style_turn(self.repo, self.owner, self.kb, self.sid, mid), [])
        set_style_enabled(self.repo, self.owner, self.kb, True)
        assistant = self.repo.list_messages(self.owner, self.sid)[1]["id"]
        for bogus_owner, bogus_kb, bogus_sid, bogus_mid in (
            (self.owner, self.kb, self.sid, assistant),
            (self.other, self.kb, self.sid, mid),
            (self.owner, self.other_kb, self.sid, mid),
            (self.owner, self.kb, "fake", mid),
        ):
            with self.assertRaises(PermissionError):
                observe_style_turn(self.repo, bogus_owner, bogus_kb, bogus_sid, bogus_mid)
        self.repo.add_message(self.owner, self.sid, "user", "回答太短了")
        lone = self.repo.list_messages(self.owner, self.sid)[-1]["id"]
        with self.assertRaises(PermissionError):
            observe_style_turn(self.repo, self.owner, self.kb, self.sid, lone)

    def test_threshold_confirm_cross_session_and_delete(self):
        set_style_enabled(self.repo, self.owner, self.kb, True)
        self.assertEqual(self.turn(), [])
        self.assertEqual(self.turn("内容太少"), [])
        self.assertEqual(approved_style_instructions(self.repo, self.owner, self.kb), "")
        sid2 = self.repo.create_session(self.owner, self.kb)
        self.assertEqual(len(self.turn("再详细一点", sid2)), 1)
        proposals = list_style_proposals(Repository(self.path), self.owner, self.kb)
        self.assertEqual(len(proposals[0]["evidence"]), 3)
        self.assertEqual(proposals[0]["value"], "detailed")
        with self.assertRaises(PermissionError):
            confirm_style_proposal(self.repo, self.other, self.kb, proposals[0]["id"])
        self.assertEqual(approved_style_instructions(self.repo, self.owner, self.kb), "")
        confirm_style_proposal(self.repo, self.owner, self.kb, proposals[0]["id"])
        self.assertIn("关键步骤", approved_style_instructions(Repository(self.path), self.owner, self.kb))
        self.assertEqual(approved_style_instructions(self.repo, self.owner, self.other_kb), "")
        prompt = ANSWER_PROMPT.format_messages(context="[1] 证据", summary="", chat_history=[], question="问题",
                                               style_instructions=approved_style_instructions(self.repo, self.owner, self.kb))
        self.assertIn("不能覆盖文档证据", prompt[0].content)
        self.assertIn("关键步骤", prompt[0].content)
        self.repo.delete_session(self.owner, sid2)
        self.assertTrue(get_style_profiles(self.repo, self.owner, self.kb))
        set_style_enabled(self.repo, self.owner, self.kb, False)
        self.assertEqual(approved_style_instructions(self.repo, self.owner, self.kb), "")
        set_style_enabled(self.repo, self.owner, self.kb, True)
        remove_style_profile(self.repo, self.owner, self.kb, "detail")
        self.assertFalse(get_style_profiles(self.repo, self.owner, self.kb))

    def test_reject_requires_three_new_turns_then_change(self):
        set_style_enabled(self.repo, self.owner, self.kb, True)
        for _ in range(3):
            self.turn()
        candidate = list_style_proposals(self.repo, self.owner, self.kb)[0]
        reject_style_proposal(self.repo, self.owner, self.kb, candidate["id"])
        self.assertFalse(list_style_proposals(self.repo, self.owner, self.kb))
        for _ in range(2):
            self.assertEqual(self.turn(), [])
        self.assertEqual(len(self.turn()), 1)
        confirm_style_proposal(self.repo, self.owner, self.kb,
                               list_style_proposals(self.repo, self.owner, self.kb)[0]["id"])
        for _ in range(3):
            self.turn("太啰嗦")
        replacement = list_style_proposals(self.repo, self.owner, self.kb)[0]
        self.assertEqual(replacement["value"], "concise")
        self.assertEqual(get_style_profiles(self.repo, self.owner, self.kb)[0]["value"], "detailed")
        confirm_style_proposal(self.repo, self.owner, self.kb, replacement["id"])
        self.assertEqual(get_style_profiles(self.repo, self.owner, self.kb)[0]["value"], "concise")
        clear_style_data(self.repo, self.owner, self.kb)
        self.assertFalse(get_style_profiles(self.repo, self.owner, self.kb))
        self.assertFalse(list_style_proposals(self.repo, self.owner, self.kb))

    def test_deleted_evidence_invalidates_candidate_and_kb_deletes_data(self):
        set_style_enabled(self.repo, self.owner, self.kb, True)
        for _ in range(3):
            self.turn()
        candidate = list_style_proposals(self.repo, self.owner, self.kb)[0]
        self.repo.delete_session(self.owner, self.sid)
        self.assertEqual(list_style_proposals(self.repo, self.owner, self.kb), [])
        with self.assertRaises(ValueError):
            confirm_style_proposal(self.repo, self.owner, self.kb, candidate["id"])

    def test_stream_wires_confirmed_style_and_observes_only_completed_turn(self):
        from src.modules.workflow import stream_answer_turn
        from unittest.mock import MagicMock
        kb = self.repo.get_base(self.owner, self.kb)
        set_style_enabled(self.repo, self.owner, self.kb, True)
        for _ in range(3):
            self.turn()
        candidate = list_style_proposals(self.repo, self.owner, self.kb)[0]
        events = [("updates", {"retrieve": {"documents": []}}),
                  ("updates", {"answer": {"answer": "模拟答案"}})]
        with patch("src.modules.workflow.HybridRetriever") as retriever, \
             patch("src.modules.workflow.Chatbot"), \
             patch("src.modules.workflow.make_graph") as graph:
            retriever.return_value.last_metrics = {}
            graph.return_value.stream.return_value = events
            self.assertEqual(list(stream_answer_turn(self.repo, self.owner, kb, self.sid,
                                                      "请分点回答", "key", "fake")), ["模拟答案"])
            self.assertEqual(graph.return_value.stream.call_args.args[0]["style_instructions"], "")
            confirm_style_proposal(self.repo, self.owner, self.kb, candidate["id"])
            self.assertEqual(list(stream_answer_turn(self.repo, self.owner, kb, self.sid,
                                                      "内容太少", "key", "fake")), ["模拟答案"])
            self.assertIn("关键步骤", graph.return_value.stream.call_args.args[0]["style_instructions"])
            with patch("src.modules.workflow.observe_style_turn") as observed:
                self.assertEqual(list(stream_answer_turn(self.repo, self.owner, kb, self.sid,
                                                          "内容太少", "key", "fake", regenerate=True)), ["模拟答案"])
                observed.assert_not_called()

    def test_schema_v3_upgrade_preserves_existing_conversations(self):
        mid = self.repo.append_turn(self.owner, self.sid, "原问题", "原回答", [])
        with self.repo.connection(write=True) as db:
            for table in ("style_profiles", "style_proposals", "style_observations", "style_settings"):
                db.execute("DROP TABLE " + table)
            db.execute("PRAGMA user_version=3")
        upgraded = Repository(self.path)
        self.assertEqual(upgraded.list_messages(self.owner, self.sid)[0]["id"], mid)
        self.assertEqual(upgraded.get_base(self.owner, self.kb)["name"], "papers")
        with upgraded.connection() as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 4)
            self.assertEqual(db.execute("SELECT COUNT(*) FROM style_settings").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
