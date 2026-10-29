"""Repository migration and tenant-isolation tests; all data stays in a temp dir."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.modules.repository import Repository


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "test.sqlite3"
        self.repo = Repository(self.path)
        self.alice = self.repo.register("alice", "strong-password-123")
        self.bob = self.repo.register("bob_01", "another-password-123")
        self.kb = self.repo.create_base(self.alice, "papers", "paper", "academic notes",
                                        "text-embedding-v4", "company", "https://example.invalid/v1")

    def test_chinese_username_can_register_and_log_in(self):
        user_id = self.repo.register("石小叶", "a-long-unique-password")
        self.assertEqual(self.repo.authenticate("石小叶", "a-long-unique-password")["id"], user_id)
        with self.assertRaises(ValueError):
            self.repo.register("石小叶", "a-long-unique-password")

    def test_password_change_requires_existing_password(self):
        with self.assertRaises(ValueError):
            self.repo.change_password(self.alice, "wrong", "new-long-password")
        self.repo.change_password(self.alice, "strong-password-123", "new-long-password")
        self.assertIsNone(self.repo.authenticate("alice", "strong-password-123"))
        self.assertEqual(self.repo.authenticate("alice", "new-long-password")["id"], self.alice)

    def test_login_hash_and_idempotent_schema_migration(self):
        self.assertIsNone(self.repo.authenticate("alice", "wrong-password"))
        self.assertEqual(self.repo.authenticate("Alice", "strong-password-123")["id"], self.alice)
        self.assertNotIn("strong-password", self.path.read_bytes().decode("latin1"))
        again = Repository(self.path)
        self.assertEqual(again.get_base(self.alice, self.kb)["name"], "papers")
        self.assertEqual(again.stats(self.alice)["bases"], 1)
        with self.assertRaisesRegex(ValueError, "存在"):
            again.register("alice", "strong-password-123")

    def test_login_ticket_is_one_time_and_revoke_survives_restart(self):
        ticket = self.repo.create_login_ticket(self.alice, lifetime_seconds=600)
        token, lifetime = self.repo.consume_login_ticket(ticket)
        self.assertEqual(lifetime, 600)
        self.assertIsNone(self.repo.consume_login_ticket(ticket))
        self.assertIsNone(self.repo.user_from_login_token("forged-token"))
        self.assertEqual(Repository(self.path).user_from_login_token(token)["id"], self.alice)
        self.repo.revoke_login_token(token)
        self.assertIsNone(Repository(self.path).user_from_login_token(token))
        fresh_ticket = self.repo.create_login_ticket(self.alice)
        fresh_token, _ = self.repo.consume_login_ticket(fresh_ticket)
        self.repo.change_password(self.alice, "strong-password-123", "new-long-password")
        self.assertIsNone(self.repo.user_from_login_token(fresh_token))

    def test_v1_database_migration_preserves_users_and_bases(self):
        with self.repo.connection(write=True) as db:
            db.execute("DROP TABLE auth_tickets")
            db.execute("DROP TABLE auth_sessions")
            db.execute("PRAGMA user_version=1")
        migrated = Repository(self.path)
        self.assertEqual(migrated.get_base(self.alice, self.kb)["name"], "papers")
        self.assertEqual(migrated.authenticate("alice", "strong-password-123")["id"], self.alice)
        with migrated.connection() as db:
            self.assertEqual(db.execute("PRAGMA user_version").fetchone()[0], 6)
        ticket = migrated.create_login_ticket(self.alice)
        self.assertIsNotNone(migrated.consume_login_ticket(ticket))
    def test_tenant_isolation_for_bases_sessions_and_messages(self):
        self.assertEqual(self.repo.list_bases(self.bob), [])
        self.assertIsNone(self.repo.get_base(self.bob, self.kb))
        with self.assertRaises(PermissionError):
            self.repo.create_session(self.bob, self.kb)
        sid = self.repo.create_session(self.alice, self.kb)
        self.repo.add_message(self.alice, sid, "user", "how does it work?")
        self.repo.add_message(self.alice, sid, "assistant", "see source [1]", [{"name": "paper.pdf"}])
        self.assertEqual(len(Repository(self.path).list_messages(self.alice, sid)), 2)
        self.assertEqual(self.repo.list_sessions(self.bob), [])
        with self.assertRaises(PermissionError):
            self.repo.list_messages(self.bob, sid)
        with self.assertRaises(PermissionError):
            self.repo.add_message(self.bob, sid, "user", "forbidden")
        self.repo.delete_session(self.bob, sid)
        self.assertIsNotNone(self.repo.get_session(self.alice, sid))
        self.assertEqual(self.repo.stats(self.alice)["conversations"], 1)

    def test_never_downgrades_future_database(self):
        with self.repo.connection(write=True) as db:
            db.execute("PRAGMA user_version=99")
        with self.assertRaises(RuntimeError):
            Repository(self.path)
    def test_document_versions_are_idempotent_and_scoped(self):
        doc, created = self.repo.create_document(self.alice, self.kb, "readme.md", "hash1", 10, "safe.md")
        self.assertTrue(created)
        repeated, created = self.repo.create_document(self.alice, self.kb, "readme.md", "hash1", 10, "safe.md")
        self.assertFalse(created)
        self.assertEqual(doc["id"], repeated["id"])
        self.repo.job_status(self.alice, doc["id"], "processing")
        self.repo.complete_document(self.alice, doc["id"], [{"id": "chunk1", "location": "paragraph 1", "content": "evidence"}])
        self.assertEqual(self.repo.stats(self.alice)["documents"], 1)
        self.assertEqual(self.repo.list_chunks(self.alice, self.kb)[0]["content"], "evidence")
        with self.assertRaises(PermissionError):
            self.repo.list_chunks(self.bob, self.kb)
        new, created = self.repo.create_document(self.alice, self.kb, "readme.md", "hash2", 15, "new.md")
        self.assertTrue(created)
        self.assertEqual(new["version"], 2)
        replaced = self.repo.complete_document(self.alice, new["id"], [{"id": "chunk2", "location": "paragraph 2", "content": "new"}])
        self.assertEqual(replaced, [doc["id"]])
        self.assertEqual(self.repo.list_chunks(self.alice, self.kb)[0]["content"], "new")


if __name__ == "__main__":
    unittest.main()