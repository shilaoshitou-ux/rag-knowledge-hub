"""CLI benchmark saves only local results and never treats a label set as a score."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts import run_rag_eval as cli
from src.modules.repository import Repository


class EvalCliTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = Repository(self.root / "db.sqlite3")
        self.owner = self.repo.register("tester", "long-test-password")
        kb_id = self.repo.create_base(self.owner, "demo", "enterprise", "", "offline-embed",
                                      "company", "https://example.invalid/v1")
        self.kb = self.repo.get_base(self.owner, kb_id)
        doc, _ = self.repo.create_document(self.owner, kb_id, "guide.txt", "digest", 50,
                                           str(self.root / "guide.txt"))
        self.repo.job_status(self.owner, doc["id"], "processing")
        self.repo.complete_document(self.owner, doc["id"], [
            {"id": "active", "content": "The process uses retrieval.", "location": "1"},
        ])
        self.cases = self.root / "cases.json"
        self.cases.write_text(json.dumps([{"question": "How?", "relevant_documents": ["guide.txt"],
                                            "relevant_chunks": ["active"]}]), encoding="utf-8")

    def test_manifest_and_offline_evaluation_metadata(self):
        with patch.object(cli, "Repository", return_value=self.repo), \
                patch.object(cli, "DATA_DIR", self.root / "data"), \
                patch.object(cli.getpass, "getpass", return_value="long-test-password"), \
                patch.dict(cli.os.environ, {"RAG_EVAL_API_KEY": "private-token"}), \
                patch.object(cli, "evaluate", return_value=[{"strategy": "vector", "recall_at_k": 1}]), \
                patch.object(cli, "summarize", return_value=[{"Recall@K": 1}]):
            manifest = cli.run(["--username", "tester", "--kb-id", self.kb["id"], "--manifest"])
            self.assertEqual(json.loads(manifest.read_text(encoding="utf-8"))["active_chunks"][0]["chunk_id"],
                             "active")
            result = cli.run(["--username", "tester", "--kb-id", self.kb["id"],
                              "--model", "offline-chat", "--cases", str(self.cases)])
            payload = json.loads(result.read_text(encoding="utf-8"))
            self.assertEqual(payload["case_count"], 1)
            self.assertEqual(payload["evidence_labelled"], 1)
            self.assertEqual(len(payload["dataset_sha256"]), 64)
            self.assertNotIn("private-token", result.read_text(encoding="utf-8"))
            self.assertTrue(result.is_relative_to(self.root / "data"))

    def test_invalid_credentials_and_missing_key_do_not_create_results(self):
        with patch.object(cli, "Repository", return_value=self.repo), \
                patch.object(cli, "DATA_DIR", self.root / "data"), \
                patch.object(cli.getpass, "getpass", return_value="wrong-password"):
            with self.assertRaises(PermissionError):
                cli.run(["--username", "tester", "--kb-id", self.kb["id"], "--manifest"])
        with patch.object(cli, "Repository", return_value=self.repo), \
                patch.object(cli, "DATA_DIR", self.root / "data"), \
                patch.object(cli.getpass, "getpass", return_value="long-test-password"), \
                patch.dict(cli.os.environ, {"RAG_EVAL_API_KEY": ""}):
            with self.assertRaises(ValueError):
                cli.run(["--username", "tester", "--kb-id", self.kb["id"],
                         "--model", "offline-chat", "--cases", str(self.cases)])
        self.assertFalse(list((self.root / "data" / "evaluation").glob("*.json")))


if __name__ == "__main__":
    unittest.main()
