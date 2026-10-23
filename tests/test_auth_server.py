"""HttpOnly cookie exchange rejects cross-origin, invalid and replayed tickets."""

import asyncio
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from asgi_app import exchange_login
from modules.repository import Repository


class AuthExchangeTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Repository(Path(self.temp.name) / "auth.sqlite3")
        self.user = self.repo.register("browser_test", "long-disposable-test-password")
        mocked = patch("modules.repository.DB_PATH", self.repo.path)
        mocked.start()
        self.addCleanup(mocked.stop)

    @staticmethod
    def request(ticket, *, origin=b"http://localhost:8507"):
        body = json.dumps({"ticket": ticket}).encode()
        scope = {"type": "http", "method": "POST", "path": "/api/auth/exchange",
                 "scheme": "http", "server": ("localhost", 8507),
                 "headers": [(b"host", b"localhost:8507"), (b"origin", origin),
                             (b"content-type", b"application/json")]}
        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}
        return Request(scope, receive)

    def test_https_reverse_proxy_origin_sets_secure_cookie(self):
        ticket = self.repo.create_login_ticket(self.user, lifetime_seconds=3600)
        with patch.dict("os.environ", {"RAG_PUBLIC_ORIGIN": "https://rag-demo.example.org"}):
            wrong = asyncio.run(exchange_login(self.request(ticket)))
            self.assertEqual(wrong.status_code, 403)
            result = asyncio.run(exchange_login(
                self.request(ticket, origin=b"https://rag-demo.example.org")))
            self.assertEqual(result.status_code, 200)
            self.assertIn("Secure", result.headers["set-cookie"])
            self.assertIn("HttpOnly", result.headers["set-cookie"])
    def test_cookie_is_http_only_and_ticket_cannot_be_replayed(self):
        ticket = self.repo.create_login_ticket(self.user, lifetime_seconds=3600)
        foreign = asyncio.run(exchange_login(self.request(ticket, origin=b"https://attacker.invalid")))
        self.assertEqual(foreign.status_code, 403)
        response = asyncio.run(exchange_login(self.request(ticket)))
        self.assertEqual(response.status_code, 200)
        cookie = response.headers["set-cookie"]
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=strict", cookie)
        self.assertIn("Max-Age=3600", cookie)
        self.assertNotIn(self.user, cookie)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(asyncio.run(exchange_login(self.request(ticket))).status_code, 401)


if __name__ == "__main__":
    unittest.main()