"""Round-trip tests use only disposable fake keys and a temporary encrypted file."""

import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.modules.credentials import CredentialStore, CredentialStoreError


@unittest.skipUnless(os.name == "nt", "DPAPI is Windows-only")
class CredentialTests(unittest.TestCase):
    def test_roundtrip_is_encrypted_and_isolated_by_endpoint(self):
        with TemporaryDirectory() as directory:
            store = CredentialStore(Path(directory) / "credentials.json")
            store.save("default|https://one.example/v1", "fake-one-only")
            store.save("custom|https://two.example/v1", "fake-two-only")
            self.assertEqual(store.load("default|https://one.example/v1"), "fake-one-only")
            self.assertEqual(store.load("custom|https://two.example/v1"), "fake-two-only")
            self.assertIsNone(store.load("custom|https://other.example/v1"))
            on_disk = store.path.read_text(encoding="utf-8")
            self.assertNotIn("fake-one-only", on_disk)
            self.assertNotIn("fake-two-only", on_disk)
            store.delete("default|https://one.example/v1")
            self.assertIsNone(store.load("default|https://one.example/v1"))
            self.assertEqual(store.load("custom|https://two.example/v1"), "fake-two-only")

    def test_bad_ciphertext_fails_without_printing_key(self):
        with TemporaryDirectory() as directory:
            store = CredentialStore(Path(directory) / "credentials.json")
            store.save("default", "fake-secret-only")
            store.path.write_text('{"version": 1, "items": {"' + store._id("default") + '": "bad!"}}', encoding="utf-8")
            with self.assertRaises(CredentialStoreError) as caught:
                store.load("default")
            self.assertNotIn("fake-secret-only", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
