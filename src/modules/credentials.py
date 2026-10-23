"""Opt-in, per-Windows-user encrypted API key storage (never in the repo)."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import tempfile
from ctypes import wintypes
from pathlib import Path


class CredentialStoreError(RuntimeError):
    """Present a safe error without echoing a key or ciphertext."""


class _Blob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(data: bytes, *, decrypt: bool) -> bytes:
    if os.name != "nt":
        raise CredentialStoreError("本机加密保存目前仅支持 Windows。可改用本机 .env 配置密钥。")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    function = crypt32.CryptUnprotectData if decrypt else crypt32.CryptProtectData
    function.argtypes = [
        ctypes.POINTER(_Blob),
        ctypes.POINTER(wintypes.LPWSTR) if decrypt else wintypes.LPCWSTR,
        ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p,
        wintypes.DWORD, ctypes.POINTER(_Blob),
    ]
    function.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    buffer = ctypes.create_string_buffer(data)
    source = _Blob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte)))
    result = _Blob()
    if not function(ctypes.byref(source), None if decrypt else "RAG Assistant", None,
                    None, None, 0x01, ctypes.byref(result)):
        raise CredentialStoreError("无法使用当前 Windows 用户账户读取或保存密钥。")
    try:
        return ctypes.string_at(result.pbData, result.cbData)
    finally:
        kernel32.LocalFree(ctypes.cast(result.pbData, ctypes.c_void_p))


class CredentialStore:
    """Credentials are DPAPI-encrypted for the signed-in Windows user."""

    def __init__(self, path: Path | None = None):
        local_app_data = os.getenv("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        self.path = path if path is not None else Path(local_app_data) / "RAGAssistant" / "credentials-v1.json"

    @staticmethod
    def _id(scope: str) -> str:
        return hashlib.sha256(scope.encode("utf-8")).hexdigest()

    def _items(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("version") != 1 or not isinstance(payload.get("items"), dict):
                raise ValueError("unexpected format")
            return payload["items"]
        except (OSError, ValueError, AttributeError) as exc:
            raise CredentialStoreError("本机保存的密钥文件无法读取；可检查文件权限。") from exc

    def has(self, scope: str) -> bool:
        return self._id(scope) in self._items()

    def load(self, scope: str) -> str | None:
        encoded = self._items().get(self._id(scope))
        if encoded is None:
            return None
        try:
            return _dpapi(base64.b64decode(encoded, validate=True), decrypt=True).decode("utf-8")
        except (TypeError, ValueError, UnicodeError, CredentialStoreError) as exc:
            raise CredentialStoreError("保存的密钥无法解密；可能切换了 Windows 用户账户。") from exc

    def save(self, scope: str, key: str) -> None:
        if not key or len(key) > 8192:
            raise CredentialStoreError("请输入有效的 API 密钥后再保存。")
        items = self._items()
        items[self._id(scope)] = base64.b64encode(_dpapi(key.encode("utf-8"), decrypt=False)).decode("ascii")
        self._write(items)

    def delete(self, scope: str) -> None:
        items = self._items()
        if items.pop(self._id(scope), None) is not None:
            self._write(items)

    def _write(self, items: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                prefix=".credentials-", suffix=".tmp", delete=False,
            ) as handle:
                tmp = Path(handle.name)
                json.dump({"version": 1, "items": items}, handle)
            os.replace(tmp, self.path)
        except OSError as exc:
            raise CredentialStoreError("无法写入本机密钥文件；请检查当前 Windows 用户的目录权限。") from exc
        finally:
            if tmp is not None:
                tmp.unlink(missing_ok=True)
