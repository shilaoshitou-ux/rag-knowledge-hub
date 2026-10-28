"""SQLite metadata store for managed knowledge bases; never touches legacy Chroma."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(ROOT / "data"))).resolve()
DB_PATH = Path(os.getenv("RAG_DB_PATH", str(DATA_DIR / "knowledge.sqlite3"))).resolve()
LOGIN_COOKIE_NAME = "rag_login_v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def uid() -> str:
    return uuid.uuid4().hex


class Repository:
    def __init__(self, path: str | Path | None = None):
        self.path = Path(path).resolve() if path else DB_PATH
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @contextmanager
    def connection(self, *, write: bool = False):
        conn = sqlite3.connect(str(self.path), timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=15000")
        try:
            if write:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if write:
                conn.commit()
        except Exception:
            if write:
                conn.rollback()
            raise
        finally:
            conn.close()

    def migrate(self):
        with self.connection(write=True) as db:
            version = db.execute("PRAGMA user_version").fetchone()[0]
            if version > 3:
                raise RuntimeError("数据库版本高于当前程序，请先升级程序，禁止降级写入。")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE, salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id),
                    name TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN ('enterprise','paper')),
                    description TEXT NOT NULL DEFAULT '', embedding_model TEXT NOT NULL,
                    embedding_provider TEXT NOT NULL, base_url TEXT NOT NULL,
                    chunk_size INTEGER NOT NULL, chunk_overlap INTEGER NOT NULL,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL, deleted_at TEXT
                );
                CREATE INDEX IF NOT EXISTS kb_owner ON knowledge_bases(owner_id,deleted_at);
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, kb_id TEXT NOT NULL REFERENCES knowledge_bases(id),
                    name TEXT NOT NULL, version INTEGER NOT NULL, content_hash TEXT NOT NULL,
                    bytes INTEGER NOT NULL, storage_path TEXT NOT NULL, status TEXT NOT NULL,
                    error TEXT NOT NULL DEFAULT '', chunk_count INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(kb_id,name,version)
                );
                CREATE INDEX IF NOT EXISTS docs_kb ON documents(kb_id,active,status);
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY, kb_id TEXT NOT NULL REFERENCES knowledge_bases(id),
                    document_id TEXT NOT NULL REFERENCES documents(id), location TEXT NOT NULL,
                    content TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS chunks_kb ON chunks(kb_id,document_id);
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id),
                    kb_id TEXT NOT NULL REFERENCES knowledge_bases(id), title TEXT NOT NULL,
                    summary TEXT NOT NULL DEFAULT '', summary_upto INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS sessions_owner ON sessions(owner_id,kb_id,updated_at);
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant')),
                    content TEXT NOT NULL, sources_json TEXT NOT NULL DEFAULT '[]', created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id,created_at);
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
                    status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_settings (
                    owner_id TEXT NOT NULL REFERENCES users(id),
                    kb_id TEXT NOT NULL REFERENCES knowledge_bases(id),
                    enabled INTEGER NOT NULL DEFAULT 0 CHECK(enabled IN (0,1)),
                    PRIMARY KEY(owner_id,kb_id)
                );
                CREATE TABLE IF NOT EXISTS personal_memories (
                    id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL REFERENCES users(id),
                    kb_id TEXT NOT NULL REFERENCES knowledge_bases(id),
                    source_session_id TEXT REFERENCES sessions(id) ON DELETE SET NULL,
                    category TEXT NOT NULL CHECK(category IN ('preference','goal','constraint','fact')),
                    quote TEXT NOT NULL, content_hash TEXT NOT NULL, importance REAL NOT NULL,
                    indexed INTEGER NOT NULL DEFAULT 0 CHECK(indexed IN (0,1)),
                    created_at TEXT NOT NULL, UNIQUE(owner_id,kb_id,content_hash)
                );
                CREATE INDEX IF NOT EXISTS memories_scope ON personal_memories(owner_id,kb_id,indexed,created_at);
                CREATE TABLE IF NOT EXISTS auth_tickets (
                    ticket_hash TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id),
                    expires_at INTEGER NOT NULL, session_seconds INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY, owner_id TEXT NOT NULL REFERENCES users(id),
                    expires_at INTEGER NOT NULL, created_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS auth_sessions_owner ON auth_sessions(owner_id, expires_at);
            """)
            if version < 2:
                db.execute("PRAGMA user_version=2")
            if version < 3:
                db.execute("PRAGMA user_version=3")
            db.execute("PRAGMA journal_mode=WAL")

    def register(self, username: str, password: str) -> str:
        username = username.strip().lower()
        if not re.fullmatch(r"[a-z0-9_\u4e00-\u9fff]{2,32}", username) or len(password) < 10:
            raise ValueError("用户名须为 2–32 位中文、英文字母、数字或下划线，密码至少 10 位。")
        salt = secrets.token_bytes(16)
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310000)
        user_id = uid()
        try:
            with self.connection(write=True) as db:
                db.execute("INSERT INTO users VALUES (?,?,?,?,?)", (user_id, username, salt, hashed, now()))
        except sqlite3.IntegrityError as exc:
            raise ValueError("用户名已经存在。") from exc
        return user_id

    def authenticate(self, username: str, password: str) -> dict | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM users WHERE username=?", (username.strip().lower(),)).fetchone()
        if not row:
            return None
        hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), row["salt"], 310000)
        return {"id": row["id"], "username": row["username"]} if hmac.compare_digest(hashed, row["password_hash"]) else None

    def change_password(self, owner: str, old_password: str, new_password: str) -> None:
        if len(new_password) < 10:
            raise ValueError("新密码至少 10 位。")
        with self.connection(write=True) as db:
            row = db.execute("SELECT salt,password_hash FROM users WHERE id=?", (owner,)).fetchone()
            if not row or not hmac.compare_digest(
                hashlib.pbkdf2_hmac("sha256", old_password.encode(), row["salt"], 310000),
                row["password_hash"],
            ):
                raise ValueError("旧密码不正确。")
            salt = secrets.token_bytes(16)
            db.execute("UPDATE users SET salt=?,password_hash=? WHERE id=?",
                       (salt, hashlib.pbkdf2_hmac("sha256", new_password.encode(), salt, 310000), owner))
            db.execute("DELETE FROM auth_sessions WHERE owner_id=?", (owner,))
            db.execute("DELETE FROM auth_tickets WHERE owner_id=?", (owner,))

    @staticmethod
    def _token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_login_ticket(self, owner: str, *, lifetime_seconds: int = 7 * 86400) -> str:
        """One-time, short-lived browser exchange; never put passwords in cookies."""
        ticket = secrets.token_urlsafe(32)
        with self.connection(write=True) as db:
            db.execute("DELETE FROM auth_tickets WHERE expires_at<=?", (int(time.time()),))
            db.execute("DELETE FROM auth_sessions WHERE expires_at<=?", (int(time.time()),))
            db.execute("INSERT INTO auth_tickets VALUES (?,?,?,?)",
                       (self._token_hash(ticket), owner, int(time.time()) + 90, lifetime_seconds))
        return ticket

    def consume_login_ticket(self, ticket: str) -> tuple[str, int] | None:
        if not isinstance(ticket, str) or len(ticket) > 128 or len(ticket) < 32:
            return None
        now_seconds = int(time.time())
        with self.connection(write=True) as db:
            row = db.execute("SELECT owner_id,expires_at,session_seconds FROM auth_tickets WHERE ticket_hash=?",
                             (self._token_hash(ticket),)).fetchone()
            if not row or row["expires_at"] <= now_seconds:
                return None
            db.execute("DELETE FROM auth_tickets WHERE ticket_hash=?", (self._token_hash(ticket),))
            token = secrets.token_urlsafe(32)
            lifetime = min(max(int(row["session_seconds"]), 60), 30 * 86400)
            db.execute("INSERT INTO auth_sessions VALUES (?,?,?,?)",
                       (self._token_hash(token), row["owner_id"], now_seconds + lifetime, now_seconds))
        return token, lifetime

    def user_from_login_token(self, token: str | None) -> dict | None:
        if not isinstance(token, str) or len(token) > 128 or len(token) < 32:
            return None
        with self.connection() as db:
            row = db.execute("""SELECT u.id,u.username FROM auth_sessions a
                JOIN users u ON u.id=a.owner_id WHERE a.token_hash=? AND a.expires_at>?""",
                (self._token_hash(token), int(time.time()))).fetchone()
        return dict(row) if row else None

    def revoke_login_token(self, token: str | None) -> None:
        if isinstance(token, str) and 32 <= len(token) <= 128:
            with self.connection(write=True) as db:
                db.execute("DELETE FROM auth_sessions WHERE token_hash=?", (self._token_hash(token),))

    def get_user(self, user_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute("SELECT id,username FROM users WHERE id=?", (user_id,)).fetchone()
        return dict(row) if row else None
    def create_base(self, owner: str, name: str, kind: str, description: str,
                    embedding_model: str, embedding_provider: str, base_url: str,
                    chunk_size: int = 1000, chunk_overlap: int = 150) -> str:
        if not name.strip() or len(name) > 100 or kind not in ("enterprise", "paper"):
            raise ValueError("请填写有效的知识库名称和类型。")
        if not (200 <= chunk_size <= 4000 and 0 <= chunk_overlap < chunk_size):
            raise ValueError("切分参数不合法。")
        kb_id, timestamp = uid(), now()
        with self.connection(write=True) as db:
            db.execute("""INSERT INTO knowledge_bases VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,NULL)""", (kb_id, owner, name.strip(), kind,
                description.strip()[:1000], embedding_model, embedding_provider, base_url,
                chunk_size, chunk_overlap, timestamp, timestamp))
        return kb_id

    def get_base(self, owner: str, kb_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute("SELECT * FROM knowledge_bases WHERE id=? AND owner_id=? AND deleted_at IS NULL",
                             (kb_id, owner)).fetchone()
        return dict(row) if row else None

    def list_bases(self, owner: str) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("""SELECT k.*, COUNT(d.id) AS document_count FROM knowledge_bases k
                LEFT JOIN documents d ON d.kb_id=k.id AND d.active=1 AND d.status='success'
                WHERE k.owner_id=? AND k.deleted_at IS NULL GROUP BY k.id ORDER BY k.updated_at DESC""", (owner,)).fetchall()
        return [dict(row) for row in rows]

    def update_base(self, owner: str, kb_id: str, name: str, description: str):
        if not name.strip() or len(name) > 100:
            raise ValueError("知识库名称不能为空或超过 100 字符。")
        with self.connection(write=True) as db:
            result = db.execute("""UPDATE knowledge_bases SET name=?,description=?,updated_at=?
                WHERE id=? AND owner_id=? AND deleted_at IS NULL""",
                (name.strip(), description.strip()[:1000], now(), kb_id, owner))
            if not result.rowcount:
                raise PermissionError("知识库不存在或无权访问。")

    def stats(self, owner: str) -> dict:
        with self.connection() as db:
            return {
                "bases": db.execute("SELECT COUNT(*) FROM knowledge_bases WHERE owner_id=? AND deleted_at IS NULL", (owner,)).fetchone()[0],
                "documents": db.execute("""SELECT COUNT(*) FROM documents d JOIN knowledge_bases k ON k.id=d.kb_id
                    WHERE k.owner_id=? AND k.deleted_at IS NULL AND d.active=1 AND d.status='success'""", (owner,)).fetchone()[0],
                "uploads": db.execute("""SELECT COUNT(*) FROM documents d JOIN knowledge_bases k ON k.id=d.kb_id
                    WHERE k.owner_id=? AND k.deleted_at IS NULL AND d.active=1 AND d.status!='deleted'""", (owner,)).fetchone()[0],
                "conversations": db.execute("SELECT COUNT(*) FROM sessions WHERE owner_id=?", (owner,)).fetchone()[0],
            }

    def create_session(self, owner: str, kb_id: str, title: str = "新会话") -> str:
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权访问。")
        sid, timestamp = uid(), now()
        with self.connection(write=True) as db:
            db.execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?)",
                       (sid, owner, kb_id, title[:80], "", 0, timestamp, timestamp))
        return sid

    def get_session(self, owner: str, sid: str) -> dict | None:
        with self.connection() as db:
            row = db.execute("""SELECT s.* FROM sessions s JOIN knowledge_bases k ON k.id=s.kb_id
                WHERE s.id=? AND s.owner_id=? AND k.deleted_at IS NULL""", (sid, owner)).fetchone()
        return dict(row) if row else None

    def list_sessions(self, owner: str, kb_id: str | None = None) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("""SELECT s.*, k.name AS kb_name FROM sessions s JOIN knowledge_bases k ON k.id=s.kb_id
                WHERE s.owner_id=? AND k.deleted_at IS NULL AND (? IS NULL OR s.kb_id=?)
                ORDER BY s.updated_at DESC LIMIT 100""", (owner, kb_id, kb_id)).fetchall()
        return [dict(r) for r in rows]

    def add_message(self, owner: str, sid: str, role: str, content: str, sources: list | None = None):
        if role not in ("user", "assistant") or not self.get_session(owner, sid):
            raise PermissionError("会话不存在或无权写入。")
        with self.connection(write=True) as db:
            db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)",
                       (uid(), sid, role, content, json.dumps(sources or [], ensure_ascii=False), now()))
            db.execute("UPDATE sessions SET updated_at=?,title=CASE WHEN title='新会话' AND ?='user' THEN ? ELSE title END WHERE id=?",
                       (now(), role, content[:50], sid))

    def append_turn(self, owner: str, sid: str, question: str, answer: str, sources: list):
        if not self.get_session(owner, sid):
            raise PermissionError("会话不存在或无权访问。")
        timestamp = now()
        with self.connection(write=True) as db:
            db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)",
                       (uid(), sid, "user", question, "[]", timestamp))
            db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)",
                       (uid(), sid, "assistant", answer, json.dumps(sources, ensure_ascii=False), timestamp))
            db.execute("UPDATE sessions SET updated_at=?,title=CASE WHEN title='新会话' THEN ? ELSE title END WHERE id=?",
                       (timestamp, question[:50], sid))
    def list_messages(self, owner: str, sid: str) -> list[dict]:
        if not self.get_session(owner, sid):
            raise PermissionError("会话不存在或无权访问。")
        with self.connection() as db:
            rows = db.execute("SELECT * FROM messages WHERE session_id=? ORDER BY rowid", (sid,)).fetchall()
        return [{**dict(r), "sources": json.loads(r["sources_json"])} for r in rows]

    def delete_session(self, owner: str, sid: str):
        with self.connection(write=True) as db:
            db.execute("DELETE FROM sessions WHERE id=? AND owner_id=?", (sid, owner))

    def set_summary(self, owner: str, sid: str, summary: str, upto: int):
        if not self.get_session(owner, sid):
            raise PermissionError("会话不存在或无权访问。")
        with self.connection(write=True) as db:
            db.execute("UPDATE sessions SET summary=?,summary_upto=? WHERE id=? AND owner_id=?",
                       (summary[:4000], upto, sid, owner))
    def memory_enabled(self, owner: str, kb_id: str) -> bool:
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权访问。")
        with self.connection() as db:
            row = db.execute("SELECT enabled FROM memory_settings WHERE owner_id=? AND kb_id=?",
                             (owner, kb_id)).fetchone()
        return bool(row and row["enabled"])

    def set_memory_enabled(self, owner: str, kb_id: str, enabled: bool):
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权访问。")
        with self.connection(write=True) as db:
            db.execute("""INSERT INTO memory_settings(owner_id,kb_id,enabled) VALUES(?,?,?)
                ON CONFLICT(owner_id,kb_id) DO UPDATE SET enabled=excluded.enabled""",
                (owner, kb_id, int(enabled)))

    def list_memories(self, owner: str, kb_id: str) -> list[dict]:
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权访问。")
        with self.connection() as db:
            rows = db.execute("""SELECT m.* FROM personal_memories m
                JOIN knowledge_bases k ON k.id=m.kb_id AND k.owner_id=m.owner_id
                WHERE m.owner_id=? AND m.kb_id=? AND k.deleted_at IS NULL
                ORDER BY m.created_at DESC, m.rowid DESC LIMIT 100""", (owner, kb_id)).fetchall()
        return [dict(row) for row in rows]

    def reserve_memory(self, owner: str, kb_id: str, sid: str | None,
                       category: str, quote: str, importance: float) -> tuple[str, bool]:
        if not self.memory_enabled(owner, kb_id):
            raise PermissionError("请先启用当前知识库的长期记忆。")
        if sid is not None:
            session = self.get_session(owner, sid)
            if not session or session["kb_id"] != kb_id:
                raise PermissionError("来源会话不存在或与知识库不匹配。")
        if category not in ("preference", "goal", "constraint", "fact") or not 8 <= len(quote) <= 180:
            raise ValueError("记忆内容或类型不合法（须为 8–180 字）。")
        if not 0 <= importance <= 1:
            raise ValueError("重要性不合法。")
        digest = hashlib.sha256(quote.strip().casefold().encode("utf-8")).hexdigest()
        with self.connection(write=True) as db:
            count = db.execute("SELECT COUNT(*) FROM personal_memories WHERE owner_id=? AND kb_id=?",
                               (owner, kb_id)).fetchone()[0]
            existing = db.execute("""SELECT id FROM personal_memories
                WHERE owner_id=? AND kb_id=? AND content_hash=?""", (owner, kb_id, digest)).fetchone()
            if existing:
                return existing["id"], False
            if count >= 100:
                raise ValueError("此知识库最多保存 100 条长期记忆。")
            memory_id = uid()
            db.execute("""INSERT INTO personal_memories
                (id,owner_id,kb_id,source_session_id,category,quote,content_hash,importance,indexed,created_at)
                VALUES(?,?,?,?,?,?,?,?,0,?)""",
                (memory_id, owner, kb_id, sid, category, quote.strip(), digest, importance, now()))
        return memory_id, True

    def activate_memory(self, owner: str, kb_id: str, memory_id: str):
        with self.connection(write=True) as db:
            result = db.execute("""UPDATE personal_memories SET indexed=1 WHERE id=? AND owner_id=? AND kb_id=?
                AND EXISTS(SELECT 1 FROM memory_settings WHERE owner_id=? AND kb_id=? AND enabled=1)""",
                (memory_id, owner, kb_id, owner, kb_id))
            if not result.rowcount:
                raise PermissionError("长期记忆已停用或不存在。")

    def delete_memory_record(self, owner: str, kb_id: str, memory_id: str):
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权访问。")
        with self.connection(write=True) as db:
            result = db.execute("DELETE FROM personal_memories WHERE id=? AND owner_id=? AND kb_id=?",
                                (memory_id, owner, kb_id))
            if not result.rowcount:
                raise PermissionError("长期记忆不存在或无权删除。")

    def create_document(self, owner: str, kb_id: str, name: str, digest: str, size: int,
                        storage_path: str) -> tuple[dict, bool]:
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权访问。")
        if not name or size <= 0:
            raise ValueError("文档为空或名称无效。")
        with self.connection(write=True) as db:
            existing = db.execute("""SELECT * FROM documents WHERE kb_id=? AND name=? AND content_hash=?
                AND active=1 AND status IN ('waiting','processing','success') ORDER BY version DESC LIMIT 1""",
                (kb_id, name, digest)).fetchone()
            if existing:
                return dict(existing), False
            version = db.execute("SELECT COALESCE(MAX(version),0)+1 FROM documents WHERE kb_id=? AND name=?",
                                 (kb_id, name)).fetchone()[0]
            doc_id, timestamp = uid(), now()
            db.execute("""INSERT INTO documents VALUES
                (?,?,?,?,?,?,?,?,?,?,?,?,?)""", (doc_id, kb_id, name, version, digest, size,
                storage_path, "waiting", "", 0, 1, timestamp, timestamp))
            db.execute("INSERT INTO jobs VALUES (?,?,?,?,?,?,?)", (uid(), doc_id, "waiting", 0, "", timestamp, timestamp))
            db.execute("UPDATE knowledge_bases SET updated_at=? WHERE id=?", (timestamp, kb_id))
            return dict(db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()), True

    def get_document(self, owner: str, document_id: str) -> dict | None:
        with self.connection() as db:
            row = db.execute("""SELECT d.* FROM documents d JOIN knowledge_bases k ON k.id=d.kb_id
                WHERE d.id=? AND k.owner_id=? AND k.deleted_at IS NULL""", (document_id, owner)).fetchone()
        return dict(row) if row else None

    def list_documents(self, owner: str, kb_id: str, *, active_only: bool = True) -> list[dict]:
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权访问。")
        with self.connection() as db:
            rows = db.execute("""SELECT d.*,j.attempts FROM documents d LEFT JOIN jobs j ON j.document_id=d.id
                WHERE d.kb_id=? AND (?=0 OR d.active=1) ORDER BY d.created_at DESC,d.version DESC""",
                (kb_id, int(active_only))).fetchall()
        return [dict(r) for r in rows]

    def job_status(self, owner: str, document_id: str, status: str, error: str = ""):
        if status not in {"waiting", "processing", "success", "failed"} or not self.get_document(owner, document_id):
            raise PermissionError("文档不存在或状态无效。")
        with self.connection(write=True) as db:
            db.execute("UPDATE documents SET status=?,error=?,updated_at=? WHERE id=?",
                       (status, error[:400], now(), document_id))
            db.execute("""UPDATE jobs SET status=?, error=?, attempts=attempts+?, updated_at=?
                WHERE document_id=?""", (status, error[:400], int(status == "processing"), now(), document_id))

    def complete_document(self, owner: str, document_id: str, chunks: list[dict]) -> list[str] | None:
        document = self.get_document(owner, document_id)
        if not document:
            raise PermissionError("文档不存在或无权访问。")
        with self.connection(write=True) as db:
            # A newer version may finish before an older upload. Never let the late
            # worker replace its successor or publish older chunks to keyword search.
            current = db.execute("SELECT status,active FROM documents WHERE id=?", (document_id,)).fetchone()
            if not current or current["status"] not in {"processing", "waiting"} or not current["active"]:
                raise ValueError("文档状态已变化，取消过期索引提交。")
            newer = db.execute("""SELECT id FROM documents WHERE kb_id=? AND name=?
                AND version>? AND active=1 AND status='success' LIMIT 1""",
                (document["kb_id"], document["name"], document["version"])).fetchone()
            if newer:
                db.execute("UPDATE documents SET active=0,status='deleted',updated_at=? WHERE id=?", (now(), document_id))
                db.execute("UPDATE jobs SET status='success',updated_at=? WHERE document_id=?", (now(), document_id))
                return None
            old = db.execute("""SELECT id FROM documents WHERE kb_id=? AND name=? AND id<>?
                AND active=1 AND status='success'""", (document["kb_id"], document["name"], document_id)).fetchall()
            old_ids = [r[0] for r in old]
            db.executemany("INSERT OR REPLACE INTO chunks(id,kb_id,document_id,location,content) VALUES(?,?,?,?,?)",
                           [(c["id"], document["kb_id"], document_id, c["location"], c["content"]) for c in chunks])
            db.execute("UPDATE documents SET status='success',chunk_count=?,error='',updated_at=? WHERE id=?",
                       (len(chunks), now(), document_id))
            for old_id in old_ids:
                db.execute("UPDATE documents SET active=0 WHERE id=?", (old_id,))
                db.execute("DELETE FROM chunks WHERE document_id=?", (old_id,))
            db.execute("UPDATE jobs SET status='success',error='',updated_at=? WHERE document_id=?", (now(), document_id))
            return old_ids

    def interrupted_documents(self, owner: str, kb_id: str) -> list[dict]:
        """List pending work for explicit recovery after a process restart."""
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权访问。")
        with self.connection() as db:
            rows = db.execute("""SELECT * FROM documents WHERE kb_id=? AND active=1
                AND status IN ('waiting','processing') ORDER BY created_at""", (kb_id,)).fetchall()
        return [dict(row) for row in rows]

    def list_chunks(self, owner: str, kb_id: str, document_ids: list[str] | None = None) -> list[dict]:
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权访问。")
        with self.connection() as db:
            rows = db.execute("""SELECT c.*, d.name AS source, d.version FROM chunks c
                JOIN documents d ON d.id=c.document_id WHERE c.kb_id=?
                AND d.active=1 AND d.status='success' ORDER BY c.rowid""", (kb_id,)).fetchall()
        return [dict(r) for r in rows if document_ids is None or r["document_id"] in document_ids]

    def get_chunk(self, owner: str, kb_id: str, chunk_id: str) -> dict | None:
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权访问。")
        with self.connection() as db:
            row = db.execute("""SELECT c.*,d.name AS source,d.version FROM chunks c
                JOIN documents d ON d.id=c.document_id
                WHERE c.id=? AND c.kb_id=? AND d.active=1 AND d.status='success'""",
                (chunk_id, kb_id)).fetchone()
        return dict(row) if row else None

    def replace_last_answer(self, owner: str, sid: str, answer: str, sources: list):
        if not self.get_session(owner, sid):
            raise PermissionError("会话不存在或无权访问。")
        with self.connection(write=True) as db:
            row = db.execute("SELECT id,role FROM messages WHERE session_id=? ORDER BY rowid DESC LIMIT 1", (sid,)).fetchone()
            if not row or row["role"] != "assistant":
                raise ValueError("没有可重新生成的回答。")
            db.execute("UPDATE messages SET content=?,sources_json=? WHERE id=?",
                       (answer, json.dumps(sources, ensure_ascii=False), row["id"]))
            db.execute("UPDATE sessions SET updated_at=? WHERE id=?", (now(), sid))
    def mark_deleted(self, owner: str, document_id: str):
        document = self.get_document(owner, document_id)
        if not document:
            raise PermissionError("文档不存在或无权访问。")
        with self.connection(write=True) as db:
            db.execute("DELETE FROM chunks WHERE document_id=?", (document_id,))
            db.execute("UPDATE documents SET active=0,status='deleted',updated_at=? WHERE id=?",
                       (now(), document_id))
            db.execute("UPDATE knowledge_bases SET updated_at=? WHERE id=?", (now(), document["kb_id"]))

    def delete_base(self, owner: str, kb_id: str):
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权删除。")
        with self.connection(write=True) as db:
            db.execute("DELETE FROM chunks WHERE kb_id=?", (kb_id,))
            db.execute("DELETE FROM jobs WHERE document_id IN (SELECT id FROM documents WHERE kb_id=?)", (kb_id,))
            db.execute("DELETE FROM documents WHERE kb_id=?", (kb_id,))
            db.execute("DELETE FROM personal_memories WHERE kb_id=? AND owner_id=?", (kb_id, owner))
            db.execute("DELETE FROM memory_settings WHERE kb_id=? AND owner_id=?", (kb_id, owner))
            db.execute("DELETE FROM sessions WHERE kb_id=? AND owner_id=?", (kb_id, owner))
            db.execute("DELETE FROM knowledge_bases WHERE id=? AND owner_id=?", (kb_id, owner))
    def archive_base(self, owner: str, kb_id: str):
        if not self.get_base(owner, kb_id):
            raise PermissionError("知识库不存在或无权访问。")
        with self.connection(write=True) as db:
            db.execute("UPDATE knowledge_bases SET deleted_at=?,updated_at=? WHERE id=? AND owner_id=?",
                       (now(), now(), kb_id, owner))