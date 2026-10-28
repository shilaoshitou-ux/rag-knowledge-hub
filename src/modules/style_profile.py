"""Opt-in, human-confirmed response style profile from repeated explicit user feedback.

Only persisted user messages are observed. Proposals never affect generation before
confirmation; approved instructions come from a fixed allow-list, not raw chat text.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

from .repository import Repository, now, uid

WINDOW_DAYS = 30
MIN_OBSERVATIONS = 3
MAX_OBSERVATIONS = 300
STYLE_OPTIONS = {
    "detail": {
        "detailed": ("回答更详细", "在证据充分时说明关键步骤与结论，避免只给一句概括；不凭空补充事实。"),
        "concise": ("回答更精炼", "优先直接回答并减少重复；必要事实和引用仍要保留。"),
    },
    "layout": {
        "bullets": ("优先分点呈现", "适合分点的问题优先使用清晰的条目，不能遗漏必要证据。"),
        "paragraphs": ("优先自然段", "适合叙述的问题优先使用短段落，而非机械列清单。"),
        "table": ("对比时优先表格", "适合对比的问题可使用 Markdown 表格；不适合时仍用普通段落。"),
    },
    "language": {
        "zh": ("优先中文", "优先使用中文作答，专业术语可保留英文。"),
        "en": ("优先英文", "优先使用英文作答；引用与证据约束保持不变。"),
    },
}
_FEEDBACK = {
    "detail": {
        "detailed": re.compile(r"太短(?:了)?|内容太少|不够(?:详细|全面|具体)|再(?:详细|具体|全面)(?:一点|些)?|讲(?:得)?更详细|展开(?:讲|说明)?(?:一点)?"),
        "concise": re.compile(r"太长(?:了)?|太啰嗦|过于啰嗦|再(?:简短|精炼|简洁)(?:一点|些)?|更(?:简洁|精炼)(?:一点|些)?|一句话(?:概括|回答)"),
    },
    "layout": {
        "bullets": re.compile(r"(?<!不要)(?<!不用)(?:请|用|按|改成|更适合)?分点(?:回答|说明|列出|写|讲)?|(?:用|按)条目(?:回答|列出)?|列成要点"),
        "paragraphs": re.compile(r"不要分点|不用分点|改成(?:自然段|段落)|用(?:自然段|段落)(?:回答|说明)?"),
        "table": re.compile(r"(?:用|以|改成)表格(?:展示|回答|列出|呈现|形式)?|整理成表格|表格形式(?:展示|回答)?"),
    },
    "language": {
        "zh": re.compile(r"(?:请|以后请|以后都|优先)?用中文(?:回答|解释|写|讲|作答|$)|回答(?:请)?用中文"),
        "en": re.compile(r"(?:请|以后请|以后都|优先)?用英文(?:回答|解释|写|讲|作答|$)|回答(?:请)?用英文"),
    },
}


def detect_feedback(message: str) -> dict[str, tuple[str, str]]:
    """Extract limited style signals; conflicts in one turn are ignored."""
    if len(message) > 1200:
        return {}
    result = {}
    for dimension, options in _FEEDBACK.items():
        found = [(value, match.group()) for value, pattern in options.items()
                 if (match := pattern.search(message)) and not (value != "paragraphs" and
                    re.search(r"(?:别|不要|不用|不必|无需|避免|禁止|不想)(?:再)?$",
                              message[max(0, match.start()-5):match.start()]))]
        if len(found) == 1:
            result[dimension] = found[0]
    return result


def _require_base(repo: Repository, owner: str, kb_id: str):
    if not repo.get_base(owner, kb_id):
        raise PermissionError("知识库不存在或无权访问。")


def style_enabled(repo: Repository, owner: str, kb_id: str) -> bool:
    _require_base(repo, owner, kb_id)
    with repo.connection() as db:
        row = db.execute("SELECT enabled FROM style_settings WHERE owner_id=? AND kb_id=?",
                         (owner, kb_id)).fetchone()
    return bool(row and row["enabled"] == 1)


def set_style_enabled(repo: Repository, owner: str, kb_id: str, enabled: bool):
    _require_base(repo, owner, kb_id)
    with repo.connection(write=True) as db:
        db.execute("""INSERT INTO style_settings(owner_id,kb_id,enabled) VALUES(?,?,?)
            ON CONFLICT(owner_id,kb_id) DO UPDATE SET enabled=excluded.enabled""",
            (owner, kb_id, int(enabled)))


def _cutoff() -> str:
    return (datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS)).isoformat(timespec="seconds")


def observe_style_turn(repo: Repository, owner: str, kb_id: str, sid: str, message_id: str) -> list[str]:
    """Read a committed user message by ID; propose only after 3 compatible turns."""
    _require_base(repo, owner, kb_id)
    with repo.connection(write=True) as db:
        message = db.execute("""SELECT m.role,m.content FROM messages m
            JOIN sessions s ON s.id=m.session_id
            WHERE m.id=? AND m.session_id=? AND s.owner_id=? AND s.kb_id=?
            AND EXISTS (SELECT 1 FROM messages a WHERE a.rowid=m.rowid+1
                        AND a.session_id=m.session_id AND a.role='assistant')""",
            (message_id, sid, owner, kb_id)).fetchone()
        if not message or message["role"] != "user":
            raise PermissionError("只能观察当前账号当前会话的用户原话。")
        enabled = db.execute("SELECT enabled FROM style_settings WHERE owner_id=? AND kb_id=?",
                             (owner, kb_id)).fetchone()
        if not enabled or not enabled["enabled"]:
            return []
        signals = detect_feedback(message["content"])
        if not signals:
            return []
        cutoff = _cutoff()
        db.execute("DELETE FROM style_observations WHERE owner_id=? AND kb_id=? AND created_at<?",
                   (owner, kb_id, cutoff))
        current_count = db.execute("SELECT COUNT(*) FROM style_observations WHERE owner_id=? AND kb_id=?",
                                   (owner, kb_id)).fetchone()[0]
        created = []
        for dimension, (value, excerpt) in signals.items():
            if current_count >= MAX_OBSERVATIONS:
                break
            inserted = db.execute("""INSERT OR IGNORE INTO style_observations
                (id,owner_id,kb_id,message_id,dimension,value,evidence,created_at)
                VALUES(?,?,?,?,?,?,?,?)""",
                (uid(), owner, kb_id, message_id, dimension, value, excerpt[:80], now()))
            if not inserted.rowcount:
                continue
            current_count += 1
            latest = db.execute("""SELECT rowid AS serial,id,value FROM style_observations
                WHERE owner_id=? AND kb_id=? AND dimension=? AND created_at>=?
                ORDER BY rowid DESC LIMIT ?""", (owner, kb_id, dimension, cutoff, MIN_OBSERVATIONS)).fetchall()
            if len(latest) < MIN_OBSERVATIONS or any(row["value"] != value for row in latest):
                continue
            profile = db.execute("""SELECT value FROM style_profiles
                WHERE owner_id=? AND kb_id=? AND dimension=?""", (owner, kb_id, dimension)).fetchone()
            if profile and profile["value"] == value:
                continue
            existing = db.execute("""SELECT * FROM style_proposals
                WHERE owner_id=? AND kb_id=? AND dimension=?""", (owner, kb_id, dimension)).fetchone()
            if existing and existing["status"] == "rejected" and any(
                    row["serial"] <= existing["last_observation_rowid"] for row in latest):
                continue
            evidence_ids = [row["id"] for row in latest]
            if existing and existing["status"] == "pending" and existing["value"] == value and _proposal_evidence(db, owner, kb_id, dict(existing)):
                continue
            proposal_id = existing["id"] if existing else uid()
            db.execute("""INSERT INTO style_proposals
                (id,owner_id,kb_id,dimension,value,status,evidence_json,last_observation_rowid,created_at,resolved_at)
                VALUES(?,?,?,?,?,'pending',?,0,?,NULL)
                ON CONFLICT(owner_id,kb_id,dimension) DO UPDATE SET
                    value=excluded.value,status='pending',evidence_json=excluded.evidence_json,
                    created_at=excluded.created_at,resolved_at=NULL""",
                (proposal_id, owner, kb_id, dimension, value,
                 json.dumps(evidence_ids), now()))
            created.append(proposal_id)
        return created


def _proposal_evidence(db, owner: str, kb_id: str, proposal: dict) -> list[dict]:
    evidence_ids = json.loads(proposal["evidence_json"])
    if not isinstance(evidence_ids, list) or len(evidence_ids) != MIN_OBSERVATIONS:
        return []
    rows = db.execute("""SELECT o.id,o.value,o.evidence,o.created_at FROM style_observations o
        JOIN messages m ON m.id=o.message_id AND m.role='user'
        WHERE o.owner_id=? AND o.kb_id=? AND o.dimension=? AND o.created_at>=?
        AND o.id IN (?,?,?)""", (owner, kb_id, proposal["dimension"], _cutoff(), *evidence_ids)).fetchall()
    found = {row["id"]: dict(row) for row in rows}
    if len(found) != MIN_OBSERVATIONS or any(found[mid]["value"] != proposal["value"] for mid in evidence_ids if mid in found):
        return []
    return [found[mid] for mid in evidence_ids]


def list_style_proposals(repo: Repository, owner: str, kb_id: str) -> list[dict]:
    _require_base(repo, owner, kb_id)
    with repo.connection() as db:
        rows = db.execute("""SELECT * FROM style_proposals WHERE owner_id=? AND kb_id=?
            AND status='pending' ORDER BY created_at DESC""", (owner, kb_id)).fetchall()
        return [{**dict(row), "evidence": evidence} for row in rows
                if (evidence := _proposal_evidence(db, owner, kb_id, dict(row)))]


def confirm_style_proposal(repo: Repository, owner: str, kb_id: str, proposal_id: str):
    if not style_enabled(repo, owner, kb_id):
        raise PermissionError("请先开启风格观察。")
    with repo.connection(write=True) as db:
        row = db.execute("""SELECT * FROM style_proposals WHERE id=? AND owner_id=? AND kb_id=?
            AND status='pending'""", (proposal_id, owner, kb_id)).fetchone()
        if not row or row["value"] not in STYLE_OPTIONS.get(row["dimension"], {}):
            raise PermissionError("候选画像不存在或无权确认。")
        proposal = dict(row)
        if len(_proposal_evidence(db, owner, kb_id, proposal)) != MIN_OBSERVATIONS:
            raise ValueError("候选的原始反馈已过期或被删除，请继续对话积累新反馈。")
        latest = db.execute("""SELECT value FROM style_observations
            WHERE owner_id=? AND kb_id=? AND dimension=? AND created_at>=?
            ORDER BY rowid DESC LIMIT ?""", (owner, kb_id, row["dimension"], _cutoff(), MIN_OBSERVATIONS)).fetchall()
        if len(latest) < MIN_OBSERVATIONS or any(item["value"] != row["value"] for item in latest):
            raise ValueError("近期出现了相反的风格反馈，候选已失效，请拒绝后继续观察。")
        db.execute("""INSERT INTO style_profiles(owner_id,kb_id,dimension,value,confirmed_at)
            VALUES(?,?,?,?,?) ON CONFLICT(owner_id,kb_id,dimension) DO UPDATE SET
            value=excluded.value,confirmed_at=excluded.confirmed_at""",
            (owner, kb_id, row["dimension"], row["value"], now()))
        db.execute("UPDATE style_proposals SET status='accepted',resolved_at=? WHERE id=?",
                   (now(), proposal_id))


def reject_style_proposal(repo: Repository, owner: str, kb_id: str, proposal_id: str):
    _require_base(repo, owner, kb_id)
    with repo.connection(write=True) as db:
        row = db.execute("""SELECT dimension FROM style_proposals
            WHERE id=? AND owner_id=? AND kb_id=? AND status='pending'""",
            (proposal_id, owner, kb_id)).fetchone()
        if not row:
            raise PermissionError("候选画像不存在或无权拒绝。")
        serial = db.execute("""SELECT COALESCE(MAX(rowid),0) FROM style_observations
            WHERE owner_id=? AND kb_id=? AND dimension=?""", (owner, kb_id, row["dimension"])).fetchone()[0]
        db.execute("""UPDATE style_proposals SET status='rejected',resolved_at=?,last_observation_rowid=?
            WHERE id=?""", (now(), serial, proposal_id))


def get_style_profiles(repo: Repository, owner: str, kb_id: str) -> list[dict]:
    _require_base(repo, owner, kb_id)
    with repo.connection() as db:
        rows = db.execute("""SELECT dimension,value,confirmed_at FROM style_profiles
            WHERE owner_id=? AND kb_id=? ORDER BY dimension""", (owner, kb_id)).fetchall()
    return [dict(row) for row in rows]


def approved_style_instructions(repo: Repository, owner: str, kb_id: str) -> str:
    if not style_enabled(repo, owner, kb_id):
        return ""
    return " ".join(STYLE_OPTIONS[row["dimension"]][row["value"]][1]
                    for row in get_style_profiles(repo, owner, kb_id)
                    if row["value"] in STYLE_OPTIONS.get(row["dimension"], {}))


def remove_style_profile(repo: Repository, owner: str, kb_id: str, dimension: str):
    _require_base(repo, owner, kb_id)
    if dimension not in STYLE_OPTIONS:
        raise ValueError("风格维度不存在。")
    with repo.connection(write=True) as db:
        db.execute("DELETE FROM style_profiles WHERE owner_id=? AND kb_id=? AND dimension=?",
                   (owner, kb_id, dimension))
        db.execute("DELETE FROM style_proposals WHERE owner_id=? AND kb_id=? AND dimension=?",
                   (owner, kb_id, dimension))
        db.execute("DELETE FROM style_observations WHERE owner_id=? AND kb_id=? AND dimension=?",
                   (owner, kb_id, dimension))


def clear_style_data(repo: Repository, owner: str, kb_id: str):
    _require_base(repo, owner, kb_id)
    with repo.connection(write=True) as db:
        for table in ("style_profiles", "style_proposals", "style_observations"):
            db.execute(f"DELETE FROM {table} WHERE owner_id=? AND kb_id=?", (owner, kb_id))
