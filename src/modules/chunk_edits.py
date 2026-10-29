"""Owner-scoped, single-chunk knowledge-base revisions (original uploads stay untouched)."""

from __future__ import annotations

import hashlib
import logging

from langchain_core.documents import Document

from .managed_index import vector_store
from .repository import Repository, uid
from .revision import REVISION_SUFFIX
from .vector_cleanup import queue_cleanup

LOGGER = logging.getLogger(__name__)
MAX_EDIT_CHARS = 4000


def _editable(repo: Repository, owner: str, kb_id: str, document_id: str, chunk_id: str):
    kb = repo.get_base(owner, kb_id)
    doc = repo.get_document(owner, document_id)
    if not kb or not doc or doc["kb_id"] != kb_id or not doc["active"] or doc["status"] != "success":
        raise PermissionError("文档不存在、尚未处理成功或无权编辑。")
    chunk = repo.get_chunk(owner, kb_id, chunk_id)
    if not chunk or chunk["document_id"] != document_id:
        raise ValueError("片段已被更新或删除，请刷新后重试。")
    if any(other["id"] != document_id and other["name"] == doc["name"]
           and other["status"] in {"waiting", "processing"}
           for other in repo.list_documents(owner, kb_id)):
        raise ValueError("同名文档有版本正在处理，请完成后再修改片段。")
    return kb, doc, chunk


def _require_key(kb: dict, api_key: str):
    if kb["embedding_provider"] == "company" and not api_key:
        raise ValueError("在线向量化需要当前知识库对应服务的密钥。")


def _delete_vector(store, chunk_id: str, repo: Repository, owner: str, kb_id: str) -> bool:
    """SQLite is authoritative; failed physical cleanup must not expose stale chunks."""
    try:
        store.delete(ids=[chunk_id])
        return False
    except Exception:
        queue_cleanup(repo, owner, kb_id, [chunk_id])
        LOGGER.warning("Chunk vector cleanup queued: %s", chunk_id, exc_info=False)
        return True


def update_chunk(repo: Repository, owner: str, kb_id: str, document_id: str,
                 chunk_id: str, new_content: str, api_key: str) -> tuple[str, bool]:
    """Embed one revised excerpt, then atomically switch the active SQLite row."""
    kb, doc, chunk = _editable(repo, owner, kb_id, document_id, chunk_id)
    _require_key(kb, api_key)
    content = new_content.strip()
    if not content or len(content) > MAX_EDIT_CHARS:
        raise ValueError("片段不能为空或超过 4000 字符。")
    if content == chunk["content"]:
        raise ValueError("片段内容未变化。")
    new_id = hashlib.sha256(f"{chunk_id}:{uid()}:{content}".encode("utf-8")).hexdigest()
    location = chunk["location"]
    if not location.endswith(REVISION_SUFFIX):
        location += REVISION_SUFFIX
    store = vector_store(kb, api_key)
    revised = Document(page_content=content, metadata={
        "source": doc["name"], "location": location, "chunk_id": new_id,
        "document_id": document_id, "version": doc["version"], "edited": True,
        "kb_id": kb_id,
    })
    try:
        store.add_documents([revised], ids=[new_id])
    except Exception as exc:
        _delete_vector(store, new_id, repo, owner, kb_id)  # The store may have written before raising.
        raise RuntimeError("片段向量化失败，旧内容未改变；请检查向量模型后重试。") from exc
    try:
        repo.replace_chunk(owner, kb_id, document_id, chunk_id, new_id, content, location)
    except Exception:
        _delete_vector(store, new_id, repo, owner, kb_id)
        raise
    return new_id, _delete_vector(store, chunk_id, repo, owner, kb_id)


def delete_chunk(repo: Repository, owner: str, kb_id: str, document_id: str,
                 chunk_id: str, api_key: str) -> bool:
    """Remove one active excerpt without removing or rewriting the uploaded file."""
    kb, _, _ = _editable(repo, owner, kb_id, document_id, chunk_id)
    _require_key(kb, api_key)
    # Validate the vector configuration before publishing the deletion.
    store = vector_store(kb, api_key)
    repo.remove_chunk(owner, kb_id, document_id, chunk_id)
    return _delete_vector(store, chunk_id, repo, owner, kb_id)
