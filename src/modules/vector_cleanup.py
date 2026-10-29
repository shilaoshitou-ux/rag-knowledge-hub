"""Durable compensating cleanup and scoped Chroma/SQLite index audit."""

from __future__ import annotations

import logging
import re

from langchain_core.documents import Document

from .managed_index import vector_store
from .repository import Repository
from .revision import REVISION_SUFFIX

LOGGER = logging.getLogger(__name__)


def queue_cleanup(repo: Repository, owner: str, kb_id: str, ids: list[str]) -> None:
    """Persist failed physical deletes without storing document text or service errors."""
    if not ids:
        return
    repo.enqueue_vector_cleanup(owner, kb_id, ids)


def _idle(repo: Repository, owner: str, kb_id: str) -> None:
    if any(doc["status"] in {"waiting", "processing"}
           for doc in repo.list_documents(owner, kb_id)):
        raise ValueError("文档仍在建索引；请等处理完成后再对账或重试向量清理。")


def retry_cleanup(repo: Repository, owner: str, kb: dict, api_key: str,
                  *, limit: int = 100) -> dict:
    """Never delete an ID published as active again; retry only stale vectors."""
    _idle(repo, owner, kb["id"])
    tasks = repo.list_vector_cleanup(owner, kb["id"], limit)
    result = {"pending": len(tasks), "deleted": 0, "failed": 0, "active_skipped": 0}
    if not tasks:
        return result
    active = {row["id"] for row in repo.list_chunks(owner, kb["id"])}
    store = vector_store(kb, api_key)
    for task in tasks:
        cid = task["chunk_id"]
        if cid in active:
            repo.finish_vector_cleanup(owner, kb["id"], cid)
            result["active_skipped"] += 1
            continue
        try:
            store.delete(ids=[cid])  # Deleting an already-gone ID is idempotent.
        except Exception as exc:
            repo.finish_vector_cleanup(owner, kb["id"], cid, failed=True)
            result["failed"] += 1
            LOGGER.warning("Vector cleanup retry failed (%s)", type(exc).__name__)
        else:
            repo.finish_vector_cleanup(owner, kb["id"], cid)
            result["deleted"] += 1
    return result


def _scan_ids(store, page_size: int, max_vectors: int) -> set[str]:
    page_size = max(1, min(page_size, 1000))
    vector_ids: set[str] = set()
    offset = 0
    while offset < max_vectors:
        batch = store.get(include=[], limit=min(page_size, max_vectors - offset), offset=offset)
        ids = batch["ids"]
        vector_ids.update(ids)
        offset += len(ids)
        if len(ids) < page_size:
            break
    if offset >= max_vectors:
        raise ValueError("向量数量超出单次对账上限，请分批运维，不要将不完整结果当作对账结论。")
    return vector_ids


def audit_vectors(repo: Repository, owner: str, kb: dict, api_key: str,
                  *, page_size: int = 500, max_vectors: int = 100_000) -> dict:
    """Compare only IDs in this KB's collection; enqueue orphan vectors, report missing."""
    _idle(repo, owner, kb["id"])
    vector_ids = _scan_ids(vector_store(kb, api_key), page_size, max_vectors)
    active_ids = {row["id"] for row in repo.list_chunks(owner, kb["id"])}
    stale = sorted(vector_ids - active_ids)
    missing = sorted(active_ids - vector_ids)
    queue_cleanup(repo, owner, kb["id"], stale)
    return {"vectors": len(vector_ids), "active_chunks": len(active_ids),
            "stale_queued": len(stale), "missing_active": len(missing),
            "missing_ids": missing[:20]}


def repair_missing(repo: Repository, owner: str, kb: dict, api_key: str, *, limit: int = 20) -> dict:
    """Re-embed only authorized, currently active SQLite chunks missing from Chroma."""
    _idle(repo, owner, kb["id"])
    store = vector_store(kb, api_key)
    existing = _scan_ids(store, 500, 100_000)
    active = {row["id"]: row for row in repo.list_chunks(owner, kb["id"])}
    missing = sorted(set(active) - existing)
    repaired = 0
    for cid in missing[:max(1, min(limit, 50))]:
        row = repo.get_chunk(owner, kb["id"], cid)
        if not row:
            continue  # A concurrent edit removed the candidate; never re-publish old text.
        metadata = {"source": row["source"], "location": row["location"],
                    "chunk_id": cid, "document_id": row["document_id"], "kb_id": kb["id"],
                    "version": row["version"], "edited": row["location"].endswith(REVISION_SUFFIX)}
        page = re.fullmatch(r"第 (\d+) 页", row["location"])
        if page:
            metadata["page"] = int(page.group(1))
        store.add_documents([Document(page_content=row["content"], metadata=metadata)], ids=[cid])
        repaired += 1
    return {"missing": len(missing), "repaired": repaired, "remaining_estimate": len(missing) - repaired}
