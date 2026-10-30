"""Async, versioned document ingestion into new per-knowledge-base Chroma collections."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock

from langchain_chroma import Chroma
from openai import APIConnectionError
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embedder import ONLINE_BATCH_SIZE, UploadedDocument, display_name, make_embeddings, parse_document
from .repository import DATA_DIR, Repository, uid

LOGGER = logging.getLogger(__name__)
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="rag-document")
_pending: set[str] = set()
_lock = Lock()
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".markdown", ".csv"}


def vector_store(kb: dict, api_key: str) -> Chroma:
    return Chroma(
        collection_name=f"kb_{kb['id']}",
        persist_directory=str(DATA_DIR / "vectors"),
        embedding_function=make_embeddings(kb["embedding_provider"], kb["embedding_model"], api_key, kb["base_url"]),
    )


def _safe_upload_path(owner: str, kb_id: str, file_id: str, suffix: str) -> Path:
    root = (DATA_DIR / "uploads").resolve()
    path = (root / owner / kb_id / f"{file_id}{suffix}").resolve()
    if not path.is_relative_to(root):
        raise ValueError("文件路径不安全。")
    return path


def queue_upload(repo: Repository, owner: str, kb: dict, file: UploadedDocument,
                 api_key: str) -> tuple[dict, bool]:
    name = display_name(file.name)
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("只支持 PDF、Word、TXT、Markdown 和 CSV。")
    # Validate size/content before any persistent writes.
    if not file.content or len(file.content) > 100 * 1024 * 1024:
        raise ValueError("文件为空或超过 100 MB。")
    if kb["embedding_provider"] == "company" and not api_key:
        raise ValueError("在线向量化需要当前服务的密钥。")
    path = _safe_upload_path(owner, kb["id"], uid(), suffix)
    digest = hashlib.sha256(file.content).hexdigest()
    document, created = repo.create_document(owner, kb["id"], name, digest, len(file.content), str(path))
    if not created:
        return document, False
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_bytes(file.content)
        os.replace(temp_path, path)
    except OSError as exc:
        repo.job_status(owner, document["id"], "failed", "本机文件写入失败。")
        raise ValueError("本机文件写入失败，请检查数据目录权限。") from exc
    enqueue(repo.path, owner, kb["id"], document["id"], api_key)
    return document, True


def enqueue(database: str | Path, owner: str, kb_id: str, document_id: str, api_key: str) -> bool:
    with _lock:
        if document_id in _pending:
            return False
        _pending.add(document_id)
    def submit(attempt: int):
        future = _executor.submit(_process, Path(database), owner, kb_id, document_id, api_key)

        def done(result):
            resubmitted = False
            try:
                retryable = result.result()
                if retryable and attempt < 3:
                    current = Repository(database)
                    doc = current.get_document(owner, document_id)
                    if doc and doc["active"] and doc["status"] == "failed":
                        time.sleep(min(2 ** attempt, 8))
                        current.job_status(owner, document_id, "waiting", "短暂服务故障，自动重试中。")
                        submit(attempt + 1)
                        resubmitted = True
            except Exception:
                LOGGER.exception("Document worker exited unexpectedly: %s", document_id)
                try:
                    Repository(database).job_status(owner, document_id, "failed", "后台任务异常退出，请手动重试。")
                except Exception:
                    LOGGER.exception("Could not persist failed job status: %s", document_id)
            finally:
                if not resubmitted:
                    with _lock:
                        _pending.discard(document_id)
        future.add_done_callback(done)

    try:
        submit(1)
    except Exception:
        with _lock:
            _pending.discard(document_id)
        raise
    return True


def _process(database: Path, owner: str, kb_id: str, document_id: str, api_key: str):
    repo = Repository(database)
    kb = repo.get_base(owner, kb_id)
    document = repo.get_document(owner, document_id)
    if not kb or not document or document["status"] == "deleted":
        return
    ids_added: list[str] = []
    store = None
    stage = "parse"
    try:
        repo.job_status(owner, document_id, "processing")
        raw = Path(document["storage_path"]).read_bytes()
        parsed = parse_document(UploadedDocument(document["name"], raw))
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=kb["chunk_size"], chunk_overlap=kb["chunk_overlap"],
        ).split_documents(parsed)
        chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
        if not chunks:
            raise ValueError("没有可提取的文本；扫描件需先 OCR。")
        records = []
        for number, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(f"{document_id}:{number}:{chunk.page_content}".encode()).hexdigest()
            chunk.metadata.update({"kb_id": kb_id, "document_id": document_id,
                                   "version": document["version"], "chunk_id": chunk_id})
            records.append({"id": chunk_id, "location": chunk.metadata.get("location", "全文"),
                            "content": chunk.page_content})
        stage = "embed"
        store = vector_store(kb, api_key)
        size = ONLINE_BATCH_SIZE if kb["embedding_provider"] == "company" else 64
        for start in range(0, len(chunks), size):
            batch = chunks[start:start + size]
            batch_ids = [record["id"] for record in records[start:start + size]]
            # Some vector stores may write a prefix before raising; include the
            # entire attempted batch in compensating cleanup.
            ids_added.extend(batch_ids)
            store.add_documents(batch, ids=batch_ids)
        previous = [d for d in repo.list_documents(owner, kb_id) if
                    d["name"] == document["name"] and d["id"] != document_id and
                    d["status"] == "success"]
        previous_ids = [row["id"] for row in repo.list_chunks(owner, kb_id)
                        if row["document_id"] in {d["id"] for d in previous}]
        stage = "commit"
        completed = repo.complete_document(owner, document_id, records)
        if completed is None:
            store.delete(ids=ids_added)
            return
        if previous_ids:
            try:
                store.delete(ids=previous_ids)
            except Exception:
                from .vector_cleanup import queue_cleanup
                queue_cleanup(repo, owner, kb_id, previous_ids)
                LOGGER.warning("Old vector cleanup queued for document %s", document_id)
    except Exception as exc:
        if store and ids_added:
            try:
                store.delete(ids=ids_added)
            except Exception:
                from .vector_cleanup import queue_cleanup
                queue_cleanup(repo, owner, kb_id, ids_added)
                LOGGER.warning("Partial vector cleanup queued for document %s", document_id)
        # Never persist model-service exception bodies; gateways may echo document text or keys.
        status_code = getattr(exc, "status_code", None)
        if isinstance(exc, ValueError) and str(exc).startswith("没有可提取的文本"):
            message = "没有可提取的文本；扫描版 PDF 需先 OCR。"
        elif status_code:
            message = f"向量服务返回 HTTP {status_code}（模型：{kb['embedding_model']}）。请检查权限、额度或网关状态。"
        elif stage == "parse":
            message = "文件解析失败，请检查是否损坏、加密或没有可提取文本。"
        elif kb["embedding_provider"] == "ollama" and stage == "embed":
            message = f"本地向量化失败，请检查 Ollama 是否运行并已下载模型 {kb['embedding_model']}。"
        elif isinstance(exc, APIConnectionError):
            message = "连接向量服务失败或超时，请检查网络/接口地址并重试。"
        else:
            message = "向量化或索引写入失败，请核对向量模型、服务额度并重试。"
        repo.job_status(owner, document_id, "failed", message)
        LOGGER.warning("Document processing failed: id=%s, class=%s, status=%s",
                       document_id, type(exc).__name__, status_code)
        return status_code in {429, 500, 502, 503, 504} or isinstance(exc, APIConnectionError)


def retry_document(repo: Repository, owner: str, kb_id: str, document_id: str, api_key: str):
    doc = repo.get_document(owner, document_id)
    kb = repo.get_base(owner, kb_id)
    if not doc or not kb or doc["kb_id"] != kb_id or doc["status"] != "failed" or not doc["active"]:
        raise ValueError("只有当前知识库中处理失败的文档可以重试。")
    if kb["embedding_provider"] == "company" and not api_key:
        raise ValueError("重试在线向量化需要对应服务的密钥。")
    if not Path(doc["storage_path"]).is_file():
        raise ValueError("本机原文件不存在，需重新上传。")
    with _lock:
        if document_id in _pending:
            raise ValueError("文档已在自动重试队列中，请等待任务结束。")
    repo.job_status(owner, document_id, "waiting")
    enqueue(repo.path, owner, kb_id, document_id, api_key)


def interrupted_count(repo: Repository, owner: str, kb_id: str) -> int:
    """Count only unfinished records with no worker active in this process."""
    documents = repo.interrupted_documents(owner, kb_id)
    with _lock:
        return sum(doc["id"] not in _pending for doc in documents)


def recover_interrupted(repo: Repository, owner: str, kb: dict, api_key: str) -> int:
    """Requeue orphaned jobs on demand, e.g. after restart; never persist API keys."""
    if not repo.get_base(owner, kb["id"]):
        raise PermissionError("知识库不存在或无权访问。")
    if kb["embedding_provider"] == "company" and not api_key:
        raise ValueError("恢复在线任务需要当前服务的密钥。")
    count = 0
    for doc in repo.interrupted_documents(owner, kb["id"]):
        with _lock:
            running = doc["id"] in _pending
        if running:
            continue
        if not Path(doc["storage_path"]).is_file():
            repo.job_status(owner, doc["id"], "failed", "原文件已丢失，请重新上传。")
            continue
        if enqueue(repo.path, owner, kb["id"], doc["id"], api_key):
            count += 1
    return count


def delete_document(repo: Repository, owner: str, kb: dict, document_id: str, api_key: str):
    doc = repo.get_document(owner, document_id)
    if not doc or doc["kb_id"] != kb["id"] or not doc["active"]:
        raise PermissionError("文档不存在或无权删除。")
    with _lock:
        if any(other["name"] == doc["name"] and other["id"] in _pending
               for other in repo.list_documents(owner, kb["id"])):
            raise ValueError("此文件有版本正在处理，请完成后再删除。")
    ids = [row["id"] for row in repo.list_chunks(owner, kb["id"]) if row["document_id"] == document_id]
    # Publish deletion first: retrieval rejects the old IDs even if physical cleanup fails.
    repo.mark_deleted(owner, document_id)
    if ids:
        try:
            vector_store(kb, api_key).delete(ids=ids)
        except Exception as exc:
            from .vector_cleanup import queue_cleanup
            queue_cleanup(repo, owner, kb["id"], ids)
            LOGGER.warning("Document vector cleanup queued (%s)", type(exc).__name__)
    path = Path(doc["storage_path"]).resolve()
    if path.is_relative_to((DATA_DIR / "uploads").resolve()):
        path.unlink(missing_ok=True)

def delete_base(repo: Repository, owner: str, kb: dict, api_key: str):
    if not repo.get_base(owner, kb["id"]):
        raise PermissionError("知识库不存在或无权删除。")
    docs = repo.list_documents(owner, kb["id"], active_only=False)
    with _lock:
        if any(doc["id"] in _pending for doc in docs):
            raise ValueError("仍有文档处理任务，请等待完成后再删除知识库。")
    if any(repo.pending_turn(owner, session["id"]) for session in repo.list_sessions(owner, kb["id"])):
        raise ValueError("该知识库仍有待恢复问答；请先继续或放弃任务，再删除知识库。")
    from .long_term_memory import delete_memory_collection
    # Collection deletion is local and does not embed or call the model service.
    delete_key = api_key or "delete-only"
    delete_memory_collection(repo, owner, kb, delete_key)
    vector_store(kb, delete_key).delete_collection()
    repo.delete_base(owner, kb["id"])
    for doc in docs:
        path = Path(doc["storage_path"]).resolve()
        if path.is_relative_to((DATA_DIR / "uploads").resolve()):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("File cleanup failed after KB deletion: %s", doc["id"])
