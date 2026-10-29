"""Owner-scoped retrieval of original uploaded PDF pages for citation inspection."""

from pathlib import Path
from pypdf import PdfReader

from .repository import DATA_DIR, Repository
from .revision import REVISION_SUFFIX


def original_page(repo: Repository, owner: str, kb_id: str, source: dict) -> str | None:
    if source.get("edited") or str(source.get("location", "")).endswith(REVISION_SUFFIX):
        return None  # Knowledge-base overlays must never be presented as original PDF text.
    document_id = source.get("document_id")
    doc = repo.get_document(owner, document_id) if document_id else None
    if not doc or doc["kb_id"] != kb_id or not doc["active"] or doc["status"] != "success":
        return None
    page = source.get("page")
    if not isinstance(page, int):
        location = source.get("location", "")
        if location.startswith("第 ") and location.endswith(" 页"):
            try:
                page = int(location[2:-2].strip())
            except ValueError:
                return None
    if not isinstance(page, int) or page < 1 or not doc["name"].lower().endswith(".pdf"):
        return None
    root = (DATA_DIR / "uploads").resolve()
    path = Path(doc["storage_path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        return None
    reader = PdfReader(path)
    return reader.pages[page - 1].extract_text() if page <= len(reader.pages) else None
