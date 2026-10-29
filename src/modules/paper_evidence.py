"""Section-aware evidence coverage for paper questions using authorized original chunks."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Sequence

from langchain_core.documents import Document

from .revision import REVISION_SUFFIX


def paper_intents(question: str) -> tuple[str, ...]:
    """Recognize reading intents; unrelated questions retain normal retrieval."""
    text = question.lower()
    overview = bool(re.search(r"主要讲|讲了什么|讲的什么|论文速览|概述|总结|梳理|摘要|overview|summari[sz]|what is this paper about", text))
    method = bool(re.search(r"方法|算法|怎么实现|如何实现|如何做|怎么做|method|algorithm|approach|framework|pipeline", text))
    contribution = bool(re.search(r"贡献|创新|新意|contribution|novelty|innovation", text))
    results = bool(re.search(r"实验|结果|结论|效果|评估|experiment|results?|findings?", text))
    if overview:
        return ("abstract", "contributions", "mechanism", "algorithm", "results")
    if method:
        return ("mechanism", "algorithm") + (("contributions",) if contribution else ())
    if contribution:
        return ("contributions", "mechanism")
    if results:
        return ("results", "abstract")
    return ()


def keyword_query(question: str) -> str:
    """Add English BM25 anchors to Chinese reading questions about English papers."""
    if not re.search(r"[\u3400-\u9fff]", question):
        return question
    intent = paper_intents(question)
    if not intent:
        return question
    anchors = {
        "abstract": "abstract research problem proposed",
        "contributions": "main contributions proposed",
        "mechanism": "proposed method framework reward model",
        "algorithm": "algorithm policy training procedure",
        "results": "simulation results experimental evaluation",
    }
    return question + " " + " ".join(anchors[part] for part in intent)


def _normalized(text: str) -> str:
    # PDF extractors often hyphenate headings across line breaks (contri- butions).
    return re.sub(r"(?<=\w)-\s+(?=\w)", "", text.lower())


def _score(section: str, content: str, position: float) -> int:
    text = _normalized(content)
    lead = text[:350]
    if re.search(r"\b(author contributions|references|bibliography|related work)\b", lead):
        return 0
    if section == "abstract" and position <= .12:
        return 1 + 2 * bool(re.search(r"\babstract\b", text[:950])) + 8 * bool(re.search(
            r"this (paper|work|study) (proposes|presents|introduces)|we (propose|present|introduce)", text)) + 2 * bool(re.search(r"\b(results|outperforms|simulation)\b", text))
    if section == "contributions" and position <= .35:
        return 5 * bool(re.search(r"(main|key|our) contributions|contributions of this paper", text)) + 2 * bool(re.search(r"\b(we propose|we develop|proposed)\b", text))
    if section == "mechanism" and .22 <= position <= .77:
        return 5 * bool(re.search(r"\b(our|proposed) (method|approach|framework|algorithm)|\b\d+(?:\.\d+)?\s+(?:[\w-]+\s+){0,5}(?:method|framework|algorithm)", text)) + 3 * bool(re.search(r"\b(reward|objective|optimization) (function|design|formulation)|\btraining and (decentralized|distributed) execution", text)) + 2 * bool(re.search(r"\b(actor|critic|policy|virtual queue)\b", text))
    if section == "algorithm" and .22 <= position <= .77:
        return 9 * bool(re.search(r"\balgorithm\s*\d+\b", text)) + 4 * bool(re.search(r"\b(require|ensure)\s*:", text)) + 2 * bool(re.search(r"\b(pseudocode|procedure|update|replay buffer|training step)\b", text))
    if section == "results" and .55 <= position <= .88:
        return 5 * bool(re.search(r"\b(simulation|experimental|evaluation) results\b|\bresults (show|demonstrate|indicate)\b", text)) + 2 * bool(re.search(r"\b(baseline|comparison|outperform)\b", text))
    return 0


def section_evidence(question: str, rows: Sequence[dict], limit: int) -> list[Document]:
    """Reserve distinct original excerpts for requested sections of a single paper."""
    sections = paper_intents(question)
    if not sections or limit <= 0:
        return []
    by_document: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_document[row["document_id"]].append(row)
    # Without a selected document, do not mix different papers for "this paper".
    if len(by_document) != 1:
        return []
    chunks = next(iter(by_document.values()))
    selected: list[Document] = []
    seen: set[str] = set()
    for section in sections:
        ranked = sorted(
            ((_score(section, row["content"], index / max(len(chunks)-1, 1)), index, row)
             for index, row in enumerate(chunks) if row["id"] not in seen),
            key=lambda item: (-item[0], item[1]),
        )
        if not ranked or ranked[0][0] <= 0:
            continue
        row = ranked[0][2]
        seen.add(row["id"])
        selected.append(Document(page_content=row["content"], metadata={
            "source": row["source"], "location": row["location"], "chunk_id": row["id"],
            "document_id": row["document_id"], "version": row["version"],
            "edited": row["location"].endswith(REVISION_SUFFIX),
        }))
        if len(selected) >= limit:
            break
    return selected