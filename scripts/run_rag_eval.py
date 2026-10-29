"""Run a labelled RAG benchmark locally; do not commit private results or API keys."""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.modules.evaluation import chunk_manifest, evaluate, parse_cases, summarize
from src.modules.repository import DATA_DIR, Repository


def run(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Run an owner-scoped, labelled RAG evaluation")
    parser.add_argument("--username", required=True)
    parser.add_argument("--kb-id", required=True, help="Current knowledge base ID (not its display name)")
    parser.add_argument("--cases", type=Path, help="UTF-8 JSON questions with relevant_documents/chunks")
    parser.add_argument("--model", help="Chat model ID for rerank and optional generation")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--judge", action="store_true", help="Optional extra paid LLM judgement")
    parser.add_argument("--manifest", action="store_true", help="Export active chunk IDs for manual labelling")
    args = parser.parse_args(argv)
    if not args.manifest and (not args.cases or not args.model):
        parser.error("evaluation requires --cases and --model; --manifest requires neither")
    repo = Repository()
    account = repo.authenticate(args.username, getpass.getpass("Local account password: "))
    if not account:
        raise PermissionError("Invalid account credentials")
    kb = repo.get_base(account["id"], args.kb_id)
    if not kb:
        raise PermissionError("Knowledge base not found or not owned by this account")
    output_dir = DATA_DIR / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if args.manifest:
        payload = {"kb_id": kb["id"], "generated_at_utc": timestamp,
                   "active_chunks": chunk_manifest(repo, account["id"], kb["id"])}
        output = output_dir / f"chunk-manifest-{timestamp}.json"
    else:
        api_key = os.environ.get("RAG_EVAL_API_KEY", "")
        if not api_key:
            raise ValueError("Set RAG_EVAL_API_KEY in the current process; never pass it in CLI args")
        raw = args.cases.read_bytes()
        cases = parse_cases(raw, repo, account["id"], kb["id"])
        details = evaluate(repo, account["id"], kb, cases, api_key, args.model,
                           args.k, semantic_judge=args.judge)
        payload = {"kb_id": kb["id"], "kb_name": kb["name"],
                   "generated_at_utc": timestamp, "dataset_sha256": hashlib.sha256(raw).hexdigest(),
                   "case_count": len(cases), "evidence_labelled": sum(bool(c.relevant_chunk_ids) for c in cases),
                   "chat_model": args.model, "embedding_model": kb["embedding_model"],
                   "k": args.k, "semantic_judge": args.judge,
                   "summary": summarize(details), "details": details}
        output = output_dir / f"rag-evaluation-{timestamp}.json"
    # Save only under the ignored local data tree; never write credentials into results.
    staged = output.with_suffix(".tmp")
    try:
        staged.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(staged, output)
    finally:
        staged.unlink(missing_ok=True)
    print(f"Private local result: {output}")
    if not args.manifest:
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return output


if __name__ == "__main__":
    run()
