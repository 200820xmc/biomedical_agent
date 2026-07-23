"""使用已有解析 Markdown 逐篇重建 Milvus 索引。

该脚本不会调用 xParse，也不会删除 collection。每篇文献复用
``VectorIndexService.index_content`` 的“先写新版本、再删除旧版本”流程，
并在每篇完成后校验稳定元数据和逻辑 Chunk 去重。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from app.core.milvus_client import milvus_manager
from app.services.vector_index_service import vector_index_service


ORIGINALS_DIR = PROJECT_ROOT / "uploads" / "originals"
PARSED_DIR = PROJECT_ROOT / "uploads" / "parsed"
DEFAULT_PROGRESS = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "KB_MAINTENANCE_20260722"
    / "reindex_progress.json"
)
REQUIRED_METADATA = {"source_id", "chunk_index", "content_hash", "chunk_id"}


def _query_document_rows(document_id: str) -> list[dict[str, Any]]:
    collection = milvus_manager.get_collection()
    rows_by_id: dict[str, dict[str, Any]] = {}
    for field in ("source_id", "_document_id"):
        rows = collection.query(
            expr=f'metadata["{field}"] == "{document_id}"',
            output_fields=["id", "content", "metadata"],
            limit=10000,
        )
        for row in rows:
            rows_by_id[str(row["id"])] = row
    return list(rows_by_id.values())


def _wait_for_complete_rows(
    document_id: str,
    expected_count: int,
    attempts: int = 10,
    delay_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    """等待Milvus写入对查询可见，并验证行数和稳定元数据。"""
    collection = milvus_manager.get_collection()
    collection.flush()
    last_rows: list[dict[str, Any]] = []
    for _ in range(attempts):
        last_rows = _query_document_rows(document_id)
        if len(last_rows) == expected_count and _is_complete(last_rows):
            return last_rows
        time.sleep(delay_seconds)
    return last_rows


def _is_complete(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    chunk_ids = []
    for row in rows:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict) or not REQUIRED_METADATA.issubset(metadata):
            return False
        chunk_ids.append(str(metadata["chunk_id"]))
    return len(chunk_ids) == len(set(chunk_ids))


def _document_inputs() -> list[dict[str, Any]]:
    documents = []
    for original_dir in sorted(ORIGINALS_DIR.glob("doc_*")):
        if not original_dir.is_dir():
            continue
        document_id = original_dir.name
        pdfs = sorted(original_dir.glob("*.pdf"))
        markdowns = sorted((PARSED_DIR / document_id).glob("*.md"))
        if len(pdfs) != 1 or len(markdowns) != 1:
            raise RuntimeError(
                f"{document_id} 文件不唯一: pdf={len(pdfs)}, markdown={len(markdowns)}"
            )
        if not markdowns[0].read_text(encoding="utf-8").strip():
            raise RuntimeError(f"{document_id} Markdown为空")
        documents.append(
            {
                "document_id": document_id,
                "pdf": pdfs[0],
                "markdown": markdowns[0],
            }
        )
    return documents


def _load_progress(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "started_at": datetime.now().astimezone().isoformat(),
            "completed": {},
            "failed": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def _save_progress(path: Path, progress: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    progress["updated_at"] = datetime.now().astimezone().isoformat()
    path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-count", type=int, default=68)
    parser.add_argument("--progress", default=str(DEFAULT_PROGRESS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    progress_path = Path(args.progress).resolve()
    progress_path.relative_to(PROJECT_ROOT.resolve())
    documents = _document_inputs()
    if len(documents) != args.expected_count:
        raise RuntimeError(
            f"可重建文献数为{len(documents)}，与expected-count={args.expected_count}不一致"
        )

    _ = milvus_manager.connect()
    before_complete = sum(
        1 for item in documents if _is_complete(_query_document_rows(item["document_id"]))
    )
    print(
        json.dumps(
            {
                "execute": args.execute,
                "document_count": len(documents),
                "already_complete": before_complete,
                "pending": len(documents) - before_complete,
                "progress": str(progress_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.execute:
        return

    progress = _load_progress(progress_path)
    for position, item in enumerate(documents, 1):
        document_id = item["document_id"]
        existing_rows = _query_document_rows(document_id)
        if _is_complete(existing_rows):
            progress["completed"][document_id] = {
                "status": "already_complete",
                "chunk_count": len(existing_rows),
            }
            progress["failed"].pop(document_id, None)
            _save_progress(progress_path, progress)
            print(f"[{position}/{len(documents)}] {document_id}: already_complete")
            continue

        source_values = {
            str(row.get("metadata", {}).get("_source", ""))
            for row in existing_rows
            if isinstance(row.get("metadata"), dict) and row["metadata"].get("_source")
        }
        if len(source_values) > 1:
            raise RuntimeError(f"{document_id} 无法确定唯一旧_source: {sorted(source_values)}")
        logical_source = (
            next(iter(source_values))
            if source_values
            else str(item["pdf"].relative_to(PROJECT_ROOT))
        )

        print(
            f"[{position}/{len(documents)}] {document_id}: "
            f"reindex old_rows={len(existing_rows)} file={item['pdf'].name}"
        )
        try:
            markdown_content = item["markdown"].read_text(encoding="utf-8")
            chunk_count = vector_index_service.index_content(
                content=markdown_content,
                logical_source=logical_source,
                display_filename=item["pdf"].name,
                parsed_source=str(item["markdown"].relative_to(PROJECT_ROOT)),
                extra_metadata={
                    "_parser": "TextIn xParse",
                    "_document_id": document_id,
                },
            )
            rows_after = _wait_for_complete_rows(document_id, chunk_count)
            if not _is_complete(rows_after):
                raise RuntimeError("写入后稳定元数据或逻辑Chunk唯一性校验失败")
            if len(rows_after) != chunk_count:
                raise RuntimeError(
                    f"写入后行数{len(rows_after)}与分块数{chunk_count}不一致"
                )

            progress["completed"][document_id] = {
                "status": "reindexed",
                "old_row_count": len(existing_rows),
                "chunk_count": chunk_count,
            }
            progress["failed"].pop(document_id, None)
            _save_progress(progress_path, progress)
            print(f"[{position}/{len(documents)}] {document_id}: complete chunks={chunk_count}")
        except Exception as exc:
            progress["failed"][document_id] = {"error": str(exc)}
            _save_progress(progress_path, progress)
            print(f"[{position}/{len(documents)}] {document_id}: failed error={exc}")
            raise

    print(
        f"reindex_complete=true documents={len(progress['completed'])} "
        f"failed={len(progress['failed'])} progress={progress_path}"
    )


if __name__ == "__main__":
    main()
