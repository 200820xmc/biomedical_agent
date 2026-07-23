"""清理本地存在但未进入 Milvus 的 PDF 文献。

默认仅输出预览；传入 ``--execute`` 才会删除。删除范围仅包括：

- ``uploads/originals/{document_id}``
- ``uploads/parsed/{document_id}``
- ``uploads/jobs`` 中属于这些 document_id 的任务 JSON

执行前会在指定位置写入包含文件 SHA-256 的审计清单。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from pymilvus import Collection, connections, utility


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORIGINALS_DIR = PROJECT_ROOT / "uploads" / "originals"
PARSED_DIR = PROJECT_ROOT / "uploads" / "parsed"
JOBS_DIR = PROJECT_ROOT / "uploads" / "jobs"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_within(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(parent.resolve())
    return resolved


def _milvus_document_ids() -> set[str]:
    alias = "cleanup_unindexed_readonly"
    connections.connect(alias=alias, host="localhost", port="19530", timeout=10)
    if not utility.has_collection("biz", using=alias):
        raise RuntimeError("Milvus collection 'biz' 不存在，拒绝执行清理")

    collection = Collection("biz", using=alias)
    collection.load()
    rows = collection.query(
        expr="id != ''",
        output_fields=["metadata"],
        limit=16384,
    )
    document_ids: set[str] = set()
    for row in rows:
        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            continue
        document_id = metadata.get("source_id") or metadata.get("_document_id")
        if document_id:
            document_ids.add(str(document_id))
    return document_ids


def _load_jobs() -> list[tuple[Path, dict[str, Any]]]:
    jobs = []
    for path in sorted(JOBS_DIR.glob("*.json")):
        try:
            jobs.append((path, json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return jobs


def build_manifest() -> dict[str, Any]:
    original_dirs = {
        path.name: path
        for path in ORIGINALS_DIR.iterdir()
        if path.is_dir() and path.name.startswith("doc_")
    }
    milvus_ids = _milvus_document_ids()
    target_ids = sorted(set(original_dirs) - milvus_ids)
    jobs = _load_jobs()

    documents = []
    for document_id in target_ids:
        original_dir = _assert_within(original_dirs[document_id], ORIGINALS_DIR)
        parsed_dir = _assert_within(PARSED_DIR / document_id, PARSED_DIR)
        pdfs = sorted(original_dir.glob("*.pdf"))
        related_jobs = [
            (path, data)
            for path, data in jobs
            if str(data.get("document_id", "")) == document_id
        ]
        documents.append(
            {
                "document_id": document_id,
                "original_dir": str(original_dir),
                "pdfs": [
                    {
                        "name": path.name,
                        "size": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in pdfs
                ],
                "parsed_dir": str(parsed_dir) if parsed_dir.exists() else None,
                "job_files": [str(_assert_within(path, JOBS_DIR)) for path, _ in related_jobs],
                "job_statuses": [str(data.get("status", "unknown")) for _, data in related_jobs],
                "latest_error": next(
                    (
                        str(data.get("error_message", ""))[:500]
                        for _, data in reversed(related_jobs)
                        if data.get("error_message")
                    ),
                    "",
                ),
            }
        )

    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "collection": "biz",
        "milvus_document_count": len(milvus_ids),
        "local_original_count_before": len(original_dirs),
        "target_count": len(target_ids),
        "target_document_ids": target_ids,
        "documents": documents,
    }


def execute_cleanup(manifest: dict[str, Any], expected_count: int) -> None:
    if manifest["target_count"] != expected_count:
        raise RuntimeError(
            f"目标数量为 {manifest['target_count']}，与 --expected-count={expected_count} 不一致，拒绝删除"
        )

    for document in manifest["documents"]:
        original_dir = _assert_within(Path(document["original_dir"]), ORIGINALS_DIR)
        if original_dir.exists():
            shutil.rmtree(original_dir)

        parsed_value = document.get("parsed_dir")
        if parsed_value:
            parsed_dir = _assert_within(Path(parsed_value), PARSED_DIR)
            if parsed_dir.exists():
                shutil.rmtree(parsed_dir)

        for job_value in document["job_files"]:
            job_path = _assert_within(Path(job_value), JOBS_DIR)
            if job_path.exists():
                job_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-count", type=int, default=47)
    parser.add_argument(
        "--manifest",
        default="evaluation/results/KB_MAINTENANCE_20260722/deleted_unindexed_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_manifest()
    print(
        json.dumps(
            {
                "execute": args.execute,
                "milvus_document_count": manifest["milvus_document_count"],
                "local_original_count_before": manifest["local_original_count_before"],
                "target_count": manifest["target_count"],
                "target_document_ids": manifest["target_document_ids"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.execute:
        return

    manifest_path = _assert_within(PROJECT_ROOT / args.manifest, PROJECT_ROOT)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    execute_cleanup(manifest, args.expected_count)

    remaining = [path for path in ORIGINALS_DIR.iterdir() if path.is_dir() and path.name.startswith("doc_")]
    print(f"cleanup_complete=true remaining_original_documents={len(remaining)} manifest={manifest_path}")


if __name__ == "__main__":
    main()
