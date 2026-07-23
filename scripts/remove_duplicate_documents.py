"""删除已确认的书目重复来源，并保留删除审计记录。"""

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
TARGETS = {
    "doc_49476e": "doc_edffea",
    "doc_26b3f0": "doc_2954af",
}
DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "KB_MAINTENANCE_20260722"
    / "deleted_duplicate_manifest.json"
)


def _within(path: Path, parent: Path) -> Path:
    resolved = path.resolve()
    resolved.relative_to(parent.resolve())
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _rows_for(collection: Collection, document_id: str) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    escaped = _escape(document_id)
    for field in ("source_id", "_document_id"):
        for row in collection.query(
            expr=f'metadata["{field}"] == "{escaped}"',
            output_fields=["id", "metadata"],
            limit=10000,
        ):
            rows[str(row["id"])] = row
    return list(rows.values())


def _connect() -> Collection:
    alias = "remove_duplicate_documents"
    connections.connect(alias=alias, host="localhost", port="19530", timeout=10)
    if not utility.has_collection("biz", using=alias):
        raise RuntimeError("Milvus collection 'biz' 不存在")
    collection = Collection("biz", using=alias)
    collection.load()
    return collection


def _job_files(document_id: str) -> list[Path]:
    matches: list[Path] = []
    for path in sorted(JOBS_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(data.get("document_id", "")) == document_id:
            matches.append(_within(path, JOBS_DIR))
    return matches


def _manifest(collection: Collection) -> dict[str, Any]:
    documents = []
    for remove_id, keep_id in TARGETS.items():
        original_dir = _within(ORIGINALS_DIR / remove_id, ORIGINALS_DIR)
        parsed_dir = _within(PARSED_DIR / remove_id, PARSED_DIR)
        if not original_dir.is_dir() or not parsed_dir.is_dir():
            raise RuntimeError(f"待删除来源文件不完整: {remove_id}")
        remove_rows = _rows_for(collection, remove_id)
        keep_rows = _rows_for(collection, keep_id)
        if not remove_rows or not keep_rows:
            raise RuntimeError(
                f"Milvus来源不存在: remove={remove_id}:{len(remove_rows)}, "
                f"keep={keep_id}:{len(keep_rows)}"
            )
        files = sorted(path for path in original_dir.rglob("*") if path.is_file())
        files.extend(sorted(path for path in parsed_dir.rglob("*") if path.is_file()))
        documents.append(
            {
                "remove_document_id": remove_id,
                "keep_document_id": keep_id,
                "milvus_chunk_ids": sorted(str(row["id"]) for row in remove_rows),
                "milvus_chunk_count": len(remove_rows),
                "kept_chunk_count": len(keep_rows),
                "files": [
                    {
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "size": path.stat().st_size,
                        "sha256": _sha256(path),
                    }
                    for path in files
                ],
                "job_files": [str(path.relative_to(PROJECT_ROOT)) for path in _job_files(remove_id)],
            }
        )
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "collection": "biz",
        "documents": documents,
    }


def _delete_milvus_rows(collection: Collection, ids: list[str]) -> None:
    for offset in range(0, len(ids), 100):
        batch = ids[offset : offset + 100]
        literals = ", ".join(json.dumps(value) for value in batch)
        collection.delete(expr=f"id in [{literals}]")
    collection.flush()


def execute(collection: Collection, manifest: dict[str, Any]) -> None:
    for document in manifest["documents"]:
        remove_id = document["remove_document_id"]
        keep_id = document["keep_document_id"]
        _delete_milvus_rows(collection, document["milvus_chunk_ids"])
        if _rows_for(collection, remove_id):
            raise RuntimeError(f"Milvus删除后仍存在来源: {remove_id}")
        if not _rows_for(collection, keep_id):
            raise RuntimeError(f"误删了应保留来源: {keep_id}")

        original_dir = _within(ORIGINALS_DIR / remove_id, ORIGINALS_DIR)
        parsed_dir = _within(PARSED_DIR / remove_id, PARSED_DIR)
        if original_dir.exists():
            shutil.rmtree(original_dir)
        if parsed_dir.exists():
            shutil.rmtree(parsed_dir)
        for job_value in document["job_files"]:
            job_path = _within(PROJECT_ROOT / job_value, JOBS_DIR)
            if job_path.exists():
                job_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest_path.relative_to(PROJECT_ROOT.resolve())
    collection = _connect()
    manifest = _manifest(collection)
    summary = {
        "execute": args.execute,
        "remove": list(TARGETS),
        "keep": list(TARGETS.values()),
        "chunk_count_to_delete": sum(
            item["milvus_chunk_count"] for item in manifest["documents"]
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not args.execute:
        return

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    execute(collection, manifest)
    manifest["completed_at"] = datetime.now().astimezone().isoformat()
    manifest["remaining_original_documents"] = len(
        [path for path in ORIGINALS_DIR.glob("doc_*") if path.is_dir()]
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        "duplicate_cleanup_complete=true "
        f"remaining_original_documents={manifest['remaining_original_documents']} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
