from __future__ import annotations

import json
import asyncio
from datetime import datetime, timedelta

from app.models.pdf_ingestion import IngestionJob
from app.services import pdf_ingestion_service as ingestion_module
from app.services.pdf_ingestion_service import PDFIngestionService
from app.services.vector_store_manager import VectorStoreManager


class _FakeCollection:
    def __init__(self, rows_by_field: dict[str, list[dict]]):
        self.rows_by_field = rows_by_field

    def query(self, expr: str, output_fields: list[str], limit: int):
        field = "source_id" if 'metadata["source_id"]' in expr else "_document_id"
        return self.rows_by_field.get(field, [])


class _FakeMilvusManager:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self):
        return self.collection


class _FakeDeleteResult:
    delete_count = 1


class _RecordingDeleteCollection:
    def __init__(self):
        self.query_expr = ""
        self.delete_expr = ""

    def query(self, expr: str, output_fields: list[str], limit: int):
        self.query_expr = expr
        return [{"id": "row-1", "metadata": {}}]

    def delete(self, expr: str):
        self.delete_expr = expr
        return _FakeDeleteResult()


def _write_job(path, job: IngestionJob) -> None:
    path.write_text(
        json.dumps(job.model_dump(mode="json"), ensure_ascii=False),
        encoding="utf-8",
    )


def test_get_document_chunk_ids_supports_new_and_legacy_metadata():
    collection = _FakeCollection(
        {
            "source_id": [{"id": "new-1"}, {"id": "shared"}],
            "_document_id": [{"id": "legacy-1"}, {"id": "shared"}],
        }
    )
    manager = VectorStoreManager(milvus_client_manager=_FakeMilvusManager(collection))

    assert manager.get_document_chunk_ids("doc_abc123") == ["legacy-1", "new-1", "shared"]


def test_delete_by_source_escapes_windows_path():
    collection = _RecordingDeleteCollection()
    manager = VectorStoreManager(milvus_client_manager=_FakeMilvusManager(collection))

    deleted = manager.delete_by_source(r'uploads\originals\doc_abc123\paper"name.pdf')

    assert deleted == 1
    assert r"uploads\\originals\\doc_abc123\\paper\"name.pdf" in collection.query_expr
    assert collection.delete_expr == 'id in ["row-1"]'


def test_submit_returns_indexed_job_when_milvus_already_has_document(
    tmp_path,
    monkeypatch,
):
    originals = tmp_path / "originals"
    jobs = tmp_path / "jobs"
    doc_dir = originals / "doc_abc123"
    doc_dir.mkdir(parents=True)
    jobs.mkdir()
    (doc_dir / "paper.pdf").write_bytes(b"%PDF-1.4\n")

    indexed_job = IngestionJob(
        job_id="job_indexed",
        document_id="doc_abc123",
        original_filename="paper.pdf",
        status="indexed",
        progress=100,
        updated_at=datetime.now(),
    )
    _write_job(jobs / "job_indexed.json", indexed_job)

    monkeypatch.setattr(ingestion_module, "ORIGINALS_DIR", originals)
    monkeypatch.setattr(ingestion_module, "JOBS_DIR", jobs)
    monkeypatch.setattr(
        ingestion_module.vector_store_manager,
        "has_document",
        lambda document_id: document_id == "doc_abc123",
    )

    service = PDFIngestionService.__new__(PDFIngestionService)
    service._active_jobs = {}

    result = asyncio.run(service.submit("doc_abc123"))

    assert result.job_id == "job_indexed"
    assert len(list(jobs.glob("*.json"))) == 1


def test_list_pending_uses_milvus_truth_and_latest_failed_status(
    tmp_path,
    monkeypatch,
):
    originals = tmp_path / "originals"
    jobs = tmp_path / "jobs"
    originals.mkdir()
    jobs.mkdir()

    for document_id in ("doc_indexed", "doc_failed"):
        doc_dir = originals / document_id
        doc_dir.mkdir()
        (doc_dir / f"{document_id}.pdf").write_bytes(b"%PDF-1.4\n")

    now = datetime.now()
    _write_job(
        jobs / "job_old.json",
        IngestionJob(
            job_id="job_old",
            document_id="doc_failed",
            status="failed",
            updated_at=now - timedelta(minutes=5),
        ),
    )
    _write_job(
        jobs / "job_latest.json",
        IngestionJob(
            job_id="job_latest",
            document_id="doc_failed",
            status="failed",
            updated_at=now,
        ),
    )

    monkeypatch.setattr(ingestion_module, "ORIGINALS_DIR", originals)
    monkeypatch.setattr(ingestion_module, "JOBS_DIR", jobs)
    monkeypatch.setattr(
        ingestion_module.vector_store_manager,
        "has_document",
        lambda document_id: document_id == "doc_indexed",
    )

    service = PDFIngestionService.__new__(PDFIngestionService)
    pending = asyncio.run(service.list_pending())

    assert [item.document_id for item in pending] == ["doc_failed"]
    assert pending[0].status == "failed"
