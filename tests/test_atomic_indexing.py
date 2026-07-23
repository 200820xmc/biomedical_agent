from __future__ import annotations

import asyncio
import json

import pytest
from langchain_core.documents import Document

from app.services.vector_index_service import VectorIndexService
from app.services.vector_store_manager import VectorStoreManager


class _Splitter:
    @staticmethod
    def split_document(content: str, source: str):
        return [Document(page_content=content, metadata={"_source": source})]


class _Store:
    def __init__(self, *, add_error=None, verified=True, delete_error=None):
        self.add_error = add_error
        self.verified = verified
        self.delete_error = delete_error
        self.events: list[str] = []
        self.documents = []

    def add_documents(self, documents):
        self.events.append("add")
        self.documents = documents
        if self.add_error:
            raise self.add_error

    def verify_source_version(self, source, version, chunk_ids):
        self.events.append("verify")
        return self.verified

    def delete_by_source(self, source, exclude_version=""):
        self.events.append("delete_old")
        if self.delete_error:
            raise self.delete_error
        return 1

    def delete_source_version(self, source, version):
        self.events.append("rollback_new")
        return 1


def _service(store: _Store) -> VectorIndexService:
    return VectorIndexService(
        store_manager=store,
        splitter_service=_Splitter(),
        version_factory=lambda: "version-new",
    )


def test_atomic_switch_writes_verifies_then_deletes_old(tmp_path) -> None:
    source = tmp_path / "paper.md"
    source.write_text("new evidence", encoding="utf-8")
    store = _Store()

    assert _service(store).index_single_file(str(source)) == 1
    assert store.events == ["add", "verify", "delete_old"]
    metadata = store.documents[0].metadata
    assert metadata["_version_id"] == "version-new"
    assert metadata["chunk_index"] == 0
    assert metadata["chunk_id"].startswith(f"{source.as_posix()}:")


@pytest.mark.parametrize(
    ("store", "expected_events"),
    [
        (_Store(add_error=RuntimeError("embedding failed")), ["add", "rollback_new"]),
        (_Store(verified=False), ["add", "verify", "rollback_new"]),
        (
            _Store(delete_error=RuntimeError("old delete failed")),
            ["add", "verify", "delete_old", "rollback_new"],
        ),
    ],
)
def test_atomic_switch_failure_rolls_back_new_version_without_retrying_old(
    tmp_path,
    store,
    expected_events,
) -> None:
    source = tmp_path / "paper.md"
    source.write_text("new evidence", encoding="utf-8")

    with pytest.raises(RuntimeError):
        _service(store).index_single_file(str(source))

    assert store.events == expected_events


def test_md_upload_failure_restores_previous_local_file(tmp_path, monkeypatch) -> None:
    from app.api import file as file_api

    old_file = tmp_path / "paper.md"
    old_file.write_bytes(b"old evidence")
    monkeypatch.setattr(file_api, "UPLOAD_DIR", tmp_path)

    def fail_index(path: str) -> None:
        raise RuntimeError("index failed")

    monkeypatch.setattr(file_api.vector_index_service, "index_single_file", fail_index)

    response = asyncio.run(file_api._handle_md_txt_upload("paper.md", b"new evidence"))
    payload = json.loads(response.body)

    assert response.status_code == 207
    assert payload["data"]["upload_success"] is True
    assert payload["data"]["index_success"] is False
    assert old_file.read_bytes() == b"old evidence"


class _VersionCollection:
    def __init__(self, rows):
        self.rows = rows
        self.flush_calls = 0

    def flush(self):
        self.flush_calls += 1

    def query(self, expr: str, output_fields: list[str], limit: int):
        return self.rows


class _Milvus:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self):
        return self.collection


def test_verify_source_version_checks_count_ids_and_required_metadata() -> None:
    metadata = {
        "_version_id": "version-new",
        "source_id": "doc",
        "chunk_index": 0,
        "content_hash": "hash",
        "chunk_id": "doc:hash",
    }
    collection = _VersionCollection(
        [{"id": "row-1", "content": "evidence", "metadata": metadata}]
    )
    manager = VectorStoreManager(milvus_client_manager=_Milvus(collection))

    assert manager.verify_source_version(
        "paper.md", "version-new", ["doc:hash"], attempts=1, delay_seconds=0
    )
    assert collection.flush_calls == 1

    broken = dict(metadata)
    broken.pop("chunk_index")
    collection.rows = [{"id": "row-1", "content": "evidence", "metadata": broken}]
    assert not manager.verify_source_version(
        "paper.md", "version-new", ["doc:hash"], attempts=1, delay_seconds=0
    )
