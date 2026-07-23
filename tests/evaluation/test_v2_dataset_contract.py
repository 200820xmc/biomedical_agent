from __future__ import annotations

import json

from evaluation.build_ragas_50_v2 import (
    DATASET_PATH,
    SUMMARY_PATH,
    _load_review_overrides,
)
from evaluation.validate_ragas_50_v2 import validate


def test_model_cleanup_overrides_cover_all_questions() -> None:
    overrides = _load_review_overrides()

    assert len(overrides) == 50
    assert set(overrides) == {f"rq{index:03d}" for index in range(1, 51)}
    assert sum(
        len(row["acceptable_chunk_ids"]) for row in overrides.values()
    ) == 91


def test_cleaned_dataset_remains_pending_human_review() -> None:
    rows = [
        json.loads(line)
        for line in DATASET_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(rows) == 50
    assert all(row["model_cleanup_applied"] is True for row in rows)
    assert all(row["reference_reviewed"] is False for row in rows)
    assert all(row["review_status"] == "pending_human_review" for row in rows)
    assert sum(len(row["acceptable_chunk_ids"]) for row in rows) == 91


def test_cleaned_dataset_summary_and_static_validation() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert summary["automatic_candidate_chunk_count"] == 143
    assert summary["cleaned_candidate_chunk_count"] == 91
    assert summary["removed_false_gold_chunk_count"] == 52
    assert summary["ready_for_formal_ragas_context_recall"] is False
    assert validate()["passed"] is True
