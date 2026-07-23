from __future__ import annotations

import csv

import pytest

from evaluation.formal_eval_contract import (
    DEFAULT_REVIEW_CSV,
    build_completion_status,
    load_formal_review_rows,
)


def test_human_review_confirmation_is_bound_to_current_csv() -> None:
    rows, contract = load_formal_review_rows(DEFAULT_REVIEW_CSV)

    assert len(rows) == 50
    assert contract["human_review_gate_passed"] is True
    assert contract["dataset_sha256"] == (
        "5836fbc05daacb0741d234d196938d5c409d2a40f839f730989e96184d02a393"
    )
    assert contract["question_confirmation_manifest"] is not None


def test_tampered_review_csv_fails_confirmation_hash(tmp_path) -> None:
    rows = []
    with DEFAULT_REVIEW_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows.extend(csv.DictReader(handle))
    rows[0]["reference_candidate"] += " tampered"
    path = tmp_path / "tampered.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(ValueError, match="CSV哈希"):
        load_formal_review_rows(path)


def test_completion_requires_one_tool_call_and_both_ragas_metrics() -> None:
    valid_row = {
        "answer": "answer",
        "tool_call_count": 1,
        "retrieved_contexts_json": '["evidence"]',
        "ragas_faithfulness": 1.0,
        "ragas_context_recall": 1.0,
    }

    status = build_completion_status(
        [valid_row],
        expected_count=1,
        required_ragas_metrics=("faithfulness", "context_recall"),
    )
    assert status["status"] == "valid"

    invalid = dict(valid_row, tool_call_count=2, ragas_context_recall=None)
    status = build_completion_status(
        [invalid],
        expected_count=1,
        required_ragas_metrics=("faithfulness", "context_recall"),
    )
    assert status["status"] == "invalid"
    assert status["invalid_tool_call_count"] == 1
    assert status["ragas_missing_count"]["context_recall"] == 1
