"""Shared preflight and completion checks for formal Full/BL-1 evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEW_CSV = ROOT / "evaluation" / "ragas_50_v2_review.csv"
CONFIRMATION_PATH = (
    ROOT / "evaluation" / "ragas_50_v2_human_review_confirmation.json"
)
TRUE_VALUES = {"1", "true", "yes", "y", "是", "通过"}


def _is_true(value: Any) -> bool:
    return str(value or "").strip().casefold() in TRUE_VALUES


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_formal_review_rows(
    review_csv: str | Path,
    *,
    limit: int = 0,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    path = Path(review_csv).resolve()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        all_rows = list(csv.DictReader(handle))

    if len(all_rows) != 50:
        raise ValueError(f"正式评测集必须为50题，实际为{len(all_rows)}题")
    question_ids = [row.get("question_id", "").strip() for row in all_rows]
    if len(set(question_ids)) != 50 or any(not value for value in question_ids):
        raise ValueError("正式评测集必须包含50个非空且唯一的question_id")

    for row in all_rows:
        qid = row["question_id"]
        document_id = row.get("document_id", "").strip()
        reference = (row.get("reference") or row.get("reference_candidate") or "").strip()
        chunk_ids = [
            value.strip()
            for value in (row.get("acceptable_chunk_ids") or "").split(";")
            if value.strip()
        ]
        if not document_id or not reference or not chunk_ids:
            raise ValueError(f"{qid}缺少document_id、reference或acceptable_chunk_ids")
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError(f"{qid}存在重复acceptable_chunk_ids")
        if any(not value.startswith(f"{document_id}:") for value in chunk_ids):
            raise ValueError(f"{qid}存在跨文献或格式错误的Gold Chunk ID")
        if not _is_true(row.get("reference_answer_reviewed")):
            raise ValueError(f"{qid}参考答案尚未人工审核通过")
        if not _is_true(row.get("acceptable_chunks_reviewed")):
            raise ValueError(f"{qid}Gold Chunk尚未人工审核通过")

    csv_hash = _sha256(path)
    per_row_questions_confirmed = all(
        _is_true(row.get("question_supported")) for row in all_rows
    )
    confirmation_used = False
    if not per_row_questions_confirmed:
        confirmation = json.loads(CONFIRMATION_PATH.read_text(encoding="utf-8"))
        if confirmation.get("dataset_sha256") != csv_hash:
            raise ValueError("问题支持性确认绑定的CSV哈希与当前文件不一致")
        if confirmation.get("question_count") != 50:
            raise ValueError("问题支持性确认的题目数不是50")
        if confirmation.get("question_supported_all") is not True:
            raise ValueError("尚未确认50道问题均可由对应文献直接回答")
        confirmation_used = True

    selected_rows = all_rows[:limit] if limit > 0 else all_rows
    return selected_rows, {
        "dataset_path": str(path),
        "dataset_sha256": csv_hash,
        "full_question_count": len(all_rows),
        "selected_question_count": len(selected_rows),
        "question_confirmation_manifest": (
            str(CONFIRMATION_PATH) if confirmation_used else None
        ),
        "human_review_gate_passed": True,
    }


def build_completion_status(
    details: list[dict[str, Any]],
    *,
    expected_count: int,
    required_ragas_metrics: tuple[str, ...],
) -> dict[str, Any]:
    error_count = sum(bool(row.get("error")) for row in details)
    empty_answer_count = sum(
        not str(row.get("answer", "")).strip()
        or row.get("answer") == "（无回答）"
        for row in details
    )
    invalid_tool_call_count = sum(
        row.get("tool_call_count") != 1 for row in details
    )
    empty_context_count = sum(
        not str(row.get("retrieved_contexts_json", "")).strip()
        or row.get("retrieved_contexts_json") == "[]"
        for row in details
    )
    ragas_missing: dict[str, int] = {}
    for metric in required_ragas_metrics:
        ragas_missing[metric] = sum(
            row.get(f"ragas_{metric}") is None for row in details
        )
    valid = (
        len(details) == expected_count
        and error_count == 0
        and empty_answer_count == 0
        and invalid_tool_call_count == 0
        and empty_context_count == 0
        and all(count == 0 for count in ragas_missing.values())
    )
    return {
        "status": "valid" if valid else "invalid",
        "expected_question_count": expected_count,
        "detail_count": len(details),
        "error_count": error_count,
        "empty_answer_count": empty_answer_count,
        "invalid_tool_call_count": invalid_tool_call_count,
        "empty_context_count": empty_context_count,
        "ragas_missing_count": ragas_missing,
    }
