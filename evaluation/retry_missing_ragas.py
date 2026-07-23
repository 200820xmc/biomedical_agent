"""Retry only missing Ragas scores from a saved formal evaluation run.

The source run is never modified.  A new directory receives copied details,
recomputed aggregates, and an audit record that is bound to the source files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import time
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


_fake_vertexai = types.ModuleType("langchain_community.chat_models.vertexai")


class _FakeVertexAI:
    pass


_fake_vertexai.ChatVertexAI = _FakeVertexAI
sys.modules["langchain_community.chat_models.vertexai"] = _fake_vertexai

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault(
    "DASHSCOPE_API_BASE",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)

from datasets import Dataset as HFDataset
from langchain_openai import ChatOpenAI
from ragas import evaluate as ragas_evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import context_recall, faithfulness

from app.config import config
from app.services.vector_embedding_service import get_vector_embedding_service
from evaluation.common import write_csv, write_json
from evaluation.formal_eval_contract import build_completion_status


CST = timezone(timedelta(hours=8))
METRICS = {
    "faithfulness": faithfulness,
    "context_recall": context_recall,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _ragas_safe_text(value: str) -> str:
    """Prevent LaTeX backslashes from becoming invalid JSON escapes in judge output."""
    return value.replace("\\", "")


def retry_missing(
    source_dir: Path,
    output_dir: Path,
    review_csv: Path,
    max_attempts: int,
) -> dict[str, Any]:
    source_summary_path = source_dir / "review_eval_summary.json"
    source_details_path = source_dir / "review_eval_details.csv"
    summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    details = _load_csv(source_details_path)
    references = {
        row["question_id"]: row.get("reference") or row.get("reference_candidate") or ""
        for row in _load_csv(review_csv)
    }

    eval_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=config.dashscope_model,
            temperature=0.0,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key=config.dashscope_api_key,
        )
    )
    attempts: list[dict[str, Any]] = []

    for row in details:
        for metric_name, metric in METRICS.items():
            column = f"ragas_{metric_name}"
            if _finite(row.get(column)) is not None:
                continue
            qid = row["question_id"]
            raw_contexts = json.loads(row["retrieved_contexts_json"])
            sample = {
                "user_input": _ragas_safe_text(row["question"]),
                "response": _ragas_safe_text(row["answer"]),
                "retrieved_contexts": [_ragas_safe_text(item) for item in raw_contexts],
                "reference": _ragas_safe_text(references[qid]),
            }
            for attempt in range(1, max_attempts + 1):
                record: dict[str, Any] = {
                    "question_id": qid,
                    "metric": metric_name,
                    "attempt": attempt,
                    "scoring_input_normalization": "latex_backslashes_removed",
                }
                try:
                    result = ragas_evaluate(
                        dataset=HFDataset.from_list([sample]),
                        metrics=[metric],
                        llm=eval_llm,
                        embeddings=get_vector_embedding_service(),
                    )
                    value = _finite(list(result[metric_name])[0])
                    record["value"] = value
                    record["status"] = "success" if value is not None else "missing"
                    attempts.append(record)
                    if value is not None:
                        row[column] = value
                        break
                except Exception as exc:
                    record["status"] = "error"
                    record["error"] = str(exc)
                    attempts.append(record)
                if attempt < max_attempts:
                    time.sleep(1.0)

    normalized: list[dict[str, Any]] = []
    for row in details:
        item = dict(row)
        item["tool_call_count"] = int(row["tool_call_count"])
        for metric_name in METRICS:
            item[f"ragas_{metric_name}"] = _finite(row.get(f"ragas_{metric_name}"))
        normalized.append(item)

    summary["completion"] = build_completion_status(
        normalized,
        expected_count=len(normalized),
        required_ragas_metrics=tuple(METRICS),
    )
    summary["ragas_metrics"] = {}
    for metric_name in METRICS:
        values = [
            value
            for row in details
            if (value := _finite(row.get(f"ragas_{metric_name}"))) is not None
        ]
        if values:
            summary["ragas_metrics"][metric_name] = {
                "mean": round(statistics.mean(values), 4),
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "count": len(values),
            }

    repair = {
        "kind": "missing_ragas_only",
        "created_at": datetime.now(CST).isoformat(),
        "source_directory": str(source_dir.resolve()),
        "source_summary_sha256": _sha256(source_summary_path),
        "source_details_sha256": _sha256(source_details_path),
        "review_csv": str(review_csv.resolve()),
        "review_csv_sha256": _sha256(review_csv),
        "max_attempts": max_attempts,
        "attempts": attempts,
    }
    summary["ragas_retry"] = repair
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json(summary, str(output_dir / "review_eval_summary.json"))
    write_csv(details, str(output_dir / "review_eval_details.csv"))
    write_json(repair, str(output_dir / "ragas_retry_audit.json"))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--review-csv",
        default=str(ROOT / "evaluation" / "ragas_50_v2_review.csv"),
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    summary = retry_missing(
        Path(args.source),
        Path(args.output),
        Path(args.review_csv),
        args.max_attempts,
    )
    print(json.dumps(summary["completion"], ensure_ascii=False, indent=2))
    if summary["completion"]["status"] != "valid":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
