"""Static contract validation for the versioned 50-question v2 candidate set."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = ROOT / "evaluation"
DATASET_PATH = EVALUATION_DIR / "ragas_50_v2_dataset.jsonl"
MAPPING_PATH = EVALUATION_DIR / "ragas_50_v2_mapping.jsonl"
REVIEW_PATH = EVALUATION_DIR / "ragas_50_v2_review.csv"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate() -> dict:
    dataset = _read_jsonl(DATASET_PATH)
    mapping = _read_jsonl(MAPPING_PATH)
    with REVIEW_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        review = list(csv.DictReader(handle))

    if not len(dataset) == len(mapping) == len(review) == 50:
        raise ValueError("dataset, mapping, and review must each contain 50 rows")

    question_ids = [row["question_id"] for row in dataset]
    if len(set(question_ids)) != 50:
        raise ValueError("question_id values must be unique")
    if question_ids != [row["question_id"] for row in mapping]:
        raise ValueError("dataset and mapping order differs")
    if question_ids != [row["question_id"] for row in review]:
        raise ValueError("dataset and review order differs")

    for row in dataset:
        acceptable = row["acceptable_chunk_ids"]
        if not acceptable or len(acceptable) != len(set(acceptable)):
            raise ValueError(f"{row['question_id']} has invalid acceptable chunks")
        if row["strict_chunk_id"] not in acceptable:
            raise ValueError(f"{row['question_id']} strict chunk is not acceptable")
        if any(not chunk_id.startswith(f"{row['document_id']}:") for chunk_id in acceptable):
            raise ValueError(f"{row['question_id']} contains a cross-document gold chunk")
        claims = row["reference_claims"]
        if not claims or not row["reference"].strip():
            raise ValueError(f"{row['question_id']} has no reference candidate")
        claim_ids = [claim["claim_id"] for claim in claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError(f"{row['question_id']} has duplicate claim IDs")
        for claim in claims:
            claim_alternatives = claim["acceptable_chunk_ids"]
            if not claim["claim_text"].strip() or not claim_alternatives:
                raise ValueError(f"{claim['claim_id']} is incomplete")
            if not set(claim_alternatives).issubset(acceptable):
                raise ValueError(f"{claim['claim_id']} uses an undeclared chunk")
        if row["reference_reviewed"] is not False:
            raise ValueError("candidate rows must remain unreviewed")
        if row["review_status"] != "pending_human_review":
            raise ValueError("candidate rows must remain pending human review")
        if row.get("model_cleanup_applied") is not True:
            raise ValueError(f"{row['question_id']} has not received model cleanup")

    acceptable_counts = [len(row["acceptable_chunk_ids"]) for row in dataset]
    return {
        "passed": True,
        "question_count": len(dataset),
        "unique_document_count": len({row["document_id"] for row in dataset}),
        "migrated_document_count": sum(
            row["document_id"] != row["source_document_id"] for row in dataset
        ),
        "questions_with_multiple_acceptable_chunks": sum(
            count > 1 for count in acceptable_counts
        ),
        "model_cleanup_applied_count": sum(
            row.get("model_cleanup_applied") is True for row in dataset
        ),
        "cleaned_acceptable_chunk_count": sum(acceptable_counts),
        "acceptable_chunk_count_min": min(acceptable_counts),
        "acceptable_chunk_count_max": max(acceptable_counts),
    }


def main() -> None:
    print(json.dumps(validate(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
