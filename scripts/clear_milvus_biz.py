"""在精确实体数保护下清空 Milvus ``biz`` collection。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from pymilvus import Collection, connections, utility


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPORT = (
    PROJECT_ROOT
    / "evaluation"
    / "results"
    / "KB_MAINTENANCE_20260722"
    / "milvus_clear_report.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--expected-entities", type=int, required=True)
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report).resolve()
    report_path.relative_to(PROJECT_ROOT.resolve())

    alias = "clear_biz_maintenance"
    connections.connect(alias=alias, host="localhost", port="19530", timeout=10)
    if not utility.has_collection("biz", using=alias):
        raise RuntimeError("collection 'biz' 不存在")

    collection = Collection("biz", using=alias)
    collection.load()
    rows = collection.query(expr="id != ''", output_fields=["id"], limit=16384)
    ids = sorted({str(row["id"]) for row in rows})
    summary = {
        "checked_at": datetime.now().astimezone().isoformat(),
        "collection": "biz",
        "schema_fields": [field.name for field in collection.schema.fields],
        "vector_dim": next(
            (
                int(field.params["dim"])
                for field in collection.schema.fields
                if field.name == "vector"
            ),
            None,
        ),
        "entities_before": len(ids),
        "execute": args.execute,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if len(ids) != args.expected_entities:
        raise RuntimeError(
            f"实际实体数{len(ids)}与expected-entities={args.expected_entities}不一致，拒绝清空"
        )
    if not args.execute:
        return

    deleted = 0
    for offset in range(0, len(ids), 100):
        batch = ids[offset : offset + 100]
        values = ", ".join(json.dumps(value, ensure_ascii=False) for value in batch)
        result = collection.delete(expr=f"id in [{values}]")
        deleted += int(getattr(result, "delete_count", 0))
    collection.flush()

    remaining = collection.query(expr="id != ''", output_fields=["id"], limit=1)
    if remaining:
        raise RuntimeError("清空后仍能查询到实体")

    summary.update(
        {
            "completed_at": datetime.now().astimezone().isoformat(),
            "delete_count_reported": deleted,
            "entities_after": 0,
        }
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"clear_complete=true deleted={deleted} remaining=0 report={report_path}")


if __name__ == "__main__":
    main()
