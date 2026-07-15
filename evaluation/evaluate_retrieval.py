"""检索效果评测脚本

绕过 Agent，直接调用向量存储的 similarity_search。
一次性检索 Top-15 快照，分别计算 Hit@K 和 Recall@K。

用法：
    python evaluation/evaluate_retrieval.py --questions evaluation/questions.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger

from evaluation.common import (
    load_questions,
    normalize_file_name,
    validate_relevant_files,
    compute_hit_at_k,
    compute_recall_at_k,
    write_json,
    write_csv,
    build_run_metadata,
    safe_milvus_preflight,
    EvalError,
)


def _get_milvus_normalized_sources() -> List[str]:
    """获取 Milvus 中所有规范化来源列表。"""
    from app.core.milvus_client import milvus_manager
    from pymilvus import utility

    milvus_manager.connect()
    if not utility.has_collection("biz"):
        return []

    collection = milvus_manager.get_collection()
    collection.load()

    entity_count = collection.num_entities
    if entity_count == 0:
        return []

    sources: set = set()
    offset = 0
    page_size = 1000
    while offset < entity_count:
        results = collection.query(
            expr="id != ''",
            output_fields=["metadata"],
            offset=offset,
            limit=min(page_size, entity_count - offset),
        )
        for row in results:
            metadata = row.get("metadata", {})
            if isinstance(metadata, dict):
                fn = metadata.get("_file_name", "")
                if fn:
                    sources.add(normalize_file_name(str(fn)))
        offset += page_size

    return sorted(sources)


def run_retrieval_evaluation(
    questions_csv: str,
    top_k_snapshot: int = 15,
    output_dir: str = "evaluation/results",
) -> Dict[str, Any]:
    """执行检索效果评测。

    Args:
        questions_csv: 问题集 CSV 路径
        top_k_snapshot: 一次性检索的数量（默认 15）
        output_dir: 结果输出目录

    Returns:
        汇总结果字典
    """
    from app.services.vector_store_manager import vector_store_manager

    # ── 准备 ──
    run_meta = build_run_metadata(question_file=questions_csv)
    run_id = run_meta["run_id"]
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    questions, csv_errors = load_questions(questions_csv)
    if csv_errors:
        for e in csv_errors:
            logger.warning(f"CSV 校验: {e}")

    milvus_sources = _get_milvus_normalized_sources()
    logger.info(f"Milvus 规范化来源数: {len(milvus_sources)}")

    # ── 逐题检索 ──
    details: List[Dict[str, Any]] = []
    has_valid_labels = False

    for q in questions:
        qid = q["question_id"]
        question = q["question"]
        relevant_str = q["relevant_files"]
        category = q["category"]

        # 匹配标准答案
        matched, label_status, ambig = validate_relevant_files(relevant_str, milvus_sources)

        if label_status == "needs_review":
            # 未标注，仍然检索但不计入指标
            pass
        elif label_status == "missing":
            logger.warning(f"[{qid}] 标准答案全部未匹配: {relevant_str}")
        elif label_status == "ambiguous":
            logger.warning(f"[{qid}] 标准答案模糊匹配: {ambig}")

        # 执行检索
        try:
            docs = vector_store_manager.similarity_search(question, k=top_k_snapshot)
        except Exception as e:
            logger.error(f"[{qid}] 检索失败: {e}")
            details.append({
                "run_id": run_id,
                "question_id": qid,
                "question": question,
                "category": category,
                "rank": 0,
                "raw_file_name": "",
                "normalized_file_name": "",
                "score": 0,
                "is_relevant": False,
                "relevant_files": relevant_str,
                "label_status": label_status,
                "error": str(e),
            })
            continue

        # 提取规范化来源
        retrieved_sources = []
        for doc in docs:
            fn = doc.metadata.get("_file_name", "")
            retrieved_sources.append(normalize_file_name(str(fn)))

        # 写入明细（每个 rank 一行）
        for rank, (doc, src) in enumerate(zip(docs, retrieved_sources), start=1):
            is_rel = src in matched if matched else False
            details.append({
                "run_id": run_id,
                "question_id": qid,
                "question": question,
                "category": category,
                "rank": rank,
                "raw_file_name": doc.metadata.get("_file_name", ""),
                "normalized_file_name": src,
                "score": doc.metadata.get("score", 0) if hasattr(doc.metadata, "get") else 0,
                "is_relevant": is_rel,
                "relevant_files": relevant_str,
                "label_status": label_status,
                "error": "",
            })

        if label_status == "matched":
            has_valid_labels = True

    # ── 汇总 ──
    # 按问题分组计算指标
    valid_questions = [q for q in questions if q["question_id"] in {
        d["question_id"] for d in details if d["label_status"] == "matched"
    }]

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "total_questions": len(questions),
        "valid_questions": len(valid_questions),
        "skipped_questions": len(questions) - len(valid_questions),
        "has_valid_labels": has_valid_labels,
        "metrics": {},
        "by_category": {},
    }

    if has_valid_labels:
        for label, metric_name in [("hit", "hit_at_k"), ("recall", "recall_at_k")]:
            summary["metrics"][metric_name] = {}
            for k in [1, 3, 5, 10]:
                values = []
                for q in valid_questions:
                    qid = q["question_id"]
                    q_details = [d for d in details if d["question_id"] == qid]
                    if not q_details:
                        continue
                    retrieved_sources = [d["normalized_file_name"] for d in q_details if d["rank"] <= k]
                    matched, _, _ = validate_relevant_files(q["relevant_files"], milvus_sources)

                    if label == "hit":
                        values.append(compute_hit_at_k(retrieved_sources, matched, k))
                    else:
                        values.append(compute_recall_at_k(retrieved_sources, matched, k))

                avg = sum(values) / len(values) if values else 0
                summary["metrics"][metric_name][f"@{k}"] = round(avg, 4)

        # 按类别分组
        categories: Dict[str, List[str]] = {}
        for q in valid_questions:
            cat = q["category"]
            categories.setdefault(cat, []).append(q["question_id"])

        for cat, qids in categories.items():
            cat_hits = {k: [] for k in [1, 3, 5]}
            for qid in qids:
                q = next((x for x in valid_questions if x["question_id"] == qid), None)
                if not q:
                    continue
                q_details = [d for d in details if d["question_id"] == qid]
                matched, _, _ = validate_relevant_files(q["relevant_files"], milvus_sources)
                for k in [1, 3, 5]:
                    sources = [d["normalized_file_name"] for d in q_details if d["rank"] <= k]
                    cat_hits[k].append(compute_hit_at_k(sources, matched, k))

            summary["by_category"][cat] = {
                f"hit@{k}": round(sum(v) / len(v), 4) if v else 0
                for k, v in cat_hits.items()
            }
    else:
        summary["metrics"] = {
            "hit_at_k": "not_available",
            "recall_at_k": "not_available",
        }
        summary["note"] = "问题集尚未人工标注，无法计算正式 Hit/Recall。请标注 questions.csv 后重新运行。"

    # ── 写入 ──
    write_csv(details, str(out_dir / "retrieval_details.csv"))
    write_json(summary, str(out_dir / "retrieval_summary.json"))

    # 更新元数据
    from datetime import datetime, timezone, timedelta
    run_meta["finished_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    run_meta["status"] = "completed" if has_valid_labels else "needs_review"
    write_json(run_meta, str(out_dir / "run_metadata.json"))

    logger.info(f"检索评测完成, run_id={run_id}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="检索效果评测（绕过 Agent，直接调向量搜索）")
    parser.add_argument("--questions", default="evaluation/questions.csv", help="问题集 CSV 路径")
    parser.add_argument("--output-dir", default="evaluation/results", help="结果输出目录")
    parser.add_argument("--top-k", type=int, default=15, help="一次性检索快照数量")
    args = parser.parse_args()

    # 检查问题集
    qpath = Path(args.questions)
    if not qpath.exists():
        print(f"[BLOCKED] 问题集不存在: {args.questions}")
        sys.exit(2)

    # 检查 Milvus
    preflight = safe_milvus_preflight()
    if not preflight["ok"]:
        print(f"[BLOCKED] Milvus 不可用: {preflight['error']}")
        sys.exit(3)

    try:
        summary = run_retrieval_evaluation(
            questions_csv=args.questions,
            top_k_snapshot=args.top_k,
            output_dir=args.output_dir,
        )
        if not summary.get("has_valid_labels"):
            print("[WARNING] 问题集尚未人工标注，Hit/Recall 标记为 not_available")
            print("请标注 questions.csv 后重新运行本脚本。")
            sys.exit(4)
        print(f"检索评测完成, 结果保存在: {args.output_dir}")
    except Exception as e:
        logger.error(f"检索评测失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
