"""去重策略对比评测脚本

对比基线（直接取 Top-5 分片）与优化（Top-15 召回 → 按 _file_name 去重 → Top-5 论文）。

用法：
    python evaluation/evaluate_deduplication.py --questions evaluation/questions.csv
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger

from evaluation.common import (
    load_questions,
    normalize_file_name,
    validate_relevant_files,
    compute_hit_at_k,
    compute_recall_at_k,
    compute_source_coverage,
    compute_duplicate_ratio,
    write_json,
    write_csv,
    build_run_metadata,
    safe_milvus_preflight,
)


def _dedup_by_source(docs: List[Any], top_n: int = 5) -> List[str]:
    """按规范化 _file_name 去重，每篇论文保留排名最高的分片。

    Args:
        docs: 检索返回的 Document 列表
        top_n: 去重后保留的论文数

    Returns:
        去重后的规范化来源列表
    """
    seen: set = set()
    result: List[str] = []
    for doc in docs:
        fn = doc.metadata.get("_file_name", "")
        norm = normalize_file_name(str(fn))
        if norm not in seen:
            seen.add(norm)
            result.append(norm)
        if len(result) >= top_n:
            break
    return result


def _get_milvus_sources() -> List[str]:
    """获取 Milvus 中所有规范化来源。"""
    from app.core.milvus_client import milvus_manager
    from pymilvus import utility

    milvus_manager.connect()
    if not utility.has_collection("biz"):
        return []

    collection = milvus_manager.get_collection()
    collection.load()

    sources: set = set()
    offset = 0
    page_size = 1000
    entity_count = collection.num_entities
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


def run_deduplication_evaluation(
    questions_csv: str,
    output_dir: str = "evaluation/results",
) -> Dict[str, Any]:
    """执行去重策略对比评测。

    Args:
        questions_csv: 问题集 CSV 路径
        output_dir: 结果输出目录

    Returns:
        汇总结果字典
    """
    from app.services.vector_store_manager import vector_store_manager

    run_meta = build_run_metadata(question_file=questions_csv)
    run_id = run_meta["run_id"]
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    questions, _ = load_questions(questions_csv)
    milvus_sources = _get_milvus_sources()

    details: List[Dict[str, Any]] = []

    # 汇总累加器
    baseline_coverage: List[int] = []
    baseline_dup_ratio: List[float] = []
    optimized_coverage: List[int] = []
    optimized_dup_ratio: List[float] = []
    baseline_recall5: List[float] = []
    baseline_hit5: List[int] = []
    optimized_recall5: List[float] = []
    optimized_hit5: List[int] = []

    for q in questions:
        qid = q["question_id"]
        question = q["question"]
        relevant_str = q["relevant_files"]
        category = q["category"]

        matched, label_status, _ = validate_relevant_files(relevant_str, milvus_sources)

        try:
            docs = vector_store_manager.similarity_search(question, k=15)
        except Exception as e:
            logger.error(f"[{qid}] 检索失败: {e}")
            details.append({
                "question_id": qid,
                "question": question,
                "category": category,
                "label_status": label_status,
                "baseline_top5_sources": "",
                "baseline_coverage": 0,
                "baseline_dup_ratio": None,
                "optimized_top5_sources": "",
                "optimized_coverage": 0,
                "optimized_dup_ratio": None,
                "error": str(e),
            })
            continue

        # 提取原始来源
        raw_sources = [normalize_file_name(str(d.metadata.get("_file_name", ""))) for d in docs]

        # 基线：直接 Top-5
        baseline_top5 = raw_sources[:5]
        b_cov = compute_source_coverage(baseline_top5, 5)
        b_dup = compute_duplicate_ratio(baseline_top5, 5)

        # 优化：Top-15 → 去重 → Top-5
        optimized_top5 = _dedup_by_source(docs, top_n=5)
        o_cov = len(optimized_top5)
        o_dup = 1.0 - (o_cov / 5) if o_cov > 0 else None

        detail = {
            "question_id": qid,
            "question": question,
            "category": category,
            "label_status": label_status,
            "baseline_top5_sources": "; ".join(baseline_top5),
            "baseline_coverage": b_cov,
            "baseline_dup_ratio": b_dup,
            "optimized_top5_sources": "; ".join(optimized_top5),
            "optimized_coverage": o_cov,
            "optimized_dup_ratio": o_dup,
            "relevant_files": relevant_str,
            "error": "",
        }

        # 计算 Recall/Hit（如有标注）
        if label_status == "matched":
            b_hit5 = compute_hit_at_k(baseline_top5, matched, 5)
            b_rec5 = compute_recall_at_k(baseline_top5, matched, 5)
            o_hit5 = compute_hit_at_k(optimized_top5, matched, 5)
            o_rec5 = compute_recall_at_k(optimized_top5, matched, 5)

            detail["baseline_hit5"] = b_hit5
            detail["baseline_recall5"] = round(b_rec5, 4)
            detail["optimized_hit5"] = o_hit5
            detail["optimized_recall5"] = round(o_rec5, 4)

            baseline_hit5.append(b_hit5)
            baseline_recall5.append(b_rec5)
            optimized_hit5.append(o_hit5)
            optimized_recall5.append(o_rec5)

        details.append(detail)

        if b_cov > 0 or o_cov > 0:
            baseline_coverage.append(b_cov)
            optimized_coverage.append(o_cov)
            if b_dup is not None:
                baseline_dup_ratio.append(b_dup)
            if o_dup is not None:
                optimized_dup_ratio.append(o_dup)

    # ── 汇总 ──
    summary: Dict[str, Any] = {
        "run_id": run_id,
        "total_questions": len(questions),
        "comparison": {
            "avg_source_coverage": {
                "baseline": round(sum(baseline_coverage) / len(baseline_coverage), 2) if baseline_coverage else 0,
                "optimized": round(sum(optimized_coverage) / len(optimized_coverage), 2) if optimized_coverage else 0,
            },
            "avg_duplicate_ratio": {
                "baseline": round(sum(baseline_dup_ratio) / len(baseline_dup_ratio), 4) if baseline_dup_ratio else 0,
                "optimized": round(sum(optimized_dup_ratio) / len(optimized_dup_ratio), 4) if optimized_dup_ratio else 0,
            },
        },
    }

    if baseline_hit5 and optimized_hit5:
        summary["comparison"]["hit@5"] = {
            "baseline": round(sum(baseline_hit5) / len(baseline_hit5), 4),
            "optimized": round(sum(optimized_hit5) / len(optimized_hit5), 4),
        }
        summary["comparison"]["recall@5"] = {
            "baseline": round(sum(baseline_recall5) / len(baseline_recall5), 4),
            "optimized": round(sum(optimized_recall5) / len(optimized_recall5), 4),
        }
    else:
        summary["note"] = "问题集未标注，Hit@5/Recall@5 不可用"

    # ── 写入 ──
    write_csv(details, str(out_dir / "deduplication_details.csv"))
    write_json(summary, str(out_dir / "deduplication_summary.json"))

    from datetime import datetime, timezone, timedelta
    run_meta["finished_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    run_meta["status"] = "completed"
    write_json(run_meta, str(out_dir / "run_metadata.json"))

    logger.info(f"去重评测完成, run_id={run_id}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="去重策略对比评测")
    parser.add_argument("--questions", default="evaluation/questions.csv", help="问题集 CSV 路径")
    parser.add_argument("--output-dir", default="evaluation/results", help="结果输出目录")
    args = parser.parse_args()

    qpath = Path(args.questions)
    if not qpath.exists():
        print(f"[BLOCKED] 问题集不存在: {args.questions}")
        sys.exit(2)

    preflight = safe_milvus_preflight()
    if not preflight["ok"]:
        print(f"[BLOCKED] Milvus 不可用: {preflight['error']}")
        sys.exit(3)

    try:
        summary = run_deduplication_evaluation(
            questions_csv=args.questions,
            output_dir=args.output_dir,
        )
        print(f"去重评测完成, 结果保存在: {args.output_dir}")
    except Exception as e:
        logger.error(f"去重评测失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
