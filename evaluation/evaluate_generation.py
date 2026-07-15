"""生成回答与引用评测脚本

调用 POST /api/chat 获取完整回答，提取引用标签，评估引用有效率。
引用支持率必须人工核验后填入（本脚本只生成核验表）。

用法：
    python evaluation/evaluate_generation.py --questions evaluation/questions.csv
"""

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import httpx
from loguru import logger

from evaluation.common import (
    load_questions,
    normalize_file_name,
    write_json,
    write_csv,
    build_run_metadata,
    EvalError,
)


CITATION_PATTERNS = [
    # (Author et al. Year) 或 (Author Year)
    re.compile(r"\(([A-Z][a-z]+(?:\s+et\s+al\.?)?(?:\s+\d{4})?)\)"),
    # [1], [2], ...
    re.compile(r"\[(\d+)\]"),
]


def _extract_citations(text: str) -> List[str]:
    """从回答文本中提取所有引用标签。

    Args:
        text: 回答全文

    Returns:
        引用标签列表（保留出现顺序）
    """
    citations: List[str] = []
    for pattern in CITATION_PATTERNS:
        for match in pattern.finditer(text):
            citations.append(match.group(0))
    return citations


def _extract_numbered_citations(text: str) -> List[Tuple[str, int]]:
    """提取 [N] 形式的引用，返回 (原始文本, 序号)。"""
    results: List[Tuple[str, int]] = []
    for match in re.finditer(r"\[(\d+)\]", text):
        results.append((match.group(0), int(match.group(1))))
    return results


def _map_citation_to_source(
    citation: str,
    retrieved_sources: List[str],
    numbered_map: Optional[Dict[int, str]] = None,
) -> Tuple[bool, str]:
    """判断一个引用是否能映射到检索来源。

    Args:
        citation: 引用标签文本（如 "(Zhou et al. 2023)" 或 "[1]"）
        retrieved_sources: 当次检索返回的规范化来源列表
        numbered_map: [N] 到来源的映射（从检索结果顺序构建）

    Returns:
        (是否有效, 匹配到的来源或空字符串)
    """
    # [N] 格式
    num_match = re.match(r"\[(\d+)\]", citation)
    if num_match and numbered_map:
        idx = int(num_match.group(1))
        if idx in numbered_map:
            return True, numbered_map[idx]
        return False, ""

    # (Author et al. Year) 格式
    # 提取作者和年份
    inner = citation.strip("()")
    for src in retrieved_sources:
        # 从引用中提取关键信息
        parts = inner.lower().replace("et al.", "").replace("et al", "").strip().split()
        author = parts[0] if parts else ""
        year = parts[-1] if len(parts) >= 2 and parts[-1].isdigit() else ""

        if author and author in src:
            if year and year in src:
                return True, src
            # 只匹配到作者也算
            return True, src

    return False, ""


def _do_retrieval_for_question(question: str, k: int = 5) -> List[str]:
    """为一道问题执行检索，返回规范化来源列表。"""
    from app.services.vector_store_manager import vector_store_manager

    try:
        docs = vector_store_manager.similarity_search(question, k=k)
    except Exception:
        return []

    sources = []
    for doc in docs:
        fn = doc.metadata.get("_file_name", "")
        sources.append(normalize_file_name(str(fn)))
    return sources


def run_generation_evaluation(
    questions_csv: str,
    base_url: str = "http://localhost:9900",
    output_dir: str = "evaluation/results",
) -> Dict[str, Any]:
    """执行生成回答与引用评测。

    Args:
        questions_csv: 问题集 CSV 路径
        base_url: FastAPI 服务地址
        output_dir: 结果输出目录

    Returns:
        汇总结果字典
    """
    run_meta = build_run_metadata(question_file=questions_csv)
    run_id = run_meta["run_id"]
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    questions, _ = load_questions(questions_csv)

    answer_details: List[Dict[str, Any]] = []
    citation_rows: List[Dict[str, Any]] = []
    total_citations = 0
    valid_citations = 0
    success_count = 0
    fail_count = 0

    for q in questions:
        qid = q["question_id"]
        question = q["question"]
        category = q["category"]

        # 调用 /api/chat
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(
                    f"{base_url}/api/chat",
                    json={"Id": f"eval-{run_id}-{qid}", "Question": question},
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("data", {}).get("answer", "")
                error_msg = ""
                status = "success"
        except Exception as e:
            answer = ""
            error_msg = str(e)
            status = "failed"

        elapsed = time.perf_counter() - t0

        if status == "success":
            success_count += 1
        else:
            fail_count += 1

        # 执行"回答后检索"（reconstructed_retrieval，非 Agent 真实轨迹）
        retrieved_sources = _do_retrieval_for_question(question, k=10)

        # 提取引用
        raw_citations = _extract_citations(answer)

        # 构建 numbered_map（从检索结果顺序映射 [N]）
        numbered_map: Dict[int, str] = {}
        for i, src in enumerate(retrieved_sources, start=1):
            numbered_map[i] = src

        # 映射引用
        for cit in raw_citations:
            is_valid, matched_src = _map_citation_to_source(cit, retrieved_sources, numbered_map)
            total_citations += 1
            if is_valid:
                valid_citations += 1

            citation_rows.append({
                "question_id": qid,
                "question": question,
                "citation": cit,
                "retrieved_files": "; ".join(retrieved_sources),
                "exists_in_retrieval": is_valid,
                "claim_text": "",
                "claim_supported": "",
                "evidence": "",
                "reviewer_notes": "",
            })

        answer_details.append({
            "question_id": qid,
            "question": question,
            "category": category,
            "answer": answer[:5000],  # 截断过长回答
            "retrieved_sources": "; ".join(retrieved_sources),
            "retrieval_type": "reconstructed_retrieval",
            "citation_count": len(raw_citations),
            "elapsed_seconds": round(elapsed, 2),
            "status": status,
            "error": error_msg,
        })

    # 汇总
    citation_valid_rate = (valid_citations / total_citations) if total_citations > 0 else 0

    summary = {
        "run_id": run_id,
        "total_questions": len(questions),
        "success_count": success_count,
        "fail_count": fail_count,
        "total_citations": total_citations,
        "valid_citations": valid_citations,
        "citation_valid_rate": round(citation_valid_rate, 4),
        "citation_support_rate": "pending_manual_review",
        "note": (
            "引用有效率由脚本自动计算。"
            "引用支持率须人工核验 citation_review.csv 中的 claim_supported 字段后填入。"
            "retrieval_type=reconstructed_retrieval 表示来源通过回答后重新检索获得，非 Agent 真实工具调用轨迹。"
        ),
    }

    write_csv(answer_details, str(out_dir / "generation_answers.csv"))
    write_csv(citation_rows, str(out_dir / "citation_review.csv"))
    write_json(summary, str(out_dir / "generation_summary.json"))

    from datetime import datetime, timezone, timedelta
    run_meta["finished_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    run_meta["status"] = "completed"
    write_json(run_meta, str(out_dir / "run_metadata.json"))

    logger.info(f"生成评测完成: 成功 {success_count}/{len(questions)}, 引用有效率 {citation_valid_rate:.2%}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="生成回答与引用评测")
    parser.add_argument("--questions", default="evaluation/questions.csv", help="问题集 CSV 路径")
    parser.add_argument("--base-url", default="http://localhost:9900", help="FastAPI 服务地址")
    parser.add_argument("--output-dir", default="evaluation/results", help="结果输出目录")
    args = parser.parse_args()

    qpath = Path(args.questions)
    if not qpath.exists():
        print(f"[BLOCKED] 问题集不存在: {args.questions}")
        sys.exit(2)

    # 检查 FastAPI 是否可用
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{args.base_url}/health")
            if resp.status_code != 200:
                print(f"[BLOCKED] FastAPI 服务不可用: {args.base_url}")
                sys.exit(3)
    except Exception as e:
        print(f"[BLOCKED] 无法连接 FastAPI: {e}")
        sys.exit(3)

    logger.warning("=" * 60)
    logger.warning("即将批量调用 RAG Agent，预计产生模型 API 调用费用。")
    logger.warning(f"问题数: 待加载 CSV 后确定")
    logger.warning("=" * 60)

    try:
        summary = run_generation_evaluation(
            questions_csv=args.questions,
            base_url=args.base_url,
            output_dir=args.output_dir,
        )
        print(f"生成评测完成, 引用有效率: {summary['citation_valid_rate']}")
        print(f"结果保存在: {args.output_dir}")
    except Exception as e:
        logger.error(f"生成评测失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
