"""SSE 流式性能评测脚本

通过 /api/chat_stream 实际测试 SSE 流式响应性能。
默认每题运行 3 次，记录 TTFT、完整响应时间、成功率。

用法：
    python evaluation/benchmark_stream.py --base-url http://localhost:9900 --runs 3
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import httpx
from loguru import logger

from evaluation.common import (
    load_questions,
    parse_sse_event,
    is_content_event,
    compute_percentile,
    write_json,
    write_csv,
    build_run_metadata,
)


async def _run_single_stream(
    client: httpx.AsyncClient,
    base_url: str,
    question: str,
    session_id: str,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    """执行单次 SSE 流式请求。

    Args:
        client: httpx 异步客户端
        base_url: FastAPI 服务地址
        question: 问题文本
        session_id: 唯一会话 ID
        timeout: 超时时间（秒）

    Returns:
        {
            "ttft": float | None,
            "total_time": float,
            "has_content": bool,
            "has_done": bool,
            "has_error": bool,
            "error_type": str,
            "content_length": int,
        }
    """
    result = {
        "ttft": None,
        "total_time": 0,
        "has_content": False,
        "has_done": False,
        "has_error": False,
        "error_type": "",
        "content_length": 0,
    }

    request_start = time.perf_counter()
    first_content_at: Optional[float] = None
    content_chars: List[str] = []

    try:
        async with client.stream(
            "POST",
            f"{base_url}/api/chat_stream",
            json={"Id": session_id, "Question": question},
            timeout=timeout,
        ) as response:
            if response.status_code != 200:
                result["has_error"] = True
                result["error_type"] = f"HTTP {response.status_code}"
                result["total_time"] = time.perf_counter() - request_start
                return result

            async for line in response.aiter_lines():
                if not line:
                    continue

                event = parse_sse_event(line)
                if event is None:
                    continue

                event_type = event.get("type", "")

                if event_type == "error":
                    result["has_error"] = True
                    result["error_type"] = str(event.get("data", "unknown"))
                    break

                if event_type == "done":
                    result["has_done"] = True
                    break

                if is_content_event(event):
                    if first_content_at is None:
                        first_content_at = time.perf_counter()
                        result["has_content"] = True
                    content_chars.append(str(event.get("data", "")))

    except httpx.TimeoutException:
        result["has_error"] = True
        result["error_type"] = "TIMEOUT"
    except Exception as e:
        result["has_error"] = True
        result["error_type"] = f"HTTP_ERROR: {type(e).__name__}"

    completed_at = time.perf_counter()

    result["total_time"] = round(completed_at - request_start, 4)
    if first_content_at is not None:
        result["ttft"] = round(first_content_at - request_start, 4)
    result["content_length"] = len("".join(content_chars))

    return result


def _is_success(result: Dict[str, Any]) -> bool:
    """判定一次测试是否成功。"""
    return (
        not result["has_error"]
        and result["has_content"]
        and result["has_done"]
        and result["ttft"] is not None
    )


async def run_stream_benchmark(
    questions_csv: str,
    base_url: str = "http://localhost:9900",
    runs_per_question: int = 3,
    output_dir: str = "evaluation/results",
) -> Dict[str, Any]:
    """执行 SSE 流式性能评测。

    Args:
        questions_csv: 问题集 CSV 路径
        base_url: FastAPI 服务地址
        runs_per_question: 每题运行次数
        output_dir: 结果输出目录

    Returns:
        汇总结果字典
    """
    run_meta = build_run_metadata(question_file=questions_csv)
    run_id = run_meta["run_id"]
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    questions, _ = load_questions(questions_csv)

    details: List[Dict[str, Any]] = []
    total_runs = 0
    success_runs = 0
    all_ttft: List[float] = []
    all_total_time: List[float] = []

    async with httpx.AsyncClient() as client:
        is_first_request_of_run = True

        for q in questions:
            qid = q["question_id"]
            question = q["question"]

            for run_idx in range(runs_per_question):
                session_id = f"eval-{run_id}-{qid}-{run_idx}"
                is_first_for_q = run_idx == 0

                result = await _run_single_stream(
                    client, base_url, question, session_id
                )

                success = _is_success(result)

                detail = {
                    "run_id": run_id,
                    "question_id": qid,
                    "question": question[:200],
                    "run_index": run_idx,
                    "session_id": session_id,
                    "ttft": result["ttft"],
                    "total_time": result["total_time"],
                    "success": success,
                    "has_content": result["has_content"],
                    "has_done": result["has_done"],
                    "has_error": result["has_error"],
                    "error_type": result["error_type"],
                    "content_length": result["content_length"],
                    "is_first_run_for_question": is_first_for_q,
                    "is_first_request_of_run": is_first_request_of_run,
                }
                details.append(detail)

                total_runs += 1
                if success:
                    success_runs += 1
                    all_ttft.append(result["ttft"])
                    all_total_time.append(result["total_time"])

                is_first_request_of_run = False

                logger.info(
                    f"[{qid}] run {run_idx + 1}/{runs_per_question}: "
                    f"TTFT={result['ttft']}s, total={result['total_time']}s, "
                    f"success={success}"
                )

    # 汇总
    success_rate = success_runs / total_runs if total_runs > 0 else 0
    summary = {
        "run_id": run_id,
        "total_runs": total_runs,
        "success_runs": success_runs,
        "success_rate": round(success_rate, 4),
        "ttft": {
            "avg": round(sum(all_ttft) / len(all_ttft), 4) if all_ttft else None,
            "p50": round(compute_percentile(all_ttft, 0.5), 4) if all_ttft else None,
            "p95": round(compute_percentile(all_ttft, 0.95), 4) if all_ttft else None,
        },
        "total_time": {
            "avg": round(sum(all_total_time) / len(all_total_time), 2) if all_total_time else None,
            "p50": round(compute_percentile(all_total_time, 0.5), 2) if all_total_time else None,
            "p95": round(compute_percentile(all_total_time, 0.95), 2) if all_total_time else None,
        },
        "note": "P50/P95 仅基于成功请求计算。TTFT = 首个 content 事件到达时间。",
    }

    write_csv(details, str(out_dir / "stream_details.csv"))
    write_json(summary, str(out_dir / "stream_summary.json"))

    from datetime import datetime, timezone, timedelta
    run_meta["finished_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
    run_meta["status"] = "completed"
    write_json(run_meta, str(out_dir / "run_metadata.json"))

    logger.info(f"流式评测完成: 成功率 {success_rate:.2%}, TTFT P50={summary['ttft']['p50']}s")
    return summary


def main():
    parser = argparse.ArgumentParser(description="SSE 流式性能评测")
    parser.add_argument("--questions", default="evaluation/questions.csv", help="问题集 CSV 路径")
    parser.add_argument("--base-url", default="http://localhost:9900", help="FastAPI 服务地址")
    parser.add_argument("--runs", type=int, default=3, help="每题运行次数")
    parser.add_argument("--output-dir", default="evaluation/results", help="结果输出目录")
    parser.add_argument("--timeout", type=int, default=120, help="超时（秒）")
    args = parser.parse_args()

    qpath = Path(args.questions)
    if not qpath.exists():
        print(f"[BLOCKED] 问题集不存在: {args.questions}")
        sys.exit(2)

    # 检查 FastAPI
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{args.base_url}/health")
            if resp.status_code != 200:
                print(f"[BLOCKED] FastAPI 服务不可用: {args.base_url}")
                sys.exit(3)
    except Exception as e:
        print(f"[BLOCKED] 无法连接 FastAPI: {e}")
        sys.exit(3)

    # 加载问题集查看数量
    questions, _ = load_questions(args.questions)
    est_calls = len(questions) * args.runs
    logger.warning("=" * 60)
    logger.warning(f"即将执行 SSE 流式性能评测: {len(questions)} 题 × {args.runs} 次 = {est_calls} 次 Agent 调用")
    logger.warning("=" * 60)

    try:
        summary = asyncio.run(
            run_stream_benchmark(
                questions_csv=args.questions,
                base_url=args.base_url,
                runs_per_question=args.runs,
                output_dir=args.output_dir,
            )
        )
        print(f"流式评测完成, 成功率: {summary['success_rate']:.2%}")
        print(f"TTFT P50: {summary['ttft']['p50']}s, P95: {summary['ttft']['p95']}s")
        print(f"结果保存在: {args.output_dir}")
    except Exception as e:
        logger.error(f"流式评测失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
