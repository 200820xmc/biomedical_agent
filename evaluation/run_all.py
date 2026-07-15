"""评测编排脚本

按顺序运行所有评测模块，生成最终报告。

用法：
    python evaluation/run_all.py --questions evaluation/questions.csv
    python evaluation/run_all.py --questions evaluation/questions.csv --safe-only
    python evaluation/run_all.py --questions evaluation/questions.csv --allow-external-calls
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger

from evaluation.common import write_json, build_run_metadata, safe_milvus_preflight, CST

CST = timezone(timedelta(hours=8))


def _run_step(name: str, args: List[str]) -> Dict[str, Any]:
    """运行一个评测步骤。

    Args:
        name: 步骤名称
        args: 命令行参数列表

    Returns:
        {"name": ..., "status": "completed|skipped|blocked|failed", "exit_code": int, "error": str}
    """
    logger.info(f"===== 开始: {name} =====")
    t0 = time.perf_counter()

    try:
        result = subprocess.run(
            [sys.executable] + args,
            cwd=str(_PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        elapsed = time.perf_counter() - t0

        status = "completed" if result.returncode == 0 else "failed"
        # 特殊退出码映射
        if result.returncode == 4:
            status = "needs_review"
        elif result.returncode == 3:
            status = "blocked"

        logger.info(f"===== 完成: {name} (exit={result.returncode}, {elapsed:.0f}s) =====")

        return {
            "name": name,
            "status": status,
            "exit_code": result.returncode,
            "stdout": result.stdout[-2000:],
            "stderr": result.stderr[-1000:],
            "elapsed_seconds": round(elapsed, 1),
        }
    except subprocess.TimeoutExpired:
        logger.error(f"===== 超时: {name} =====")
        return {
            "name": name,
            "status": "failed",
            "exit_code": -1,
            "error": "timeout (600s)",
            "elapsed_seconds": 600,
        }
    except Exception as e:
        logger.error(f"===== 失败: {name}: {e} =====")
        return {
            "name": name,
            "status": "failed",
            "exit_code": -1,
            "error": str(e),
            "elapsed_seconds": time.perf_counter() - t0,
        }


def main():
    parser = argparse.ArgumentParser(description="AVF RAG 评测编排")
    parser.add_argument("--questions", default="evaluation/questions.csv", help="问题集 CSV 路径")
    parser.add_argument("--output-dir", default="evaluation/results", help="结果输出目录")
    parser.add_argument("--base-url", default="http://localhost:9900", help="FastAPI 服务地址")
    parser.add_argument("--safe-only", action="store_true", help="只运行本地盘点和索引 dry-run")
    parser.add_argument("--allow-external-calls", action="store_true", help="允许 Embedding、Agent、SSE 调用")
    parser.add_argument("--skip-generation", action="store_true", help="跳过生成回答评测")
    parser.add_argument("--skip-stream", action="store_true", help="跳过流式性能评测")
    parser.add_argument("--fail-fast", action="store_true", help="任一阶段失败立即停止")
    args = parser.parse_args()

    questions_csv = args.questions
    output_dir = args.output_dir
    base_url = args.base_url

    # 生成 run_id
    run_meta = build_run_metadata(question_file=questions_csv)
    run_id = run_meta["run_id"]
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"评测开始, run_id={run_id}")
    logger.info(f"safe_only={args.safe_only}, allow_external_calls={args.allow_external_calls}")

    steps: List[Dict[str, Any]] = []

    # ── 阶段 1: 单元测试 ──
    step = _run_step("单元测试", ["-m", "pytest", "tests/evaluation/", "-q", "-o", "addopts=", "-p", "no:cacheprovider"])
    steps.append(step)
    if args.fail_fast and step["status"] == "failed":
        logger.error("单元测试失败，--fail-fast 终止")
        sys.exit(1)

    # 所有子脚本共享同一个 run_id
    common_args = ["--run-id", run_id, "--output-dir", output_dir]

    # ── 阶段 2: 知识库盘点 ──
    step = _run_step("知识库盘点", ["evaluation/inventory.py"] + common_args)
    steps.append(step)

    # ── 阶段 3: 索引 dry-run ──
    step = _run_step("索引 dry-run", ["evaluation/benchmark_indexing.py", "--dry-run"] + common_args)
    steps.append(step)

    if args.safe_only:
        logger.info("--safe-only 模式，跳过外部调用阶段")
        _write_report(steps, run_id, out_dir, output_dir)
        return

    # ── 阶段 4: 检索评测 ──
    step = _run_step("检索评测", [
        "evaluation/evaluate_retrieval.py",
        "--questions", questions_csv,
    ] + common_args)
    steps.append(step)

    # ── 阶段 5: 去重评测 ──
    step = _run_step("去重评测", [
        "evaluation/evaluate_deduplication.py",
        "--questions", questions_csv,
    ] + common_args)
    steps.append(step)

    if not args.allow_external_calls:
        logger.info("未授权外部调用，跳过生成评测和流式评测。使用 --allow-external-calls 启用。")
        _write_report(steps, run_id, out_dir, output_dir)
        return

    # ── 阶段 6: 生成回答评测 ──
    if not args.skip_generation:
        step = _run_step("生成回答评测", [
            "evaluation/evaluate_generation.py",
            "--questions", questions_csv,
            "--base-url", base_url,
        ] + common_args)
        steps.append(step)
    else:
        steps.append({"name": "生成回答评测", "status": "skipped", "exit_code": 0})

    # ── 阶段 7: 流式性能评测 ──
    if not args.skip_stream:
        step = _run_step("流式性能评测", [
            "evaluation/benchmark_stream.py",
            "--questions", questions_csv,
            "--base-url", base_url,
            "--runs", "3",
        ] + common_args)
        steps.append(step)
    else:
        steps.append({"name": "流式性能评测", "status": "skipped", "exit_code": 0})

    # ── 写入最终报告 ──
    _write_report(steps, run_id, out_dir, output_dir)

    # 退出码
    failed = [s for s in steps if s["status"] == "failed"]
    blocked = [s for s in steps if s["status"] == "blocked"]
    if failed:
        logger.error(f"{len(failed)} 个阶段失败: {[s['name'] for s in failed]}")
        sys.exit(1)
    if blocked:
        logger.warning(f"{len(blocked)} 个阶段被阻塞: {[s['name'] for s in blocked]}")
    logger.info(f"全部评测完成, run_id={run_id}")


def _write_report(steps: List[Dict[str, Any]], run_id: str, out_dir: Path, output_dir: str):
    """生成最终报告 markdown。"""
    lines = [
        f"# AVF RAG 项目评测报告",
        f"",
        f"**Run ID**: `{run_id}`",
        f"**生成时间**: {datetime.now(CST).isoformat()}",
        f"",
        f"## 阶段执行状态",
        f"",
        f"| 阶段 | 状态 | 退出码 | 耗时 |",
        f"|------|------|--------|------|",
    ]

    for s in steps:
        status_emoji = {
            "completed": "OK",
            "skipped": "SKIP",
            "blocked": "BLOCKED",
            "failed": "FAIL",
            "needs_review": "REVIEW",
        }.get(s["status"], "??")
        lines.append(f"| {s['name']} | {status_emoji} | {s.get('exit_code', '')} | {s.get('elapsed_seconds', '')}s |")

    # 简历候选
    lines.extend([
        f"",
        f"## 简历候选描述",
        f"",
        f"> 以下数据来自本次正式评测，可直接用于简历：",
        f"",
        f"```text",
        f"构建包含 XX 篇 AVF 论文、XX 个文本分片的医学文献知识库；",
        f"基于 XX 道人工标注问题进行检索评测，Hit@5 达到 XX%，Recall@5 达到 XX%。",
        f"",
        f"通过候选分片超额召回与论文级去重，将 Top-5 平均来源覆盖数",
        f"由 X.X 篇提升至 X.X 篇，重复来源占比下降 XX 个百分点。",
        f"",
        f"SSE 流式问答 TTFT P50 为 X.X 秒，P95 为 X.X 秒，成功率 XX%。",
        f"```",
        f"",
        f"> 注：XX 部分需从各模块汇总 JSON 中提取实际数值后填入。",
        f"",
        f"## 限制与偏差",
        f"",
        f"- 引用评测使用 reconstructed_retrieval（回答后重新检索），非 Agent 真实工具调用轨迹",
        f"- 引用支持率须人工核验后填入",
        f"- 流式评测在同一网络环境下执行，结果受网络波动影响",
        f"- 索引性能为 dry-run 结果，不包含 Embedding API 调用和 Milvus 写入耗时",
        f"",
        f"## 结果文件",
        f"",
    ])

    for f in sorted(out_dir.glob("*")):
        if f.is_file():
            lines.append(f"- `{f.name}`")

    report_path = out_dir / "final_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"最终报告已生成: {report_path}")


if __name__ == "__main__":
    main()
