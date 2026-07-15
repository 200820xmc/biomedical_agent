"""索引性能评测脚本

默认 dry-run 模式：只读取文件、分块统计，不调用 Embedding，不写入 Milvus。
正式写入须显式授权并使用独立临时 collection。

用法：
    python evaluation/benchmark_indexing.py --input-dir uploads --dry-run
    python evaluation/benchmark_indexing.py --input-dir uploads --write-mode temporary --confirm-external-calls
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger

from evaluation.common import write_json, write_csv, build_run_metadata, compute_percentile


def run_dry_run(input_dir: str, output_dir: str) -> Dict[str, Any]:
    """执行索引 dry-run：文件读取 + 分块统计。

    Args:
        input_dir: 输入目录（默认 uploads）
        output_dir: 结果输出目录

    Returns:
        汇总结果字典
    """
    from app.services.document_splitter_service import document_splitter_service

    in_path = Path(input_dir)
    if not in_path.exists():
        logger.error(f"输入目录不存在: {input_dir}")
        return {"error": f"输入目录不存在: {input_dir}"}

    files = sorted([f for f in in_path.iterdir() if f.suffix.lower() in (".md", ".txt")])

    details: List[Dict[str, Any]] = []
    total_chars = 0
    total_chunks = 0
    total_split_time = 0.0
    success_count = 0
    fail_count = 0

    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = f.read_text(encoding="gbk")
            except Exception:
                details.append({
                    "file_name": f.name,
                    "file_size_bytes": f.stat().st_size,
                    "char_count": 0,
                    "chunk_count": 0,
                    "split_time_seconds": 0,
                    "status": "failed",
                    "error": "无法读取文件编码",
                })
                fail_count += 1
                continue

        char_count = len(text)

        # 分块计时
        t0 = time.perf_counter()
        try:
            chunks = document_splitter_service.split_document(text, str(f))
            split_time = time.perf_counter() - t0
            chunk_count = len(chunks)
            status = "success"
            error = ""
        except Exception as e:
            split_time = time.perf_counter() - t0
            chunk_count = 0
            status = "failed"
            error = str(e)

        details.append({
            "file_name": f.name,
            "file_size_bytes": f.stat().st_size,
            "char_count": char_count,
            "chunk_count": chunk_count,
            "split_time_seconds": round(split_time, 4),
            "status": status,
            "error": error,
        })

        if status == "success":
            total_chars += char_count
            total_chunks += chunk_count
            total_split_time += split_time
            success_count += 1
        else:
            fail_count += 1

    # 汇总
    split_times = [d["split_time_seconds"] for d in details if d["status"] == "success"]

    summary = {
        "mode": "dry_run",
        "total_files": len(files),
        "success_count": success_count,
        "fail_count": fail_count,
        "total_chars": total_chars,
        "total_chunks": total_chunks,
        "avg_chunks_per_file": round(total_chunks / success_count, 1) if success_count else 0,
        "split_time_seconds": {
            "total": round(total_split_time, 2),
            "avg": round(sum(split_times) / len(split_times), 4) if split_times else None,
            "p50": round(compute_percentile(split_times, 0.5), 4) if split_times else None,
            "p95": round(compute_percentile(split_times, 0.95), 4) if split_times else None,
        },
        "chars_per_10k_index_time": None,  # dry-run 无正式索引耗时
    }

    # 写入结果
    run_id = build_run_metadata(question_file="N/A")["run_id"]
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(details, str(out_dir / "indexing_details.csv"))
    write_json(summary, str(out_dir / "indexing_summary.json"))

    logger.info(f"Dry-run 完成: {success_count}/{len(files)} 成功, 总分片 {total_chunks}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="索引性能评测")
    parser.add_argument("--input-dir", default="uploads", help="输入目录")
    parser.add_argument("--output-dir", default="evaluation/results", help="输出目录")
    parser.add_argument("--dry-run", action="store_true", default=True, help="只分块不写入（默认）")
    parser.add_argument("--write-mode", choices=["temporary"], help="写入独立临时 collection")
    parser.add_argument("--confirm-external-calls", action="store_true", help="确认外部调用")
    args = parser.parse_args()

    if args.write_mode and not args.confirm_external_calls:
        print("[BLOCKED] 正式索引写入需要 --confirm-external-calls 参数确认")
        sys.exit(5)

    if args.write_mode == "temporary":
        print("[BLOCKED] 临时 collection 写入模式尚未实现，请使用 --dry-run")
        sys.exit(5)

    try:
        result = run_dry_run(args.input_dir, args.output_dir)
        if "error" in result:
            print(f"[FAILED] {result['error']}")
            sys.exit(1)
        print(f"Dry-run 完成, 结果保存在: {args.output_dir}")
    except Exception as e:
        logger.error(f"索引评测失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
