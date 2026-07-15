"""知识库规模盘点脚本（只读操作）

统计：
- uploads/ 中 .md/.txt 文件数、大小、字符数
- 疑似重复文件
- Milvus collection 实体数、来源数、分片分布
- 分片字符长度分布

用法：
    python evaluation/inventory.py
    python evaluation/inventory.py --output-dir evaluation/results/my_run
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger

from evaluation.common import (
    normalize_file_name,
    write_json,
    write_csv,
    build_run_metadata,
    safe_milvus_preflight,
    compute_percentile,
    EvalError,
)


def _scan_uploads(uploads_dir: str) -> List[Dict[str, Any]]:
    """扫描 uploads/ 目录，返回文件信息列表。

    Args:
        uploads_dir: uploads 目录路径

    Returns:
        文件信息列表，每个元素包含字段：
        - file_name: 原始文件名
        - normalized_name: 规范化文件名
        - extension: 扩展名 (.md/.txt)
        - size_bytes: 文件字节数
        - char_count: 文本字符数
        - is_suspected_duplicate: 是否疑似重复
        - duplicate_group: 重复组编号（0 表示唯一）
    """
    uploads_path = Path(uploads_dir)
    if not uploads_path.exists():
        logger.error(f"uploads 目录不存在: {uploads_dir}")
        return []

    files = sorted(
        [f for f in uploads_path.iterdir() if f.suffix.lower() in (".md", ".txt")]
    )

    records = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            try:
                text = f.read_text(encoding="gbk")
            except Exception:
                text = ""

        records.append({
            "file_name": f.name,
            "normalized_name": normalize_file_name(f.name),
            "extension": f.suffix.lower(),
            "size_bytes": f.stat().st_size,
            "char_count": len(text),
            "is_suspected_duplicate": False,
            "duplicate_group": 0,
        })

    # 检测疑似重复（按规范化名称分组）
    name_groups: Dict[str, List[int]] = {}
    for i, rec in enumerate(records):
        norm = rec["normalized_name"]
        name_groups.setdefault(norm, []).append(i)

    dup_group = 1
    for norm, indices in name_groups.items():
        if len(indices) > 1:
            for idx in indices:
                records[idx]["is_suspected_duplicate"] = True
                records[idx]["duplicate_group"] = dup_group
            dup_group += 1

    return records


def _scan_milvus() -> Dict[str, Any]:
    """只读扫描 Milvus collection，返回统计信息。

    Returns:
        {
            "ok": bool,
            "entity_count": int,
            "unique_sources": int,
            "chunks_per_source": dict,
            "chunk_lengths": list,
            "error": str | None,
        }
    """
    try:
        from pymilvus import utility

        from app.core.milvus_client import milvus_manager
        milvus_manager.connect()

        if not utility.has_collection("biz"):
            return {"ok": False, "error": "Collection 'biz' 不存在", "entity_count": 0}

        collection = milvus_manager.get_collection()
        collection.load()

        entity_count = collection.num_entities
        logger.info(f"Milvus 实体总数: {entity_count}")

        if entity_count == 0:
            return {
                "ok": True,
                "entity_count": 0,
                "unique_sources": 0,
                "chunks_per_source": {},
                "chunk_lengths": [],
                "error": None,
            }

        # 分页读取所有实体（只读 metadata._file_name 和 content）
        chunk_lengths: List[int] = []
        source_chunks: Dict[str, int] = {}

        offset = 0
        page_size = 1000
        while offset < entity_count:
            results = collection.query(
                expr="id != ''",
                output_fields=["content", "metadata"],
                offset=offset,
                limit=min(page_size, entity_count - offset),
            )
            for row in results:
                content = row.get("content", "")
                chunk_lengths.append(len(content))

                metadata = row.get("metadata", {})
                if isinstance(metadata, dict):
                    file_name = metadata.get("_file_name", "unknown")
                else:
                    file_name = "unknown"
                norm = normalize_file_name(str(file_name))
                source_chunks[norm] = source_chunks.get(norm, 0) + 1

            offset += page_size

        return {
            "ok": True,
            "entity_count": entity_count,
            "unique_sources": len(source_chunks),
            "chunks_per_source": source_chunks,
            "chunk_lengths": chunk_lengths,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Milvus 扫描失败: {e}")
        return {"ok": False, "error": str(e), "entity_count": 0}


def run_inventory(
    uploads_dir: str = "uploads",
    output_dir: str = "evaluation/results",
    run_id: str | None = None,
) -> str:
    """执行知识库规模盘点。

    Args:
        uploads_dir: uploads 目录路径
        output_dir: 结果输出目录

    Returns:
        run_id
    """
    run_meta = build_run_metadata(question_file="N/A")
    if run_id is None:
        run_id = run_meta["run_id"]
    run_meta["run_id"] = run_id
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"===== 知识库规模盘点开始 (run_id={run_id}) =====")

    # ── 1. 本地文件统计 ──
    logger.info("[1/3] 扫描本地 uploads 目录...")
    file_records = _scan_uploads(uploads_dir)
    local_file_count = len(file_records)
    local_dup_count = sum(1 for r in file_records if r["is_suspected_duplicate"])
    local_total_chars = sum(r["char_count"] for r in file_records)

    logger.info(f"  本地文件数: {local_file_count}, 疑似重复: {local_dup_count}, 总字符数: {local_total_chars}")

    # ── 2. Milvus 统计 ──
    logger.info("[2/3] 扫描 Milvus collection...")
    milvus_info = _scan_milvus()

    # ── 3. 汇总 ──
    logger.info("[3/3] 生成汇总...")

    chunk_lens = milvus_info.get("chunk_lengths", [])
    chunks_per_source = milvus_info.get("chunks_per_source", {})
    chunk_counts = list(chunks_per_source.values())

    inventory = {
        "run_id": run_id,
        "local_files": {
            "total_files": local_file_count,
            "suspected_duplicates": local_dup_count,
            "total_chars": local_total_chars,
        },
        "milvus": {
            "ok": milvus_info["ok"],
            "entity_count": milvus_info.get("entity_count", 0),
            "unique_sources": milvus_info.get("unique_sources", 0),
            "avg_chunks_per_source": (
                round(sum(chunk_counts) / len(chunk_counts), 1) if chunk_counts else 0
            ),
            "chunk_length": {
                "min": min(chunk_lens) if chunk_lens else 0,
                "avg": round(sum(chunk_lens) / len(chunk_lens), 1) if chunk_lens else 0,
                "median": compute_percentile([float(x) for x in chunk_lens], 0.5) if chunk_lens else 0,
                "p95": compute_percentile([float(x) for x in chunk_lens], 0.95) if chunk_lens else 0,
                "max": max(chunk_lens) if chunk_lens else 0,
            },
            "error": milvus_info.get("error"),
        },
    }

    # 写入结果
    write_json(inventory, str(out_dir / "inventory.json"))

    # 分片明细 CSV
    if chunk_lens:
        chunk_rows = [
            {"chunk_index": i, "char_count": cl} for i, cl in enumerate(chunk_lens)
        ]
        write_csv(chunk_rows, str(out_dir / "inventory_chunks.csv"))

    # 疑似重复报告
    dup_records = [r for r in file_records if r["is_suspected_duplicate"]]
    if dup_records:
        write_csv(dup_records, str(out_dir / "inventory_duplicates.csv"))

    # 每个来源的分片数
    source_rows = [
        {"normalized_source": src, "chunk_count": cnt}
        for src, cnt in sorted(chunks_per_source.items(), key=lambda x: -x[1])
    ]
    if source_rows:
        write_csv(source_rows, str(out_dir / "inventory_sources.csv"))

    # 更新元数据
    run_meta["finished_at"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone(__import__("datetime").timedelta(hours=8))
    ).isoformat()
    run_meta["status"] = "completed"
    write_json(run_meta, str(out_dir / "run_metadata.json"))

    logger.info(f"===== 盘点完成，结果保存在: {out_dir} =====")
    return run_id


def main():
    parser = argparse.ArgumentParser(description="知识库规模盘点（只读）")
    parser.add_argument("--uploads-dir", default="uploads", help="uploads 目录路径")
    parser.add_argument("--output-dir", default="evaluation/results", help="结果输出目录")
    parser.add_argument("--run-id", default=None, help="指定 run_id（不指定则自动生成）")
    args = parser.parse_args()

    # 检查 Milvus
    preflight = safe_milvus_preflight()
    if not preflight["ok"]:
        logger.error(f"Milvus 不可用: {preflight['error']}")
        print(f"[BLOCKED] Milvus 不可用: {preflight['error']}")
        sys.exit(3)

    try:
        run_id = run_inventory(
            uploads_dir=args.uploads_dir,
            output_dir=args.output_dir,
        )
        print(f"盘点完成, run_id={run_id}")
        print(f"结果目录: {args.output_dir}/{run_id}")
    except Exception as e:
        logger.error(f"盘点失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
