"""批量 PDF 入库脚本 — 将 PDF 目录下所有文件上传并解析入库

流程：
1. 扫描 PDF 目录下所有 .pdf 文件
2. 通过 API 上传每个 PDF（保存到 originals + 生成 document_id）
3. 提交入库任务（xParse 解析 → 分块 → Embedding → Milvus）
4. 免费 API 要求串行，逐个等待完成
5. 输出汇总报告

用法：
    python scripts/batch_ingest_pdfs.py
"""

import asyncio
import hashlib
import shutil
import sys
import time
from pathlib import Path

# 修复 Windows GBK 控制台 Unicode 编码问题
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 确保项目根在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.pdf_ingestion_service import pdf_ingestion_service, ORIGINALS_DIR

# ── 配置 ──────────────────────────────────────────────────
PDF_SOURCE_DIR = PROJECT_ROOT / "PDF"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB（免费 API 限制）


def sanitize_filename(name: str) -> str:
    """规范化文件名"""
    sanitized = name.replace(" ", "_")
    for char in ['\\', '/', ':', '*', '?', '"', '<', '>', '|']:
        sanitized = sanitized.replace(char, "_")
    return sanitized


def register_pdf(pdf_path: Path) -> str | None:
    """将 PDF 复制到 originals 并返回 document_id

    跳过已存在且哈希相同的文件。

    Args:
        pdf_path: 源 PDF 路径

    Returns:
        document_id 或 None（跳过）
    """
    # 计算 SHA-256
    content = pdf_path.read_bytes()
    sha256_hash = hashlib.sha256(content).hexdigest()
    document_id = f"doc_{sha256_hash[:6]}"

    # 检查大小
    if len(content) > MAX_FILE_SIZE:
        print(f"  [SKIP] 超过 10MB: {pdf_path.name} ({len(content) / 1024 / 1024:.1f}MB)")
        return None

    # 验证 PDF 文件头
    if not content.startswith(b"%PDF-"):
        print(f"  [SKIP] 非 PDF: {pdf_path.name}")
        return None

    # 保存到 originals
    doc_dir = ORIGINALS_DIR / document_id
    safe_name = sanitize_filename(pdf_path.name)
    dest_path = doc_dir / safe_name

    if dest_path.exists():
        existing_hash = hashlib.sha256(dest_path.read_bytes()).hexdigest()
        if existing_hash == sha256_hash:
            print(f"  [EXISTS] {safe_name} (id={document_id})")
            return document_id

    doc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf_path, dest_path)
    print(f"  [REG] {safe_name} (id={document_id})")
    return document_id


async def ingest_one(pdf_path: Path, index: int, total: int) -> dict:
    """处理单个 PDF：登记 → 入库 → 等待完成"""
    result = {
        "filename": pdf_path.name,
        "success": False,
        "chunks": 0,
        "duration_s": 0,
        "error": "",
    }

    start = time.time()

    try:
        # 1. 登记 PDF
        doc_id = register_pdf(pdf_path)
        if doc_id is None:
            result["error"] = "skipped"
            return result

        # 2. 提交入库任务
        job = await pdf_ingestion_service.submit(
            document_id=doc_id,
            force_reindex=False,  # 已 indexed 的跳过
        )

        if job.status == "queued" and job.chunk_count is None:
            # 新任务，等待完成
            print(f"  [PARSING] ", end="", flush=True)
            for _ in range(120):  # 最多等 10 分钟
                await asyncio.sleep(5)
                job = await pdf_ingestion_service.get_status(job.job_id)
                if job.status == "indexed":
                    result["success"] = True
                    result["chunks"] = job.chunk_count or 0
                    break
                elif job.status == "failed":
                    result["error"] = job.error_message or "unknown"
                    break
                print(".", end="", flush=True)
            print()
        else:
            # 已入库，跳过
            result["success"] = True
            result["chunks"] = job.chunk_count or 0
            result["error"] = "already indexed"
            print(f"  [OK] 已入库({result['chunks']} chunks)，跳过")

    except Exception as e:
        result["error"] = str(e)

    result["duration_s"] = time.time() - start
    return result


async def main():
    print("=" * 60)
    print("批量 PDF 入库")
    print("=" * 60)

    if not PDF_SOURCE_DIR.exists():
        print(f"[ERROR] PDF 目录不存在: {PDF_SOURCE_DIR}")
        return

    # 收集所有 PDF
    pdf_files = sorted(
        [f for f in PDF_SOURCE_DIR.glob("*.pdf") if f.is_file()],
        key=lambda f: f.stat().st_size,  # 小的先处理
    )
    total = len(pdf_files)
    print(f"\n找到 {total} 个 PDF 文件\n")

    if total == 0:
        print("没有 PDF 文件需要处理。")
        return

    # 统计信息
    total_size = sum(f.stat().st_size for f in pdf_files)
    too_large = sum(1 for f in pdf_files if f.stat().st_size > MAX_FILE_SIZE)
    print(f"总大小: {total_size / 1024 / 1024:.1f} MB")
    print(f"超过 10MB（将跳过）: {too_large} 个")
    print(f"预计可用: {total - too_large} 个")
    print()

    # 自动确认（可通过命令行参数 --yes 跳过交互）
    auto_yes = "--yes" in sys.argv
    if not auto_yes:
        confirm = input("开始批量入库？(y/N): ").strip().lower()
        if confirm != "y":
            print("已取消。")
            return

    # ── 串行处理 ──────────────────────────────────────────
    results = []
    success_count = 0
    skip_count = 0
    fail_count = 0
    total_chunks = 0
    overall_start = time.time()

    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{total}] {pdf_path.name[:80]}")

        result = await ingest_one(pdf_path, i, total)
        results.append(result)

        if result["error"] == "skipped":
            skip_count += 1
        elif result["success"]:
            success_count += 1
            total_chunks += result["chunks"]
            dur = result["duration_s"]
            print(f"  [OK] 成功({result['chunks']} chunks, {dur:.0f}s)")
        else:
            fail_count += 1
            print(f"  [FAIL] {result['error'][:100]}")

    # ── 汇总报告 ──────────────────────────────────────────
    overall_dur = time.time() - overall_start
    print("\n" + "=" * 60)
    print("批量入库完成！")
    print("=" * 60)
    print(f"  总计: {total} 个")
    print(f"  成功: {success_count} 个")
    print(f"  跳过: {skip_count} 个")
    print(f"  失败: {fail_count} 个")
    print(f"  总chunks: {total_chunks} 个")
    print(f"  总耗时: {overall_dur / 60:.1f} 分钟")
    print("=" * 60)

    if fail_count > 0:
        print("\n失败文件:")
        for r in results:
            if not r["success"] and r["error"] != "skipped" and r["error"] != "already indexed":
                print(f"  - {r['filename']}: {r['error'][:80]}")


if __name__ == "__main__":
    asyncio.run(main())
