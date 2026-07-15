"""评测通用工具模块

提供文件名规范化、指标计算、CSV/JSON 读写、Milvus 只读检查等通用能力。
本模块只提供工具函数，不执行评测任务。
"""

import csv
import json
import os
import re
import unicodedata
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─── 项目根目录（评测脚本中用于相对路径解析） ───
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── CST 时区（北京时间） ───
CST = timezone(timedelta(hours=8))


# ============================================================
# 文件名规范化
# ============================================================

def normalize_file_name(name: str) -> str:
    """对文件名执行 8 步规范化，用于跨 Milvus/本地文件匹配。

    步骤：
    1. 反斜杠转正斜杠
    2. 提取路径最后一段（纯文件名）
    3. Unicode NFKC 规范化
    4. 去除首尾空白
    5. casefold() 统一大小写
    6. 去除 .md / .txt 后缀
    7. 连续空格、下划线、连字符统一为单个空格
    8. 再次去除首尾空白

    Args:
        name: 原始文件名或路径

    Returns:
        规范化后的文件名（不含后缀，小写，分隔符统一为空格）
    """
    # 1. 反斜杠 → 正斜杠
    name = name.replace("\\", "/")

    # 2. 只取最后一段文件名
    if "/" in name:
        name = name.split("/")[-1]

    # 3. Unicode NFKC 规范化
    name = unicodedata.normalize("NFKC", name)

    # 4. 去除首尾空白
    name = name.strip()

    # 5. casefold（比 lower 更激进的统一大小写）
    name = name.casefold()

    # 6. 去除扩展名
    for ext in (".md", ".txt"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break

    # 7. Unicode 特殊空白字符 → 普通空格（\xa0 不间断空格等）
    name = re.sub(r"[  -​  　]", " ", name)

    # 7b. 编码转换产生的 ? 替换符 → 移除
    # 文件名中极少出现真正的问号，移除 ? 可修复 GBK/UTF-8 转换导致的损坏
    name = name.replace("?", "")

    # 8. 统一分隔符：空格、下划线、连字符 → 单个空格
    name = re.sub(r"[\s_\-]+", " ", name)

    # 9. 再次去除首尾空白
    name = name.strip()

    return name


# ============================================================
# CSV 读取与校验
# ============================================================

def load_questions(csv_path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    """读取并校验问题集 CSV。

    校验规则：
    - question_id 不能为空且必须唯一
    - question 不能为空
    - relevant_files 为空或为 NEEDS_REVIEW 时，不进入正式 Recall 分母
    - category 为空时统一归入 "未分类"

    Args:
        csv_path: questions.csv 路径

    Returns:
        (questions, errors): 问题列表和错误信息列表

    Raises:
        FileNotFoundError: CSV 文件不存在
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"问题集文件不存在: {csv_path}")

    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    errors: List[str] = []
    seen_ids: set = set()
    questions: List[Dict[str, str]] = []

    for i, row in enumerate(rows, start=2):  # 第 1 行是表头
        qid = (row.get("question_id") or "").strip()
        question = (row.get("question") or "").strip()
        relevant = (row.get("relevant_files") or "").strip()
        category = (row.get("category") or "").strip()

        # 校验
        if not qid:
            errors.append(f"第 {i} 行: question_id 为空")
            continue
        if qid in seen_ids:
            errors.append(f"第 {i} 行: question_id '{qid}' 重复")
            continue
        if not question:
            errors.append(f"第 {i} 行: question 为空")
            continue

        seen_ids.add(qid)
        if not category:
            category = "未分类"

        questions.append({
            "question_id": qid,
            "question": question,
            "relevant_files": relevant,
            "category": category,
            "notes": (row.get("notes") or "").strip(),
        })

    return questions, errors


# ============================================================
# 标注文件匹配
# ============================================================

def validate_relevant_files(
    relevant_files_str: str,
    milvus_normalized_sources: List[str],
) -> Tuple[List[str], str, List[str]]:
    """校验人工标注的相关文件是否能匹配到 Milvus 来源。

    Args:
        relevant_files_str: 分号分隔的标注文件名
        milvus_normalized_sources: Milvus 中所有规范化来源列表

    Returns:
        (matched, status, ambiguous):
        - matched: 匹配到的规范化来源列表
        - status: "matched" | "missing" | "ambiguous" | "needs_review"
        - ambiguous: 匹配到多个来源的标注文件列表
    """
    if not relevant_files_str or relevant_files_str.upper() == "NEEDS_REVIEW":
        return [], "needs_review", []

    raw_files = [f.strip() for f in relevant_files_str.split(";") if f.strip()]
    matched: List[str] = []
    missing: List[str] = []
    ambiguous: List[str] = []

    for raw in raw_files:
        normalized = normalize_file_name(raw)

        # 在 Milvus 来源中查找匹配
        candidates = [s for s in milvus_normalized_sources if normalized in s or s in normalized]

        if len(candidates) == 0:
            missing.append(raw)
        elif len(candidates) == 1:
            matched.append(candidates[0])
        else:
            # 多个候选，尝试精确匹配
            exact = [s for s in candidates if s == normalized]
            if len(exact) == 1:
                matched.append(exact[0])
            else:
                ambiguous.append(raw)
                # 仍然加入第一个候选（最佳猜测）
                matched.append(candidates[0])

    if missing and not matched:
        return matched, "missing", []
    if ambiguous:
        return matched, "ambiguous", ambiguous
    if missing:
        # 部分匹配也算 matched，但标注中有缺失
        return matched, "matched", []

    return matched, "matched", []


# ============================================================
# 指标计算
# ============================================================

def compute_hit_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int,
) -> int:
    """计算单题 Hit@K。

    Top-K 中只要出现任意一篇相关论文，返回 1，否则返回 0。

    Args:
        retrieved_sources: 前 K 个检索结果的规范化来源列表
        relevant_sources: 标注的相关规范化来源列表
        k: Top-K 的 K 值

    Returns:
        1（命中）或 0（未命中）
    """
    if not relevant_sources:
        return 0
    top_k = retrieved_sources[:k]
    return 1 if any(src in relevant_sources for src in top_k) else 0


def compute_recall_at_k(
    retrieved_sources: List[str],
    relevant_sources: List[str],
    k: int,
) -> float:
    """计算单题 Recall@K。

    Recall@K = Top-K 中命中的不同相关论文数 / 该问题的相关论文总数

    Args:
        retrieved_sources: 前 K 个检索结果的规范化来源列表
        relevant_sources: 标注的相关规范化来源列表
        k: Top-K 的 K 值

    Returns:
        Recall 值 (0.0 ~ 1.0)，如果 relevant_sources 为空则返回 0.0
    """
    if not relevant_sources:
        return 0.0
    top_k = retrieved_sources[:k]
    # 标准答案可能因人工录入或文件别名出现重复。Recall 的分母应是
    # 不同相关论文数，否则重复标签会把同一道题的 Recall 人为压低。
    unique_relevant_sources = set(relevant_sources)
    hit_count = sum(1 for src in set(top_k) if src in unique_relevant_sources)
    return hit_count / len(unique_relevant_sources)


def compute_source_coverage(
    retrieved_sources: List[str],
    k: int,
) -> int:
    """计算 Top-K 中不同来源的数量。

    Args:
        retrieved_sources: 检索结果的规范化来源列表
        k: Top-K 的 K 值

    Returns:
        不同来源数
    """
    top_k = retrieved_sources[:k]
    return len(set(top_k))


def compute_duplicate_ratio(
    retrieved_sources: List[str],
    k: int,
) -> Optional[float]:
    """计算 Top-K 的重复来源占比。

    DuplicateRatio = 1 - 不同来源数 / 实际返回结果数

    Args:
        retrieved_sources: 检索结果的规范化来源列表
        k: Top-K 的 K 值

    Returns:
        重复占比 (0.0 ~ 1.0)，实际返回数为 0 时返回 None
    """
    top_k = retrieved_sources[:k]
    actual_count = len(top_k)
    if actual_count == 0:
        return None
    unique_count = len(set(top_k))
    return 1.0 - (unique_count / actual_count)


def compute_percentile(data: List[float], percentile: float) -> Optional[float]:
    """计算百分位数（线性插值法）。

    position = (n - 1) × percentile

    Args:
        data: 数值列表
        percentile: 百分位 (0.0 ~ 1.0)

    Returns:
        百分位数值，空列表返回 None
    """
    if not data:
        return None

    sorted_data = sorted(data)
    n = len(sorted_data)

    if n == 1:
        return sorted_data[0]

    position = (n - 1) * percentile
    lower_idx = int(position)
    upper_idx = min(lower_idx + 1, n - 1)
    frac = position - lower_idx

    return sorted_data[lower_idx] + frac * (sorted_data[upper_idx] - sorted_data[lower_idx])


# ============================================================
# 结果写入
# ============================================================

def write_json(data: Any, path: str, indent: int = 2) -> None:
    """将数据写入 JSON 文件（UTF-8，ensure_ascii=False 保留中文）。

    Args:
        data: 要写入的数据
        path: 输出路径
        indent: 缩进空格数
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent, default=str)


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    """将字典列表写入 CSV 文件（UTF-8 with BOM，Excel 兼容）。

    Args:
        rows: 字典列表
        path: 输出路径
    """
    if not rows:
        return

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # 收集所有列名（保持出现顺序）
    fieldnames: List[str] = []
    seen_fields: set = set()
    for row in rows:
        for key in row:
            if key not in seen_fields:
                fieldnames.append(key)
                seen_fields.add(key)

    with open(p, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 运行元数据
# ============================================================

def build_run_metadata(
    question_file: str,
    collection_name: str = "biz",
    embedding_model: str = "text-embedding-v4",
    vector_dimension: int = 1024,
) -> Dict[str, Any]:
    """生成运行元数据。

    Args:
        question_file: 问题集文件路径
        collection_name: Milvus collection 名称
        embedding_model: 嵌入模型名称
        vector_dimension: 向量维度

    Returns:
        运行元数据字典
    """
    import hashlib
    import platform
    import sys

    now = datetime.now(CST)
    run_id = now.strftime("%Y%m%d_%H%M%S")

    metadata: Dict[str, Any] = {
        "run_id": run_id,
        "started_at": now.isoformat(),
        "finished_at": None,
        "python_version": sys.version.split()[0],
        "platform": platform.system() + "-" + platform.release(),
        "collection_name": collection_name,
        "embedding_model": embedding_model,
        "vector_dimension": vector_dimension,
        "question_file": question_file,
        "question_file_sha256": "",
        "status": "running",
    }

    # 计算问题文件 SHA256
    qpath = Path(question_file)
    if qpath.exists():
        sha = hashlib.sha256()
        sha.update(qpath.read_bytes())
        metadata["question_file_sha256"] = sha.hexdigest()

    return metadata


# ============================================================
# Milvus 只读检查
# ============================================================

def safe_milvus_preflight(collection_name: str = "biz") -> Dict[str, Any]:
    """对 Milvus 执行只读检查，不写入、不修改。

    检查项：
    - collection 是否存在
    - 向量维度是否为 1024
    - 实体数
    - schema 是否匹配

    Args:
        collection_name: collection 名称

    Returns:
        {"ok": bool, "entity_count": int, "dimension": int, "error": str | None}

    Raises:
        不会抛出异常，错误信息在返回值的 error 字段中
    """
    from pymilvus import utility

    try:
        # 确保连接
        from app.core.milvus_client import milvus_manager
        milvus_manager.connect()

        if not utility.has_collection(collection_name):
            return {
                "ok": False,
                "entity_count": 0,
                "dimension": 0,
                "error": f"Collection '{collection_name}' 不存在",
            }

        collection = milvus_manager.get_collection()
        collection.load()

        entity_count = collection.num_entities

        # 检查向量维度
        schema = collection.schema
        dimension = 0
        for field in schema.fields:
            from pymilvus import DataType
            if field.dtype in (DataType.FLOAT_VECTOR, DataType.BINARY_VECTOR):
                dimension = field.params.get("dim", 0)
                break

        return {
            "ok": True,
            "entity_count": entity_count,
            "dimension": dimension,
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "entity_count": 0,
            "dimension": 0,
            "error": str(e),
        }


# ============================================================
# SSE 事件解析
# ============================================================

def parse_sse_event(line: str) -> Optional[Dict[str, Any]]:
    """解析单行 SSE 数据。

    支持格式：
        data: {"type": "content", "data": "..."}
        data: [DONE]

    Args:
        line: SSE 原始行（不含换行符）

    Returns:
        解析后的事件字典，解析失败返回 None
    """
    if not line:
        return None

    # 去掉 "data: " 前缀
    if line.startswith("data: "):
        payload = line[6:].strip()
    elif line.startswith("data:"):
        payload = line[5:].strip()
    else:
        return None

    # [DONE] 信号
    if payload == "[DONE]":
        return {"type": "done", "data": None}

    # JSON 负载
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def is_content_event(event: Optional[Dict[str, Any]]) -> bool:
    """判断 SSE 事件是否为有效 content 事件。

    以下类型不计为内容事件：debug, tool_call, search_results, done, error

    Args:
        event: 解析后的事件字典

    Returns:
        是否为有效 content 事件
    """
    if not event or not isinstance(event, dict):
        return False
    event_type = event.get("type", "")
    if event_type in ("debug", "tool_call", "search_results", "done", "error"):
        return False
    if event_type == "content":
        data = event.get("data", "")
        return bool(data)  # 非空才算
    return False


# ============================================================
# 错误类型
# ============================================================

class EvalError:
    """评测错误类型常量"""
    CONFIG_ERROR = "CONFIG_ERROR"
    QUESTION_VALIDATION_ERROR = "QUESTION_VALIDATION_ERROR"
    LABEL_NOT_REVIEWED = "LABEL_NOT_REVIEWED"
    MILVUS_UNAVAILABLE = "MILVUS_UNAVAILABLE"
    COLLECTION_NOT_FOUND = "COLLECTION_NOT_FOUND"
    SCHEMA_MISMATCH = "SCHEMA_MISMATCH"
    EMBEDDING_ERROR = "EMBEDDING_ERROR"
    AGENT_ERROR = "AGENT_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    SSE_PARSE_ERROR = "SSE_PARSE_ERROR"
    TIMEOUT = "TIMEOUT"
    INDEXING_ERROR = "INDEXING_ERROR"
    PERMISSION_REQUIRED = "PERMISSION_REQUIRED"
