"""测试 common.py 中的纯计算逻辑（不依赖 DashScope 或 Milvus）。"""

import json
import os
import tempfile
from pathlib import Path

import pytest

# 将项目根目录加入 sys.path（测试运行目录可能在 tests/evaluation/ 下）
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.common import (
    normalize_file_name,
    load_questions,
    validate_relevant_files,
    parse_sse_event,
    is_content_event,
    compute_hit_at_k,
    compute_recall_at_k,
    compute_source_coverage,
    compute_duplicate_ratio,
    compute_percentile,
)


# ============================================================
# normalize_file_name
# ============================================================

class TestNormalizeFileName:
    """文件名规范化测试"""

    def test_basic_chinese_name(self):
        """中文文件名：去除后缀，casefold"""
        result = normalize_file_name("Zhou 等 - 2023 - Deep learning.MD")
        assert "zhou" in result
        assert "deep learning" in result
        assert not result.endswith(".md")

    def test_windows_backslash_path(self):
        """Windows 反斜杠路径 → 只保留文件名"""
        result = normalize_file_name(r"uploads\Zhou 等 - 2023.md")
        assert "zhou" in result
        assert "uploads" not in result
        assert "\\" not in result

    def test_unix_path(self):
        """Unix 正斜杠路径"""
        result = normalize_file_name("uploads/sub/Zhou 2023.md")
        assert "zhou" in result
        assert "uploads" not in result

    def test_underscore_to_space(self):
        """下划线 → 空格"""
        result = normalize_file_name("Park_et_al_-_2022.md")
        assert "park et al" in result
        assert "_" not in result

    def test_hyphen_to_space(self):
        """连字符 → 空格"""
        result = normalize_file_name("Ota-等-2020-Evaluation.md")
        assert "ota" in result
        # 连字符被替换为空格
        assert "ota 等 2020 evaluation" in result

    def test_casefold(self):
        """大小写统一"""
        r1 = normalize_file_name("AVF_Stenosis.MD")
        r2 = normalize_file_name("avf_stenosis.md")
        assert r1 == r2

    def test_nfkc_normalization(self):
        """NFKC 规范化：全角字符 → 半角"""
        # 全角字母
        result = normalize_file_name("Ｔｅｓｔ.md")
        assert "test" in result

    def test_whitespace_trim(self):
        """首尾空白去除"""
        result = normalize_file_name("   test paper 2023.md   ")
        assert result == "test paper 2023"

    def test_txt_extension(self):
        """.txt 后缀也去除"""
        result = normalize_file_name("document.txt")
        assert result == "document"

    def test_multiple_separators(self):
        """连续空格、下划线、连字符 → 单个空格"""
        result = normalize_file_name("test___paper---2023   final.md")
        # 所有分隔符压缩为单个空格
        assert result == "test paper 2023 final"

    def test_no_extension(self):
        """无后缀的文件名"""
        result = normalize_file_name("no_extension_file")
        assert "no extension file" in result


# ============================================================
# load_questions
# ============================================================

class TestLoadQuestions:
    """问题集 CSV 读取与校验测试"""

    def _write_csv(self, content: str) -> str:
        """写入临时 CSV 并返回路径"""
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8-sig"
        )
        tmp.write(content)
        tmp.close()
        return tmp.name

    def test_valid_csv(self):
        """正常 CSV 读取"""
        csv_content = (
            "question_id,question,relevant_files,category,notes\n"
            'q001,哪些研究使用深度学习？,Zhou 2023.md;Park 2022.md,深度学习,\n'
            'q002,血流声音如何采集？,NEEDS_REVIEW,听诊,\n'
            'q003,STFT 参数如何选择？,,时频分析,'
        )
        path = self._write_csv(csv_content)
        try:
            questions, errors = load_questions(path)
            assert len(questions) == 3
            assert len(errors) == 0
            assert questions[0]["relevant_files"] == "Zhou 2023.md;Park 2022.md"
            assert questions[1]["relevant_files"] == "NEEDS_REVIEW"
            assert questions[2]["category"] == "时频分析"
        finally:
            os.unlink(path)

    def test_empty_question_id(self):
        """空 question_id 报错"""
        csv_content = (
            "question_id,question,relevant_files,category,notes\n"
            ",test question,,,"
        )
        path = self._write_csv(csv_content)
        try:
            questions, errors = load_questions(path)
            assert len(questions) == 0
            assert len(errors) >= 1
            assert "question_id" in errors[0].lower()
        finally:
            os.unlink(path)

    def test_duplicate_id(self):
        """重复 question_id 报错"""
        csv_content = (
            "question_id,question,relevant_files,category,notes\n"
            "q001,test1,,,\n"
            "q001,test2,,,"
        )
        path = self._write_csv(csv_content)
        try:
            questions, errors = load_questions(path)
            assert len(questions) == 1
            assert len(errors) >= 1
            assert "重复" in errors[0]
        finally:
            os.unlink(path)

    def test_empty_question(self):
        """空 question 报错"""
        csv_content = (
            "question_id,question,relevant_files,category,notes\n"
            "q001,,,"
        )
        path = self._write_csv(csv_content)
        try:
            questions, errors = load_questions(path)
            assert len(errors) >= 1
        finally:
            os.unlink(path)

    def test_empty_category_default(self):
        """空 category 归入 '未分类'"""
        csv_content = (
            "question_id,question,relevant_files,category,notes\n"
            "q001,test,,,"
        )
        path = self._write_csv(csv_content)
        try:
            questions, errors = load_questions(path)
            assert questions[0]["category"] == "未分类"
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        """不存在的文件抛出 FileNotFoundError"""
        with pytest.raises(FileNotFoundError):
            load_questions("/nonexistent/path.csv")


# ============================================================
# validate_relevant_files
# ============================================================

class TestValidateRelevantFiles:
    """标注文件匹配测试"""

    MILVUS_SOURCES = [
        "zhou 等 2023 deep learning analysis of blood flow sounds",
        "park 等 2022 feasibility of deep learning based analysis",
        "song 等 2023 an effective ai model",
        "ota 等 2020 evaluation of hemodialysis arteriovenous bruit",
    ]

    def test_needs_review(self):
        """NEEDS_REVIEW 标记"""
        matched, status, ambig = validate_relevant_files(
            "NEEDS_REVIEW", self.MILVUS_SOURCES
        )
        assert status == "needs_review"
        assert matched == []

    def test_empty_string(self):
        """空字符串"""
        matched, status, ambig = validate_relevant_files("", self.MILVUS_SOURCES)
        assert status == "needs_review"

    def test_matched(self):
        """正常匹配"""
        matched, status, ambig = validate_relevant_files(
            "Zhou 等 - 2023 - Deep learning.md;Park 等 - 2022.md",
            self.MILVUS_SOURCES,
        )
        assert status == "matched"
        assert len(matched) == 2

    def test_missing(self):
        """全部未匹配"""
        matched, status, ambig = validate_relevant_files(
            "NonExistent 2024.md",
            self.MILVUS_SOURCES,
        )
        assert status == "missing"

    def test_ambiguous(self):
        """模糊匹配（多候选但非精确）"""
        sources = ["test 2023.md", "test 2023 final.md"]
        matched, status, ambig = validate_relevant_files("test 2023.md", sources)
        # 两个规范化后可能都包含 "test 2023"
        # 取决于规范化结果
        assert status in ("matched", "ambiguous")


# ============================================================
# parse_sse_event
# ============================================================

class TestParseSSEEvent:
    """SSE 事件解析测试"""

    def test_content_event(self):
        event = parse_sse_event('data: {"type":"content","data":"你好"}')
        assert event is not None
        assert event["type"] == "content"
        assert event["data"] == "你好"

    def test_done_signal(self):
        event = parse_sse_event("data: [DONE]")
        assert event is not None
        assert event["type"] == "done"

    def test_tool_call_event(self):
        event = parse_sse_event(
            'data: {"type":"tool_call","data":{"tool":"retrieve_knowledge"}}'
        )
        assert event is not None
        assert event["type"] == "tool_call"

    def test_empty_line(self):
        event = parse_sse_event("")
        assert event is None

    def test_no_data_prefix(self):
        event = parse_sse_event('{"type":"content"}')
        assert event is None

    def test_invalid_json(self):
        event = parse_sse_event("data: not valid json")
        assert event is None

    def test_search_results_event(self):
        event = parse_sse_event(
            'data: {"type":"search_results","data":{"count":5}}'
        )
        assert event is not None
        assert event["type"] == "search_results"


# ============================================================
# is_content_event
# ============================================================

class TestIsContentEvent:
    """SSE content 事件判定测试"""

    def test_valid_content(self):
        assert is_content_event({"type": "content", "data": "你好"}) is True

    def test_empty_content(self):
        """空 data 不计为有效 content"""
        assert is_content_event({"type": "content", "data": ""}) is False

    def test_debug_ignored(self):
        assert is_content_event({"type": "debug", "data": "..."}) is False

    def test_tool_call_ignored(self):
        assert is_content_event({"type": "tool_call", "data": {}}) is False

    def test_search_results_ignored(self):
        assert is_content_event({"type": "search_results", "data": []}) is False

    def test_done_ignored(self):
        assert is_content_event({"type": "done", "data": None}) is False

    def test_error_ignored(self):
        assert is_content_event({"type": "error", "data": "something"}) is False

    def test_none_event(self):
        assert is_content_event(None) is False

    def test_non_dict(self):
        assert is_content_event("not a dict") is False
