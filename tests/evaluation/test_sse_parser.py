"""测试 SSE 解析与 TTFT 计时逻辑（纯计算，不依赖外部服务）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.common import parse_sse_event, is_content_event


class TestSSETimingLogic:
    """SSE 计时相关逻辑测试"""

    # 模拟一次完整 SSE 流
    FULL_STREAM = [
        ('data: {"type":"debug","node":"agent","message_type":"start"}', "debug"),
        ('data: {"type":"tool_call","data":{"tool":"retrieve_knowledge"}}', "tool_call"),
        ('data: {"type":"search_results","data":[{"_file_name":"test.md"}]}', "search_results"),
        ('data: {"type":"content","data":"根据"}', "content"),           # ← 首个 content
        ('data: {"type":"content","data":"研究显示"}', "content"),
        ('data: {"type":"content","data":"..."}', "content"),
        ('data: {"type":"done","data":{"answer":"完整答案"}}', "done"),
    ]

    def test_first_content_identification(self):
        """正确识别首个 content 事件"""
        first_content_idx = None
        for i, (line, _) in enumerate(self.FULL_STREAM):
            event = parse_sse_event(line)
            if is_content_event(event):
                first_content_idx = i
                break
        assert first_content_idx == 3  # 第 4 个事件是首个 content

    def test_debug_not_content(self):
        """debug 事件不计为 content"""
        event = parse_sse_event(self.FULL_STREAM[0][0])
        assert is_content_event(event) is False

    def test_tool_call_not_content(self):
        """tool_call 事件不计为 content"""
        event = parse_sse_event(self.FULL_STREAM[1][0])
        assert is_content_event(event) is False

    def test_search_results_not_content(self):
        """search_results 事件不计为 content"""
        event = parse_sse_event(self.FULL_STREAM[2][0])
        assert is_content_event(event) is False

    def test_done_not_content(self):
        """done 事件不计为 content"""
        event = parse_sse_event(self.FULL_STREAM[6][0])
        assert is_content_event(event) is False

    def test_empty_content_not_counted(self):
        """空 content data 不计为有效 content"""
        event = {"type": "content", "data": ""}
        assert is_content_event(event) is False
        event = {"type": "content", "data": "有效内容"}
        assert is_content_event(event) is True


class TestSSEErrorHandling:
    """SSE 错误和异常事件处理测试"""

    def test_error_event_parsed(self):
        event = parse_sse_event('data: {"type":"error","data":"timeout"}')
        assert event is not None
        assert event["type"] == "error"

    def test_error_not_content(self):
        event = parse_sse_event('data: {"type":"error","data":"timeout"}')
        assert is_content_event(event) is False

    def test_malformed_then_valid(self):
        """畸形事件后仍能正常解析"""
        # 畸形行
        assert parse_sse_event("garbage without prefix") is None
        # 随后的正常行
        event = parse_sse_event('data: {"type":"content","data":"ok"}')
        assert event is not None
        assert event["type"] == "content"

    def test_done_signal_not_content(self):
        """[DONE] 信号不计为 content"""
        event = parse_sse_event("data: [DONE]")
        assert is_content_event(event) is False


class TestSSEFailureModes:
    """SSE 失败判定测试"""

    def test_no_content_before_done(self):
        """没有 content 但有 done → 算失败（无有效内容）"""
        events = [
            'data: {"type":"debug","node":"start"}',
            'data: {"type":"tool_call","data":{}}',
            'data: {"type":"done","data":null}',
        ]
        has_content = any(
            is_content_event(parse_sse_event(line))
            for line in events
        )
        assert has_content is False

    def test_only_error(self):
        """只有 error → 无 content → 失败"""
        events = [
            'data: {"type":"debug","node":"start"}',
            'data: {"type":"error","data":"connection refused"}',
        ]
        has_content = any(
            is_content_event(parse_sse_event(line))
            for line in events
        )
        assert has_content is False

    def test_content_before_error(self):
        """先有 content 后有 error → 有 content → 算有内容"""
        events = [
            'data: {"type":"content","data":"部分回答"}',
            'data: {"type":"error","data":"connection lost"}',
        ]
        has_content = any(
            is_content_event(parse_sse_event(line))
            for line in events
        )
        assert has_content is True
