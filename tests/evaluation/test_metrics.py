"""测试 common.py 中的指标计算函数（纯计算，不依赖外部服务）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.common import (
    compute_binary_recall_at_k,
    compute_hit_at_k,
    compute_reciprocal_rank,
    compute_recall_at_k,
    compute_source_coverage,
    compute_duplicate_ratio,
    compute_percentile,
)


class TestBinaryRecallAndMRR:
    def test_binary_recall_only_checks_whether_any_answer_is_in_top_k(self):
        retrieved = ["wrong-1", "wrong-2", "answer-b", "answer-a", "wrong-3"]
        acceptable = ["answer-a", "answer-b"]

        assert compute_binary_recall_at_k(retrieved, acceptable, k=3) == 1
        assert compute_binary_recall_at_k(retrieved, ["answer-a"], k=3) == 0
        assert compute_binary_recall_at_k(retrieved, ["answer-a"], k=5) == 1

    def test_reciprocal_rank_uses_first_correct_result(self):
        retrieved = ["wrong", "answer-b", "answer-a"]

        assert compute_reciprocal_rank(
            retrieved, ["answer-a", "answer-b"]
        ) == 0.5

    def test_reciprocal_rank_is_zero_when_not_found(self):
        assert compute_reciprocal_rank(["wrong"], ["answer"]) == 0.0


# ============================================================
# Hit@K
# ============================================================

class TestHitAtK:
    """Hit@K 计算测试"""

    def test_hit_top1(self):
        """第 1 位命中"""
        assert compute_hit_at_k(["A", "B", "C", "D", "E"], ["A"], k=1) == 1

    def test_hit_top3(self):
        """第 3 位命中"""
        assert compute_hit_at_k(["A", "B", "C", "D", "E"], ["C"], k=3) == 1

    def test_hit_top5_not_in_top3(self):
        """在 Top-5 但不在 Top-3"""
        assert compute_hit_at_k(["A", "B", "C", "D", "E"], ["E"], k=3) == 0

    def test_miss(self):
        """完全未命中"""
        assert compute_hit_at_k(["A", "B", "C", "D", "E"], ["X"], k=5) == 0

    def test_multiple_relevant_one_in_top_k(self):
        """多个相关论文，其中一个在 Top-K"""
        assert compute_hit_at_k(["A", "B", "C"], ["A", "X", "Y"], k=3) == 1

    def test_empty_relevant(self):
        """空标准答案 → 0"""
        assert compute_hit_at_k(["A", "B"], [], k=3) == 0

    def test_empty_retrieved(self):
        """空检索结果 → 0"""
        assert compute_hit_at_k([], ["A"], k=3) == 0

    def test_hit_at_k_greater_than_retrieved(self):
        """K 大于检索结果数"""
        assert compute_hit_at_k(["A", "B"], ["B"], k=10) == 1

    def test_duplicate_sources_in_retrieved(self):
        """检索结果有重复来源，命中只计一次"""
        # 如果 A 出现在第 1 位和第 3 位
        retrieved = ["A", "B", "A", "C", "D"]
        assert compute_hit_at_k(retrieved, ["A"], k=5) == 1


# ============================================================
# Recall@K
# ============================================================

class TestRecallAtK:
    """Recall@K 计算测试"""

    def test_full_recall(self):
        """全部召回"""
        assert compute_recall_at_k(["A", "B", "C"], ["A", "B"], k=3) == 1.0

    def test_partial_recall(self):
        """部分召回"""
        assert compute_recall_at_k(["A", "B", "C"], ["A", "X", "Y"], k=3) == 1.0 / 3.0

    def test_zero_recall(self):
        """零召回"""
        assert compute_recall_at_k(["A", "B", "C"], ["X", "Y"], k=3) == 0.0

    def test_recall_at_smaller_k(self):
        """K 较小，只取前几个结果"""
        result = compute_recall_at_k(["A", "B", "C", "D"], ["B", "D"], k=2)
        assert result == 0.5  # 只命中 B

    def test_empty_relevant(self):
        """空标准答案 → 0"""
        assert compute_recall_at_k(["A", "B"], [], k=3) == 0.0

    def test_empty_retrieved(self):
        """空检索结果 → 0"""
        assert compute_recall_at_k([], ["A", "B"], k=3) == 0.0

    def test_duplicate_in_retrieved(self):
        """检索结果有重复，去重计算"""
        retrieved = ["A", "A", "B", "C"]
        # Top-3 中不同来源: A, B
        # 相关: A, X → 命中 1/2
        result = compute_recall_at_k(retrieved, ["A", "X"], k=3)
        assert result == 0.5

    def test_duplicate_in_relevant(self):
        """标准答案中有重复，自动去重"""
        retrieved = ["A", "B", "C"]
        result = compute_recall_at_k(retrieved, ["A", "A", "B"], k=3)
        # relevant 有 2 个唯一值: A, B。Top-3 命中 A, B
        assert result == 1.0


# ============================================================
# Source Coverage
# ============================================================

class TestSourceCoverage:
    """来源覆盖数测试"""

    def test_all_unique(self):
        assert compute_source_coverage(["A", "B", "C", "D", "E"], k=5) == 5

    def test_with_duplicates(self):
        assert compute_source_coverage(["A", "A", "B", "C", "C"], k=5) == 3

    def test_k_smaller_than_data(self):
        assert compute_source_coverage(["A", "B", "C", "D", "E"], k=3) == 3

    def test_empty(self):
        assert compute_source_coverage([], k=5) == 0

    def test_k_greater_than_data(self):
        assert compute_source_coverage(["A", "B"], k=10) == 2


# ============================================================
# Duplicate Ratio
# ============================================================

class TestDuplicateRatio:
    """重复来源占比测试"""

    def test_no_duplicates(self):
        assert compute_duplicate_ratio(["A", "B", "C", "D", "E"], k=5) == 0.0

    def test_all_same(self):
        assert compute_duplicate_ratio(["A", "A", "A", "A", "A"], k=5) == 0.8  # 1 - 1/5

    def test_mixed(self):
        # 3 个不同来源 / 5 个结果 → 重复占比 = 1 - 3/5 = 0.4
        result = compute_duplicate_ratio(["A", "A", "B", "C", "C"], k=5)
        assert result == 0.4

    def test_k_smaller(self):
        result = compute_duplicate_ratio(["A", "B", "C", "D", "E"], k=3)
        assert result == 0.0

    def test_empty(self):
        assert compute_duplicate_ratio([], k=5) is None


# ============================================================
# Percentile
# ============================================================

class TestPercentile:
    """百分位数计算测试"""

    def test_p50_odd(self):
        """奇数个元素 P50（中位数）"""
        assert compute_percentile([1.0, 2.0, 3.0, 4.0, 5.0], 0.50) == 3.0

    def test_p50_even(self):
        """偶数个元素 P50"""
        result = compute_percentile([1.0, 2.0, 3.0, 4.0], 0.50)
        assert result == 2.5

    def test_p95(self):
        """P95 计算"""
        data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        result = compute_percentile(data, 0.95)
        # position = 9 * 0.95 = 8.55
        # sorted[8] + 0.55 * (sorted[9] - sorted[8])
        # = 9.0 + 0.55 * (10.0 - 9.0) = 9.55
        assert abs(result - 9.55) < 0.01

    def test_empty(self):
        assert compute_percentile([], 0.50) is None

    def test_single_element(self):
        assert compute_percentile([42.0], 0.50) == 42.0
        assert compute_percentile([42.0], 0.95) == 42.0

    def test_p0(self):
        assert compute_percentile([5.0, 1.0, 3.0], 0.0) == 1.0

    def test_p100(self):
        assert compute_percentile([5.0, 1.0, 3.0], 1.0) == 5.0

    def test_unsorted_input(self):
        """未排序的输入也能正确计算"""
        result = compute_percentile([5.0, 1.0, 4.0, 2.0, 3.0], 0.50)
        assert result == 3.0


# ============================================================
# 一致性测试
# ============================================================

class TestConsistency:
    """指标计算一致性测试（使用固定明细验证）"""

    FIXED_RETRIEVED = ["A", "A", "B", "C", "D", "E", "E", "F", "G", "H"]
    FIXED_RELEVANT = ["A", "B", "X", "Y"]  # 4 篇相关论文

    def test_hit_consistency(self):
        """Hit@K 一致性"""
        # Top-1: A 命中 → 1
        assert compute_hit_at_k(self.FIXED_RETRIEVED, self.FIXED_RELEVANT, k=1) == 1
        # Top-3: A, A, B → 去重后有 A, B 命中 → 1
        assert compute_hit_at_k(self.FIXED_RETRIEVED, self.FIXED_RELEVANT, k=3) == 1
        # Top-10: 全部命中 A, B → 1
        assert compute_hit_at_k(self.FIXED_RETRIEVED, self.FIXED_RELEVANT, k=10) == 1

    def test_recall_consistency(self):
        """Recall@K 一致性"""
        # Top-1: 命中 A → 1/4
        assert compute_recall_at_k(self.FIXED_RETRIEVED, self.FIXED_RELEVANT, k=1) == 0.25
        # Top-3: 去重后 A, B → 2/4
        assert compute_recall_at_k(self.FIXED_RETRIEVED, self.FIXED_RELEVANT, k=3) == 0.5
        # Top-10: 去重后 A, B, C, D, E, F, G, H → 命中 A, B → 2/4
        assert compute_recall_at_k(self.FIXED_RETRIEVED, self.FIXED_RELEVANT, k=10) == 0.5

    def test_coverage_consistency(self):
        """来源覆盖一致性"""
        # Top-5: "A","A","B","C","D"，不同来源为 A、B、C、D
        assert compute_source_coverage(self.FIXED_RETRIEVED, k=5) == 4

    def test_duplicate_ratio_consistency(self):
        """重复占比一致性"""
        # Top-5: "A","A","B","C","D" → 4 个不同 / 5 = 0.8 → ratio = 0.2
        result = compute_duplicate_ratio(self.FIXED_RETRIEVED, k=5)
        assert result == pytest.approx(0.2)
