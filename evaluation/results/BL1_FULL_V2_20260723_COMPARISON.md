# BL-1 与 Full 正式评测对比（2026-07-23）

## 1. 有效性

- 数据集：`evaluation/ragas_50_v2_review.csv`
- 数据集 SHA-256：`5836fbc05daacb0741d234d196938d5c409d2a40f839f730989e96184d02a393`
- 题目：50题、50篇对应文献
- 人工门禁：参考答案、可接受 Gold Chunk、`question_supported` 均为 50/50 通过
- 执行顺序：BL-1 完成后再运行 Full，未并行
- 两套最终结果：50条明细、0执行错误、0空回答、0异常工具调用、0空上下文
- Ragas 完整性：Faithfulness 50/50，Context Recall 50/50

原始两次正式运行都在 `rq035` 的 Faithfulness 判分中遇到一次 Ragas JSON 解析失败，原始结果已原样保留。随后仅使用已保存的回答和上下文定向重评该缺失项，没有重跑检索或回答。重评时只在送入评分器的临时副本中移除 LaTeX 反斜杠，避免评估模型产生非法 JSON 转义；原回答、上下文和原始结果均未修改。两套重评过程分别保存了源文件哈希和逐次尝试记录。

## 2. 核心结果

| 指标 | BL-1 | Full | Full - BL-1 |
|---|---:|---:|---:|
| Acceptable-Chunk Recall@3 | 46.00% | 66.00% | +20.00个百分点 |
| Acceptable-Chunk Recall@5 | 56.00% | 76.00% | +20.00个百分点 |
| Acceptable-Chunk MRR | 0.3797 | 0.4953 | +0.1156 |
| Ragas Faithfulness | 0.8916 | 0.9172 | +0.0256 |
| Ragas Context Recall | 0.7850 | 0.9233 | +0.1383 |

Recall@K 按“前 K 条中是否出现任一人工认可的 Gold Chunk”逐题记 0/1 后取平均。MRR 按第一个可接受 Gold Chunk 的倒数排名计分，未命中为0。

## 3. 辅助检索结果

| 指标 | BL-1 | Full | 差值 |
|---|---:|---:|---:|
| Doc-Hit | 88.00%（44/50） | 98.00%（49/50） | +10.00个百分点 |
| Doc-Hit@1 | 64.00% | 88.00% | +24.00个百分点 |
| Doc-Hit@3 | 84.00% | 94.00% | +10.00个百分点 |
| Doc-Hit@5 | 88.00% | 98.00% | +10.00个百分点 |
| 任一可接受 Chunk 命中 | 28/50 | 39/50 | +11题 |

## 4. 结论边界

在本次同一50题、同一语料和同一 Gold 标注下，Full 相比 BL-1 明显改善了可接受证据在前3条和前5条中的出现率，并将第一个正确证据整体前移；Context Recall 的均值提升也较明显。Faithfulness 均值小幅提升，但 Full 的最低单题 Faithfulness 为0.1429，低于 BL-1 的0.2000，因此不能表述为每一道题或所有生成质量指标都提升。

Full 的最终上下文可能包含 Top-5 基础证据的索引邻居，所以“任一可接受 Chunk 命中”与 Recall@5 不是同一指标。正式对外比较应优先使用上表明确约定的 Recall@3、Recall@5 和 MRR。

## 5. 结果目录

- BL-1 原始正式运行：`evaluation/results/BL1_V2_20260723_FORMAL/`
- BL-1 有效重评结果：`evaluation/results/BL1_V2_20260723_FORMAL_RAGAS_RETRY/`
- Full 原始正式运行：`evaluation/results/FULL_V2_20260723_FORMAL/`
- Full 有效重评结果：`evaluation/results/FULL_V2_20260723_FORMAL_RAGAS_RETRY/`

每个有效重评目录包含 `review_eval_summary.json`、`review_eval_details.csv` 和 `ragas_retry_audit.json`。
