# Ragas 50 题安全评测集说明

> 当前状态（2026-07-17）：50题Evidence-first数据、50个真实Milvus逻辑Chunk映射、Full 50题和BL-1 50题均已完成。文档级ID指标可公平比较，但两组Ragas汇总均值因未过滤NaN而不可引用。Full的Strict-Chunk-Hit低于BL-1，后续需要增加人工审核的多可接受Chunk标注。

正式评测输入：

```text
evaluation/ragas_50_actual_chunk_review.csv
```

不要使用原始 `evidence_context_id` 直接计算Milvus命中。真实逻辑ID格式为 `{document_id}:{sha256(chunk_content)[:16]}`。

## 1. 目标

本评测集用于验证科研文献 RAG/Agent 的检索、排序、证据利用和回答忠实度。

本次采用“证据先行、答案后审”的生成方式：

1. 先冻结一个可追踪的参考证据块。
2. 仅根据该证据块编写一个单跳问题。
3. 自动检查证据块是否包含人工指定的关键证据词。
4. 暂不自动生成 `reference`，避免未经审核的答案成为错误真值。
5. 人工审核通过后，再补充参考答案或原子主张。

## 2. 已生成文件

| 文件 | 用途 |
|---|---|
| `evaluation/ragas_50_dataset.jsonl` | Ragas 单轮评测核心数据，共 50 条 |
| `evaluation/ragas_50_manifest.jsonl` | 题号、类别、来源、证据哈希和审核状态 |
| `evaluation/ragas_50_review.csv` | 人工逐题审核表 |
| `evaluation/ragas_50_summary.json` | 数量、类别分布和自动校验结果 |
| `evaluation/generate_ragas_50.py` | 可重复生成并验证上述文件 |
| `evaluation/ragas_50_actual_chunks.jsonl` | 已映射到当前 Milvus 实际 chunk 的正式评测数据 |
| `evaluation/ragas_50_actual_chunk_mapping.jsonl` | 参考证据 ID、逻辑 chunk ID 与 Milvus UUID 主键映射 |
| `evaluation/ragas_50_actual_chunk_review.csv` | 实际 chunk 映射人工复核表 |
| `evaluation/build_actual_chunk_mapping.py` | 从当前 Milvus 重建和校验实际 chunk 映射 |

## 3. 数据字段

`ragas_50_dataset.jsonl` 的每条记录包含：

```json
{
  "user_input": "问题",
  "response": null,
  "retrieved_contexts": [],
  "retrieved_context_ids": [],
  "reference": null,
  "reference_contexts": ["经截取和清洗的参考证据块"],
  "reference_context_ids": ["doc_xxxxxx:证据哈希前16位"]
}
```

字段使用方式：

- `user_input`：当前评测问题。
- `response`：运行 Agent 后写入最终回答。
- `retrieved_contexts`：按实际排序写入 Agent 召回的 chunk 文本。
- `retrieved_context_ids`：按实际排序写入召回的稳定 chunk ID。
- `reference_contexts`：人工审核前的候选金标准证据块。
- `reference_context_ids`：候选金标准证据 ID。
- `reference`：当前有意保持为空，人工确认事实后再填写。

Ragas 当前的 `SingleTurnSample` 原生支持上述文本上下文、上下文 ID、回答和参考答案字段。官方 schema 见：

- <https://docs.ragas.io/en/stable/references/evaluation_schema/>

## 4. 为什么不自动生成参考答案

科研文献容易出现 OCR 错误、表格错位、符号损坏、结论条件丢失和模型过度归纳。如果直接使用模型生成答案并写入 `reference`，评测可能稳定地奖励错误答案。

因此，本数据集将事实真值拆成两层：

1. `reference_contexts`：证据层真值，可追踪、可哈希、可复核。
2. `reference`：答案层真值，必须经过人工审核。

在答案层审核完成前，不应运行依赖参考答案正确性的指标。

## 5. 自动安全检查

生成脚本会执行以下检查，任一失败即停止生成：

- 题目总数必须为 50。
- 50 个问题必须完全唯一。
- 50 个来源文档必须唯一。
- 每个来源目录必须只有一个 Markdown。
- 每个证据块长度不得低于 180 字符。
- 每个证据块必须包含该题预先指定的关键证据词。
- 科研表达中的 `P<0.001`、角度 `<30°` 等内容不得被误当作 HTML 删除。
- 每个证据块生成 SHA-256，用于发现语料或切块变化。

当前结果：

- 问题数：50
- 唯一来源数：50
- 自动证据检查通过：50/50
- 自动生成参考答案：0
- 待人工审核：50

## 6. 人工复核流程

打开 `evaluation/ragas_50_review.csv`，逐题检查：

1. `question_supported`：证据是否能直接、完整地回答问题。
2. `evidence_clean`：证据是否存在严重 OCR、表格错位或语义截断。
3. `reference_answer_reviewed`：参考答案是否已经由人工依据证据填写并复核。
4. `accept`：该题是否进入正式基准集。
5. `reviewer_notes`：记录修改原因、歧义或适用条件。

建议至少由一名领域人员审核全部 50 题；涉及临床结论、数值和比较排序的题目建议双人复核。

### 接受标准

一题只有同时满足下列条件才能标记为接受：

- 问题不依赖证据块之外的信息。
- 证据中存在明确答案，而不是仅说明“本文研究了该问题”。
- 问题没有把相关性误写成因果关系。
- 数值、单位、时间和样本量均能在证据中定位。
- 问题没有要求文献未给出的确定性排序。
- 证据块不是整篇文档，也不是只包含标题的片段。

## 7. 与运行时 chunk 对齐

原始证据集中的 `reference_context_ids` 是由文档 ID 和参考证据文本哈希组成的证据 ID，
不能直接与 Milvus chunk 比较。正式评测应使用
`evaluation/ragas_50_actual_chunks.jsonl`，其中逻辑 chunk ID 为：

```text
document_id + ":" + sha256(chunk_text)[:16]
```

完整映射保存在：

```text
evidence_context_id
    -> actual_chunk_id
    -> Milvus UUID primary key
```

不要退化为文件级命中。一个文件被召回不代表回答所需的具体 chunk 已被召回。

## 8. 推荐评测顺序

### 阶段 A：参考答案审核前

优先使用：

1. ID-based Context Precision
2. ID-based Context Recall
3. Context Utilization
4. Faithfulness
5. Response Relevancy

Ragas 的 Context Precision 关注相关 chunk 是否排在前面；官方也提供直接比较上下文 ID 的 Precision。Context Utilization 可在没有参考答案时，用生成回答判断召回上下文的利用情况：

- <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/>

Faithfulness 比较 `response` 与 `retrieved_contexts`，检查回答中的主张是否受到召回证据支持：

- <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/>

### 阶段 B：参考答案或原子主张审核后

再加入：

1. LLM-based Context Recall
2. Answer Correctness / Factual Correctness
3. 按原子主张计算的 Claim Recall

Ragas 的 LLM-based Context Recall 会把 `reference` 拆成主张，再检查这些主张是否被召回上下文覆盖，因此必须先保证参考答案准确：

- <https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_recall/>

## 9. 运行时数据回填

对每一道题调用一次 Agent，并将结果写回：

```text
response               <- Agent 最终回答
retrieved_contexts     <- rerank 后实际交给模型的 chunk 文本，保持排序
retrieved_context_ids  <- 与上述文本一一对应的 chunk ID
```

不要把以下内容写入 `retrieved_contexts`：

- 调试数字、流式事件编号或 SSE 元数据。
- 工具调用参数。
- 未实际提供给生成模型的超额召回候选。
- 文档标题列表或文件级摘要。

如需单独评估“超额召回”和“rerank”，应保存两套字段：

```text
recall_candidate_ids   <- 初始超额召回结果
retrieved_context_ids  <- rerank 后最终上下文
```

然后分别计算候选集 Recall@K、最终 Precision@K、MRR、nDCG@K 和答案层指标。

## 10. 版本管理

以下任一变化都应生成新的评测集版本，不应覆盖旧哈希：

- 文献重新解析。
- Markdown 清洗规则变化。
- chunk 最大长度或 overlap 变化。
- 表格、公式或参考文献过滤策略变化。
- 问题文本或参考证据变化。
- 人工参考答案发生修改。

建议版本号：

```text
ragas_avf_50_v1_evidence_only
ragas_avf_50_v2_reviewed_reference
```
