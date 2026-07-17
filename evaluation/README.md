# AVF RAG 评测模块

本目录同时保存旧版论文级检索实验、50题Agent全链路评测和BL-1受控基线。运行或引用指标时必须先确认问题集、检索链路和结果目录，不能跨版本混用。

## 1. 评测分层

| 层级 | 输入 | 用途 | 当前状态 |
|---|---|---|---|
| Legacy | `questions_25.csv` | 25题、93条论文相关性标注的旧版检索与去重消融 | 已完成，保留历史结果 |
| Full | `ragas_50_actual_chunk_review.csv` | 50题、真实Milvus逻辑Chunk ID、Agent同次调用全链路 | 已完成一次正式运行 |
| BL-1 | `ragas_50_actual_chunk_review.csv` | Dense Top-5唯一Chunk受控基线 | 已完成50题正式运行 |

## 2. 50题数据生成与映射

### 2.1 Evidence-first生成

```powershell
python evaluation/generate_ragas_50.py
```

主要输出：

- `ragas_50_dataset.jsonl`：Ragas风格数据；
- `ragas_50_manifest.jsonl`：题目与证据清单；
- `ragas_50_review.csv`：人工审核表；
- `ragas_50_summary.json`：生成摘要。

该阶段不自动生成参考答案。参考证据不等于经过人工确认的医学标准答案。

### 2.2 映射真实Milvus Chunk ID

```powershell
python evaluation/build_actual_chunk_mapping.py
```

正式Full和BL-1必须使用：

```text
evaluation/ragas_50_actual_chunk_review.csv
```

不要使用 `ragas_50_review.csv` 或 `ragas_50_dataset.jsonl` 中的原始证据ID直接计算Milvus Chunk命中。

逻辑ID格式：

```text
{document_id}:{sha256(chunk_content)[:16]}
```

当前映射结果：50/50文档重建Chunk哈希集合与Milvus完全一致，50题各映射1个严格目标Chunk；评测涉及文档中统计到467条重复Milvus UUID行。

## 3. Full全链路评测

先运行3题冒烟测试：

```powershell
python evaluation/evaluate_review.py --limit 3
```

再运行完整50题：

```powershell
python evaluation/evaluate_review.py
```

Full通过 `RagAgentService.query_with_trace()` 获取同一次Agent调用实际使用的检索Artifact，避免在回答前后额外检索导致上下文不一致。

最新正式结果：

```text
evaluation/results/20260717_045223/
```

有效ID指标：

| 指标 | 结果 |
|---|---:|
| Doc-Hit | 49/50 = 98.0% |
| Doc-Hit@1 | 82.0% |
| Doc-Hit@3 | 94.0% |
| Doc-Hit@5 | 98.0% |
| Doc-Hit@10 | 98.0% |
| Doc平均排名 | 1.3 |
| Strict-Chunk-Hit | 18/50 |

`Strict-Chunk-Hit`只检查唯一指定目标Chunk。32道严格Chunk未命中中有31道命中了正确文献中的其他Chunk，因此不能将18/50解释为答案正确率。

## 4. BL-1基线

BL-1定义：

```text
Dense检索
→ 删除完全重复Chunk
→ 取前5个唯一Chunk
→ 与Full相同的ContextBuilder、Agent模型、系统提示词和Ragas评判
```

BL-1不使用：

- LLM Rerank；
- 0.65阈值；
- 来源多样性；
- 相邻Chunk扩展；
- Query Rewrite；
- Multi-query。

冒烟测试：

```powershell
python evaluation/evaluate_bl1.py --limit 3
```

完整50题：

```powershell
python evaluation/evaluate_bl1.py
```

BL-1结果目录：

```text
evaluation/results/BL1_20260717_045844/   # 3题冒烟
evaluation/results/BL1_20260717_052107/   # 50题正式结果
```

50题结果：Doc-Hit 94%、Doc-Hit@1 60%、Doc-Hit@3 88%、Doc-Hit@5 94%、平均排名1.6、Strict-Chunk-Hit 56%。

与Full对比：Full的Doc-Hit、Doc-Hit@1、Doc-Hit@3、Doc-Hit@5分别提升4、22、6、4个百分点，平均排名由1.6改善为1.3；但Strict-Chunk-Hit从56%下降到36%。因此当前只能报告“目标文献更容易进入前列”，不能报告“所有检索和回答指标全面提升”。

## 5. Ragas指标边界

当前接入：

- Faithfulness；
- Answer Relevancy。

尚未启用或不具备可靠前提：

- Context Recall：需要人工审核的参考答案或声明；
- Answer Correctness：需要人工审核的参考答案。

已知问题：当前Full和BL-1的50题汇总仅排除 `None`，没有排除 `NaN`，因此Faithfulness和Answer Relevancy均值不可直接引用。现有详情还只保存回答和上下文预览，无法从截断数据无损重算。修复后应保存每题完整回答、完整检索上下文和每题Ragas分数。

## 6. Legacy评测

旧版脚本仍可使用 `questions_25.csv`：

```powershell
python evaluation/evaluate_retrieval.py --questions evaluation/questions_25.csv
python evaluation/evaluate_deduplication.py --questions evaluation/questions_25.csv
python evaluation/evaluate_generation.py --questions evaluation/questions_25.csv
```

旧版结果保存在2026-07-15对应的run目录。它评估的是论文级标签和旧检索策略，不代表当前Agent全链路。

## 7. 输出契约

Full与BL-1均输出：

```text
evaluation/results/{run_id}/
├── review_eval_summary.json
└── review_eval_details.csv
```

核心指标字段保持一致：

```text
question_count
id_based_metrics.Doc-Hit
id_based_metrics.Doc-Hit@1
id_based_metrics.Doc-Hit@3
id_based_metrics.Doc-Hit@5
id_based_metrics.Doc-Hit@10
id_based_metrics.Doc_mean_rank
id_based_metrics.Chunk-Hit
id_based_metrics.Chunk_mean_rank
ragas_metrics.faithfulness
ragas_metrics.answer_relevancy
```

## 8. 运行安全

- `generate_ragas_50.py`主要读取本地文献并生成评测文件。
- Chunk映射会只读查询Milvus。
- Full和BL-1会调用Milvus、生成模型和Ragas评判模型，可能产生费用和限流。
- Full与BL-1不应并行运行，否则延迟、限流和结果可比性会受影响。
- 调试检索且不希望产生Ragas费用时，可使用 `--skip-ragas`；正式比较不得使用该参数。
- 评测不得修改 `.env`、`uploads/`、`volumes/` 或重建 `biz` collection。

## 9. 正式对比前验收

Full和BL-1都必须满足：

- `question_count = 50`；
- 50个唯一问题ID；
- 无执行错误；
- 无空答案；
- 使用同一问题文件和同一目标Chunk ID；
- 使用同一Milvus collection、Embedding、生成模型和Ragas评判配置；
- Ragas均值只聚合有限数值；
- 保存足以复核的完整Trace。

当前两组50题的ID指标已经满足同题对比条件，可以报告文档级绝对提升；回答层提升仍需先修复NaN和Trace持久化。新增多可接受Chunk标注前，Strict-Chunk-Hit只能作为严格单金标准指标单独解释。
