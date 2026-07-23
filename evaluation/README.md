# AVF RAG 评测说明

本目录同时保留当前50题正式评测、检索消融和Legacy实验。引用指标前必须确认数据集、索引版本、检索链路和结果目录，禁止跨版本混用。

## 1. 当前正式评测

正式输入：

```text
evaluation/ragas_50_v2_review.csv
```

数据集包含50题、50篇对应文献。参考答案、可接受Gold Chunk和`question_supported`均已人工复核通过。每题可以有多个可接受Chunk，避免唯一摘要Chunk被更直接证据替代时产生误判。

### Full

Full使用Agent同一次调用实际看到的检索Artifact，生产链路为：

```text
Dense Top-20
→ 精确Chunk去重
→ qwen3-rerank Top-10
→ 0.65阈值
→ Top-5基础证据
→ chunk_index邻接扩展
→ Agent回答与Ragas评判
```

先运行3题冒烟测试：

```powershell
python evaluation/evaluate_review.py --limit 3
```

正式运行并指定独立结果目录：

```powershell
python evaluation/evaluate_review.py --output evaluation/results/{run_name}
```

当前正式结果位于`results/FULL_QWEN3_V2_20260723_FORMAL/`：

| 指标 | 结果 |
|---|---:|
| 有效题目 | 50/50 |
| Recall@3 | 88% |
| Recall@5 | 88% |
| MRR | 0.8067 |
| Doc-Hit@5 | 100% |
| Faithfulness | 0.9382 |
| Context Recall | 0.9650 |
| Rerank降级 | 0次 |

### BL-1

BL-1只执行Dense检索、精确去重并保留Top-5唯一Chunk；不使用Rerank、阈值、邻接扩展、Query Rewrite、Multi-query或混合检索。

```powershell
python evaluation/evaluate_bl1.py --limit 3
python evaluation/evaluate_bl1.py --output evaluation/results/{run_name}
```

当前有效结果位于`results/BL1_V2_20260723_FORMAL_RAGAS_RETRY/`：

| 指标 | 结果 |
|---|---:|
| 有效题目 | 50/50 |
| Recall@3 | 46% |
| Recall@5 | 56% |
| MRR | 0.3797 |
| Doc-Hit@5 | 88% |
| Faithfulness | 0.8916 |
| Context Recall | 0.7850 |

当前Full相对BL-1的Recall@3、Recall@5和MRR分别提升42个百分点、32个百分点和0.4270。该结论只适用于上述同一v2数据集和当前索引。

## 2. 指标定义

- `Recall@3`：前3条中出现任一人工审核的可接受Chunk，该题记1，否则记0，再对50题取平均。
- `Recall@5`：前5条中出现任一人工审核的可接受Chunk，该题记1，否则记0，再对50题取平均。
- `MRR`：每题按`1 / 第一个可接受Chunk的名次`计分，未命中记0，再取平均。
- `Doc-Hit@K`：目标文献是否出现在前K条；它不等于Chunk级Recall。
- `Faithfulness`：Ragas判断回答中的可验证主张是否得到实际检索上下文支持。
- `Context Recall`：Ragas判断人工审核参考答案中的主张是否能由实际检索上下文覆盖。

`Strict-Chunk-Hit`是旧版唯一指定Chunk指标，不等于答案正确率，也不应替代多Gold Chunk的Recall@K。

## 3. 正式运行门禁

正式结果必须同时满足：

- 使用`ragas_50_v2_review.csv`完整50题；
- 数据集人工审核门禁通过；
- 50个唯一问题ID，且明细为50条；
- 无执行错误、空回答、非法工具调用或空上下文；
- Full和BL-1使用同一Milvus collection、Embedding、生成模型和Ragas配置；
- Ragas均值只聚合有限数值；
- 保存完整回答、检索上下文、ID指标和运行配置；
- Full与BL-1串行运行，不得并行竞争外部模型资源。

3题`--limit 3`仅用于冒烟测试，不能作为正式基线或对外结果。

## 4. 缺失Ragas评分重试

只有检索和回答已经成功、但单个Ragas字段因解析失败而缺失时，才可使用`retry_missing_ragas.py`。重试必须：

- 复用已保存的完整回答和上下文，不重跑检索或回答；
- 只补缺失字段，不覆盖已有有效分数；
- 保存源结果哈希、逐次尝试和规范化说明；
- 输出到新的run-specific目录，保留原始结果。

BL-1的`rq035`曾按此规则补评；当前Full结果50题两个Ragas指标均完整，无需重试。

## 5. 结果与归档

```text
evaluation/
├── ragas_50_v2_review.csv       # 当前正式人工审核输入
├── evaluate_review.py           # Full
├── evaluate_bl1.py              # BL-1
├── retry_missing_ragas.py       # 仅补缺失Ragas字段
├── results/                     # 原始结果、消融、对比和审计
└── archive/                     # 过时评测说明，禁止作为当前入口
```

必须保留所有原始JSON/CSV、run-specific结果和失败样本。Legacy 25题论文级实验及旧版50题结果只用于追溯，不代表当前生产链路；说明见`archive/README.md`。

## 6. 成本与运行安全

Full和BL-1正式评测会调用Milvus、生成模型和Ragas评判模型，50题会产生多次外部模型调用、费用和限流风险。开始前必须说明题目数、调用类型、预计成本以及是否有其他评测进程。

文档整理、静态检查和结果复核不得顺带启动评测。评测也不得修改`.env`、`uploads/`、`volumes/`或重建Milvus `biz` collection。
