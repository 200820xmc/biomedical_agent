# AVF RAG 评测结果索引

> 当前结论更新时间：2026-07-17
> 项目输出仅用于科研、教学和算法验证，不构成临床诊断意见。

本文区分两代评测：2026-07-15的Legacy论文级检索实验，以及2026-07-17的50题Agent全链路评测。两者问题集、检索链路和指标含义不同，不得直接拼接或当作同一实验的前后结果。

## 1. 当前可用结论

### 1.1 Full 50题全链路

结果目录：

```text
evaluation/results/20260717_045223/
```

输入：

```text
evaluation/ragas_50_actual_chunk_review.csv
```

运行事实：

| 项目 | 结果 |
|---|---:|
| 题目数 | 50 |
| 唯一问题ID | 50 |
| 执行错误 | 0 |
| 调用检索工具 | 50/50 |
| 空答案 | 0 |
| 进入Ragas队列 | 49/50 |

ID指标：

| 指标 | 结果 |
|---|---:|
| Doc-Hit | 49/50 = 98.0% |
| Doc-Hit@1 | 0.82 |
| Doc-Hit@3 | 0.94 |
| Doc-Hit@5 | 0.98 |
| Doc-Hit@10 | 0.98 |
| Doc平均排名 | 1.3 |
| Strict-Chunk-Hit | 18/50 |
| Strict-Chunk平均排名 | 1.5 |

解释：Strict-Chunk-Hit只检查每题唯一指定目标Chunk。32道严格Chunk未命中问题中，31道仍命中了目标论文中的其他Chunk。因此18/50不能解释成系统只有36%有效，也不能替代答案正确率。

### 1.2 真实Chunk映射质量

结果：`evaluation/ragas_50_actual_chunk_summary.json`

| 指标 | 结果 |
|---|---:|
| 问题数 | 50 |
| 映射完成 | 50/50 |
| 自动映射检查通过 | 50/50 |
| 文档重建哈希集合与Milvus完全一致 | 50/50 |
| 每题严格目标Chunk | 1 |
| 评测文档重复Milvus UUID行 | 467 |

逻辑Chunk ID：

```text
{document_id}:{sha256(chunk_content)[:16]}
```

## 2. 当前不可对外使用的结论

### 2.1 Ragas均值

Full结果中Faithfulness和Answer Relevancy各有49条进入聚合，但当前实现只排除了 `None`，没有排除 `NaN`，导致均值为NaN。

现有详情还只持久化：

- 回答前500字符；
- 上下文前180字符预览；
- 未保存每题Ragas完整得分。

因此不能从当前截断结果无损重算均值。修复并重新评测前，不得在README、简历或报告中引用Full 50题的Faithfulness和Answer Relevancy平均分。

### 2.2 Full相对BL-1提升

BL-1正式50题目录：

```text
evaluation/results/BL1_20260717_052107/
```

两组均使用相同50题、真实逻辑Chunk标注、Milvus collection、Embedding、生成模型、系统提示词和Ragas评判配置。ID指标对比：

| 指标 | BL-1 | Full | 绝对变化 |
|---|---:|---:|---:|
| Doc-Hit | 94% | 98% | +4个百分点 |
| Doc-Hit@1 | 60% | 82% | +22个百分点 |
| Doc-Hit@3 | 88% | 94% | +6个百分点 |
| Doc-Hit@5 | 94% | 98% | +4个百分点 |
| Doc-Hit@10 | 94% | 98% | +4个百分点 |
| Doc平均排名 | 1.6 | 1.3 | 改善0.3 |
| Strict-Chunk-Hit | 56% | 36% | -20个百分点 |
| Strict-Chunk平均排名 | 2.6 | 1.5 | 命中样本中改善1.1 |

结论：Full提高了目标文献进入结果和排在前列的概率，尤其Doc-Hit@1提升22个百分点；但唯一严格目标Chunk命中下降20个百分点。这说明Rerank、多样性和预算控制可能选中了正确论文中的其他证据，或把单一Gold Chunk排除。当前可以报告文档级提升，但不能报告“所有检索指标全面提升”。

Full和BL-1的Ragas均值都存在NaN，回答层提升仍不可用。

## 3. Full失败样本

`rq007`：

```text
问题：定量血管声学分析能够估计哪些动脉血流和几何参数？
Recall最高分：0.5555
Rerank最高分：0.6000
阈值：0.65
过滤结果：12 → 0
```

该题最终没有检索上下文，是当前“固定0.65阈值无保底”问题的真实证据。回答没有伪造文献结论，但该样本不应被描述为成功检索。

## 4. Legacy：2026-07-15论文级检索实验

适用范围：旧版25题、93条相关论文标签，检索直接调用向量存储，不经过当前Agent全链路。

历史结果目录：

```text
evaluation/results/20260715_184942/
evaluation/results/20260715_185026/
evaluation/results/20260715_185054/
```

当时知识库快照：

| 指标 | 结果 |
|---|---:|
| Milvus实体 | 308 |
| 去重来源 | 30 |
| 平均Chunk数/来源 | 10.3 |
| 平均Chunk长度 | 4473.9字符 |

旧版原始向量检索：

| 指标 | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|
| Hit@K | 52.0% | 72.0% | 76.0% | 84.0% |
| Recall@K | 16.9% | 34.9% | 42.5% | 62.1% |

旧版论文去重消融：

| 指标 | 基线 | 优化 |
|---|---:|---:|
| 平均来源覆盖数 | 3.00 | 4.76 |
| 重复来源占比 | 40.0% | 4.8% |
| Hit@5 | 76.0% | 88.0% |
| Recall@5 | 42.5% | 62.1% |

这些数据可以作为历史消融实验引用，但必须注明“25题Legacy检索实验”，不能写成最新50题Full结果。

## 5. 当前简历可用指标

可以使用：

```text
构建50题、50篇独立科研文献的真实Chunk ID全链路评测集；
50题执行成功率100%、工具调用率100%、空答案率0%，
目标文献命中率49/50（98%），Doc-Hit@1/@3/@5分别为82%/94%/98%，
目标文献平均排名1.3。
```

也可以使用同题基线对比：

```text
与Dense Top-5基线相比，两阶段全链路在相同50题上将Doc-Hit@1
由60%提升至82%（+22个百分点），Doc-Hit@5由94%提升至98%，
目标文献平均排名由1.6改善至1.3。
```

面试时必须同时说明：Strict-Chunk-Hit由56%下降至36%，当前单题只标注一个严格Gold Chunk，后续需要增加人工审核的可接受Chunk集合。

可以作为Legacy消融实验单独使用：

```text
在25题、93条相关论文标注的检索消融实验中，
平均来源覆盖由3.00提升至4.76，重复来源占比由40.0%降至4.8%，
Hit@5由76.0%提升至88.0%，Recall@5由42.5%提升至62.1%。
```

暂时不能使用：

- Full 50题Faithfulness均值；
- Full 50题Answer Relevancy均值；
- Full相对BL-1的回答层提升；
- “Full所有指标全面优于BL-1”；
- 将Strict-Chunk-Hit 18/50描述为答案正确率。

## 6. 下一次回答层对比要求

1. 修复Ragas聚合：只聚合 `math.isfinite()` 的得分；
2. 持久化完整回答、完整上下文和每题Ragas得分；
3. 为每题人工审核多个可独立支持答案的 `acceptable_chunk_ids`；
4. 保留Strict-Chunk-Hit，同时新增可接受Chunk命中；
5. 修复后重新运行Full与BL-1相同50题；
6. 再计算Faithfulness、Answer Relevancy和可接受Chunk指标的提升。

## 7. 结果文件索引

| 文件 | 说明 |
|---|---|
| `20260717_045223/review_eval_summary.json` | Full 50题汇总 |
| `20260717_045223/review_eval_details.csv` | Full 50题明细 |
| `BL1_20260717_045844/review_eval_summary.json` | BL-1 3题冒烟汇总 |
| `BL1_20260717_045844/review_eval_details.csv` | BL-1 3题冒烟明细 |
| `BL1_20260717_052107/review_eval_summary.json` | BL-1 50题正式汇总 |
| `BL1_20260717_052107/review_eval_details.csv` | BL-1 50题正式明细 |
| `../ragas_50_actual_chunk_summary.json` | 真实Chunk映射摘要 |
| `20260715_185026/retrieval_summary.json` | Legacy检索结果 |
| `20260715_185054/deduplication_summary.json` | Legacy去重消融结果 |
