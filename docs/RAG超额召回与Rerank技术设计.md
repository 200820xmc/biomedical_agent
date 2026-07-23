# RAG 检索与 Rerank 技术设计

> 状态：当前实现
> 更新时间：2026-07-23

## 1. 目标

检索链路优先保证正确证据进入前列，并在外部Reranker不可用时继续返回非空结果。当前不以扩大文件来源覆盖为目标，也不执行来源多样性限制。

## 2. 当前参数

| 参数 | 值 |
|---|---:|
| Dense候选 | 20 |
| Rerank模型 | `qwen3-rerank` |
| Rerank保留 | 10 |
| Rerank阈值 | 0.65 |
| 阈值全失败保底 | Top-3 |
| 基础证据 | Top-5 |
| 邻接扩展窗口 | `chunk_index ± 1` |
| 邻接锚点 | 排名前3条基础证据 |
| 上下文预算 | 12,000字符 |

## 3. 检索流程

```text
Agent query
→ Dense Top-20
→ 精确去重
→ qwen3-rerank Top-10
→ 0.65阈值
→ Top-5
→ 前3个锚点的索引邻居
→ ContextBuilder
```

生产工具只接收Agent生成的单路query。原问题+Agent query双路RRF仅作为消融实验保留，没有进入在线链路。

## 4. 精确去重

按以下优先级识别同一逻辑Chunk：

```text
chunk_id
→ content_hash
→ 完全相同正文
```

不按文件名或来源限制同一论文的候选数量。

## 5. 专用Reranker

`RerankService`一次向DashScope `qwen3-rerank`提交20个候选的完整正文，要求返回Top-10的原候选索引和相关性分数。

成功条件：

- HTTP状态成功；
- 返回结果非空；
- 返回数量等于期望Top-N；
- 索引合法且不重复。

成功后按`relevance_score`降序排列，并在元数据中记录Rerank状态与模型。

## 6. 故障降级

以下情况整次降级：

- 30秒超时；
- SDK或网络异常；
- 非成功HTTP响应；
- 空结果；
- 部分结果、非法索引或重复索引。

降级行为：

```text
候选按vector_score降序
→ 保留Top-10
→ rerank_score置空
→ rerank_applied=false
→ 跳过0.65专用分数阈值
→ 继续Top-5与邻接扩展
```

这保证P0-4不会因为Rerank失败产生零上下文。

## 7. 阈值与保底

0.65阈值只用于真实专用Rerank分数，不能用于向量分数。若Rerank真实成功但10个结果全部低于阈值，保留排序最前的Top-3。

## 8. 邻接扩展

基础证据确定后，使用`source_id + chunk_index`直接查询同文献的前后Chunk，不在Dense召回池里寻找邻居。当前66篇统一重建索引已通过顺序连续性审计；新增或恢复数据必须重新审计。

## 9. 上下文构建

`ContextBuilder`在12,000字符预算内统一计算证据、分隔符和参考文献长度。Artifact保存模型实际可见的正文；只有剩余预算不足时才尝试按`RAG_MAX_CHARS_PER_EVIDENCE`截断。

## 10. 评测证据

| 方案 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|
| BL-1 Dense Top-5 | 46% | 56% | 0.3797 |
| 旧Full：Listwise Rerank | 66% | 76% | 0.4953 |
| 当前Full：`qwen3-rerank` | **88%** | **88%** | **0.8067** |

当前Full的50次Rerank全部成功，0降级；Doc-Hit@5为100%。

结果：

- `evaluation/results/FULL_QWEN3_V2_20260723_FORMAL/`
- `evaluation/results/FULL_QWEN3_V2_20260723_COMPARISON.md`
- `evaluation/results/RERANKER_ABLATION_20260723.md`

## 11. 尚未进入生产的方向

- 双路Query与RRF；
- Query Rewrite；
- Multi-query；
- Hybrid Search。

这些能力只能作为实验或后续规划描述，不得作为当前线上功能。
