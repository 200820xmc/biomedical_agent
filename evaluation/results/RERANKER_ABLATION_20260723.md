# 专用 Reranker 消融报告（2026-07-23）

## 结论

DashScope `qwen3-rerank` 在相同50题、相同当前索引和相同Gold Chunk标注上显著优于原 `qwen-max` Listwise评分。生产链路已仅替换Reranker；召回数量、阈值、Top-K和邻接扩展未改变。

以下消融复用正式Full保存的Agent query，不重新运行Agent生成回答，也不计算Ragas。因此这些数据是检索级结果，不是替换后新的Full全链路结果。

| 方案 | Recall@3 | Recall@5 | MRR | Doc-Hit@5 | 有效性 |
|---|---:|---:|---:|---:|---|
| BL-1 Dense Top-5 | 46% | 56% | 0.3797 | 94% | 50/50有效 |
| 原Full：Agent query + `qwen-max` Listwise | 66% | 76% | 0.4953 | 98% | 50/50有效 |
| 原问题直接检索 + 原Listwise | 62% | 72% | 0.5340 | 98% | 50/50有效 |
| 双路RRF + 原Listwise | 68% | 74% | 0.5483 | 96% | 50/50有效 |
| Agent query单路 + `qwen3-rerank` | **88%** | **90%** | **0.8207** | **100%** | 50/50有效，0降级 |
| 双路RRF + `qwen3-rerank` | **92%** | **92%** | **0.8495** | **100%** | 50/50有效，0降级 |

## 逐步诊断

1. 12道原Full Recall@5失败中，`rq007`、`rq050`是真正的召回失败；其余多为部分支撑或多Chunk联合支撑。
2. 原问题直接检索没有稳定超过Agent query，不能直接禁止Agent改写。
3. 原双路RRF中，`rq020`、`rq031`的Gold已进入Dense候选，却被Listwise Rerank淘汰，证明瓶颈在重排稳定性。
4. `gte-rerank-v2`三题门槛只命中1/3，不继续50题。
5. `qwen3-rerank`双路三题门槛全部排第1，随后50题达到Recall@5 92%。
6. 为避免引入跨层原问题传递，追加生产同构的Agent-query单路实验，Recall@5仍达到90%，相对原Full新增命中7题、丢失0题。

Agent-query单路剩余Recall@5失败：`rq007`、`rq012`、`rq017`、`rq028`、`rq050`。

双路剩余Recall@5失败：`rq006`（Gold实际第7）、`rq007`、`rq028`、`rq050`。

## 生产落地范围

- 使用DashScope `AioTextReRank`和模型 `qwen3-rerank`；
- 20个候选一次请求，保留Top-10；
- 传入完整Chunk正文，不再截断到前800字符；
- 30秒超时；
- 超时、异常、非成功HTTP、空结果和部分结果均整次降级为向量Top-10；
- 降级结果不应用0.65专用Rerank阈值；
- 专用Rerank全部低于0.65时仍使用既有Top-3保底；
- 最终Top-K仍为5，邻接扩展仍按 `source_id + chunk_index` 查询。

## 结果位置

- 原问题消融：`evaluation/results/DIRECT_QUERY_V2_20260723_FORMAL/`
- 原双路RRF：`evaluation/results/DUAL_RRF_V2_20260723_FORMAL/`
- `gte-rerank-v2`三题：`evaluation/results/DUAL_RRF_GTE_DIAG_RETRY_20260723/`
- `qwen3-rerank`双路三题：`evaluation/results/DUAL_RRF_QWEN3_DIAG_20260723/`
- `qwen3-rerank`双路50题：`evaluation/results/DUAL_RRF_QWEN3_V2_20260723_FORMAL/`
- `qwen3-rerank`Agent单路三题：`evaluation/results/AGENT_QWEN3_DIAG_20260723/`
- `qwen3-rerank`Agent单路50题：`evaluation/results/AGENT_QWEN3_V2_20260723_FORMAL/`

## 正式Full复跑结果

替换后的生产代码已完成50题正式Full全链路和Ragas：Recall@3 88%、Recall@5 88%、MRR 0.8067、Faithfulness 0.9382、Context Recall 0.9650、Doc-Hit@5 100%。50题全部有效，两个Ragas指标均为50/50有限值，无需retry。详见 `evaluation/results/FULL_QWEN3_V2_20260723_COMPARISON.md`。
