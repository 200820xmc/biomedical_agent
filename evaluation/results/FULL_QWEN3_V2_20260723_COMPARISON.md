# 正式 Full：专用 Reranker 替换前后对比

评测日期：2026-07-23

两次运行使用相同的50题人工审核数据集、相同当前Milvus索引、相同Top-20 / Rerank Top-10 / final Top-5链路。新运行由Agent重新执行，因此Agent生成的检索query也可能存在正常的模型波动；结果不是仅替换缓存排序的离线回放。

| 指标 | 旧Full：`qwen-max` Listwise | 新Full：`qwen3-rerank` | 变化 |
|---|---:|---:|---:|
| Doc-Hit | 98% | 100% | +2个百分点 |
| Doc-Hit@1 | 88% | 96% | +8个百分点 |
| Doc-Hit@3 | 94% | 100% | +6个百分点 |
| Doc-Hit@5 | 98% | 100% | +2个百分点 |
| Recall@3 | 66% | 88% | **+22个百分点** |
| Recall@5 | 76% | 88% | **+12个百分点** |
| MRR | 0.4953 | 0.8067 | **+0.3114** |
| Chunk-Hit | 39/50 | 44/50 | +5题 |
| Faithfulness | 0.9172 | 0.9382 | +0.0210 |
| Context Recall | 0.9233 | 0.9650 | +0.0417 |

新Full在Recall@5新增命中：`rq016`、`rq021`、`rq026`、`rq037`、`rq039`、`rq045`；丢失0题。

新Full的Recall@5失败题：`rq006`、`rq007`、`rq012`、`rq017`、`rq028`、`rq050`。

## 完整性检查

- 50/50题完成；
- 0题错误；
- 每题恰好1次知识检索工具调用；
- 0空回答、0空上下文；
- 50次专用Rerank全部成功，0降级；
- Faithfulness 50/50有限值；
- Context Recall 50/50有限值；
- 主运行直接有效，无需Ragas retry。

## 结果文件

- 旧Full：`evaluation/results/FULL_V2_20260723_FORMAL_RAGAS_RETRY/`
- 新Full：`evaluation/results/FULL_QWEN3_V2_20260723_FORMAL/`
