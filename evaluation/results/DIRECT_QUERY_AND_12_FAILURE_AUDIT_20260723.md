# 原问题直接检索消融与12道失败样本审计

日期：2026-07-23

## 1. 实验约束

- 数据集：`evaluation/ragas_50_v2_review.csv`
- SHA-256：`5836fbc05daacb0741d234d196938d5c409d2a40f839f730989e96184d02a393`
- 题目：50题
- 对照：已保存的 Full Agent-query 正式结果
- 实验：人工审核后的原问题直接进入同一 Full 检索链路，绕过 Agent 生成工具query
- 保持不变：Top-20 Dense召回、一次性Rerank、Top-10保留、0.65阈值、Top-5基础证据、`chunk_index`邻接扩展、12000字符预算
- 未执行：回答生成、Ragas、入库、清库或索引重建
- 直接检索运行状态：valid，50/50，0错误，0空结果

## 2. 总体指标

| 指标 | Agent-query Full | 原问题直接检索 | 原问题 - Agent |
|---|---:|---:|---:|
| Doc-Hit | 98% | 98% | 0 |
| Doc-Hit@1 | 88% | 88% | 0 |
| Doc-Hit@3 | 94% | 96% | +2个百分点 |
| Doc-Hit@5 | 98% | 98% | 0 |
| Acceptable-Chunk Recall@3 | 66% | 62% | -4个百分点 |
| Acceptable-Chunk Recall@5 | 76% | 72% | -4个百分点 |
| Acceptable-Chunk MRR | 0.4953 | 0.5340 | +0.0387 |

结论：原问题直接检索提高了第一个正确Chunk的整体排名，但降低了Recall@3和Recall@5。不能简单禁止Agent改写；更合理的方向是保留原问题，同时将Agent关键词query作为第二路检索，再进行融合。

本实验与历史Full是两次独立的LLM Rerank运行，因此逐题变化同时含有少量Rerank非确定性。总体结果可用于判断方向，但不应把单题差异全部归因于query文本。

## 3. Recall@5命中状态转移

| 题号 | Agent-query | 原问题 | 解释 |
|---|---:|---:|---|
| rq012 | 0 | 1（Gold第2） | 原问题保留完整时间变化问法，修复命中 |
| rq008 | 1（第3） | 0 | Agent关键词化query更有利 |
| rq020 | 1（第3） | 0（Gold第6） | Agent明确强化CTA/MRA“敏感性对比” |
| rq031 | 1（第4） | 0 | Agent保留“体外狭窄模型、带宽、动态范围”等关键词 |

Agent-query净增加2道Recall@5命中，但原问题在若干已命中题中将Gold提前，因此MRR更高。

## 4. 原Full的12道Recall@5失败审计

审计标准：只有单个Chunk能够直接、完整支撑参考答案的全部核心限定，才能直接加入当前“任一Gold即命中”的 `acceptable_chunk_ids`。多个Chunk合并后才能完整回答的，标为“联合支撑”，不能把其中任一部分证据单独补成Gold。

| 题号 | 审计结论 | 前5证据情况 | 处理建议 |
|---|---|---|---|
| rq006 | 部分/接近完整 | `doc_2954af:d2810a053d148c02`与`doc_2954af:a2c40d6beef9e86b`覆盖PVDF、2 mm膜片、硅胶背衬、听诊器比较、平坦频响和多通道监测，但对受控狭窄模型的限定不完整；原Gold在第6 | 不直接补单Gold；优先调整邻居排序，使Gold从第6进入前5 |
| rq007 | 真正失败 | 前5没有目标文献`doc_29cc5e`，无法支持直径、流速、局部湍流强度和壁面压力波动四项估计参数 | 需要关键词/混合检索或双Query |
| rq012 | 两个Chunk联合完整支撑 | `doc_3e2e29:fbc42a3184ad89ce`覆盖狭窄扩展、速度规则化和近壁扰动下降；`doc_3e2e29:affa3b318800e261`覆盖6个月成熟、1年狭窄和1.5年失败 | 建立“Gold证据集合/claim级命中”，不能把两个部分Chunk任一单独记为完整Gold |
| rq016 | 部分支撑 | `doc_edffea:fecf145f91edb146`完整描述14项特征、CWT、AR与ASC/ASF，但未完整给出线性相关及同时估计位置和程度的结果 | 保持失败；需命中摘要或结果Chunk |
| rq017 | 部分支撑 | `doc_57bca3:ad4d56c57bf974b6`覆盖临床/CFD研究计数与30–70°结论；其他Chunk覆盖部分角度和VasQ信息，但未完整覆盖三类边界及40–50°限定 | 不补单Gold；需要章节级/claim级证据集合 |
| rq021 | 部分支撑 | `doc_6f1019:a1c1b625578a0063`明确支持起始时间关系反转和`Td<0`，但缺少+22 ms及−20至−38 ms | 保持失败；数字敏感的关键词/稀疏检索 |
| rq026 | 部分支撑 | 前5描述湍流压力、管壁振动、周围组织介质和耗散，但没有完整给出谐波线力、流固耦合、附加惯性及耗散链路 | 保持失败；需命中理论模型核心Chunk |
| rq028 | 两个Chunk联合完整支撑 | `doc_8b08d9:9035a152765d99f3`给出低/振荡WSS位置；`doc_8b08d9:086fa93224e7536b`给出RRT与狭窄位置、低WSS/振荡剪切和IH关系 | 改为claim级或Gold集合命中，不应单独补任一部分Chunk |
| rq037 | 部分支撑 | 前5充分描述S变换二维时频特征，但没有召回87.84% PPV与89.24%敏感度 | 保持失败；需数字和指标关键词检索 |
| rq039 | 两个Chunk联合完整支撑 | `doc_9f299f:65997a9665db1705`覆盖面积效应；`doc_9f299f:f34634895f9b670e`覆盖约43°、58°和逆流 | 改为Gold集合/claim级命中；当前单Chunk命中口径低估实际证据完整性 |
| rq045 | 部分支撑 | `doc_adffac:87cf57b4d6b84d5c`支持MRI+速度数据+CFD方法；`doc_adffac:170bf25243ad1d29`支持静脉扰动/WSS不均与非均匀重塑，但缺少完整3人、三个时间点、MRA/MRV边界条件 | 保持失败；需方法Chunk与结果Chunk联合召回 |
| rq050 | 真正失败 | 只召回目标文献中的NefDiag实现Chunk，未出现三个子模块和38名患者 | 需要摘要路由或精确关键词检索 |

审计汇总：

- 两个或多个前5 Chunk联合完整支撑：3题（rq012、rq028、rq039）
- 仅部分或接近完整支撑：7题
- 真正没有所需答案证据：2题（rq007、rq050）
- 可无争议直接追加为“单一替代Gold”的Chunk：0题

因此，不应通过简单扩充 `acceptable_chunk_ids` 把Recall@5人为抬高。下一版评测应增加claim级Gold或Gold证据集合指标，同时保留当前严格单Chunk指标作为诊断项。

## 5. 下一步依据

本轮证据支持的最小改动方向是“双路Query融合”，而不是关闭Agent改写：

1. Query A固定使用完整原问题；
2. Query B使用Agent生成的关键词query；
3. 两路召回结果去重后用RRF融合；
4. 总候选仍截断为20，再执行当前一次性Rerank；
5. 对含数字、缩写和性能指标的问题增加BM25/稀疏检索候选；
6. 另行新增claim级/Gold集合Recall，不改写或删除当前严格指标。

优先验证对象：rq007、rq021、rq037、rq050，以及邻居已命中但排第6的rq006。

## 6. 结果文件

- 原问题直接检索摘要：`evaluation/results/DIRECT_QUERY_V2_20260723_FORMAL/direct_retrieval_summary.json`
- 原问题直接检索逐题明细：`evaluation/results/DIRECT_QUERY_V2_20260723_FORMAL/direct_retrieval_details.csv`
- Agent-query Full逐题明细：`evaluation/results/FULL_V2_20260723_FORMAL_RAGAS_RETRY/review_eval_details.csv`
