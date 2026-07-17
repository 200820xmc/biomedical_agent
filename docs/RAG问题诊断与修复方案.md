# RAG 当前问题状态与后续修复方案

## 1. 文档信息

| 项目 | 内容 |
|---|---|
| 文档名称 | RAG 当前问题状态与后续修复方案 |
| 适用项目 | AVF Research Assistant |
| 更新时间 | 2026-07-17 |
| 文档状态 | 当前代码静态检查版 |
| 检查范围 | RAG Agent、流式输出、超额召回、Rerank、上下文构建、文档分块、Milvus 入库、上传接口 |

## 1.1 2026-07-17全链路评测补充证据

最新50题Full结果位于 `evaluation/results/20260717_045223/`：

- 50/50执行完成，无执行错误，工具调用率100%，空答案率0%；
- Doc-Hit 49/50（98%），Doc-Hit@1/@3/@5为82%/94%/98%，平均排名1.3；
- Strict-Chunk-Hit为18/50，但32道严格Chunk未命中中有31道仍命中正确论文；
- `rq007` 的Rerank最高分为0.60，低于0.65阈值，12个候选被过滤为0，证明“固定阈值无保底”仍是实际P0问题；
- 50篇评测文档中统计到467条重复Milvus UUID行；
- 新索引已经写入稳定 `chunk_id`、`content_hash` 和 `chunk_index`，旧索引仍需重建才能保证真实邻接；
- Full的Faithfulness和Answer Relevancy均值被NaN污染，当前详情又没有保存完整回答、完整上下文和每题Ragas分数，无法无损重算；
- BL-1已完成相同50题：Doc-Hit 94%、Doc-Hit@1 60%、Doc-Hit@3 88%、Doc-Hit@5 94%、平均排名1.6、Strict-Chunk-Hit 56%；
- Full相对BL-1的文档级指标提升有效，但Strict-Chunk-Hit从56%下降到36%，说明单一Gold Chunk与多样性/预算选择之间存在冲突，不能笼统宣称全面提升；
- BL-1的Ragas均值同样被NaN污染，回答层仍不能比较。

因此本文后续修复优先级应增加：

1. 阈值过滤为0时的保底策略；
2. Ragas有限数值过滤与完整Trace持久化；
3. 旧索引重建和重复行清理；
4. 为每题增加人工审核的可接受Chunk集合，同时保留Strict-Chunk-Hit。

> 本文档根据当前工作区代码进行静态检查，不代表所有改动已经通过运行测试、集成测试和真实问答回归。

## 2. 当前总体结论

当前项目已经完成部分问题的代码级修复，包括：

- 为 Rerank 模型增加内部标签。
- 流式输出增加节点、标签和工具调用块过滤。
- 增加 Rerank 相关性阈值。
- 增加科研因果和影响排序约束。
- 增加 Rerank 前候选精确去重。
- ContextBuilder 遇到超长候选后继续检查后续候选。
- 增加单证据字符预算配置。
- MD/TXT 上传响应区分上传成功和索引成功。
- PDF 入库增加版本化写入的初步实现。
- 增加相邻 chunk 扩展的代码框架。

但项目仍然存在多项未解决或只解决一部分的问题。当前最重要的问题是：

1. Rerank 失败时，0.65 阈值可能过滤掉全部向量结果。
2. Rerank 是否真正成功仍然判断不准确。
3. 分批 Listwise Rerank 的分数仍然不可跨批次可靠比较。
4. 配置中的1600字符实际仍会变成3200字符。
5. 单证据1600字符目前不是硬限制。
6. 相邻 chunk 扩展缺少可靠的 `chunk_index` 数据基础。
7. 普通 MD/TXT 仍然采用“先删旧索引、再写新索引”的高风险流程。
8. 当前新增和修改内容尚未形成完整测试闭环。

## 3. 当前状态分级

| 状态 | 含义 |
|---|---|
| 已实现，待验证 | 已加入代码，但尚未通过真实运行和专项测试确认 |
| 部分解决 | 已完成部分处理，但根本问题或边界情况仍存在 |
| 未解决 | 当前代码中尚未实现或配置尚未生效 |
| 新增风险 | 修复过程中引入的新冲突或不一致 |

## 4. 已实现但仍需验证的问题

## 4.1 流式 Rerank 评分泄漏

### 原问题

用户在正式回答前看到：

```text
0:8
1:6
2:9
...
```

这些内容是工具内部 Rerank 模型的评分结果，不应该发送给用户。

### 当前实现

Rerank 模型调用已经增加：

```python
config={"tags": ["internal_rerank"]}
```

Agent 流式输出已经增加：

```python
if "internal_rerank" in tags:
    continue

if node_name != "model":
    continue

if getattr(token, "tool_call_chunks", None):
    continue
```

### 当前状态

```text
状态：已实现，待验证
```

### 仍需验证

- Rerank 评分是否完全不再进入 SSE `content`。
- 工具调用参数是否可能进入正文。
- Agent 决定调用工具前产生的普通文本是否会进入正文。
- 流式和非流式最终正文是否一致。
- 不同 LangGraph 或 LangChain 版本中的节点名称是否始终为 `model`。

### 验收用例

问题：

```text
请问动静脉瘘狭窄的原因是什么？
这些原因的影响排序是怎么样的？
```

回答不得出现：

```text
0:8
1:6
14:4
```

## 4.2 科研因果和影响排序约束

### 当前实现

系统提示词已经增加：

- 区分病因、危险因素、相关因素、病理机制和检测方法。
- 相关性不能直接解释为因果关系。
- 只有直接比较研究才能支持影响排序。
- 检测模型和诊断方法不是疾病形成原因。
- 缺少直接证据时必须说明无法可靠排序。

### 当前状态

```text
状态：已实现，待验证
```

### 仍需验证

- 模型是否真正遵守约束。
- 工具上下文包含无关检测论文时，模型是否仍可能引用。
- 没有直接比较研究时，模型是否停止生成第一、第二、第三名。
- 不同模型温度和会话历史是否会影响约束效果。

## 4.3 候选精确去重

### 当前实现

Rerank 前增加了候选去重：

```text
chunk_id
→ content_hash
→ 正文前200字符
```

重复候选保留向量分数更高的一项。

### 当前状态

```text
状态：已实现，待验证
```

### 仍需改进

- 正文前200字符不是可靠的完整内容哈希。
- 旧数据未必具有稳定 `chunk_id` 和 `content_hash`。
- 多个内容相同但章节不同的 chunk 可能被误判。
- 需要在入库时生成稳定的完整内容哈希。

## 4.4 上传状态分离

### 当前实现

MD/TXT 上传响应已经增加：

```json
{
  "upload_success": true,
  "index_success": false,
  "index_error": "..."
}
```

索引失败时使用：

```text
HTTP 207 Multi-Status
message = partial_success
```

### 当前状态

```text
状态：已实现，待前端适配和验证
```

### 仍需处理

- 前端是否展示 `index_success`。
- 前端是否会把207响应误认为普通成功。
- 响应体中的 `code=200` 与 HTTP 207 存在语义不一致。
- 上传失败、保存成功、索引失败需要明确的状态枚举。

## 5. 尚未解决的 P0 问题

## 5.1 Rerank 降级与0.65阈值冲突

### 当前逻辑

Rerank失败或不执行时：

```python
item.rerank_score = item.vector_score
```

随后统一执行：

```python
item.rerank_score >= 0.65
```

### 问题

当前向量相似度通常约为：

```text
0.50～0.56
```

因此会出现：

```text
Rerank失败
→ 回退到vector_score
→ 所有结果低于0.65
→ 最终没有证据
```

候选数量小于等于 `rerank_k` 时也会跳过真正的 Rerank，同样可能被0.65阈值全部过滤。

### 当前状态

```text
状态：未解决，新增高优先级风险
```

### 推荐方案

分别配置：

```python
rag_rerank_threshold = 0.65
rag_vector_threshold = 0.45
```

根据实际状态选择阈值：

```python
if rerank_result.applied:
    threshold = config.rag_rerank_threshold
else:
    threshold = config.rag_vector_threshold
```

不得使用同一个阈值同时解释 Rerank 分数和向量分数。

## 5.2 Rerank 是否成功的状态仍不准确

### 当前逻辑

```python
rerank_applied = (
    self.rerank.enabled
    and len(candidates) > rerank_k
)
```

### 问题

该条件只能说明满足执行条件，不能说明：

- API调用成功。
- 没有超时。
- 分数解析成功。
- 没有降级。

即使实际发生异常并退回向量排序，仍可能记录：

```text
rerank_applied = true
rerank_degraded = false
```

当前动态添加的 `rerank_degraded` 也没有稳定定义在 Artifact 数据模型中，最终序列化结果没有完整返回该字段。

### 当前状态

```text
状态：未解决
```

### 推荐方案

定义明确结果模型：

```python
@dataclass
class RerankResult:
    items: list[RetrievalItem]
    applied: bool
    degraded: bool
    model: str | None
    error: str | None
    duration_ms: float
```

由 RerankService 返回真实执行结果，RetrievalService 不再通过候选数量推测状态。

## 5.3 分批 Listwise Rerank 仍不可跨批次可靠比较

### 当前实现

```python
_RERANK_BATCH_SIZE = 15
```

每批候选独立打0～10分，随后将全部批次分数直接合并排序。

### 问题

不同批次中的9分不是同一评价环境下产生，不能保证具有相同相关性。

例如：

```text
批次A中的9分：直接回答问题
批次B中的9分：只是该批次中相对最相关
```

当前系统会将两者视为完全相同。

### 当前状态

```text
状态：未解决
```

### 推荐方案

优先使用专用 Reranker或 Cross-Encoder。

临时方案可以采用 Pointwise 结构化评分：

```json
{
  "relevance": 0.82,
  "direct_answer": true,
  "evidence_type": "cause"
}
```

候选独立评分后再全局排序。

## 5.4 Rerank 仍然只读取前800字符

### 当前实现

```python
_RERANK_CONTENT_MAX_CHARS = 800
```

### 问题

当前实际 chunk 可能达到3200字符甚至更大。关键的实验结果、结论或病因说明如果出现在后半部分，Reranker无法看到。

### 当前状态

```text
状态：未解决
```

### 推荐方案

如果实际 chunk 调整为1600字符，则同步设置：

```python
_RERANK_CONTENT_MAX_CHARS = 1600
```

更好的方案是提取与query最相关的局部窗口，而不是固定截取开头。

## 6. 分块问题

## 6.1 配置1600，实际仍是3200

### 当前配置

```python
chunk_max_size = 1600
chunk_overlap = 200
```

### 当前分块器

```python
chunk_size=self.chunk_size * 2
```

因此实际目标是：

```text
1600 × 2 = 3200字符
```

### 当前状态

```text
状态：未解决
```

### 推荐修改

去掉隐式乘法：

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=self.chunk_size,
    chunk_overlap=self.chunk_overlap,
)
```

日志只输出真实生效值。

## 6.2 小块合并可能超过最大尺寸

### 当前判断

```python
doc_size < min_size
and len(current_doc.page_content) < self.chunk_size * 2
```

### 问题

这里只检查合并前的 `current_doc`，没有检查：

```text
current_doc长度 + 新chunk长度
```

因此合并结果仍可能超过最大尺寸。

### 推荐修改

```python
merged_size = (
    len(current_doc.page_content)
    + 2
    + len(doc.page_content)
)

if doc_size < min_size and merged_size <= self.chunk_size:
    merge()
```

## 6.3 修改分块后需要重新索引

现有 Milvus 数据仍然是旧分块策略生成的。只修改代码不会改变已入库 chunk。

必须执行：

```text
修改分块代码
→ 为每个chunk增加稳定metadata
→ 清理或版本迁移旧索引
→ 重新索引全部论文
→ 重新运行检索评测
```

## 7. 上下文构建问题

## 7.1 单证据1600字符不是硬限制

### 当前逻辑

如果完整 chunk 能放入总预算，就直接放入完整内容：

```python
if total_chars + len(full_evidence) <= budget:
    append(full_evidence)
```

只有完整内容放不下时，才截断到：

```python
rag_max_chars_per_evidence
```

### 问题

第一个3200字符的chunk只要能放进12000总预算，就会完整进入上下文。

因此：

```text
rag_max_chars_per_evidence = 1600
```

当前不是硬上限。

### 当前状态

```text
状态：部分解决
```

### 推荐修改

先限制单证据，再检查总预算：

```python
content = item.content[:self._max_per_evidence]
evidence = build(content)
```

## 7.2 来源多样性和上下文预算仍然分离

当前仍然是：

```text
先选择final_k个chunk
→ 邻居扩展
→ ContextBuilder按预算保留
```

虽然 ContextBuilder 已经不会在第一个超长候选后直接停止，但来源多样性服务仍然不知道最终哪些内容会被预算保留。

### 影响

- 选择阶段的来源覆盖不等于最终上下文来源覆盖。
- 日志中的 `selected` 仍可能大于 Artifact 中的 `selected_count`。
- 邻居扩展可能挤占原本选择的其他来源证据。

### 推荐方案

将以下条件放入同一个预算感知选择阶段：

- Rerank分数。
- 相关性阈值。
- 每来源chunk上限。
- 最大来源数。
- 单证据上限。
- 总上下文预算。
- 内容重复度。

## 7.3 置信度仍然过于简单

当前主要规则：

```text
至少2个来源
最高分≥0.7
→ high
```

没有考虑：

- 平均相关性。
- 超过阈值的证据数。
- 是否直接回答问题。
- 子问题覆盖情况。
- Rerank是否成功。
- 是否发生降级。

### 当前状态

```text
状态：未解决
```

### 推荐规则

```text
high：
- Rerank成功
- 至少3个直接证据超过阈值
- 至少2个来源
- 问题主要部分均被覆盖

medium：
- 存在直接证据
- 但覆盖不完整

low：
- 没有证据超过阈值
- 或Rerank降级且向量相关性较低
```

## 8. 相邻 chunk 扩展问题

## 8.1 代码已加入，但缺少真实 chunk_index

当前扩展逻辑尝试查找：

```text
chunk_index - 1
chunk_index + 1
```

但旧数据和当前普通入库流程没有稳定写入：

```text
chunk_index
chunk_count
chunk_id
content_hash
source_id
```

RecallService在缺失时使用检索结果排名：

```python
chunk_index = metadata.get("chunk_index", idx)
```

这里的 `idx` 是相似度搜索排名，不是原论文顺序。

### 当前状态

```text
状态：代码框架已加入，实际能力未完成
```

### 推荐方案

分块完成后统一写入：

```python
for index, doc in enumerate(documents):
    doc.metadata["source_id"] = stable_source_id
    doc.metadata["chunk_index"] = index
    doc.metadata["chunk_count"] = len(documents)
    doc.metadata["content_hash"] = sha256(...)
    doc.metadata["chunk_id"] = stable_chunk_id
```

完成全量重新索引后，才能启用邻居扩展。

## 8.2 邻居扩展未再次执行相关性检查

当前邻居来自原始候选池，可能没有通过 Rerank阈值。

邻居可以作为上下文补充，但应满足：

- 只附着在高分主证据上。
- 不作为独立高可信证据。
- 计入总预算。
- 标记 `is_neighbor=true`。
- 必要时进行轻量相关性检查。

## 9. 检索功能未完成项

## 9.1 `max_sources` 尚未生效

配置已存在：

```python
rag_max_sources = 5
```

但 DiversityService 中的 `max_sources` 参数仍标记为暂未使用。

### 当前状态

```text
状态：未解决
```

## 9.2 `source_filter` 尚未实现

Agent工具已经暴露：

```python
source_filter
```

但检索服务没有应用Milvus过滤条件。

### 当前状态

```text
状态：未解决
```

## 9.3 `auto` 模式不是真正自动

`search_mode="auto"` 当前只是读取默认参数，没有内部问题类型判断。

系统仍然依赖 Agent 正确传入：

```text
focused
comparison
broad
```

### 当前状态

```text
状态：未解决
```

## 9.4 Query改写和多查询未启用

当前配置：

```python
rag_query_rewrite_enabled = False
rag_multi_query_enabled = False
```

复杂问题仍然只有一个宽泛query。

### 当前状态

```text
状态：未解决
```

## 9.5 混合检索尚未实现

当前主要依赖向量检索，没有：

- BM25关键词检索。
- 标题精确匹配。
- 作者、年份和章节metadata匹配。
- RRF多路融合。

### 当前状态

```text
状态：未解决，属于后续增强
```

## 10. 文档入库与数据安全问题

## 10.1 普通 MD/TXT 仍先删旧索引再写新索引

当前 `index_single_file()`：

```text
读取文件
→ 删除旧索引
→ 分块
→ Embedding
→ 写入新索引
```

如果Embedding或Milvus写入失败，旧索引已经丢失。

### 当前状态

```text
状态：未解决，高优先级
```

### 推荐方案

让 MD/TXT 与 PDF 共用版本化流程：

```text
新内容分块
→ 写入新version_id
→ 验证新版本
→ 删除旧版本
```

## 10.2 MD/TXT 上传仍先覆盖旧文件

当前同名文件上传：

```python
file_path.unlink()
file_path.write_bytes(content)
```

索引在文件覆盖后才执行。

### 风险

- 旧文件已经丢失。
- 旧索引可能被删除。
- 新索引可能写入失败。

### 推荐方案

```text
写入临时文件
→ 使用临时文件建立新索引
→ 验证成功
→ 原子替换正式文件
→ 清理旧索引
```

## 10.3 PDF版本化流程缺少完整验证和失败清理

PDF流程已经采用：

```text
先写新版本
→ 再删除旧版本
```

但目前没有验证：

- 实际成功写入多少条。
- 新版本是否可以查询。
- 所有Embedding批次是否完整成功。
- 删除旧版本是否成功。

如果分批写入中途失败，部分新版本可能残留。

### 推荐方案

- 记录预期chunk数量。
- 查询新版本实际数量。
- 数量一致后才切换版本。
- 写入失败时删除该 `version_id` 的部分数据。
- 删除旧版本失败时返回明确的部分成功状态。

## 10.4 删除旧版本失败会被吞掉

`delete_by_source()` 捕获异常后返回0，不继续抛出。

因此可能发生：

```text
新版本写入成功
旧版本删除失败
上层仍报告索引成功
Milvus同时存在新旧版本
```

### 当前状态

```text
状态：未解决
```

## 10.5 向量维度不匹配时自动删除collection

Milvus启动时如果检测到向量维度不匹配，会执行：

```python
drop_collection("biz")
```

这可能导致整个知识库被清空。

### 当前状态

```text
状态：未解决，高风险
```

### 推荐方案

- 禁止启动时自动删除。
- 创建新collection，例如 `biz_v2`。
- 执行数据迁移和重新索引。
- 验证完成后人工切换。

## 11. Agent和会话问题

## 11.1 消息裁剪函数仍未接入

代码定义了：

```python
trim_messages_middleware()
```

但 `create_agent()` 没有使用该逻辑。

### 当前状态

```text
状态：未解决
```

## 11.2 每轮重复提交系统消息

每次流式和非流式请求都会重新构造：

```python
SystemMessage(...)
HumanMessage(...)
```

同时使用相同的 `thread_id` 保存历史，可能导致系统消息重复积累。

### 当前状态

```text
状态：未解决
```

## 11.3 会话只保存在内存

当前使用：

```python
MemorySaver
```

服务重启后会话历史丢失。

### 当前状态

```text
状态：未解决
```

## 12. SSE接口问题

## 12.1 API支持事件，但Agent没有真正生成

API层已经能够处理：

```text
tool_start
retrieval_complete
search_results
content
done
```

但当前 `query_stream()` 主要只生成：

```text
content
complete
error
```

因此工具状态和检索完成事件目前没有真正发送。

### 当前状态

```text
状态：部分解决
```

## 12.2 `done`事件已经包含完整回答，但前端仍需适配验证

API层会累计正文并返回：

```json
{
  "type": "done",
  "data": {
    "answer": "...",
    "tool_calls": 0
  }
}
```

仍需确认前端：

- 是否读取 `done.data.answer`。
- 是否只依赖本地Token累计。
- 网络中断时能否恢复最终回答。

## 13. 引用问题

ContextBuilder已经优先读取：

```text
title
authors
year
```

但旧MD/TXT数据通常不包含这些metadata，仍会回退到文件名解析。

因此仍可能生成：

```text
(fbioe-)
```

### 当前状态

```text
状态：部分解决
```

### 推荐方案

- PDF解析时写入正式元数据。
- MD/TXT入库时从文档头部或文件映射表补充元数据。
- 无法解析时使用编号引用，不生成伪作者名称。

## 14. 工程状态和测试缺口

## 14.1 当前工作区处于未提交开发状态

当前存在：

- 多个已修改文件。
- 新增 retrieval模块。
- 新增PDF入库模块。
- 新增脚本和文档。
- 评测问题文件调整。

这些变化尚未形成稳定提交。

## 14.2 缺少专项测试

当前需要新增：

1. Rerank评分不进入流式正文。
2. 工具调用参数不进入正文。
3. Rerank失败时使用向量阈值。
4. 候选数小于 `rerank_k` 时不会被全部过滤。
5. 病因问题不返回检测模型。
6. 无直接证据时不生成确定排名。
7. 单证据严格不超过字符上限。
8. 超长候选不会阻塞后续短候选。
9. 稳定 `chunk_index` 支持真实邻居扩展。
10. MD/TXT索引失败时保留旧文件和旧索引。
11. PDF分批写入失败时清理部分版本。
12. 删除旧版本失败时返回真实状态。

## 15. 推荐修复优先级

## 15.1 第一优先级

```text
1. 修复Rerank降级与0.65阈值冲突
2. 让RerankService返回真实执行状态
3. 去掉分块器中的 * 2
4. 将单证据1600字符变成硬限制
5. 为流式过滤增加专项测试
```

## 15.2 第二优先级

```text
6. 为所有入库类型写入稳定chunk metadata
7. 全量重新索引现有文档
8. 修复普通MD/TXT的版本化安全更新
9. 完善PDF版本验证和失败清理
10. 重构置信度计算
```

## 15.3 第三优先级

```text
11. 替换分批Listwise Rerank
12. 增加Query分解和多查询
13. 实现max_sources和source_filter
14. 增加BM25/标题匹配混合检索
15. 完善SSE工具事件和前端引用展示
```

## 16. 推荐的下一轮开发任务

建议下一轮只处理以下闭环，不同时扩展其他功能：

```text
RerankResult真实状态
  ↓
Rerank/Vector双阈值
  ↓
真实1600字符分块
  ↓
单证据1600字符硬限制
  ↓
流式泄漏和无关证据回归测试
```

完成后使用固定问题：

```text
请问动静脉瘘狭窄的原因是什么？
这些原因的影响排序是怎么样的？
```

验收要求：

- 页面不出现内部评分。
- 不把检测模型作为狭窄形成原因。
- 区分病因、危险因素、相关性和检测方法。
- 没有直接比较证据时明确说明无法可靠排序。
- Rerank失败时仍能返回合理的向量检索证据。
- 每个证据不超过配置的字符上限。

## 17. 最终结论

当前项目已经针对流式泄漏、相关性阈值、因果约束、候选去重和上下文预算进行了代码修改，但尚未达到完全解决状态。

当前最关键的未解决问题是：

```text
Rerank降级状态不真实
Rerank阈值与向量分数冲突
实际chunk仍然过大
单证据上限没有硬执行
邻居扩展缺少稳定chunk_index
普通MD/TXT索引仍存在数据丢失风险
```

在这些问题修复并通过专项回归之前，当前版本应视为开发验证版本，不建议直接作为稳定科研问答版本交付。
