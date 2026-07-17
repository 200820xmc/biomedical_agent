# AVF RAG 项目评测技术要求

> **Legacy技术要求**：本文对应2026-07-15的25题论文级检索评测。当前正式50题输入、真实Chunk ID、Full和BL-1输出契约请查看 `evaluation/README.md`。本文旧路径和旧Top-K设计仅作为历史记录保留。

> 文档版本：v1.0  
> 适用项目：AVF Research Assistant v2.0.0  
> 适用范围：知识库规模、检索效果、去重策略、引用质量、流式性能和索引性能评测  
> 项目定位：本项目仅用于医学科研、教学和算法验证，不构成临床诊断意见。

---

## 1. 文档目的

本文档用于指导 AVF 医学科研文献智能问答项目的评测功能开发、执行和验收。评测系统应生成真实、可重复、可追溯的量化结果，为以下场景提供依据：

- 检查当前知识库的实际数据规模和数据质量；
- 验证向量检索是否能够找到人工标注的相关论文；
- 比较论文级去重前后的检索效果；
- 检查回答中的引用是否来自实际检索结果；
- 测量 SSE 流式回答的首字延迟和完整响应时间；
- 测量文献分块和索引流程的性能；
- 为 README、项目报告和个人简历提供真实量化描述。

所有正式指标必须来自实际运行结果。禁止预填、估算或编造最终数值。

---

## 2. 名词说明

### 2.1 知识库来源

知识库来源是指 Milvus 文档元数据中的 `_file_name`。同一篇论文可能因为文件名中的空格、下划线、大小写或路径格式不同，以多个文件形式存在。

### 2.2 文本分片

文本分片是论文经过 `DocumentSplitterService` 分割后写入 Milvus 的一个 `Document`。一个文本分片至少包含：

```text
page_content：分片正文
metadata._file_name：来源文件名
metadata._source：来源路径
metadata._extension：文件类型
```

### 2.3 Hit@K

通俗解释：查看前 K 个检索结果中是否至少出现一篇正确论文。

```text
Hit@K = 命中至少一篇相关论文的问题数 / 有效问题总数
```

每道问题的取值只能为 `0` 或 `1`，整体结果取平均值。

### 2.4 Recall@K

通俗解释：一道问题应当找到多篇相关论文时，前 K 个结果实际找回了其中多少篇。

```text
Recall@K = Top-K 中命中的不同相关论文数 / 该问题的相关论文总数
```

整体结果使用宏平均，即先计算每道有效问题的 Recall，再对所有有效问题取平均。

### 2.5 TTFT

TTFT（Time To First Token）表示从 HTTP 请求发出，到客户端收到第一个有效 `content` SSE 事件之间的时间。

以下事件不得作为首字事件：

- `debug`
- `tool_call`
- `search_results`
- `done`
- `error`

---

## 3. 评测范围

### 3.1 必须评测的指标

| 类别 | 必须输出的指标 |
|---|---|
| 知识库规模 | 本地支持文件数、疑似独立论文数、Milvus 实体数、已入库来源数、平均每篇分片数、分片长度分布 |
| 检索效果 | Hit@1、Hit@3、Hit@5、Recall@3、Recall@5、Recall@10 |
| 去重效果 | Top-5 平均来源覆盖数、重复来源占比、去重前后 Hit@5、去重前后 Recall@5 |
| 引用质量 | 回答引用数、有效引用数、引用有效率、人工核验进度、人工引用支持率 |
| 流式性能 | TTFT、完整响应时间、成功率、P50、P95、首次运行标记 |
| 索引性能 | 文件字符数、分片数、分块耗时、单篇索引耗时、索引成功率、每万字符索引耗时 |

### 3.2 不在本次范围内的内容

- 不评估医学诊断有效性；
- 不评估临床准确率；
- 不训练或重新训练深度学习模型；
- 不修改现有 Milvus schema；
- 不自动判断论文结论的医学正确性；
- 不使用大模型自动生成正式人工标准答案；
- 不将模拟结果描述为真实实验结果。

---

## 4. 总体技术原则

评测代码必须遵守以下原则：

1. 优先复用当前项目配置、文档分块服务、向量存储和 RAG Agent。
2. 评测代码放在独立的 `evaluation/` 目录中。
3. 纯计算逻辑与外部服务调用分离，保证核心指标可以单元测试。
4. 每次运行同时保存逐条明细和汇总结果。
5. 汇总指标必须能够从明细文件重新计算。
6. 评测失败时保留错误信息，不得静默跳过或伪装成成功。
7. 不得读取、打印或写出 `DASHSCOPE_API_KEY`。
8. 不得修改 `.env`、`uploads/` 或 `volumes/`。
9. 不得删除、清空或重建现有 `biz` collection。
10. 外部模型批量调用和真实索引写入必须经过用户确认。

---

## 5. 安全等级与执行门禁

### 5.1 Level 0：纯本地安全操作

允许直接执行：

- 读取 `uploads/` 中文件的名称、大小和文本内容；
- 运行文件名规范化；
- 生成候选问题；
- 校验 CSV；
- 运行指标单元测试；
- 运行 Python 语法检查；
- 执行索引 `--dry-run`。

### 5.2 Level 1：外部服务只读操作

执行前应确认 Milvus 和网络可用：

- 查询 Milvus collection schema；
- 查询 Milvus 实体和元数据；
- 执行向量相似度检索；
- 调用 DashScope Embedding 生成查询向量。

这些操作不得写入、删除或更新现有实体。

### 5.3 Level 2：可能产生明显模型调用量的操作

以下操作执行前必须向用户说明预计调用次数并获得确认：

- 批量运行完整 RAG Agent；
- 每道问题执行多次 SSE 性能测试；
- 对多篇文献执行正式向量化索引。

如果有 `Q` 道问题、每题流式测试 `R` 次，则流式测试最多产生约 `Q × R` 次完整 Agent 调用。生成质量评测还会额外产生约 `Q` 次调用。

### 5.4 Level 3：数据写入操作

默认禁止对现有 `biz` collection 执行索引性能测试。正式写入只能满足以下条件之一：

1. 使用本次评测创建的独立临时 collection；
2. 用户明确授权写入指定测试 collection。

临时 collection 名称应包含固定前缀和时间戳，例如：

```text
evaluation_biz_20260715_170000
```

不得对现有 `biz` collection 使用 `drop_old=True`，不得调用 `drop_collection("biz")`。

---

## 6. 目录与文件要求

评测模块应创建以下结构：

```text
evaluation/
├── README.md
├── questions.example.csv
├── questions.csv
├── common.py
├── inventory.py
├── evaluate_retrieval.py
├── evaluate_deduplication.py
├── evaluate_generation.py
├── benchmark_stream.py
├── benchmark_indexing.py
├── run_all.py
└── results/
    └── .gitkeep

tests/
└── evaluation/
    ├── test_common.py
    ├── test_metrics.py
    └── test_sse_parser.py
```

新增文件仅允许位于 `evaluation/` 和对应的测试目录。除非出现无法通过评测层解决的问题，否则不修改 `app/` 中的生产代码。

---

## 7. 运行结果组织方式

每次评测应生成唯一 `run_id`：

```text
YYYYMMDD_HHMMSS
```

建议每次运行使用独立子目录：

```text
evaluation/results/20260715_170000/
```

该目录至少保存：

- `run_metadata.json`
- 各评测模块的明细 CSV；
- 各评测模块的汇总 JSON；
- `final_report.md`；
- 失败阶段的错误摘要。

`run_metadata.json` 至少包含：

```json
{
  "run_id": "20260715_170000",
  "started_at": "2026-07-15T17:00:00+08:00",
  "finished_at": null,
  "python_version": "3.13.3",
  "platform": "Windows-11",
  "collection_name": "biz",
  "embedding_model": "text-embedding-v4",
  "vector_dimension": 1024,
  "question_file": "evaluation/questions.csv",
  "question_file_sha256": "...",
  "status": "running"
}
```

禁止在运行元数据中保存 API Key、访问令牌或 `.env` 内容。

---

## 8. 文件名规范化要求

### 8.1 规范化流程

文件匹配必须基于 Milvus 元数据 `_file_name`，规范化过程至少包括：

1. 将反斜杠转换为正斜杠；
2. 只提取路径最后一段文件名；
3. 执行 Unicode NFKC 规范化；
4. 去除首尾空白；
5. 使用 `casefold()` 统一大小写；
6. 去除 `.md` 或 `.txt` 后缀；
7. 将连续空格、下划线等格式分隔符统一；
8. 保留原始名称用于结果追溯。

### 8.2 匹配结果

人工标注文件名与 Milvus 来源匹配后，只能出现以下状态：

| 状态 | 含义 | 处理方式 |
|---|---|---|
| `matched` | 唯一匹配到一个规范化来源 | 可以参加正式评测 |
| `missing` | 未匹配到知识库来源 | 该题标记无效并报错 |
| `ambiguous` | 同时匹配到多个无法确认的来源 | 等待人工处理 |
| `needs_review` | 尚未填写标准答案 | 不进入 Recall 分母 |

不得在匹配失败时自动选择最相似的文件名作为标准答案。

---

## 9. 评测集技术要求

### 9.1 CSV 格式

`questions.csv` 必须使用 UTF-8 编码，并包含：

```csv
question_id,question,relevant_files,category,notes
q001,哪些研究使用深度学习识别动静脉瘘狭窄？,NEEDS_REVIEW,深度学习,待人工标注
```

### 9.2 校验规则

- `question_id` 不能为空且必须唯一；
- `question` 不能为空；
- `relevant_files` 使用英文分号分隔；
- 支持中文、英文和混合文件名；
- `category` 为空时统一归入 `未分类`；
- `relevant_files` 为空或为 `NEEDS_REVIEW` 时，不进入正式 Recall 分母；
- 同一道题中的重复相关文件应规范化去重；
- 标准答案必须来自人工确认，不能由被评测模型自动填写。

### 9.3 建议问题类别

候选问题应覆盖当前知识库的主要主题：

- 深度学习狭窄识别；
- 血流声音与听诊；
- STFT、频谱或时频分析；
- 传统机器学习与人工特征；
- CFD 与血流动力学；
- 多位置或多通道融合；
- 无创 AVF 状态监测；
- 模型性能与实验对比。

正式结果必须报告有效标注问题数，不能只报告 CSV 总行数。

---

## 10. `common.py` 技术要求

`common.py` 只提供通用能力，不执行评测任务。至少包含：

- `load_questions()`：读取并校验问题集；
- `normalize_file_name()`：规范化文件名；
- `validate_relevant_files()`：校验人工标注来源；
- `compute_hit_at_k()`：计算单题 Hit@K；
- `compute_recall_at_k()`：计算单题 Recall@K；
- `compute_source_coverage()`：计算来源覆盖数；
- `compute_duplicate_ratio()`：计算重复来源占比；
- `compute_percentile()`：计算 P50、P95；
- `write_csv()`、`write_json()`：统一写结果；
- `build_run_metadata()`：生成运行元数据；
- `safe_milvus_preflight()`：只读检查 Milvus；
- `parse_sse_event()`：解析 SSE 事件。

百分位数算法必须固定并记录。建议使用线性插值法：

```text
position = (n - 1) × percentile
```

当输入为空时，均值、P50、P95 应返回 `null`，不能返回 `0` 冒充性能结果。

---

## 11. 知识库规模评测要求

### 11.1 本地文件统计

`inventory.py` 应只统计 `uploads/` 目录下的 `.md` 和 `.txt` 文件，并记录：

- 原始文件名；
- 规范化文件名；
- 文件类型；
- 文件字节数；
- 文本字符数；
- 是否疑似重复；
- 重复组编号。

### 11.2 Milvus 统计

Milvus 统计必须是只读操作：

- 检查 `biz` collection 是否存在；
- 检查向量字段维度是否为 1024；
- 读取实体数；
- 读取 `content` 和 `metadata`；
- 按规范化 `_file_name` 统计来源；
- 统计每个来源的分片数；
- 统计分片字符长度。

不得直接调用可能在维度不匹配时删除 collection 的连接流程。发现 schema 不匹配时应立即终止并报告。

### 11.3 长度分布

分片字符长度至少输出：

- 最小值；
- 平均值；
- 中位数；
- P95；
- 最大值。

### 11.4 输出文件

```text
inventory.json
inventory_chunks.csv
inventory_duplicates.csv
```

在重复论文未人工确认前，只能称为“疑似独立论文数”或“规范化来源数”，不得直接声称为正式独立论文数。

---

## 12. 检索效果评测要求

### 12.1 调用路径

检索评测应绕过 Agent，直接复用当前向量存储的相似度检索能力，避免 LLM 是否调用工具影响检索指标。

### 12.2 检索快照

每道问题建议一次性检索 Top-15，并保存完整排序快照。后续指标使用同一快照计算：

```text
Top-1  → Hit@1
Top-3  → Hit@3、Recall@3
Top-5  → Hit@5、Recall@5
Top-10 → Recall@10
Top-15 → 去重策略评测
```

这样可以减少重复 Embedding 调用，也能避免多次查询结果波动影响对比。

### 12.3 明细字段

检索明细至少包含：

```text
run_id
question_id
question
category
rank
raw_file_name
normalized_file_name
score
is_relevant
relevant_files
label_status
error
```

如果使用 L2 距离，必须在结果中注明“数值越小表示越相似”；不得将距离直接描述为相似度百分比。

### 12.4 汇总要求

汇总结果应同时提供：

- 整体指标；
- 按 `category` 分组指标；
- 有效问题数；
- 跳过问题数；
- 匹配失败问题数。

没有有效人工标注问题时：

- 可以输出检索明细；
- 可以输出来源覆盖数；
- 必须将 Hit 和 Recall 标记为 `not_available`；
- 进程应返回能够表示“需要人工标注”的非零退出码。

---

## 13. 去重策略评测要求

### 13.1 对比策略

必须对比：

**基线策略**

```text
直接取 Top-5 文本分片
```

**优化策略**

```text
召回 Top-15 文本分片
→ 按规范化 _file_name 去重
→ 每篇论文保留排名最高的分片
→ 截取前 5 篇论文
```

### 13.2 指标

```text
SourceCoverage@5 = Top-5 中不同规范化来源数

DuplicateRatio@5 = 1 - 不同规范化来源数 / 实际返回结果数
```

当实际返回结果数为 0 时，重复来源占比返回 `null`。

### 13.3 特殊要求

- 去重键必须使用规范化 `_file_name`；
- 不能使用正文内容作为默认去重键；
- 不能删除 Milvus 或 `uploads/` 中的重复数据；
- 同时保留去重前后的完整文件排序；
- 必须区分“来源覆盖提升”和“Recall 提升”，不能混为一个指标。

---

## 14. 生成回答与引用评测要求

### 14.1 评测内容

`evaluate_generation.py` 应记录：

- 用户问题；
- Agent 最终回答；
- Agent 实际调用的检索工具；
- 工具实际返回的论文来源；
- 回答中的引用标签；
- 引用是否能够映射到检索来源；
- 完整耗时；
- 错误类型和错误信息。

### 14.2 获取真实检索来源

当前普通问答接口只返回最终答案，不能通过“回答后重新检索”冒充 Agent 实际使用的论文。

评测实现应优先从 Agent 返回消息中的 `ToolMessage.artifact` 提取 `Document`。如果当前依赖版本无法获得工具 artifact，应：

1. 将来源类型标记为 `reconstructed_retrieval`；
2. 在报告限制中说明这不是 Agent 原始工具轨迹；
3. 不得将其描述为精确的 Agent 引用来源。

### 14.3 引用标签识别

至少支持：

```text
(Zhou et al. 2023)
(Seo 2017)
[1]
[2]
```

对 `[1]` 形式的引用，必须结合当次检索结果顺序映射，不能跨问题复用。

### 14.4 引用有效率

```text
引用有效率 = 能映射到当次检索来源的引用次数 / 回答中的引用总次数
```

应明确分母采用“引用出现次数”还是“不同引用标签数”。建议主指标采用引用出现次数，并额外报告唯一引用标签有效率。

### 14.5 人工引用核验

引用核验表至少包含：

```csv
question_id,question,citation,retrieved_files,exists_in_retrieval,claim_text,claim_supported,evidence,reviewer_notes
```

以下字段默认留空：

- `claim_supported`
- `evidence`
- `reviewer_notes`

只有人工审核后，才允许计算：

```text
引用支持率 = 有原文支持的引用结论数 / 已核验引用结论总数
```

未完成审核时，最终报告必须显示“待人工核验”，不能显示 0% 或 100%。

---

## 15. SSE 流式性能评测要求

### 15.1 调用方式

`benchmark_stream.py` 必须通过真实接口调用：

```text
POST http://localhost:9900/api/chat_stream
```

建议使用项目已有的 `httpx.AsyncClient.stream()`，不新增 SSE 第三方依赖。

### 15.2 计时方式

使用 `time.perf_counter()` 记录：

```text
request_start：发送请求前
first_content_at：收到第一个有效 content 事件
completed_at：收到 done 或连接结束
```

```text
TTFT = first_content_at - request_start
完整响应时间 = completed_at - request_start
```

### 15.3 成功判定

一次测试同时满足以下条件才记为成功：

1. HTTP 状态码为 200；
2. 至少收到一个非空 `content` 事件；
3. 收到 `done` 事件；
4. 未收到 `error` 事件；
5. 连接未超时。

### 15.4 会话隔离

每次测试必须使用唯一 `session_id`：

```text
eval-{run_id}-{question_id}-{repeat_index}
```

不得复用会话导致历史消息影响响应时间和答案内容。

### 15.5 冷启动标记

至少记录：

- `is_first_run_for_question`：是否为该问题第一次运行；
- `is_first_request_of_run`：是否为本次评测进程的第一个请求。

每道题第一次运行不等同于服务进程冷启动，因此两个字段不能混用。

### 15.6 默认执行方式

- 默认每题运行 3 次；
- 默认顺序执行，避免并发限流干扰；
- 命令行允许修改运行次数和超时；
- P50、P95 只基于成功请求计算；
- 成功率分母包含所有尝试。

---

## 16. 索引性能评测要求

### 16.1 默认 dry-run

默认模式只执行：

```text
读取文件
→ 统计字符数
→ 调用现有 DocumentSplitterService
→ 统计分片数
→ 记录分块耗时
```

dry-run 不得调用文档 Embedding，不得写入 Milvus。

### 16.2 正式索引模式

正式模式必须显式提供类似参数：

```powershell
python evaluation/benchmark_indexing.py `
  --input-dir uploads `
  --write-mode temporary `
  --confirm-external-calls
```

正式模式必须：

- 使用独立临时 collection；
- 记录 collection 名称；
- 复用现有文本分块和 Embedding 配置；
- 不调用生产代码中的 `delete_by_source()` 删除现有数据；
- 将创建、写入和清理结果写入报告；
- 清理失败时明确报告残留 collection 名称。

### 16.3 每万字符耗时

```text
每万字符索引耗时 = 完整索引耗时 / 字符数 × 10000
```

当字符数为 0 或未执行正式索引时，该指标返回 `null`。

---

## 17. 命令行要求

每个脚本必须支持 `--help`，并使用清晰的中文错误信息。

```powershell
python evaluation/inventory.py --help
python evaluation/evaluate_retrieval.py --questions evaluation/questions.csv
python evaluation/evaluate_deduplication.py --questions evaluation/questions.csv
python evaluation/evaluate_generation.py --questions evaluation/questions.csv
python evaluation/benchmark_stream.py --base-url http://localhost:9900 --runs 3
python evaluation/benchmark_indexing.py --input-dir uploads --dry-run
python evaluation/run_all.py --questions evaluation/questions.csv
```

建议 `run_all.py` 额外支持：

```text
--safe-only               只运行本地盘点、问题校验、单元测试和索引 dry-run
--allow-external-calls    允许 Embedding、Agent 和 SSE 调用
--skip-generation         跳过生成回答评测
--skip-stream             跳过流式性能评测
--fail-fast               任一阶段失败后立即停止
```

### 17.1 退出码

| 退出码 | 含义 |
|---:|---|
| 0 | 所有请求执行阶段成功 |
| 1 | 程序内部错误或部分阶段失败 |
| 2 | 命令行参数或配置错误 |
| 3 | Milvus、FastAPI 或模型服务不可用 |
| 4 | 问题集尚未人工标注，无法计算正式指标 |
| 5 | 操作需要用户授权 |

`run_all.py` 即使部分阶段失败，也应尽量保存已完成阶段的明细和最终失败报告。

---

## 18. 测试要求

### 18.1 单元测试

纯计算测试不得连接 DashScope 或 Milvus，至少覆盖：

- Windows 和 Unix 路径文件名规范化；
- 中文、英文、大小写、空格和下划线处理；
- Hit@1、Hit@3、Hit@5；
- Recall@3、Recall@5、Recall@10；
- 多个相关文件；
- 重复检索来源；
- 空检索结果；
- 空标准答案和 `NEEDS_REVIEW`；
- 来源覆盖数；
- 重复来源占比；
- 空数组、单元素和多元素 P50、P95；
- SSE 多行事件解析；
- 第一个 `content` 事件识别；
- `debug`、`tool_call` 等事件不计入 TTFT；
- `error`、超时和缺少 `done` 的失败判定。

### 18.2 最小验证命令

```powershell
python -m compileall evaluation tests/evaluation
pytest tests/evaluation -q
ruff check evaluation tests/evaluation
```

如果开发依赖未安装，必须说明未运行的检查项及安装方法，不能声称已经通过。

### 18.3 一致性测试

至少选取一个固定明细样例，验证：

- JSON 汇总与 CSV 明细重新计算一致；
- 去重前后结果数量符合定义；
- 未标注问题没有进入 Recall 分母；
- 失败请求进入成功率分母；
- 引用支持率只统计已经人工审核的行。

---

## 19. 日志与错误处理

### 19.1 日志要求

评测脚本使用 Loguru 记录：

- 阶段开始和结束；
- 输入文件；
- 有效问题数；
- 外部调用次数；
- 结果输出位置；
- 服务不可用或数据异常；
- 被跳过阶段及原因。

日志不得包含：

- API Key；
- `.env` 内容；
- HTTP Authorization Header；
- 患者隐私信息。

### 19.2 错误分类

建议使用稳定错误类型：

```text
CONFIG_ERROR
QUESTION_VALIDATION_ERROR
LABEL_NOT_REVIEWED
MILVUS_UNAVAILABLE
COLLECTION_NOT_FOUND
SCHEMA_MISMATCH
EMBEDDING_ERROR
AGENT_ERROR
HTTP_ERROR
SSE_PARSE_ERROR
TIMEOUT
INDEXING_ERROR
PERMISSION_REQUIRED
```

不得使用空异常捕获或 `except: pass` 隐藏错误。

---

## 20. 总报告要求

最终生成：

```text
evaluation/results/{run_id}/final_report.md
```

报告至少包含：

1. 运行时间、运行环境和配置摘要；
2. 问题集总数、有效标注数和跳过数；
3. 本地文件与 Milvus 知识库规模；
4. 疑似重复文件报告；
5. 检索指标；
6. 按问题类别分组的检索指标；
7. 去重前后对比；
8. 引用有效率；
9. 人工引用核验进度和支持率；
10. TTFT、完整响应时间和成功率；
11. 索引 dry-run 或正式索引结果；
12. 失败项、跳过项、限制和潜在偏差；
13. 可用于简历的候选描述。

每个阶段必须显示状态：

```text
completed
skipped
blocked
needs_review
failed
```

不能把 `skipped` 或 `blocked` 的指标显示为 0。

---

## 21. 简历指标使用要求

简历描述只能引用同一次正式评测报告中的真实结果，例如：

```text
构建包含 XX 篇规范化 AVF 论文、XX 个文本分片的医学文献知识库；
基于 XX 道人工标注问题进行检索评测，Hit@5 达到 XX%，Recall@5 达到 XX%。
```

```text
通过候选分片超额召回与论文级去重，将 Top-5 平均来源覆盖数
由 X.X 篇提升至 X.X 篇，重复来源占比下降 XX 个百分点。
```

使用前必须满足：

- 问题集已人工标注；
- 指标来自正式运行；
- 报告保留明细；
- 没有将 Hit@5 写成 Recall@5；
- 没有将 dry-run 耗时写成完整索引耗时；
- 没有将引用有效率写成引用支持率；
- 没有将科研原型描述为临床诊断系统。

---

## 22. 验收标准

评测模块通过验收需要同时满足：

### 22.1 功能验收

- 所有要求的脚本和结果文件均已创建；
- 所有脚本支持 `--help`；
- 本地盘点、检索、去重、生成、流式和索引模块可以独立运行；
- `run_all.py` 可以按顺序调度并生成总报告。

### 22.2 数据验收

- 明细 CSV 和汇总 JSON 一致；
- 文件名匹配基于规范化 `_file_name`；
- 空标准答案没有进入 Recall 分母；
- 重复论文没有被静默删除；
- 人工字段没有自动填入模型判断结果。

### 22.3 安全验收

- `.env` 未被修改；
- 日志和结果中没有 API Key；
- `uploads/` 和 `volumes/` 未被修改；
- 现有 `biz` collection 未被删除、重建或重复写入；
- 正式索引只使用授权的临时 collection。

### 22.4 质量验收

- 新增纯计算逻辑具有单元测试；
- Python 语法检查通过；
- 未运行的测试有明确说明；
- 服务不可用时返回合理非零退出码；
- 最终报告包含限制和偏差；
- 没有伪造实验结果、模型性能或医学结论。

---

## 23. 推荐实施顺序

为了降低数据风险和模型调用费用，建议按以下顺序实施：

```text
阶段一：common.py + 单元测试
        ↓
阶段二：本地 inventory + 索引 dry-run
        ↓
阶段三：Milvus 只读 inventory
        ↓
阶段四：生成候选问题并进行人工标注
        ↓
阶段五：检索与去重评测
        ↓
阶段六：用户确认外部调用量
        ↓
阶段七：生成回答与引用评测
        ↓
阶段八：SSE 性能评测
        ↓
阶段九：总报告和简历候选描述
```

每完成一个阶段，应立即运行对应测试并检查输出文件，再进入下一阶段。不得在问题集尚未人工确认时直接生成正式 Recall，也不得在用户未授权时运行批量 Agent 或真实索引写入。

---

## 24. 当前已知前置条件

正式评测开始前应确认：

1. Docker Desktop 已启动；
2. Milvus、MinIO 和 etcd 容器状态正常；
3. `biz` collection 存在且向量维度为 1024；
4. FastAPI 服务运行在 `http://localhost:9900`；
5. `/health` 返回成功；
6. DashScope API Key 已在本地 `.env` 中正确配置；
7. `127.0.0.1:7890` 代理可用；
8. `questions.csv` 已完成人工标注；
9. 用户已确认预计的 Agent 和 SSE 调用次数。

任何条件不满足时，应停止对应阶段、保存错误信息并在最终报告中标记为 `blocked`，不得生成替代性虚假结果。
