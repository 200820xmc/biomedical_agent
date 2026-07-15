# Codex 执行任务：RAG 项目量化评测

## 1. 任务目标

为当前 AVF 医学科研文献智能问答项目建立一套可重复运行的评测流程，生成可用于简历和项目文档的真实量化指标。

必须完成以下指标：

1. 知识库规模：独立论文数、文本分片数、平均每篇分片数、平均分片长度。
2. 检索效果：Hit@1、Hit@3、Hit@5、Recall@3、Recall@5、Recall@10。
3. 去重策略效果：Top-5 平均来源覆盖数、重复来源占比、去重前后 Recall@5。
4. 引用质量：引用有效率、人工核验模板及引用支持率。
5. 流式性能：TTFT（首个 content 事件延迟）、完整响应时间、成功率、P50、P95。
6. 索引性能：单篇索引耗时、分片数、索引成功率、每万字符索引耗时。

所有结果必须来自实际运行，禁止编造或预填最终数值。

## 2. 工作原则

- 先阅读现有实现，复用当前配置、服务和向量存储，不另建重复业务逻辑。
- 不修改 `.env`，不输出 API Key 或其他敏感配置。
- 不删除或重建现有 Milvus collection，除非用户明确授权。
- 不覆盖 `uploads` 中的文献。
- 评测代码放在独立的 `evaluation/` 目录中。
- 每次评测保存明细 CSV 和汇总 JSON，保证结果可追溯。
- 文件名匹配应基于 Milvus 元数据 `_file_name`，并兼容斜杠、空格和大小写差异。
- 如果服务、Milvus 或模型不可用，应明确报告阻塞点，不得伪造结果。

## 3. 需要创建的目录和文件

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
```

如果 `questions.csv` 尚未具备可靠人工标注，不得自动虚构标准答案。此时应：

1. 从项目文献主题生成候选问题。
2. 将 `relevant_files` 留空或标记 `NEEDS_REVIEW`。
3. 提示用户人工确认后再计算正式 Recall@K。

## 4. 评测集格式

`questions.csv` 使用以下字段：

```csv
question_id,question,relevant_files,category,notes
q001,哪些研究使用深度学习识别动静脉瘘狭窄？,"论文A.md;论文B.md",深度学习,
```

要求：

- `question_id` 唯一。
- `relevant_files` 使用英文分号分隔。
- 一道问题可以对应多篇相关论文。
- 至少支持中文文件名。
- 空标准答案的问题不得计入 Recall 分母。

## 5. 指标定义

### 5.1 Hit@K

Top-K 返回论文中只要出现任意一篇标准相关论文，该问题记为 1，否则为 0。

```text
Hit@K = 命中至少一篇相关论文的问题数 / 有效问题总数
```

### 5.2 Recall@K

```text
Recall@K = Top-K 中命中的不同相关论文数 / 该问题的相关论文总数
```

整体结果取所有有效问题的宏平均。

### 5.3 来源覆盖数

```text
SourceCoverage@5 = Top-5 结果中不同 _file_name 的数量
```

### 5.4 重复来源占比

```text
DuplicateRatio@5 = 1 - 不同论文数 / 实际返回结果数
```

### 5.5 引用有效率

回答中的引用标签能够映射到本次检索结果，记为有效引用。

```text
引用有效率 = 有效引用数 / 回答总引用数
```

### 5.6 引用支持率

引用对应的原文能够支持引用前的结论，人工标记为 1。

```text
引用支持率 = 有原文支持的引用结论数 / 已核验引用结论总数
```

### 5.7 TTFT

```text
TTFT = HTTP 请求发出至首个 SSE content 事件到达的时间
```

`debug`、`tool_call` 和 `search_results` 事件不得作为首字事件。

## 6. 各脚本要求

### 6.1 `common.py`

提供：

- CSV 读取与校验。
- 文件名规范化。
- Milvus/VectorStore 获取方法。
- JSON、CSV 结果写入。
- 均值、P50、P95 统计。
- 统一时间戳和运行元数据。

### 6.2 `inventory.py`

只读统计：

- `uploads` 中 `.md`、`.txt` 文件数。
- 规范化文件名后的疑似重复文件。
- Milvus collection 实体数。
- `_file_name` 去重后的入库论文数。
- 平均每篇分片数。
- 分片字符长度的最小值、平均值、中位数、P95、最大值。

输出：

- `evaluation/results/inventory.json`
- `evaluation/results/inventory_chunks.csv`

不得删除重复文件，只生成重复项报告。

### 6.3 `evaluate_retrieval.py`

绕过 Agent，直接调用当前向量存储的相似度检索。

分别测试 K=1、3、5、10，计算：

- Hit@K
- Recall@K
- 每题检索结果及命中文件
- 按 `category` 分组的指标

输出：

- `evaluation/results/retrieval_details.csv`
- `evaluation/results/retrieval_summary.json`

### 6.4 `evaluate_deduplication.py`

实现并比较：

- 基线：直接返回 Top-5 分片。
- 优化：召回 Top-15 分片，按 `_file_name` 去重后保留 Top-5。

统计：

- Recall@5
- Hit@5
- 平均来源覆盖数
- 平均重复来源占比
- 每道问题去重前后变化

输出：

- `evaluation/results/deduplication_details.csv`
- `evaluation/results/deduplication_summary.json`

### 6.5 `evaluate_generation.py`

批量调用完整 RAG 问答流程，记录：

- 问题、回答、检索论文、回答引用。
- 引用是否能映射到检索论文。
- 错误信息和耗时。

输出可人工填写的核验表：

```csv
question_id,question,citation,retrieved_files,exists_in_retrieval,claim_text,claim_supported,evidence,reviewer_notes
```

其中 `claim_supported`、`evidence`、`reviewer_notes` 默认留空，等待人工审核。脚本只能自动计算引用有效率，不得自动宣称引用支持率。

输出：

- `evaluation/results/generation_answers.csv`
- `evaluation/results/citation_review.csv`
- `evaluation/results/generation_summary.json`

### 6.6 `benchmark_stream.py`

通过 `/api/chat_stream` 实际测试 SSE：

- 默认读取 `questions.csv`。
- 每题运行 3 次，支持命令行调整次数。
- 记录 TTFT、完整响应时间、是否成功、错误类型。
- 单独标记每题第一次运行，避免将冷启动与稳定状态混淆。
- 计算平均值、P50、P95、成功率。

输出：

- `evaluation/results/stream_details.csv`
- `evaluation/results/stream_summary.json`

### 6.7 `benchmark_indexing.py`

索引测试存在写入 Milvus 和覆盖数据的风险，因此默认只提供安全模式：

- `--dry-run`：只完成文件读取和分块统计。
- 真正写入测试必须使用独立临时 collection，或获得用户明确授权。
- 不得默认对现有 `biz` collection 重复写入。

记录：

- 文件大小、字符数、分片数。
- 分块耗时。
- 完整索引耗时（仅在授权执行时）。
- 成功状态和错误信息。

输出：

- `evaluation/results/indexing_details.csv`
- `evaluation/results/indexing_summary.json`

### 6.8 `run_all.py`

按以下顺序运行：

1. 环境和依赖检查。
2. 知识库规模统计。
3. 评测集校验。
4. 检索评估。
5. 去重对比。
6. 生成回答与引用核验表。
7. 流式性能测试。
8. 索引 dry-run。
9. 生成总报告。

如果评测集尚未人工标注，应跳过正式 Recall 计算并给出明确提示。

## 7. 命令行接口

每个脚本都应支持：

```powershell
python evaluation/inventory.py --help
python evaluation/evaluate_retrieval.py --questions evaluation/questions.csv
python evaluation/evaluate_deduplication.py --questions evaluation/questions.csv
python evaluation/evaluate_generation.py --questions evaluation/questions.csv
python evaluation/benchmark_stream.py --base-url http://localhost:9900 --runs 3
python evaluation/benchmark_indexing.py --input-dir uploads --dry-run
python evaluation/run_all.py --questions evaluation/questions.csv
```

脚本应返回合理退出码：成功为 `0`，配置或服务不可用为非零。

## 8. 测试要求

为不依赖真实 DashScope 和 Milvus 的纯计算逻辑添加单元测试，至少覆盖：

- 文件名规范化。
- Hit@K。
- Recall@K。
- 来源覆盖数。
- 重复来源占比。
- 空结果与空标准答案。
- P50、P95。
- SSE 事件中首个 `content` 的识别。

运行项目已有代码检查工具；如果仓库没有可运行测试，应至少运行新增评测测试和 Python 语法检查。

## 9. 总报告

生成：

```text
evaluation/results/final_report.md
```

报告必须包含：

1. 运行时间和环境信息。
2. 数据集规模。
3. 知识库规模。
4. 检索指标表。
5. 去重前后对比表。
6. 引用有效率及人工核验进度。
7. TTFT 和完整响应时间。
8. 索引 dry-run 或正式测试结果。
9. 失败项、限制和可能的偏差。
10. 可直接用于简历的候选描述。

简历候选描述只能引用报告中的真实结果，例如：

```text
构建包含 XX 篇 AVF 论文、XX 个文本分片的医学知识库；基于 XX 道人工标注问题进行评测，Top-5 论文命中率达到 XX%。

通过候选分片超额召回与论文级去重，使 Top-5 平均来源覆盖由 X.X 篇提升至 X.X 篇，重复来源占比下降 XX 个百分点。
```

## 10. 验收标准

- 所有新增文件仅位于 `evaluation/` 和对应测试目录。
- 不泄露 `.env` 或 API Key。
- 不破坏现有 API、知识库和上传文献。
- 指标公式与明细结果一致，可从明细重新计算。
- 无人工标准答案时不生成虚假的 Recall。
- 引用支持率必须经过人工核验后才能进入最终报告。
- 所有脚本有 `--help`，错误信息清晰。
- 最终汇报列出创建文件、执行命令、实际结果和未完成原因。

## 11. Codex 开始执行时的第一步

先检查以下实现和数据结构，再开始编码：

- `app/services/vector_store_manager.py`
- `app/services/vector_index_service.py`
- `app/services/document_splitter_service.py`
- `app/services/rag_agent_service.py`
- `app/tools/knowledge_tool.py`
- `app/api/chat.py`
- `app/core/milvus_client.py`
- `uploads/`

然后给出简短实施计划，创建评测脚本，运行安全的只读评测。涉及真实索引写入、外部模型大量调用或其他可能产生费用的操作时，先说明预计影响并等待用户确认。
