# 动静脉瘘狭窄分类科研助手 — 需求文档

> **Legacy历史需求**：本文记录2026-07-07的项目迁移与早期产品设想，其中 `aiops-docs`、Plan-Execute-Replan、旧MCP目录和知识库规划不代表当前实现。当前项目入口与事实请以根目录 `README.md`、`CLAUDE.md` 和 `docs/技术文档.md` 为准。

> 版本: v1.0
> 日期: 2026-07-07
> 状态: Legacy，保留项目演进记录

---

## 1. 项目背景

### 1.1 现状

课题组已有 `super_biz_agent_py` 项目，是一个基于 LangChain + LangGraph 的智能 OnCall 运维系统，具备 RAG 知识库问答和 AIOps Plan-Execute-Replan 自动故障诊断能力。

### 1.2 目标

将该项目的通用框架迁移到医学影像科研场景，构建一个**动静脉瘘（AVF）狭窄深度学习分类**课题的科研助手，帮助新人（师弟）快速入门该方向。

### 1.3 核心价值

- 新人无需逐篇查找论文、手动梳理文献，Agent 自动完成
- 课题组知识资产（论文笔记、实验记录、最佳实践）可沉淀、可检索
- Plan-Execute-Replan 模式让 Agent 像资深研究员一样系统性地调研问题

---

## 2. 用户画像与使用场景

### 2.1 用户：刚进组的师弟/师妹

| 属性 | 描述 |
|------|------|
| 身份 | 研究生一年级，刚接触 AVF 狭窄分类课题 |
| 医学背景 | 了解动静脉瘘基本概念，不熟悉影像分类文献 |
| 编程水平 | 会用 Python，了解深度学习基础概念，未做过完整项目 |
| 核心痛点 | 不知道从哪开始、不知道读哪些论文、不清楚主流方法有哪些 |

### 2.2 使用场景

| 编号 | 场景 | 典型问题 |
|------|------|---------|
| S1 | 课题入门 | "AVF狭窄用什么深度学习方法来分类？有哪些主流方向？" |
| S2 | 论文推荐 | "最近3年AVF狭窄分类有哪些重要论文？帮我列10篇必读的" |
| S3 | 数据集选择 | "做AVF狭窄分类一般用哪些数据集？超声图像数据怎么预处理？" |
| S4 | 实验设计 | "我想用ResNet做AVF狭窄二分类，实验应该怎么设计？" |
| S5 | 方法对比 | "CNN和Transformer在AVF分类上哪个效果更好？" |
| S6 | 文献综述 | "帮我写一份AVF狭窄分类的文献综述" |

---

## 3. 功能需求

### 3.1 RAG 知识库对话 Agent

**描述**：用户通过自然语言提问，Agent 从知识库中检索相关内容，结合大模型生成回答。

**输入**：自然语言问题
**输出**：基于知识库的流式回答（SSE）

**功能要求**：

| ID | 要求 | 优先级 |
|----|------|--------|
| F1.1 | 支持多轮对话，记住上下文 | P0 |
| F1.2 | 流式输出答案 | P0 |
| F1.3 | 能从知识库检索相关文档片段（Top-K） | P0 |
| F1.4 | 回答需标注信息来源（引用知识库文档） | P1 |
| F1.5 | 无相关知识时诚实告知，不编造 | P0 |

### 3.2 文献搜索 Agent（Plan-Execute-Replan）

**描述**：用户提出问题后，Agent 自动制定调研计划、搜索论文、评估结果、生成结构化报告。

**输入**：研究问题（如"帮我调研AVF狭窄的深度学习分类方法"）
**输出**：流式生成的结构化研究报告（SSE）

**功能要求**：

| ID | 要求 | 优先级 |
|----|------|--------|
| F2.1 | 自动制定调研计划（3-5步） | P0 |
| F2.2 | 调用 PubMed 检索论文 | P0 |
| F2.3 | 调用知识库检索相关内容 | P0 |
| F2.4 | 评估信息是否充分，动态调整计划 | P0 |
| F2.5 | 生成结构化 Markdown 研究报告 | P0 |
| F2.6 | 报告含：研究背景、文献汇总、方法对比、推荐阅读清单 | P1 |
| F2.7 | 防止无限循环（最多8步强制结束） | P1 |
| F2.8 | 流式展示执行过程（让用户看到"正在搜论文→正在分析→正在写报告"） | P1 |

### 3.3 知识库管理

**描述**：支持文档上传、自动索引、增量更新知识库。

**功能要求**：

| ID | 要求 | 优先级 |
|----|------|--------|
| F3.1 | 支持 Markdown 文档上传 | P0 |
| F3.2 | 自动文本分块 + 向量化 + 存入 Milvus | P0 |
| F3.3 | 支持增量更新（新增/修改文档自动重建索引） | P1 |
| F3.4 | 预置 5 篇 AVF 狭窄分类核心知识文档 | P0 |

### 3.4 Web 界面

**描述**：简洁的 Web 聊天界面，支持普通对话和文献分析两种模式。

**功能要求**：

| ID | 要求 | 优先级 |
|----|------|--------|
| F4.1 | 对话模式：快速问答 | P0 |
| F4.2 | 分析模式：文献综述调研 | P0 |
| F4.3 | 流式显示生成内容 | P0 |
| F4.4 | 展示调研过程（计划→步骤执行→报告生成） | P1 |
| F4.5 | 品牌风格：科研学术风 | P2 |

---

## 4. 非功能需求

| ID | 要求 | 说明 |
|----|------|------|
| NF1 | 响应时间 | 普通对话首次Token < 3秒 |
| NF2 | 并发支持 | 支持至少 3 个用户同时使用 |
| NF3 | 运行环境 | Windows 本地运行，无需 GPU |
| NF4 | 资源占用 | 内存 < 8GB（含 Milvus Docker） |
| NF5 | 可维护性 | 知识库文档用 Markdown，无需改代码即可更新知识 |
| NF6 | 安全性 | 本地运行，无外部 API 调用（除 PubMed，免费） |

---

## 5. 技术架构

### 5.1 整体架构图

```
┌────────────────────────────────────────────────┐
│                   用户浏览器                      │
│               http://localhost:9900             │
└──────────────────────┬─────────────────────────┘
                       │
┌──────────────────────▼─────────────────────────┐
│              FastAPI 主服务 (:9900)              │
│                                                  │
│  /api/chat        → 对话 Agent（流式）            │
│  /api/research    → 文献分析 Agent（流式）        │
│  /api/upload      → 知识库文档上传               │
│  /api/health      → 健康检查                     │
└──────┬───────────────────────────┬──────────────┘
       │                           │
┌──────▼──────────┐    ┌───────────▼──────────────┐
│  RAG Agent      │    │  Research Agent           │
│  (对话)         │    │  (Plan-Execute-Replan)    │
│                 │    │                           │
│  工具:          │    │  工具:                    │
│  - 知识库检索   │    │  - 知识库检索             │
│  - 时间工具     │    │  - PubMed 论文搜索         │
│                 │    │  - 时间工具               │
└──────┬──────────┘    └───────────┬──────────────┘
       │                           │
       └───────────┬───────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│                  共享基础层                       │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
│  │ Milvus   │  │ PubMed   │  │ 阿里千问 LLM  │  │
│  │ 向量数据库│  │ Entrez   │  │ (DashScope)   │  │
│  └──────────┘  └──────────┘  └───────────────┘  │
└─────────────────────────────────────────────────┘
```

### 5.2 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | 继承现有 |
| LLM | 阿里云 DashScope (千问) | 继承现有 |
| Agent 框架 | LangChain + LangGraph | 继承现有 |
| 向量数据库 | Milvus (Docker) | 继承现有 |
| 协议 | MCP (Model Context Protocol) | 继承现有 |
| 论文检索 | NCBI Entrez API | 新增，免费 |
| 前端 | 原生 HTML/JS/CSS | 继承现有，微调 |

### 5.3 关键依赖关系

```
LangGraph 工作流
    ├── Planner   ← 制定调研计划
    ├── Executor  ← 执行步骤（调工具）
    └── Replanner ← 评估→决定继续/调整/生成报告

工具集
    ├── retrieve_knowledge   ← 查 Milvus 向量库（已有）
    ├── get_current_time      ← 时间工具（已有）
    └── search_pubmed         ← PubMed 搜索（新增）
```

---

## 6. 迁移范围

### 6.1 复用（不改）

| 文件 | 说明 |
|------|------|
| `app/main.py` | FastAPI 入口 |
| `app/api/chat.py` | 对话接口 |
| `app/api/file.py` | 文档上传接口 |
| `app/api/health.py` | 健康检查 |
| `app/services/rag_agent_service.py` | RAG Agent（改提示词） |
| `app/services/vector_*.py` | 向量存储全套服务 |
| `app/core/*.py` | LLM 工厂 + Milvus 客户端 |
| `app/agent/mcp_client.py` | MCP 客户端 |
| `app/agent/aiops/state.py` | 通用状态定义 |
| `app/agent/aiops/utils.py` | 工具格式化 |
| `app/agent/aiops/executor.py` | 通用执行器 |
| `app/utils/logger.py` | 日志配置 |
| `app/models/request.py` `response.py` `document.py` | 通用模型 |
| `static/index.html` | Web 界面（改文案） |
| `vector-database.yml` `pyproject.toml` `uv.lock` | 配置 |
| `mcp_servers/` | MCP 服务（先保留可正常启动，后续按需扩展 PubMed MCP） |

### 6.2 修改（微调）

| 文件 | 改动 |
|------|------|
| `app/services/rag_agent_service.py` | 系统提示词：AI助手 → AVF 科研助手 |
| `app/services/aiops_service.py` | `diagnose()` 任务模板：告警报告 → 文献综述报告 |
| `app/agent/aiops/planner.py` | 提示词示例：运维场景 → 科研场景 |
| `app/agent/aiops/replanner.py` | 提示词：保持通用，微调场景描述 |
| `app/agent/aiops/executor.py` | 系统提示词：运维助手 → 科研助手 |
| `app/tools/__init__.py` | 工具集：去掉 `query_prometheus_alerts`，加入 `search_pubmed` |
| `app/config.py` | MCP 配置改名：`cls`/`monitor` → 科研MCP名 |
| `app/models/aiops.py` | 类名：`AlertInfo` → `ResearchTask` |
| `app/main.py` | 描述文案 |
| `static/index.html` | 标题、按钮、loading 文案 |
| `static/app.js` | SSE 事件文案、错误提示 |
| `static/styles.css` | 类名 `.aiops-*` → `.research-*` |

### 6.3 新增

| 文件 | 说明 |
|------|------|
| `app/tools/paper_search_tool.py` | PubMed 论文检索工具 |
| `aiops-docs/avf_overview.md` | AVF 狭窄基础知识 |
| `aiops-docs/dl_classification_methods.md` | 深度学习分类方法汇总（含模型性能对比表） |
| `aiops-docs/dataset_and_preprocessing.md` | 常用数据集与预处理指南 |
| `aiops-docs/experiment_design_guide.md` | 实验设计模板与 checklist |
| `aiops-docs/recommended_papers.md` | 推荐论文清单（含摘要和阅读建议） |

### 6.4 删除

| 文件 | 原因 |
|------|------|
| `app/tools/query_metrics_alerts.py` | Prometheus 告警查询，运维专用 |
| `aiops-docs/cpu_high_usage.md` | CPU 告警 SOP |
| `aiops-docs/memory_high_usage.md` | 内存告警 SOP |
| `aiops-docs/disk_high_usage.md` | 磁盘告警 SOP |
| `aiops-docs/service_unavailable.md` | 服务不可用 SOP |
| `aiops-docs/slow_response.md` | 响应慢 SOP |

---

## 7. 知识库文档规划

### 7.1 文档清单

| 序号 | 文件名 | 内容 | 字数目标 |
|------|--------|------|---------|
| 1 | `avf_overview.md` | AVF 定义、狭窄病因、分型、临床意义、超声诊断价值 | ~2000字 |
| 2 | `dl_classification_methods.md` | CNN/Transformer/混合方法汇总，含代表性论文和性能对比表 | ~2500字 |
| 3 | `dataset_and_preprocessing.md` | 常用超声数据集、数据增强方法、预处理 pipeline | ~2000字 |
| 4 | `experiment_design_guide.md` | 实验设计模板、评价指标详解、常见坑与最佳实践 | ~2000字 |
| 5 | `recommended_papers.md` | 10-15篇推荐论文清单（综述+经典+前沿），含摘要和阅读建议 | ~3000字 |

### 7.2 文档结构要求

每篇文档需包含：
- **标题层级分明**（`#` `##` `###`），方便 LLM 和向量检索理解结构
- **具体的数据和指标**（避免模糊描述，如"效果很好"→写"准确率92.3%"）
- **论文引用格式统一**（作者. 标题. 期刊, 年份）
- **避免占位符**（不要 "xxx@company.com" 这类运维模板遗留）

---

## 8. 实施计划

### 8.1 分阶段交付

```
Phase 1  ▌知识库 + 对话 Agent            预计 1.5h
         │  写5篇知识库文档 → 替换 aiops-docs/
         │  改 RAG Agent 系统提示词 → 重启
         │  验证: "什么是AVF狭窄？" 能正确回答
         │
Phase 2  ▌PubMed 工具 + 文献分析 Agent    预计 2h
         │  写 search_pubmed 工具代码
         │  改 Planner/Replanner/aiops_service 提示词
         │  验证: "帮我调研AVF分类方法" 能自动搜论文出报告
         │
Phase 3  ▌前端 + 配置适配                预计 1h
         │  改前端文案和样式
         │  改 config.py 和模型文件
         │  验证: 全流程跑通
         │
Phase 4  ▌测试 + 优化                    预计 1h
         │  端到端测试
         │  调优提示词
         │  补充知识库文档
```

### 8.2 验收标准

| Phase | 验收条件 |
|-------|---------|
| Phase 1 | 问"什么是AVF狭窄"，Agent 回答引用知识库内容，不编造 |
| Phase 2 | 问"调研AVF分类方法"，Agent 自动搜索 PubMed + 生成结构化报告 |
| Phase 3 | Web 界面显示"AVF科研助手"，两种模式可正常切换 |
| Phase 4 | 5种典型问题（见 2.2 节）均能给出合理回答 |

---

## 9. 风险与约束

| 风险 | 影响 | 应对 |
|------|------|------|
| PubMed API 限流 | Agent 短时间内大量请求被拒 | 工具内加 1秒间隔，单次查询 max 5篇 |
| LLM 编造论文 | 引用不存在的论文 | 系统提示词强调"只引用实际检索到的论文" |
| 知识库不够完善 | 回答质量差 | Phase 1 先写核心5篇，后续持续补充 |
| 阿里云 API 费用 | DashScope 按量计费 | 使用 qwen-max 即可，调研阶段成本很低 |

---

## 10. 附录

### A. 与运维版本的差异总结

| 维度 | 运维版 | 科研版 |
|------|--------|--------|
| 核心任务 | 告警诊断 | 文献调研 |
| 对外工具 | 查日志/监控/Prometheus | 查 PubMed/知识库 |
| 输出产物 | 诊断报告 | 文献综述报告 |
| 知识库 | 运维 SOP | 医学科研文档 |
| 用户 | OnCall 工程师 | 研究生/科研人员 |
| 品牌 | 智能OnCall助手 | AVF 科研助手 |

### B. 参考资料

- [NCBI Entrez API 文档](https://www.ncbi.nlm.nih.gov/books/NBK25501/)
- [LangGraph Plan-Execute 教程](https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/)
- [LangChain PubMed 集成](https://python.langchain.com/docs/integrations/tools/pubmed/)
