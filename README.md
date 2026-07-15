# 🔬 AVF Research Assistant

> 动静脉瘘（AVF）狭窄深度学习科研助手 — 基于 RAG 知识库的论文检索与智能问答系统

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1+-orange.svg)](https://www.langchain.com/)
[![Milvus](https://img.shields.io/badge/Milvus-2.5+-brightgreen.svg)](https://milvus.io/)

## 📖 项目简介

AVF Research Assistant 是一个面向医学影像与生物医学工程领域的**智能文献检索与问答系统**。基于 RAG（检索增强生成）架构，能够：

- 🔍 自动检索知识库中的学术论文
- 💬 基于论文内容进行自然语言问答
- 📄 支持 Markdown 论文上传与自动向量化索引
- 🌐 提供 Web 对话界面和 RESTful API

### 应用场景

- **科研调研**：快速检索 AVF 狭窄检测、血流声学、深度学习分类等相关论文
- **文献综述**：自动汇总多篇论文的方法、模型和性能指标
- **知识管理**：构建个人/课题组的论文知识库

## 🏗️ 系统架构

```
┌─────────────────────────────────┐
│   Frontend (Vanilla JS + SSE)   │  ← Web 对话界面
├─────────────────────────────────┤
│   API Layer (FastAPI)           │  ← RESTful + SSE 流式
│   /api/chat  /api/chat_stream   │
│   /api/upload  /health          │
├─────────────────────────────────┤
│   RAG Agent (LangGraph)         │  ← 对话引擎 + 工具调用
│   ChatQwen (通义千问 qwen-max)   │
├─────────────────────────────────┤
│   Knowledge Base (Milvus)       │  ← 向量存储 + 相似检索
│   Embedding: text-embedding-v4  │
└─────────────────────────────────┘
```

## 🚀 快速开始

### 环境要求

- Python 3.11+
- Docker Desktop
- 阿里云 DashScope API Key（[免费申请](https://bailian.console.aliyun.com/)）

### 1. 克隆项目

```bash
git clone https://github.com/yourusername/avf-research-assistant.git
cd avf-research-assistant
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 3. 安装依赖

```bash
pip install -e .
```

### 4. 启动 Milvus 向量数据库

```bash
docker compose -f vector-database.yml up -d
```

### 5. 启动服务

```bash
python run_server.py
```

访问 **http://localhost:9900** 即可使用。

## 📡 API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/health` | 健康检查 |
| `POST` | `/api/chat` | 非流式对话 |
| `POST` | `/api/chat_stream` | 流式对话（SSE） |
| `POST` | `/api/upload` | 上传论文（.md/.txt） |
| `GET` | `/` | Web 对话界面 |
| `GET` | `/docs` | Swagger API 文档 |

### 调用示例

```python
import requests

# 上传论文
with open("paper.md", "rb") as f:
    requests.post("http://localhost:9900/api/upload", files={"file": f})

# 提问
resp = requests.post("http://localhost:9900/api/chat", json={
    "Id": "session-1",
    "Question": "AVF狭窄的深度学习方法有哪些？"
})
print(resp.json()["data"]["answer"])
```

## 📁 项目结构

```
├── app/
│   ├── api/             # API 路由（chat, file, health）
│   ├── agent/           # Agent 管理（MCP 客户端）
│   ├── core/            # 核心组件（LLM 工厂、Milvus 客户端）
│   ├── models/          # Pydantic 数据模型
│   ├── services/        # 业务服务（RAG、向量存储、文档分割）
│   ├── tools/           # Agent 工具（知识检索、时间）
│   └── utils/           # 工具函数（日志）
├── static/              # 前端静态文件
├── scripts/             # 辅助脚本（PDF 转换、批量上传）
├── vector-database.yml  # Milvus Docker Compose
├── run_server.py        # 启动脚本
├── pyproject.toml       # 项目配置
└── .env.example         # 环境变量模板
```

## 🔧 技术栈

| 组件 | 技术 | 说明 |
|------|------|------|
| Web 框架 | FastAPI | 高性能异步 API |
| LLM | 通义千问 qwen-max | 阿里云 DashScope |
| Agent 框架 | LangChain + LangGraph | Agent 编排与工具调用 |
| 向量数据库 | Milvus 2.5 | 论文向量存储与检索 |
| 嵌入模型 | text-embedding-v4 | 1024 维文本向量化 |
| 前端 | Vanilla JS + SSE | 流式对话实时渲染 |

## 📄 许可证

MIT License
