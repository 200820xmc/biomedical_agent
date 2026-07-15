# AVF RAG 项目评测模块

为 AVF 医学科研文献智能问答项目提供可重复运行的量化评测。

## 快速开始

```powershell
# 1. 安全模式（只读，不需授权）
python evaluation/run_all.py --safe-only

# 2. 完整评测（需先确认外部调用量）
python evaluation/run_all.py --allow-external-calls
```

## 脚本说明

| 脚本 | 功能 | 安全等级 |
|------|------|----------|
| `common.py` | 通用工具（文件名规范化、指标计算、CSV/JSON 读写） | — |
| `inventory.py` | 知识库规模统计 | Level 0+1 |
| `benchmark_indexing.py` | 索引性能测试（默认 dry-run） | Level 0 |
| `evaluate_retrieval.py` | 检索效果评测（Hit@K, Recall@K） | Level 1 |
| `evaluate_deduplication.py` | 去重策略对比评测 | Level 1 |
| `evaluate_generation.py` | 生成回答与引用评测 | Level 2 |
| `benchmark_stream.py` | SSE 流式性能评测 | Level 2 |
| `run_all.py` | 编排所有评测并生成最终报告 | — |

## 独立运行

```powershell
python evaluation/inventory.py --help
python evaluation/evaluate_retrieval.py --questions evaluation/questions.csv
python evaluation/evaluate_deduplication.py --questions evaluation/questions.csv
python evaluation/evaluate_generation.py --questions evaluation/questions.csv
python evaluation/benchmark_stream.py --base-url http://localhost:9900 --runs 3
python evaluation/benchmark_indexing.py --input-dir uploads --dry-run
python evaluation/run_all.py --questions evaluation/questions.csv
```

## 运行测试

```powershell
python -m pytest tests/evaluation/ -q -o "addopts=" -p no:cacheprovider
```

## 前置条件

- Docker Desktop 已启动
- Milvus 容器正常运行
- FastAPI 服务运行在 `http://localhost:9900`
- DashScope API Key 已在 `.env` 中配置
- 代理 `127.0.0.1:7890` 可用（如需访问 DashScope）

## 输出文件

每次运行在 `evaluation/results/{run_id}/` 下生成：

```
{run_id}/
├── run_metadata.json          # 运行元数据
├── inventory.json             # 知识库规模汇总
├── inventory_chunks.csv       # 分片明细
├── inventory_sources.csv      # 来源分片数
├── inventory_duplicates.csv   # 疑似重复文件
├── indexing_details.csv       # 索引 dry-run 明细
├── indexing_summary.json      # 索引汇总
├── retrieval_details.csv      # 检索明细
├── retrieval_summary.json     # 检索汇总
├── deduplication_details.csv  # 去重对比明细
├── deduplication_summary.json # 去重汇总
├── generation_answers.csv     # 生成回答
├── citation_review.csv        # 引用核验表（待人工审核）
├── generation_summary.json    # 生成汇总
├── stream_details.csv         # 流式性能明细
├── stream_summary.json        # 流式汇总
└── final_report.md            # 最终报告
```

## 安全原则

- Level 0（纯本地）：不连接任何外部服务
- Level 1（只读）：查询 Milvus、调用 Embedding，不写入
- Level 2（外部调用）：Agent/SSE 批量调用，需用户确认
- Level 3（写入）：正式索引写入，需临时 collection + 显式授权

所有脚本不修改 `.env`、`uploads/`、`volumes/`，不删除或重建现有 `biz` collection。
