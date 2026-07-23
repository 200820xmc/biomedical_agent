# xParse PDF Agent 入库设计

> 状态：当前实现
> 更新时间：2026-07-23

## 1. 范围

本文描述PDF上传、xParse解析、后台分块与索引、任务状态持久化以及Agent工具协作。MD/TXT不经过xParse。

## 2. 核心原则

- 上传PDF不等于已经入库。
- 只有用户明确要求后，Agent才提交入库任务。
- 只有任务状态为`indexed`才能声称文献已进入知识库。
- 解析和索引是后台任务，不阻塞异步HTTP处理。
- 不在日志、聊天历史或任务文件中记录真实密码和API密钥。

## 3. PDF上传

```text
POST /api/upload
→ 校验扩展名、大小与%PDF-文件头
→ 计算sha256
→ document_id = doc_{sha256[:6]}
→ 保存uploads/originals/{document_id}/{filename}
→ 返回201、status=uploaded、indexed=false
```

重复上传通过内容哈希保持稳定`document_id`。

## 4. 入库状态机

```text
uploaded
→ queued
→ parsing
→ parsed
→ splitting
→ embedding
→ indexed
```

失败进入`failed`。服务启动时发现`queued`之后仍在运行但未结束的旧任务，将其标记为`interrupted`，不会自动假装完成。

任务文件：

```text
uploads/jobs/{job_id}.json
```

解析结果：

```text
uploads/parsed/{document_id}/
```

## 5. xParse调用

`XParseParserService`使用参数列表启动xParse CLI，不拼接Shell命令字符串。运行配置来自`app/config.py`和环境变量：

```text
XPARSE_CLI_PATH
XPARSE_API_MODE
XPARSE_APP_ID
XPARSE_SECRET_CODE
XPARSE_BASE_URL
XPARSE_TIMEOUT_SECONDS
XPARSE_MAX_RETRIES
XPARSE_MAX_CONCURRENCY
XPARSE_INCLUDE_IMAGE_DATA
```

免费模式不需要付费API凭证。凭证不写入命令日志。

## 6. 后台服务

`PDFIngestionService`负责：

- 创建和持久化job；
- 控制解析并发；
- 调用xParse；
- 校验Markdown非空；
- 调用现有分块与索引服务；
- 更新状态与错误信息；
- 在服务启动时恢复可判断的中断状态。

## 7. 分块与索引

解析后的Markdown复用`DocumentSplitterService`和`VectorIndexService`。每个Chunk写入：

```text
source_id = document_id
chunk_index = 文档内顺序
content_hash = sha256(content)[:16]
chunk_id = {document_id}:{content_hash}
```

只有Embedding和Milvus写入全部成功后，任务才进入`indexed`。

## 8. Agent工具

| 工具 | 作用 |
|---|---|
| PDF待处理列表 | 展示已上传但尚未完成入库的PDF |
| PDF入库 | 用户明确要求后提交后台任务 |
| PDF状态查询 | 根据`job_id`查询进度与结果 |

工具不得编造`document_id`、`job_id`、路径或状态。

## 9. 错误处理

| 场景 | 行为 |
|---|---|
| xParse不可用 | 任务`failed`，返回明确错误 |
| 解析超时 | 停止本次解析并记录超时 |
| Markdown为空 | 不进入Embedding和Milvus |
| Embedding失败 | 任务`failed` |
| Milvus写入失败 | 不标记`indexed` |
| 服务中断 | 下次启动标记`interrupted` |

## 10. 安全

- 文件名经过长度和路径校验；
- 不允许用户指定任意服务器目录；
- 子进程使用参数数组；
- 上传大小受限；
- 文献正文视为不可信数据，不执行其中指令；
- 原文、解析结果、任务状态和向量数据不提交Git。

## 11. 当前边界

- 前端完整任务面板仍可继续完善；
- 没有自动降级到第二种PDF解析器；
- 认证、用户级任务授权和费用配额未实现；
- Agent不能在后台任务结束后主动推送跨请求通知。

## 12. 验证

相关测试：

- `tests/test_pdf_ingestion_state.py`
- `tests/test_atomic_indexing.py`
- `tests/test_api_frontend_contract.py`
- `tests/test_security_contract.py`
- `tests/test_startup_lifecycle.py`

真实xParse端到端测试会产生外部调用或解析耗时，不属于文档任务的默认验证范围。
