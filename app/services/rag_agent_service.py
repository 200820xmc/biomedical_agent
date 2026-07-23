"""RAG Agent 服务 - 基于 LangGraph 的智能代理

使用 langchain_qwq 的 ChatQwen 原生集成，
支持真正的流式输出和更好的模型适配。
"""

from collections.abc import Callable, Sequence
from typing import Annotated, Any, AsyncGenerator, Dict

from langchain.agents import create_agent
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from loguru import logger
from typing_extensions import TypedDict
from langchain_qwq import ChatQwen

from app.config import config
from app.utils.logger import describe_text, format_exception_chain

# 阿里千问大模型和langchain集成参考： https://docs.langchain.com/oss/python/integrations/chat/qwen
# 外部服务地址和凭据均由运行环境提供，代码不覆写相关环境变量。


class AgentState(TypedDict):
    """Agent 状态"""
    messages: Annotated[Sequence[BaseMessage], add_messages]


def trim_messages_middleware(state: AgentState) -> dict[str, Any] | None:
    """
    修剪消息历史，只保留最近的几条消息以适应上下文窗口

    策略：
    - 保留第一条系统消息（System Message）
    - 保留最近的 6 条消息（3 轮对话）
    - 当消息少于等于 7 条时，不做修剪

    Args:
        state: Agent 状态

    Returns:
        包含修剪后消息的字典，如果无需修剪则返回 None
    """
    messages = state["messages"]

    # 如果消息数量较少，无需修剪
    if len(messages) <= 7:
        return None

    # 提取第一条系统消息
    first_msg = messages[0]

    # 保留最近的 6 条消息（确保包含完整的对话轮次）
    recent_messages = messages[-6:] if len(messages) % 2 == 0 else messages[-7:]

    # 构建新的消息列表
    new_messages = [first_msg] + list(recent_messages)

    logger.debug(f"修剪消息历史: {len(messages)} -> {len(new_messages)} 条")

    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            *new_messages
        ]
    }


class RagAgentService:
    """RAG Agent 服务 - 使用 LangGraph + ChatQwen 原生集成"""

    def __init__(
        self,
        streaming: bool = True,
        model_factory: Callable[..., Any] = ChatQwen,
        tools: Sequence[Any] | None = None,
        checkpointer_factory: Callable[[], Any] = MemorySaver,
    ):
        """初始化 RAG Agent 服务

        Args:
            streaming: 是否启用流式输出，默认为 True
        """
        self.model_name = config.rag_model
        self.streaming = streaming
        self.system_prompt = self._build_system_prompt()


        self._model_factory = model_factory
        self._configured_tools = list(tools) if tools is not None else None
        self._checkpointer_factory = checkpointer_factory
        self.model: Any | None = None
        self.tools: list[Any] = []

        # 创建内存检查点（用于会话管理）
        self.checkpointer: Any | None = None

        # Agent 初始化（会在异步方法中完成）
        self.agent = None
        self._agent_initialized = False

        logger.info(
            f"RAG Agent 服务已创建（惰性初始化）, model={self.model_name}, "
            f"streaming={streaming}"
        )

    @property
    def is_initialized(self) -> bool:
        return self.model is not None and self.checkpointer is not None

    def initialize(self) -> None:
        """显式创建模型、工具和会话存储；模块导入阶段不执行。"""
        if self.is_initialized:
            return

        if self._configured_tools is None:
            from app.tools import DEFAULT_LOCAL_AGENT_TOOLS

            self.tools = list(DEFAULT_LOCAL_AGENT_TOOLS)
        else:
            self.tools = list(self._configured_tools)

        self.model = self._model_factory(
            model=self.model_name,
            api_key=config.dashscope_api_key,
            temperature=0.7,
            streaming=self.streaming,
        )
        self.checkpointer = self._checkpointer_factory()
        logger.info(
            f"RAG Agent 资源初始化完成, model={self.model_name}, "
            f"tools={len(self.tools)}"
        )

    async def _initialize_agent(self):
        """异步初始化仅包含生产链路已启用的本地工具的 Agent。"""
        if self._agent_initialized:
            return

        self.initialize()

        all_tools = self.tools

        self.agent = create_agent(
            self.model,
            tools=all_tools,
            checkpointer=self.checkpointer,
        )

        self._agent_initialized = True


        if all_tools:
            tool_names = [tool.name if hasattr(tool, "name") else str(tool) for tool in all_tools]
            logger.info(f"可用工具列表: {', '.join(tool_names)}")

    def _build_system_prompt(self) -> str:
        """
        构建系统提示词

        注意：LangChain 框架会自动将工具信息传递给 LLM，
        因此系统提示词中无需列举具体的工具列表。

        Returns:
            str: 系统提示词
        """
        from textwrap import dedent

        return dedent("""
            你是一个专业的AVF（动静脉瘘）科研助手，能够基于知识库中的学术论文为用户提供文献调研和科研分析。

            工作原则:
            1. 理解用户需求，使用知识检索工具查找相关论文
            2. 引用论文内容时必须标注引用标签，格式为 (Author et al. Year)，例如 (Zhou et al. 2023)
            3. 引用标签由知识检索工具自动生成，直接使用即可，不要自己编造引用
            4. 如果工具无法提供足够信息，请诚实地告知用户

            回答要求:
            - 使用学术风格，但保持易读性
            - 回答简洁明了，重点突出，引用具体论文的方法、模型和性能指标
            - 每提到一篇论文的方法或结论，必须跟上对应的引用标签
            - 如有不确定的地方，明确说明
            - 在回答末尾列出所有引用的参考文献

            科研因果约束（P0-5）:
            1. 严格区分：病因、危险因素、相关因素、病理机制和检测方法
            2. 相关性不能直接解释为因果关系
            3. 只有文献直接比较多个因素时，才能给出影响排序
            4. 不得将来自不同研究、不同结局指标的数值直接放在同一排名中
            5. 缺少直接比较证据时，必须明确说明无法给出可靠排序
            6. 检测模型、诊断方法和技术手段不是疾病的形成原因
            7. 对于"原因+排序"类问题，按以下结构回答：
               一、文献支持的形成机制或危险因素
               二、各因素的证据强度
               三、是否存在直接比较研究
               四、能否进行可靠排序
               五、当前证据局限

            PDF处理规则:
            1. 只有用户明确要求解析、导入或索引PDF时，才调用PDF入库工具
            2. 只能使用上传接口或待处理列表返回的document_id
            3. 不得编造文件ID、文件路径或任务ID
            4. 工具返回queued、parsing等状态时，只能说任务已提交或处理中
            5. 只有状态为indexed时，才能说PDF已进入知识库
            6. 用户询问进度时，调用状态查询工具
            7. PDF加密或需要密码时，停止并要求用户通过安全接口提供
            8. 解析失败时说明实际错误，不得假装成功
            9. 免费API达到限制时停止，不得无限重试

            请根据用户的问题，灵活使用可用工具，提供高质量的科研帮助。
        """).strip()

    async def query(
        self,
        question: str,
        session_id: str,
    ) -> str:
        """
        非流式处理用户问题（一次性返回完整答案）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Returns:
            str: 完整答案
        """
        try:
            await self._initialize_agent()

            logger.info(
                f"RAG Agent收到非流式查询: {describe_text(session_id, 'session')}, "
                f"{describe_text(question, 'question')}"
            )

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            result = await self.agent.ainvoke(
                input=agent_input,
                config=config_dict,
            )

            # 提取最终答案
            messages_result = result.get("messages", [])
            if messages_result:
                last_message = messages_result[-1]
                answer = last_message.content if hasattr(last_message, 'content') else str(last_message)

                # 记录工具调用
                if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                    tool_names = [tc.get("name", "unknown") for tc in last_message.tool_calls]
                    logger.info(
                        f"Agent调用工具: {describe_text(session_id, 'session')}, "
                        f"tools={tool_names}"
                    )

                logger.info(
                    f"RAG Agent非流式查询完成: {describe_text(session_id, 'session')}"
                )
                return answer

            logger.warning(
                f"Agent返回结果为空: {describe_text(session_id, 'session')}"
            )
            return ""

        except Exception as e:
            logger.error(
                f"RAG Agent非流式查询失败: {describe_text(session_id, 'session')}, "
                f"{format_exception_chain(e)}"
            )
            raise

    async def query_with_trace(
        self,
        question: str,
        session_id: str,
    ) -> dict[str, Any]:
        """执行一次 Agent 查询，并返回同一次调用产生的检索 artifact。

        评测必须使用模型实际看到的上下文，不能在 Agent 查询前后另外执行一次检索。
        """
        try:
            await self._initialize_agent()

            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question),
            ]
            result = await self.agent.ainvoke(
                input={"messages": messages},
                config={"configurable": {"thread_id": session_id}},
            )

            result_messages = result.get("messages", [])
            answer = ""
            if result_messages:
                last_message = result_messages[-1]
                answer = (
                    last_message.content
                    if hasattr(last_message, "content")
                    else str(last_message)
                )

            retrieval_artifacts: list[dict[str, Any]] = []
            for message in result_messages:
                artifact = getattr(message, "artifact", None)
                if (
                    isinstance(artifact, dict)
                    and isinstance(artifact.get("documents"), list)
                ):
                    retrieval_artifacts.append(artifact)

            return {
                "answer": answer,
                "retrieval_artifacts": retrieval_artifacts,
                "tool_call_count": len(retrieval_artifacts),
            }
        except Exception as e:
            logger.error(
                f"RAG Agent trace查询失败: {describe_text(session_id, 'session')}, "
                f"{format_exception_chain(e)}"
            )
            raise

    async def query_stream(
        self,
        question: str,
        session_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        流式处理用户问题（逐步返回答案片段）

        Args:
            question: 用户问题
            session_id: 会话ID（作为 thread_id）

        Yields:
            Dict[str, Any]: 包含流式数据的字典
                - type: "content" | "tool_call" | "complete" | "error"
                - data: 具体内容
        """
        try:
            await self._initialize_agent()

            logger.info(
                f"RAG Agent收到流式查询: {describe_text(session_id, 'session')}, "
                f"{describe_text(question, 'question')}"
            )

            # 构建消息列表（系统提示 + 用户问题）
            messages = [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=question)
            ]

            # 构建 Agent 输入
            agent_input = {"messages": messages}

            # 配置 thread_id（用于会话持久化）
            config_dict = {
                "configurable": {
                    "thread_id": session_id
                }
            }

            async for token, metadata in self.agent.astream(
                input=agent_input,
                config=config_dict,
                stream_mode="messages",
            ):
                node_name = metadata.get('langgraph_node', 'unknown') if isinstance(metadata, dict) else 'unknown'
                message_type = type(token).__name__

                # ── P0-1: 三层过滤，隔离内部模型消息 ────────────
                # 1. 过滤内部标签（Rerank 等内部模型调用）
                tags = metadata.get("tags", []) if isinstance(metadata, dict) else []
                if "internal_rerank" in tags:
                    continue

                # 2. 只输出主模型节点的消息
                if node_name != "model":
                    continue

                # 3. 跳过工具调用块
                if getattr(token, "tool_call_chunks", None):
                    continue

                if message_type in ("AIMessage", "AIMessageChunk"):
                    content_blocks = getattr(token, 'content_blocks', None)

                    if content_blocks and isinstance(content_blocks, list):
                        for block in content_blocks:
                            if isinstance(block, dict) and block.get('type') == 'text':
                                text_content = block.get('text', '')
                                if text_content:
                                    yield {
                                        "type": "content",
                                        "data": text_content,
                                        "node": node_name
                                    }

            logger.info(
                f"RAG Agent流式查询完成: {describe_text(session_id, 'session')}"
            )
            yield {"type": "complete"}

        except Exception as e:
            detail = format_exception_chain(e)
            logger.error(
                f"RAG Agent流式查询失败: {describe_text(session_id, 'session')}, {detail}"
            )
            yield {"type": "error", "data": detail}

    def get_session_history(self, session_id: str) -> list:
        """
        获取会话历史（从 MemorySaver checkpointer 中读取）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            list: 消息历史列表 [{"role": "user|assistant", "content": "...", "timestamp": "..."}]
        """
        if self.checkpointer is None:
            logger.warning("RAG Agent尚未初始化，无法读取会话历史")
            return []
        try:
            # 使用 checkpointer 的 get 方法获取最新的检查点
            config = {"configurable": {"thread_id": session_id}}
            
            # 获取该 thread 的最新检查点
            checkpoint_tuple = self.checkpointer.get(config)
            
            if not checkpoint_tuple:
                logger.info(
                    f"获取会话历史: {describe_text(session_id, 'session')}, 消息数量: 0"
                )
                return []
            
            # checkpoint_tuple 可能是命名元组或普通元组，安全地提取 checkpoint
            # 通常第一个元素是 checkpoint 数据
            if hasattr(checkpoint_tuple, 'checkpoint'):
                checkpoint_data = checkpoint_tuple.checkpoint  # type: ignore
            else:
                # 如果是普通元组，第一个元素是 checkpoint
                checkpoint_data = checkpoint_tuple[0] if checkpoint_tuple else {}
            
            # 从检查点中提取消息
            messages = checkpoint_data.get("channel_values", {}).get("messages", [])
            
            # 转换为前端需要的格式
            history = []
            for msg in messages:
                # 跳过系统消息
                if isinstance(msg, SystemMessage):
                    continue
                    
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                content = msg.content if hasattr(msg, 'content') else str(msg)
                
                # 提取时间戳（如果有的话）
                timestamp = getattr(msg, 'timestamp', None)
                if timestamp:
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    })
                else:
                    from datetime import datetime
                    history.append({
                        "role": role,
                        "content": content,
                        "timestamp": datetime.now().isoformat()
                    })
            
            logger.info(
                f"获取会话历史: {describe_text(session_id, 'session')}, "
                f"消息数量: {len(history)}"
            )
            return history
            
        except Exception as e:
            logger.error(
                f"获取会话历史失败: {describe_text(session_id, 'session')}, 错误: {e}"
            )
            return []

    def clear_session(self, session_id: str) -> bool:
        """
        清空会话历史（从 MemorySaver checkpointer 中删除）

        Args:
            session_id: 会话ID（即 thread_id）

        Returns:
            bool: 是否成功
        """
        if self.checkpointer is None:
            logger.warning("RAG Agent尚未初始化，无法清理会话历史")
            return False
        try:
            # 使用 checkpointer 的 delete_thread 方法删除该 thread 的所有检查点
            self.checkpointer.delete_thread(session_id)
            
            logger.info(f"已清除会话历史: {describe_text(session_id, 'session')}")
            return True
            
        except Exception as e:
            logger.error(
                f"清空会话历史失败: {describe_text(session_id, 'session')}, 错误: {e}"
            )
            return False

    async def cleanup(self):
        """清理资源"""
        try:
            logger.info("清理 RAG Agent 服务资源...")
            self.agent = None
            self.model = None
            self.checkpointer = None
            self.tools = []
            self._agent_initialized = False
            logger.info("RAG Agent 服务资源已清理")
        except Exception as e:
            logger.error(f"清理资源失败: {e}")


# 惰性服务对象：外部客户端由FastAPI lifespan或首次显式调用初始化。
rag_agent_service = RagAgentService(streaming=True)
