"""
Agent 核心模块 V2 - 基于 LangChain create_agent

使用 LangChain 原生 create_agent 和 AgentMiddleware 实现。
保持所有现有功能，代码更简洁。
"""

from typing import List, Optional, Dict, Any
from langchain.agents import create_agent as lc_create_agent
from langchain.agents.middleware import (
    ToolRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver

from ollamapilot.skills import SkillRegistry
from ollamapilot.tools.builtin import (
    read_file, write_file, list_directory, search_files,
    shell_exec, shell_script, python_exec, web_search, web_fetch
)
from ollamapilot.skill_middleware import (
    SkillSelectorMiddleware,
    ToolLoggingMiddleware,
    create_skill_middlewares,
)


class OllamaPilotAgent:
    """
    OllamaPilot Agent - V2 版本

    基于 LangChain create_agent 实现，使用原生 Middleware 机制。

    特性:
    - 使用 create_agent 创建 Agent
    - Skill 通过 AgentMiddleware 实现
    - 自动工具重试和限流
    - 对话记忆持久化
    - 详细执行日志
    - Skill 可注册自定义中间件（如 GraphRAG）

    示例:
        >>> from ollamapilot import init_ollama_model, OllamaPilotAgent
        >>>
        >>> model = init_ollama_model("qwen3.5:4b")
        >>> agent = OllamaPilotAgent(model, skills_dir="skills")
        >>> response = agent.invoke("明天苏州天气怎么样？")
    """

    def __init__(
        self,
        model: BaseChatModel,
        skills_dir: Optional[str] = None,
        enable_memory: bool = True,
        max_tool_calls: int = 50,
        verbose: bool = True,
        checkpointer=None,
        tools: Optional[List[BaseTool]] = None,  # 兼容参数，实际使用内置工具
        embedding_model: Optional[str] = None,  # Embedding 模型名称
        **kwargs,  # 忽略其他参数，保持向后兼容
    ):
        """
        初始化 Agent

        Args:
            model: 聊天模型实例
            skills_dir: Skill 目录路径
            enable_memory: 是否启用对话记忆
            max_tool_calls: 最大工具调用次数
            verbose: 是否显示详细执行过程
            checkpointer: 自定义 checkpointer
            embedding_model: Embedding 模型名称（传递给 GraphRAG Skill）
        """
        self.model = model
        self.verbose = verbose

        # 构建 Skill 配置
        skill_config = {}
        if embedding_model:
            skill_config["graphrag"] = {"embedding_model": embedding_model}

        # 初始化 Skill 注册中心
        self.skill_registry = SkillRegistry(skill_config=skill_config)

        # 加载 Skill
        skill_count = 0
        if skills_dir:
            skill_count = self.skill_registry.discover_skills(skills_dir)

        # 收集所有工具（内置 + Skill）
        self.all_tools = self._get_all_tools()

        # 配置 Checkpointer
        if checkpointer:
            self.checkpointer = checkpointer
        elif enable_memory:
            self.checkpointer = MemorySaver()
        else:
            self.checkpointer = None

        # 保存最大工具调用次数（用于设置 recursion_limit）
        self.max_tool_calls = max_tool_calls

        # 构建中间件列表
        middleware = self._build_middleware(max_tool_calls)

        # 打印启动摘要
        if self.verbose:
            print(f"📦 已加载 {skill_count} 个 Skill | 🔧 {len(self.all_tools)} 个工具 | 🔒 工具过滤已启用")

        # 创建 Agent（使用 LangChain 原生 create_agent）
        self.agent = lc_create_agent(
            model=model.bind_tools(self.all_tools),
            tools=self.all_tools,
            middleware=middleware,
            checkpointer=self.checkpointer,
        )

    def _get_all_tools(self) -> List[BaseTool]:
        """获取所有工具（内置工具 + Skill 工具）"""
        # 内置工具
        tools = [
            read_file,
            write_file,
            list_directory,
            search_files,
            shell_exec,
            shell_script,
            python_exec,
            web_search,
            web_fetch,
        ]

        # 收集所有 Skill 提供的工具
        skill_tools = self.skill_registry.get_all_tools()
        tools.extend(skill_tools)

        return tools

    def _build_middleware(self, max_tool_calls: int) -> List[Any]:
        """
        构建中间件列表

        中间件执行顺序（从前到后）:
        1. SkillSelectorMiddleware - Skill 选择
        2. Skill 自定义中间件（如 GraphRAGMiddleware）
        3. ToolFilterMiddleware - 工具过滤（根据 Skill 限制可用工具）
        4. ToolLoggingMiddleware - 工具调用日志
        5. ToolRetryMiddleware - 工具重试
        6. ToolCallLimitMiddleware - 工具调用限制
        """
        middleware = []

        # 1. Skill 选择器（核心）
        self._selector = SkillSelectorMiddleware(
            self.skill_registry,
            default_skill_name="default",
            verbose=self.verbose
        )
        middleware.append(self._selector)

        # 2. Skill 自定义中间件（动态收集）
        skill_middlewares = self.skill_registry.get_all_middlewares()
        for mw in skill_middlewares:
            middleware.append(mw)

        # 3. 工具过滤中间件（根据选中 Skill 限制工具）
        tool_filter = self._selector.get_tool_filter()
        middleware.append(tool_filter)

        # 4. 工具调用日志
        if self.verbose:
            logging_mw = ToolLoggingMiddleware(verbose=True)
            middleware.append(logging_mw)

        # 5. 工具重试
        retry_mw = ToolRetryMiddleware(max_retries=2)
        middleware.append(retry_mw)

        # 6. 工具调用限制
        limit_mw = ToolCallLimitMiddleware(run_limit=max_tool_calls)
        middleware.append(limit_mw)

        return middleware

    def invoke(self, query: str, thread_id: Optional[str] = None) -> str:
        """
        执行用户查询

        Args:
            query: 用户输入
            thread_id: 对话线程 ID（用于持久化记忆）

        Returns:
            模型回复
        """
        if self.verbose:
            print(f"🤖 用户: {query}")

        # 配置（包含 recursion_limit 以防止 LangGraph 默认限制）
        config = {
            "configurable": {"thread_id": thread_id or "default"},
            "recursion_limit": self.max_tool_calls + 10  # 给一些余量
        }

        # 执行
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config
        )

        # 提取回复
        messages = result.get("messages", [])
        response = ""
        has_tool_calls = False

        if messages:
            # 检查消息历史中是否有工具调用
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    has_tool_calls = True
                    break

            # 获取最后一条消息的内容
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                response = last_message.content or ""

        # 如果有工具调用但回复为空，强制生成回复
        if has_tool_calls and not response.strip():
            if self.verbose:
                print("🔄 检测到工具调用后无回复，强制生成回复...")
            response = self._force_response(messages, config)

        if self.verbose and response:
            print(f"🤖 AI: {response[:200]}{'...' if len(response) > 200 else ''}")

        return response

    def _force_response(self, messages: List[Any], config: Dict[str, Any]) -> str:
        """
        强制生成回复（当模型调用工具后没有生成回复时）

        Args:
            messages: 当前消息历史
            config: 配置

        Returns:
            强制生成的回复
        """
        try:
            # 从 checkpointer 获取完整的对话历史（包含 ToolMessage）
            thread_id = config.get("configurable", {}).get("thread_id", "default")
            full_messages = messages

            if self.checkpointer:
                try:
                    # 尝试从 checkpointer 获取完整历史
                    checkpoint_tuple = self.checkpointer.get_tuple({"configurable": {"thread_id": thread_id}})
                    if checkpoint_tuple and checkpoint_tuple.checkpoint:
                        # 尝试不同的消息存储位置
                        checkpoint = checkpoint_tuple.checkpoint

                        # 方法1: 直接获取 messages
                        if "messages" in checkpoint:
                            full_messages = checkpoint["messages"]
                        # 方法2: 从 channel_values 获取 (LangGraph 新格式)
                        elif "channel_values" in checkpoint:
                            channel_values = checkpoint["channel_values"]
                            if "messages" in channel_values:
                                full_messages = channel_values["messages"]
                        # 方法3: 从完整状态获取
                        elif checkpoint_tuple.state and "messages" in checkpoint_tuple.state:
                            full_messages = checkpoint_tuple.state["messages"]

                        if self.verbose:
                            print(f"   [ForceResponse] 从 checkpointer 加载了 {len(full_messages)} 条消息")
                except Exception as e:
                    if self.verbose:
                        print(f"   [ForceResponse] 从 checkpointer 加载失败: {e}")

            # 过滤掉系统提示词，构建清晰的消息链
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
            cleaned_messages = []
            for msg in full_messages:
                if isinstance(msg, SystemMessage):
                    continue
                elif isinstance(msg, AIMessage):
                    # 跳过空的 AIMessage（没有content也没有tool_calls）
                    if not msg.content and not msg.tool_calls:
                        if self.verbose:
                            print(f"   [ForceResponse] 跳过空的 AIMessage")
                        continue
                    # 保留带 tool_calls 的 AIMessage，但添加说明文本
                    if msg.tool_calls and not msg.content:
                        # 创建一个新的 AIMessage，添加说明文本
                        tool_names = [tc.get('name', 'unknown') for tc in msg.tool_calls]
                        explanation = f"我需要调用工具来获取信息：{', '.join(tool_names)}"
                        # 创建新的 AIMessage，保留 tool_calls 但添加 content
                        new_msg = AIMessage(
                            content=explanation,
                            tool_calls=msg.tool_calls,
                            id=msg.id if hasattr(msg, 'id') else None
                        )
                        cleaned_messages.append(new_msg)
                        if self.verbose:
                            print(f"   [ForceResponse] 为带 tool_calls 的 AIMessage 添加说明: {explanation}")
                        continue
                    cleaned_messages.append(msg)
                elif isinstance(msg, (HumanMessage, ToolMessage)):
                    cleaned_messages.append(msg)
                elif hasattr(msg, "content"):
                    cleaned_messages.append(msg)

            # 只保留最后一次查询的消息（从最后一个 HumanMessage 开始）
            # 找到最后一个 HumanMessage 的位置
            last_human_index = -1
            for i in range(len(cleaned_messages) - 1, -1, -1):
                if isinstance(cleaned_messages[i], HumanMessage):
                    last_human_index = i
                    break

            if last_human_index >= 0:
                cleaned_messages = cleaned_messages[last_human_index:]
                if self.verbose:
                    print(f"   [ForceResponse] 截取最后一次查询，从 HumanMessage[{last_human_index}] 开始")

            if self.verbose:
                print(f"   [ForceResponse] 清理后消息数: {len(cleaned_messages)}")
                # 打印消息类型摘要
                for i, msg in enumerate(cleaned_messages):
                    msg_type = type(msg).__name__
                    has_content = bool(getattr(msg, 'content', None))
                    has_tools = bool(getattr(msg, 'tool_calls', None))
                    print(f"      [{i}] {msg_type} (content:{has_content}, tools:{has_tools})")

            # 添加强制回复提示
            force_messages = cleaned_messages + [
                HumanMessage(content="请基于上述工具执行结果，直接回答用户的问题。不要调用任何工具。")
            ]

            # 临时禁用工具调用，强制模型只生成文本回复
            result = self.model.invoke(force_messages)

            if hasattr(result, "content"):
                return result.content or "（工具执行完成，但无法生成详细回复）"

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 强制生成回复失败: {e}")
                import traceback
                traceback.print_exc()

        return "（工具执行完成，但生成回复失败）"

    def _select_skill_for_query(self, query: str) -> Optional[Any]:
        """
        根据查询选择合适的 Skill

        Args:
            query: 用户查询

        Returns:
            Skill 实例或 None
        """
        # 1. 查找匹配的特定 Skill
        matches = self.skill_registry.find_skill_by_trigger(query)

        if matches:
            return self.skill_registry.get_skill(matches[0])

        # 2. 返回默认 Skill
        default_skill = self.skill_registry.get_default_skill()
        if default_skill:
            return default_skill

        return None

    def stream(self, query: str, thread_id: Optional[str] = None):
        """
        流式执行用户查询

        Args:
            query: 用户输入
            thread_id: 对话线程 ID

        Yields:
            流式输出块
        """
        config = {"configurable": {"thread_id": thread_id or "default"}}

        for chunk in self.agent.stream(
            {"messages": [HumanMessage(content=query)]},
            config
        ):
            yield chunk

    async def astream_events(self, query: str, thread_id: Optional[str] = None):
        """
        异步流式事件输出 - 使用 LangChain 原生 astream_events

        提供结构化的事件流，包括：
        - on_tool_start: 工具开始执行
        - on_tool_end: 工具执行结束
        - on_chat_model_stream: 模型流式输出

        Args:
            query: 用户输入
            thread_id: 对话线程 ID

        Yields:
            事件字典，包含 event, name, data 等字段
        """
        # 配置（包含 recursion_limit 以防止 LangGraph 默认限制）
        config = {
            "configurable": {"thread_id": thread_id or "default"},
            "recursion_limit": self.max_tool_calls + 10  # 给一些余量
        }

        # 只传入当前用户消息，历史消息由 checkpointer 自动管理
        # LangChain Agent 会自动从 checkpointer 加载历史消息并合并
        async for event in self.agent.astream_events(
            {"messages": [HumanMessage(content=query)]},
            config,
            version="v1"
        ):
            yield event

    def get_history(self, thread_id: Optional[str] = None) -> List[Any]:
        """
        获取对话历史

        Args:
            thread_id: 对话线程 ID

        Returns:
            消息列表
        """
        if not self.checkpointer:
            return []

        config = {"configurable": {"thread_id": thread_id or "default"}}

        try:
            checkpoint_tuple = self.checkpointer.get_tuple(config)
            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                # 尝试不同的消息存储位置
                checkpoint = checkpoint_tuple.checkpoint

                # 方法1: 直接获取 messages
                if "messages" in checkpoint:
                    return checkpoint["messages"]

                # 方法2: 从 channel_values 获取 (LangGraph 新格式)
                if "channel_values" in checkpoint:
                    channel_values = checkpoint["channel_values"]
                    if "messages" in channel_values:
                        return channel_values["messages"]

                # 方法3: 从完整状态获取
                if checkpoint_tuple.state:
                    state = checkpoint_tuple.state
                    if "messages" in state:
                        return state["messages"]
        except Exception as e:
            if self.verbose:
                print(f"   [GetHistory] 获取历史失败: {e}")

        return []

    def clear_history(self, thread_id: Optional[str] = None) -> None:
        """
        清除对话历史

        Args:
            thread_id: 对话线程 ID
        """
        if not self.checkpointer:
            return

        config = {"configurable": {"thread_id": thread_id or "default"}}

        try:
            self.checkpointer.delete(config)
        except Exception:
            pass

    def get_skill_stats(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """
        获取 Skill 的统计信息

        Args:
            skill_name: Skill 名称

        Returns:
            统计信息字典，如果 Skill 不存在或不支持返回 None
        """
        skill = self.skill_registry.get_skill(skill_name)
        if not skill:
            return None

        # 检查 Skill 是否有 get_stats 方法
        if hasattr(skill, 'get_stats'):
            return skill.get_stats()

        return None

    async def force_response_after_tool(self, thread_id: Optional[str] = None):
        """
        工具调用后强制模型生成回复

        当模型在工具调用后没有生成回复时，使用此方法触发一次额外的模型调用，
        要求模型基于工具结果生成回复。

        注意：此方法会临时清空工具列表，确保只生成文本回复。

        Args:
            thread_id: 对话线程 ID

        Yields:
            模型生成的文本块
        """
        # 临时禁用工具调用，确保只生成文本回复
        # 通过设置一个标志，让 before_model 知道这是强制回复
        if self._selector:
            # 保存原始工具列表
            original_tools = self._selector.tool_filter.allowed_tools.copy()
            # 清空工具列表，禁止工具调用
            self._selector.tool_filter.set_allowed_tools([])
            if self.verbose:
                print("   [ForceResponse] 临时禁用工具调用")

        try:
            # 添加提示消息，要求模型基于工具结果生成回复
            prompt = "基于上述工具执行结果，请为用户提供清晰、有用的回复。总结关键信息并给出实用建议。"

            async for event in self.astream_events(prompt, thread_id=thread_id):
                event_type = event.get("event", "")
                if event_type == "on_chat_model_stream":
                    data = event.get("data", {})
                    chunk = data.get("chunk", None)
                    if chunk and hasattr(chunk, "content"):
                        content = chunk.content
                        if content:
                            yield content
        finally:
            # 恢复原始工具设置
            if self._selector and original_tools:
                self._selector.tool_filter.set_allowed_tools(list(original_tools))
                if self.verbose:
                    print("   [ForceResponse] 恢复工具调用")


def create_ollama_agent(
    model: BaseChatModel,
    skills_dir: Optional[str] = None,
    enable_memory: bool = True,
    max_tool_calls: Optional[int] = None,
    verbose: bool = True,
    **kwargs
) -> OllamaPilotAgent:
    """
    创建 OllamaPilot Agent

    工厂函数，快速创建 Agent 实例。

    Args:
        model: 聊天模型实例
        skills_dir: Skill 目录路径
        enable_memory: 是否启用对话记忆
        max_tool_calls: 最大工具调用次数，默认从配置文件读取 RECURSION_LIMIT
        verbose: 是否显示详细执行过程
        **kwargs: 其他参数传递给 OllamaPilotAgent

    Returns:
        OllamaPilotAgent 实例

    Example:
        >>> from ollamapilot import init_ollama_model, create_ollama_agent
        >>>
        >>> model = init_ollama_model("qwen3.5:4b")
        >>> agent = create_ollama_agent(model, skills_dir="skills")
        >>> response = agent.invoke("明天苏州天气怎么样？")
    """
    # 如果未指定 max_tool_calls，从配置文件读取
    if max_tool_calls is None:
        from ollamapilot.config import get_config
        config = get_config()
        max_tool_calls = config.get_int('RECURSION_LIMIT', 50)

    return OllamaPilotAgent(
        model=model,
        skills_dir=skills_dir,
        enable_memory=enable_memory,
        max_tool_calls=max_tool_calls,
        verbose=verbose,
        **kwargs
    )


# 保持向后兼容的别名
create_agent = create_ollama_agent
