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
from langchain_core.messages import HumanMessage
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
        **kwargs,  # 忽略其他参数
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
        """
        self.model = model
        self.verbose = verbose

        # 初始化 Skill 注册中心
        self.skill_registry = SkillRegistry()
        if skills_dir:
            count = self.skill_registry.discover_skills(skills_dir)
            if self.verbose:
                print(f"📦 已加载 {count} 个 Skill")

        # 收集所有内置工具
        self.builtin_tools = self._get_builtin_tools()
        self.all_tools = self.builtin_tools  # 兼容旧版本属性名

        # 配置 Checkpointer
        if checkpointer:
            self.checkpointer = checkpointer
        elif enable_memory:
            self.checkpointer = MemorySaver()
        else:
            self.checkpointer = None

        # 构建中间件列表
        middleware = self._build_middleware(max_tool_calls)

        # 创建 Agent（使用 LangChain 原生 create_agent）
        self.agent = lc_create_agent(
            model=model.bind_tools(self.builtin_tools),
            tools=self.builtin_tools,
            middleware=middleware,
            checkpointer=self.checkpointer,
        )

    def _get_builtin_tools(self) -> List[BaseTool]:
        """获取所有内置工具"""
        return [
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

    def _build_middleware(self, max_tool_calls: int) -> List[Any]:
        """
        构建中间件列表

        中间件执行顺序（从前到后）:
        1. SkillSelectorMiddleware - Skill 选择
        2. ToolLoggingMiddleware - 工具调用日志
        3. ToolRetryMiddleware - 工具重试
        4. ToolCallLimitMiddleware - 工具调用限制
        """
        middleware = []

        # 1. Skill 选择器（核心）
        selector = SkillSelectorMiddleware(
            self.skill_registry,
            default_skill_name="default",
            verbose=self.verbose
        )
        middleware.append(selector)

        # 2. 工具调用日志
        if self.verbose:
            logging_mw = ToolLoggingMiddleware(verbose=True)
            middleware.append(logging_mw)

        # 3. 工具重试
        retry_mw = ToolRetryMiddleware(max_retries=2)
        middleware.append(retry_mw)

        # 4. 工具调用限制
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

        # 配置
        config = {"configurable": {"thread_id": thread_id or "default"}}

        # 执行
        result = self.agent.invoke(
            {"messages": [HumanMessage(content=query)]},
            config
        )

        # 提取回复
        messages = result.get("messages", [])
        if messages:
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                response = last_message.content
                if self.verbose:
                    print(f"🤖 AI: {response[:200]}{'...' if len(response) > 200 else ''}")
                return response

        return ""

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
                return checkpoint_tuple.checkpoint.get("messages", [])
        except Exception:
            pass

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


def create_ollama_agent(
    model: BaseChatModel,
    skills_dir: Optional[str] = None,
    enable_memory: bool = True,
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
    return OllamaPilotAgent(
        model=model,
        skills_dir=skills_dir,
        enable_memory=enable_memory,
        verbose=verbose,
        **kwargs
    )


# 保持向后兼容的别名
create_agent = create_ollama_agent
