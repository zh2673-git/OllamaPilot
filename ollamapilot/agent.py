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
        if skills_dir:
            count = self.skill_registry.discover_skills(skills_dir)
            if self.verbose:
                print(f"📦 已加载 {count} 个 Skill")

        # 收集所有工具（内置 + Skill）
        self.all_tools = self._get_all_tools()

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

        if self.verbose and skill_tools:
            print(f"🔧 加载 {len(skill_tools)} 个 Skill 工具")

        return tools

    def _build_middleware(self, max_tool_calls: int) -> List[Any]:
        """
        构建中间件列表

        中间件执行顺序（从前到后）:
        1. SkillSelectorMiddleware - Skill 选择
        2. Skill 自定义中间件（如 GraphRAGMiddleware）
        3. ToolLoggingMiddleware - 工具调用日志
        4. ToolRetryMiddleware - 工具重试
        5. ToolCallLimitMiddleware - 工具调用限制
        """
        middleware = []

        # 1. Skill 选择器（核心）
        selector = SkillSelectorMiddleware(
            self.skill_registry,
            default_skill_name="default",
            verbose=self.verbose
        )
        middleware.append(selector)

        # 2. Skill 自定义中间件（动态收集）
        skill_middlewares = self.skill_registry.get_all_middlewares()
        for mw in skill_middlewares:
            middleware.append(mw)
            if self.verbose:
                print(f"🔌 加载 Skill 中间件: {mw.name if hasattr(mw, 'name') else type(mw).__name__}")

        # 3. 工具调用日志
        if self.verbose:
            logging_mw = ToolLoggingMiddleware(verbose=True)
            middleware.append(logging_mw)

        # 4. 工具重试
        retry_mw = ToolRetryMiddleware(max_retries=2)
        middleware.append(retry_mw)

        # 5. 工具调用限制
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

        # 手动选择 Skill 并显示日志（确保在 invoke 前显示）
        skill = self._select_skill_for_query(query)
        if skill and self.verbose:
            print(f"🎯 激活 Skill: {skill.name}")

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
