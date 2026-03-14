"""
Skill Middleware 模块

将 Skill 系统转换为 LangChain AgentMiddleware 实现。
保持 SKILL.md 配置方式，内部使用原生 Middleware 机制。
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Set
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage

from ollamapilot.skills.base import Skill
from ollamapilot.skills.loader import MarkdownSkill


# 内置工具白名单 - 始终允许使用
BUILTIN_TOOLS = {
    "read_file",
    "write_file",
    "list_directory",
    "search_files",
    "shell_exec",
    "shell_script",
    "python_exec",
    "web_search",
    "web_fetch",
    "web_search_setup",
}


def get_time_aware_prompt() -> str:
    """
    获取带当前时间的系统提示词
    
    注入当前时间信息，让模型始终知道"今天"、"明天"等相对时间概念。
    使用简洁格式，避免模型重复输出时间信息。
    
    Returns:
        包含时间信息的系统提示词片段
    """
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    current_weekday = now.strftime("%A")
    
    # 计算相对日期
    tomorrow = (now + timedelta(days=1)).strftime("%Y-%m-%d")
    day_after = (now + timedelta(days=2)).strftime("%Y-%m-%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    
    today = now.strftime('%Y-%m-%d')
    return f"""当前时间: {current_time} ({current_weekday})。今天是{today}，明天是{tomorrow}，后天是{day_after}，昨天是{yesterday}。"""


class ToolFilterMiddleware(AgentMiddleware):
    """
    工具过滤中间件

    根据当前激活的 Skill，限制只能调用特定的工具。
    内置工具始终允许，Skill 专属工具按需允许。

    示例:
        >>> filter_mw = ToolFilterMiddleware(verbose=True)
        >>> filter_mw.set_allowed_tools(["execute_deep_research"])
        >>> agent = create_agent(model, tools, middleware=[..., filter_mw])
    """

    def __init__(self, verbose: bool = False):
        """
        初始化工具过滤中间件

        Args:
            verbose: 是否显示详细日志
        """
        super().__init__()
        self.verbose = verbose
        self.allowed_tools: Set[str] = set()
        self._update_allowed_tools([])  # 初始只允许内置工具

    def _update_allowed_tools(self, skill_tools: List[str]):
        """
        更新允许的工具列表

        Args:
            skill_tools: Skill 专属工具名称列表
        """
        # 内置工具 + Skill 专属工具
        self.allowed_tools = BUILTIN_TOOLS | set(skill_tools)

        if self.verbose:
            print(f"🔧 允许工具 ({len(self.allowed_tools)} 个): {sorted(self.allowed_tools)}")

    def set_allowed_tools(self, skill_tools: List[str]):
        """设置允许的工具列表（供外部调用）"""
        self._update_allowed_tools(skill_tools)

    @property
    def name(self) -> str:
        return "ToolFilterMiddleware"

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        """
        包装工具调用，进行过滤（同步版本）

        Args:
            request: 工具调用请求
            handler: 处理函数

        Returns:
            工具调用结果或错误信息
        """
        return self._filter_and_call(request, handler)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """
        包装工具调用，进行过滤（异步版本）

        Args:
            request: 工具调用请求
            handler: 异步处理函数

        Returns:
            工具调用结果或错误信息
        """
        return await self._filter_and_call_async(request, handler)

    def _filter_and_call(self, request: Any, handler: Any) -> Any:
        """
        工具调用过滤逻辑（同步版本）

        Args:
            request: 工具调用请求
            handler: 处理函数

        Returns:
            工具调用结果或错误信息
        """
        # 获取工具名称
        tool_call = getattr(request, "tool_call", {})
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name", "unknown")
        else:
            tool_name = getattr(tool_call, "name", "unknown")

        # 检查工具是否允许使用
        if tool_name not in self.allowed_tools:
            available = ", ".join(sorted(self.allowed_tools))
            error_msg = (
                f"❌ 工具 '{tool_name}' 在当前 Skill 下不可用。\n\n"
                f"可用工具: {available}\n\n"
                f"提示: 如需使用此工具，请尝试用相关关键词触发对应的 Skill。"
            )

            if self.verbose:
                print(f"🚫 阻止调用: {tool_name}")

            return error_msg

        # 允许调用
        if self.verbose:
            print(f"✅ 允许调用: {tool_name}")

        return handler(request)

    async def _filter_and_call_async(self, request: Any, handler: Any) -> Any:
        """
        工具调用过滤逻辑（异步版本）

        Args:
            request: 工具调用请求
            handler: 异步处理函数

        Returns:
            工具调用结果或错误信息
        """
        # 获取工具名称
        tool_call = getattr(request, "tool_call", {})
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name", "unknown")
        else:
            tool_name = getattr(tool_call, "name", "unknown")

        # 检查工具是否允许使用
        if tool_name not in self.allowed_tools:
            available = ", ".join(sorted(self.allowed_tools))
            error_msg = (
                f"❌ 工具 '{tool_name}' 在当前 Skill 下不可用。\n\n"
                f"可用工具: {available}\n\n"
                f"提示: 如需使用此工具，请尝试用相关关键词触发对应的 Skill。"
            )

            if self.verbose:
                print(f"🚫 阻止调用: {tool_name}")

            return error_msg

        # 允许调用
        if self.verbose:
            print(f"✅ 允许调用: {tool_name}")

        # 异步调用 handler
        return await handler(request)


class SkillMiddleware(AgentMiddleware):
    """
    Skill 中间件基类

    将 Skill 包装为 LangChain AgentMiddleware。
    在 before_model 阶段注入 Skill 的系统提示词。

    示例:
        >>> skill = registry.get_skill("weather")
        >>> middleware = SkillMiddleware(skill)
        >>> agent = create_agent(model, tools, middleware=[middleware])
    """

    def __init__(self, skill: Skill, priority: int = 0):
        """
        初始化 Skill Middleware

        Args:
            skill: Skill 实例
            priority: 优先级，数值越小优先级越高
        """
        super().__init__()
        self.skill = skill
        self.priority = priority
        self._name = f"SkillMiddleware({skill.name})"

        # 注册 Skill 提供的工具
        self.tools = skill.get_tools() or []

    @property
    def name(self) -> str:
        """中间件名称"""
        return self._name

    def before_model(self, state: Any, runtime: Any) -> Optional[Dict[str, Any]]:
        """
        在模型调用前注入 Skill 的系统提示词

        Args:
            state: 当前 Agent 状态
            runtime: 运行时上下文

        Returns:
            状态更新字典
        """
        system_prompt = self.skill.get_system_prompt()
        if not system_prompt:
            return None

        # 获取当前消息列表
        messages = state.get("messages", [])

        # 检查是否已存在系统消息
        has_system = False
        for msg in messages:
            if isinstance(msg, SystemMessage):
                has_system = True
                break

        if not has_system:
            # 添加 Skill 的系统提示词
            return {
                "messages": [SystemMessage(content=system_prompt)] + messages
            }

        return None

    def should_activate(self, query: str) -> bool:
        """
        判断是否应该激活此 Skill

        Args:
            query: 用户查询

        Returns:
            是否匹配触发词
        """
        query_lower = query.lower()
        for trigger in self.skill.triggers:
            if trigger.lower() in query_lower:
                return True
        return False


class SkillSelectorMiddleware(AgentMiddleware):
    """
    Skill 选择器中间件

    根据用户查询自动选择合适的 Skill，并注入对应的系统提示词。
    这是核心中间件，应该放在中间件链的最前面。

    示例:
        >>> selector = SkillSelectorMiddleware(registry)
        >>> agent = create_agent(model, tools, middleware=[selector, ...])
    """

    def __init__(
        self,
        skill_registry: Any,
        default_skill_name: Optional[str] = "default",
        verbose: bool = False
    ):
        """
        初始化 Skill 选择器

        Args:
            skill_registry: Skill 注册中心
            default_skill_name: 默认 Skill 名称
            verbose: 是否显示详细日志
        """
        super().__init__()
        self.registry = skill_registry
        self.default_skill_name = default_skill_name
        self.verbose = verbose
        self._active_skill: Optional[str] = None
        
        # 创建工具过滤器
        self.tool_filter = ToolFilterMiddleware(verbose=verbose)

    @property
    def name(self) -> str:
        return "SkillSelectorMiddleware"

    def before_model(self, state: Any, runtime: Any) -> Optional[Dict[str, Any]]:
        """
        选择合适的 Skill 并注入系统提示词（包含时间信息）

        Args:
            state: 当前 Agent 状态
            runtime: 运行时上下文

        Returns:
            状态更新字典
        """
        # 获取最后一条用户消息
        messages = state.get("messages", [])
        if not messages:
            return None

        last_message = messages[-1]
        
        # 调试：打印消息列表结构
        if self.verbose:
            msg_types = [type(m).__name__ for m in messages]
            print(f"   [SkillSelector] 消息列表: {msg_types}")
        
        # 只处理用户消息（HumanMessage），不处理工具返回或 AI 消息
        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
        if not isinstance(last_message, HumanMessage):
            if self.verbose:
                msg_type = type(last_message).__name__
                print(f"   [SkillSelector] 跳过非用户消息: {msg_type}")
            return None
        
        query = ""
        if hasattr(last_message, "content"):
            query = str(last_message.content)

        if not query:
            return None

        # 查找匹配的 Skill
        skill = self._select_skill(query)

        # 构建时间感知的系统提示词
        time_prompt = get_time_aware_prompt()
        
        # 更新允许的工具列表
        skill_tool_names = []
        if skill:
            skill_tools = skill.get_tools()
            if skill_tools:
                skill_tool_names = [t.name for t in skill_tools]
        
        if self.verbose:
            print(f"   [SkillSelector] 设置工具白名单: {skill_tool_names if skill_tool_names else '(仅内置工具)'}")
        
        self.tool_filter.set_allowed_tools(skill_tool_names)
        
        if skill:
            # 打印 Skill 激活日志
            if self.verbose:
                print(f"🎯 激活 Skill: {skill.name}")

            # 记录当前激活的 Skill
            self._active_skill = skill.name

            skill_prompt = skill.get_system_prompt()
            
            # 组合时间提示词和 Skill 提示词
            if skill_prompt:
                full_prompt = f"{time_prompt}\n\n【Skill 上下文】\n{skill_prompt}"
            else:
                full_prompt = time_prompt
            
            # 检查是否已存在系统消息
            has_system = False
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    has_system = True
                    break

            if not has_system:
                # 没有系统消息，添加新的系统消息
                # 注意：保留所有原始消息，只在开头添加系统消息
                new_messages = [SystemMessage(content=full_prompt)] + messages
                if self.verbose:
                    print(f"   [SkillSelector] 添加系统消息，消息数: {len(messages)} -> {len(new_messages)}")
                return {
                    "messages": new_messages,
                    "active_skill": skill.name
                }
            else:
                # 已有系统消息，替换系统消息并保留其他所有消息
                # 注意：messages[1:] 保留除第一条（旧系统消息）外的所有消息
                new_messages = [SystemMessage(content=full_prompt)] + messages[1:]
                if self.verbose:
                    print(f"   [SkillSelector] 替换系统消息，消息数: {len(messages)} -> {len(new_messages)}")
                return {
                    "messages": new_messages,
                    "active_skill": skill.name
                }
        else:
            # 没有匹配 Skill，只注入时间信息
            self._active_skill = None
            
            # 检查是否已存在系统消息
            has_system = False
            for msg in messages:
                if isinstance(msg, SystemMessage):
                    has_system = True
                    break
            
            if not has_system:
                new_messages = [SystemMessage(content=time_prompt)] + messages
                if self.verbose:
                    print(f"   [SkillSelector] 无Skill匹配，添加时间系统消息，消息数: {len(messages)} -> {len(new_messages)}")
                return {
                    "messages": new_messages
                }
            else:
                # 替换现有的系统消息，保留时间信息
                new_messages = [SystemMessage(content=time_prompt)] + messages[1:]
                if self.verbose:
                    print(f"   [SkillSelector] 无Skill匹配，替换系统消息，消息数: {len(messages)} -> {len(new_messages)}")
                return {
                    "messages": new_messages
                }

    def _select_skill(self, query: str) -> Optional[Skill]:
        """
        根据查询选择合适的 Skill

        Args:
            query: 用户查询

        Returns:
            Skill 实例或 None
        """
        # 1. 查找匹配的特定 Skill
        matches = self.registry.find_skill_by_trigger(query)

        if matches:
            return self.registry.get_skill(matches[0])

        # 2. 返回默认 Skill
        if self.default_skill_name:
            return self.registry.get_skill(self.default_skill_name)

        return None

    def get_tool_filter(self) -> ToolFilterMiddleware:
        """
        获取工具过滤器实例

        供 Agent 添加到中间件链中使用。

        Returns:
            ToolFilterMiddleware 实例
        """
        return self.tool_filter


class ToolLoggingMiddleware(AgentMiddleware):
    """
    工具调用日志中间件

    记录工具调用的详细信息，便于调试。

    示例:
        >>> logging_mw = ToolLoggingMiddleware(verbose=True)
        >>> agent = create_agent(model, tools, middleware=[..., logging_mw])
    """

    def __init__(self, verbose: bool = True):
        super().__init__()
        self.verbose = verbose

    @property
    def name(self) -> str:
        return "ToolLoggingMiddleware"

    def wrap_tool_call(self, request: Any, handler: Any) -> Any:
        """
        包装工具调用，添加日志（同步版本）

        Args:
            request: 工具调用请求 (ToolCallRequest)
            handler: 处理函数

        Returns:
            工具调用结果
        """
        return self._log_tool_call(request, handler)

    async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
        """
        包装工具调用，添加日志（异步版本）

        Args:
            request: 工具调用请求 (ToolCallRequest)
            handler: 异步处理函数

        Returns:
            工具调用结果
        """
        if self.verbose:
            # 从 request.tool_call 获取工具调用信息
            tool_call = getattr(request, "tool_call", {})
            tool_name = tool_call.get("name", "unknown") if isinstance(tool_call, dict) else getattr(tool_call, "name", "unknown")
            tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
            print(f"🔧 执行工具: {tool_name}({tool_args})")

        # 异步调用 handler
        result = await handler(request)

        if self.verbose:
            result_preview = str(result)[:200]
            if len(str(result)) > 200:
                result_preview += "..."
            print(f"   ✅ 结果: {result_preview}")

        return result

    def _log_tool_call(self, request: Any, handler: Any) -> Any:
        """
        工具调用日志的核心逻辑（同步版本）

        Args:
            request: 工具调用请求
            handler: 处理函数

        Returns:
            工具调用结果
        """
        if self.verbose:
            # 从 request.tool_call 获取工具调用信息
            tool_call = getattr(request, "tool_call", {})
            tool_name = tool_call.get("name", "unknown") if isinstance(tool_call, dict) else getattr(tool_call, "name", "unknown")
            tool_args = tool_call.get("args", {}) if isinstance(tool_call, dict) else getattr(tool_call, "args", {})
            print(f"🔧 执行工具: {tool_name}({tool_args})")

        result = handler(request)

        if self.verbose:
            result_preview = str(result)[:200]
            if len(str(result)) > 200:
                result_preview += "..."
            print(f"   ✅ 结果: {result_preview}")

        return result


def create_skill_middlewares(
    skill_registry: Any,
    verbose: bool = False
) -> List[AgentMiddleware]:
    """
    从 Skill 注册中心创建中间件列表

    创建一个 SkillSelectorMiddleware 作为核心中间件。
    不需要为每个 Skill 创建单独的中间件，而是由选择器动态处理。

    Args:
        skill_registry: Skill 注册中心
        verbose: 是否显示详细日志

    Returns:
        中间件列表
    """
    middlewares = []

    # 1. Skill 选择器（核心）
    selector = SkillSelectorMiddleware(
        skill_registry,
        default_skill_name="default",
        verbose=verbose
    )
    middlewares.append(selector)

    # 2. 工具调用日志
    if verbose:
        logging_mw = ToolLoggingMiddleware(verbose=True)
        middlewares.append(logging_mw)

    return middlewares
