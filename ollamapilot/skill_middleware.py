"""
Skill Middleware 模块

将 Skill 系统转换为 LangChain AgentMiddleware 实现。
保持 SKILL.md 配置方式，内部使用原生 Middleware 机制。
"""

from typing import Any, Dict, List, Optional, Sequence
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool
from langchain_core.messages import SystemMessage

from ollamapilot.skills.base import Skill
from ollamapilot.skills.loader import MarkdownSkill


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

    @property
    def name(self) -> str:
        return "SkillSelectorMiddleware"

    def before_model(self, state: Any, runtime: Any) -> Optional[Dict[str, Any]]:
        """
        选择合适的 Skill 并注入系统提示词

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
        query = ""
        if hasattr(last_message, "content"):
            query = str(last_message.content)

        if not query:
            return None

        # 查找匹配的 Skill
        skill = self._select_skill(query)

        if skill:
            # 注：Skill 激活日志现在在 agent.invoke() 中打印，避免重复
            # if self.verbose:
            #     print(f"🎯 激活 Skill: {skill.name}")

            system_prompt = skill.get_system_prompt()
            if system_prompt:
                # 检查是否已存在系统消息
                has_system = False
                for msg in messages:
                    if isinstance(msg, SystemMessage):
                        has_system = True
                        break

                if not has_system:
                    return {
                        "messages": [SystemMessage(content=system_prompt)] + messages
                    }

        return None

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
        包装工具调用，添加日志

        Args:
            request: 工具调用请求 (ToolCallRequest)
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
