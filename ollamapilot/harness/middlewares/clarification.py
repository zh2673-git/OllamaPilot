"""
ClarificationMiddleware - 澄清请求中间件

当用户输入不明确时，请求澄清并中断执行。
继承 LangChain AgentMiddleware
"""

import re
from typing import Any, Dict, List

from langchain.agents.middleware import AgentMiddleware


class ClarificationMiddleware(AgentMiddleware):
    """
    澄清请求中间件

    检测用户输入是否需要澄清，需要时中断执行并返回澄清请求。
    
    触发条件：
    1. 输入过于模糊（如"帮我做"、"怎么做"）
    2. 缺少关键信息
    3. 多个可能的解释
    
    继承 LangChain AgentMiddleware，统一使用 LangChain 中间件机制
    
    注意：LangChain 中间件不支持直接中断，我们通过修改 state 来标记需要澄清
    """

    # 模糊输入模式
    VAGUE_PATTERNS = [
        r"^帮我做$",
        r"^怎么做$",
        r"^怎么做.+\?$",
        r"^帮我.*\?$",
        r"^这个.*",
        r"^那个.*",
    ]

    # 需要澄清的关键词
    CLARIFICATION_TRIGGERS = [
        "不知道",
        "不清楚",
        "什么意思",
        "不明白",
    ]

    def __init__(self, auto_clarify: bool = False):
        self.auto_clarify = auto_clarify

    @property
    def name(self) -> str:
        return "ClarificationMiddleware"

    def before_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any]:
        """检查是否需要澄清"""
        if not self.auto_clarify:
            return state

        messages = state.get("messages", [])
        
        # 提取用户查询（支持对象和字典两种格式）
        query = ""
        for msg in reversed(messages):
            # 支持对象格式
            if hasattr(msg, "content") and hasattr(msg, "type"):
                if msg.type == "human":
                    query = msg.content
                    break
            # 支持字典格式
            elif isinstance(msg, dict):
                if msg.get("type") == "human":
                    query = msg.get("content", "")
                    break

        if not query:
            return state

        # 检查是否需要澄清
        clarification_msg = self._check_need_clarification(query)
        if clarification_msg:
            # 在 state 中标记需要澄清（LangChain 中间件无法直接中断）
            state["needs_clarification"] = True
            state["clarification_message"] = clarification_msg
            # 修改消息，让模型知道需要澄清
            from langchain_core.messages import SystemMessage
            state["messages"].append(SystemMessage(
                content=f"系统提示：用户输入需要澄清。{clarification_msg}"
            ))

        return state

    def _check_need_clarification(self, query: str) -> str:
        """检查是否需要澄清，返回澄清消息或空字符串"""
        query_lower = query.lower().strip()

        # 检查模糊模式
        for pattern in self.VAGUE_PATTERNS:
            if re.match(pattern, query_lower):
                return "您的请求比较模糊，能否提供更多细节？比如具体想做什么、有什么要求等。"

        # 检查触发词
        for trigger in self.CLARIFICATION_TRIGGERS:
            if trigger in query_lower:
                return "我注意到您可能有些困惑。能否详细描述一下您想解决的问题或想达成的目标？"

        # 检查是否过短
        if len(query_lower) < 5:
            return "您的输入较短，能否提供更多上下文信息，让我更好地帮助您？"

        return ""
