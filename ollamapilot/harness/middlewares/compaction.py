"""
CompactionMiddleware - 上下文压缩中间件

包装现有的 ContextCompactor，在模型调用前压缩上下文。
继承 LangChain AgentMiddleware
"""

from typing import Any, Dict

from langchain.agents.middleware import AgentMiddleware


class CompactionMiddleware(AgentMiddleware):
    """
    上下文压缩中间件

    包装现有的 ContextCompactor，在 Token 超出预算时压缩上下文。
    
    压缩策略：
    1. 优先压缩历史消息
    2. 保留最近 N 条消息
    3. 保留系统消息
    
    继承 LangChain AgentMiddleware，统一使用 LangChain 中间件机制
    """

    def __init__(self, compactor: Any):
        self.compactor = compactor

    @property
    def name(self) -> str:
        return "CompactionMiddleware"

    def before_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any]:
        """在模型调用前压缩上下文"""
        messages = state.get("messages", [])
        if not messages or not self.compactor:
            return state

        # 调用现有 ContextCompactor
        try:
            compacted_messages = self.compactor.compact(messages)
            state["messages"] = compacted_messages
            state["compacted"] = len(compacted_messages) < len(messages)
        except Exception:
            # 压缩失败不影响主流程
            pass

        return state
