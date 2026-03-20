"""
CompactionMiddleware - 上下文压缩中间件

在模型调用前，检查上下文长度并触发压缩。
"""

from typing import Any, Dict, List, Optional

from langchain.agents.middleware import AgentMiddleware


class CompactionMiddleware(AgentMiddleware):
    """
    上下文压缩中间件

    职责：
    1. 监控上下文长度
    2. 超过阈值时触发压缩
    3. 优先保留重要信息

    执行时机：before_model 钩子
    """

    def __init__(self, compactor: Any):
        self.compactor = compactor

    @property
    def name(self) -> str:
        return "CompactionMiddleware"

    def before_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any]:
        messages = state.get("messages", [])
        if not messages:
            return state

        if self.compactor and hasattr(self.compactor, 'compact'):
            messages = self.compactor.compact(messages)
            state["messages"] = messages

        return state
