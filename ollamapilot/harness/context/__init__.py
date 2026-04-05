"""
上下文管理模块

包装现有的 ContextBuilder，支持 Harness 架构
"""

from ollamapilot.harness.context.state import AgentState
from ollamapilot.harness.context.wrapper import ContextBuilderWrapper

__all__ = [
    "AgentState",
    "ContextBuilderWrapper",
]
