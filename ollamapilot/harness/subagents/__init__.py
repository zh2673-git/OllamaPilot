"""
子 Agent 系统 - 可选的 Agent 委托机制

借鉴 DeerFlow 的子 Agent 设计
支持：
1. 任务分解
2. Agent 委托
3. 结果聚合
"""

from ollamapilot.harness.subagents.base import SubAgent, SubAgentResult
from ollamapilot.harness.subagents.factory import SubAgentFactory

__all__ = [
    "SubAgent",
    "SubAgentResult",
    "SubAgentFactory",
]
