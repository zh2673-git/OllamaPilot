"""
内置子 Agent

提供通用的子 Agent 实现
"""

from ollamapilot.harness.subagents.builtin.general import GeneralSubAgent
from ollamapilot.harness.subagents.builtin.research import ResearchSubAgent
from ollamapilot.harness.subagents.builtin.code import CodeSubAgent

__all__ = [
    "GeneralSubAgent",
    "ResearchSubAgent",
    "CodeSubAgent",
]
