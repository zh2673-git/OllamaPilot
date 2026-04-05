"""
OllamaPilot Harness - v0.6.0 增强型架构

核心原则：增强而非替换
- 保留：四层 Context、Skill-Middleware、文件驱动记忆、强制回复保底
- 增强：中间件链编排、Tool 生命周期管理、沙箱安全执行、LLM 事实提取、子 Agent 委托

架构借鉴：
- DeerFlow: 中间件链、虚拟路径、LLM 事实提取、子 Agent
- Claude Code: Tool 生命周期（校验/权限/执行/渲染）
- OllamaPilot: 四层 Context、强制回复保底、Token 预算优化
"""

from ollamapilot.harness.agent import OllamaPilotHarnessAgent, create_harness_agent
from ollamapilot.harness.middlewares.chain import MiddlewareChain
from ollamapilot.harness.tools.base import Tool, ToolContext, ToolResult
from ollamapilot.harness.tools.adapter import SkillToolAdapter

__all__ = [
    "OllamaPilotHarnessAgent",
    "create_harness_agent",
    "MiddlewareChain",
    "Tool",
    "ToolContext",
    "ToolResult",
    "SkillToolAdapter",
]
