"""
三层工具架构

借鉴 Claude Code 的 Tool 生命周期设计：
1. 输入校验 (validate)
2. 权限检查 (check_permission)
3. 执行 (execute)
4. 结果渲染 (render_result)

三层共存：
- Layer 1: 现有 Skill（保持不变）
- Layer 2: SkillToolAdapter（自动转换）
- Layer 3: 新 Tool 基类（完整生命周期）
"""

from ollamapilot.harness.tools.base import Tool, ToolContext, ToolResult, ValidationResult, PermissionResult
from ollamapilot.harness.tools.adapter import SkillToolAdapter
from ollamapilot.harness.tools.registry import ToolRegistry

__all__ = [
    "Tool",
    "ToolContext",
    "ToolResult",
    "ValidationResult",
    "PermissionResult",
    "SkillToolAdapter",
    "ToolRegistry",
]
