"""
Skill 模块

提供可插拔的 Skill 系统，支持模块化功能扩展。
支持两种格式：
1. Python Skill (skill.py) - 自定义工具和逻辑
2. Markdown Skill (SKILL.md) - 纯配置，使用内置工具
"""

from ollamapilot.skills.base import Skill, SkillMetadata
from ollamapilot.skills.registry import SkillRegistry
from ollamapilot.skills.default_skill import DefaultSkill
from ollamapilot.skills.loader import MarkdownSkill, load_markdown_skill

__all__ = [
    "Skill",
    "SkillMetadata",
    "SkillRegistry",
    "DefaultSkill",
    "MarkdownSkill",
    "load_markdown_skill",
]
