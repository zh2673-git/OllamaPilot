"""
Skill 模块

提供可插拔的 Skill 系统，支持模块化功能扩展。
"""

from ollamapilot.skills.base import (
    Skill,
    SkillMetadata,
    DefaultSkill,
    MarkdownSkill,
    SkillRegistry,
    load_markdown_skill,
)

__all__ = [
    "Skill",
    "SkillMetadata",
    "DefaultSkill",
    "MarkdownSkill",
    "SkillRegistry",
    "load_markdown_skill",
]
