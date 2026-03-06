"""
Skill 层 - 负责 Skill 定义、注册、路由、加载等
"""

from .skill import (
    Skill,
    SkillMetadata,
    SkillToolConfig,
)
from .registry import (
    SkillRegistry,
    simplify_tool_description,
    format_tools_for_small_model,
    auto_register,
)
from .router import SkillRouter, discover_skills_metadata
from .chunker import (
    SkillChunker,
    SkillChunk,
    ChunkMatch,
    AdaptiveSkillLoader,
)

__all__ = [
    # Skill 定义
    "Skill",
    "SkillMetadata",
    "SkillToolConfig",
    # 注册中心
    "SkillRegistry",
    "simplify_tool_description",
    "format_tools_for_small_model",
    "auto_register",
    # 路由
    "SkillRouter",
    "discover_skills_metadata",
    # 分块
    "SkillChunker",
    "SkillChunk",
    "ChunkMatch",
    "AdaptiveSkillLoader",
]
