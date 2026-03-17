"""
Context 管理层 - 构建最优 Context

基于"Context 总纲"理念，所有模块为构建最优 Context 服务。
提供三层架构：实时层、工作层、知识层。
"""

from .types import (
    ContextLayer,
    Layer,
    RuntimeContext,
    WorkingContext,
    KnowledgeContext,
    SkillContext,
    Context,
    ContextPart,
    ToolDefinition,
    Example,
)
from .builder import ContextBuilder
from .optimizer import TokenOptimizer

__all__ = [
    # 类型
    "ContextLayer",
    "Layer",
    "RuntimeContext",
    "WorkingContext",
    "KnowledgeContext",
    "SkillContext",
    "Context",
    "ContextPart",
    "ToolDefinition",
    "Example",
    # 构建器
    "ContextBuilder",
    "TokenOptimizer",
]
