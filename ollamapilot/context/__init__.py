"""
Context 管理层 - 构建最优 Context

基于"Context 总纲"理念，所有模块为构建最优 Context 服务。
提供三层架构：实时层、工作层、知识层。
支持四层 Context 架构（L3/L2/L1/L0）。
"""

from .types import (
    ContextLayer,
    Layer,
    RuntimeContext,
    WorkingContext,
    KnowledgeContext,
    SkillContext,
    Context as ContextType,
    ContextPart,
    ToolDefinition,
    Example,
)
from .builder import ContextBuilder, Context
from .optimizer import TokenOptimizer
from .compactor import ContextCompactor, CompressionResult

__all__ = [
    # 类型
    "ContextLayer",
    "Layer",
    "RuntimeContext",
    "WorkingContext",
    "KnowledgeContext",
    "SkillContext",
    "ContextType",
    "ContextPart",
    "ToolDefinition",
    "Example",
    # 四层 Context
    "Context",
    # 构建器
    "ContextBuilder",
    "TokenOptimizer",
    # 压缩器
    "ContextCompactor",
    "CompressionResult",
]