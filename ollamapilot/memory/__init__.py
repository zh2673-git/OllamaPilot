"""
系统记忆模块 - 跨会话长期记忆

与 GraphRAG 的区别：
- SystemMemory: Agent 自动维护，对用户透明
- GraphRAG: 用户主动使用，通过 Skill 触发

三种类型：
- 语义记忆：用户偏好、重要事实
- 程序记忆：Skill 使用模式
- 情景记忆：重要对话摘要

特性：
- 可选向量检索：通过 enable_vector_search 参数控制
- 统一接口：无论是否启用向量，接口完全一致
"""

from .types import MemoryType, MemoryEntry
from .system_memory import SystemMemory

__all__ = [
    "MemoryType",
    "MemoryEntry",
    "SystemMemory",
]
