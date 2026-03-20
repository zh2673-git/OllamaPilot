"""
系统记忆模块 - 跨会话长期记忆

与 GraphRAG 的区别：
- MemoryManager: 被 Context 统管，负责记忆检索和存储
- GraphRAG: 用户主动使用，通过 Skill 触发

V0.5.0 重构：简化架构，移除废弃文件依赖
"""

from .types import MemoryType, MemoryEntry
from .manager import MemoryManager, SearchResult
from .indexer import MemoryIndexer, OllamaEmbeddingWrapper

__all__ = [
    "MemoryType",
    "MemoryEntry",
    "MemoryManager",
    "SearchResult",
    "MemoryIndexer",
    "OllamaEmbeddingWrapper",
]