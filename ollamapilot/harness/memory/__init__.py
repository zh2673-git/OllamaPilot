"""
记忆增强模块 - 可选的 LLM 事实提取叠加

借鉴 DeerFlow 的 Memory 设计
支持：
1. LLM 事实提取
2. 语义检索
3. 记忆整合
"""

from ollamapilot.harness.memory.enhanced import EnhancedMemoryManager
from ollamapilot.harness.memory.extractor import FactExtractor

__all__ = [
    "EnhancedMemoryManager",
    "FactExtractor",
]
