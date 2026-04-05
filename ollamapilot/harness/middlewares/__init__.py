"""
中间件链系统 - 使用 LangChain AgentMiddleware

所有中间件继承 LangChain 的 AgentMiddleware，统一使用 LangChain 中间件机制。
"""

from ollamapilot.harness.middlewares.base import HarnessMiddleware, MiddlewareResult
from ollamapilot.harness.middlewares.context_injection import ContextInjectionMiddleware
from ollamapilot.harness.middlewares.memory_retrieval import MemoryRetrievalMiddleware
from ollamapilot.harness.middlewares.compaction import CompactionMiddleware
from ollamapilot.harness.middlewares.clarification import ClarificationMiddleware

__all__ = [
    "HarnessMiddleware",
    "MiddlewareResult",
    "ContextInjectionMiddleware",
    "MemoryRetrievalMiddleware",
    "CompactionMiddleware",
    "ClarificationMiddleware",
]
