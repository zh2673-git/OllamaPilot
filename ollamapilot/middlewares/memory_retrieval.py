"""
MemoryRetrievalMiddleware - 记忆检索中间件

在模型调用前，检索相关记忆并注入到 Context 中。
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents.middleware import AgentMiddleware


class MemoryRetrievalMiddleware(AgentMiddleware):
    """
    记忆检索中间件（优化版本）

    职责：
    1. 根据当前查询检索相关记忆
    2. 缓存检索结果，每轮只检索一次
    3. 将记忆注入到状态中

    优化点：
    1. 每轮对话只检索一次，避免重复检索开销
    2. 缓存检索结果，同一轮多次模型调用复用
    3. 线程安全：使用 thread_id 作为缓存键
    """

    def __init__(self, context_builder: Any):
        self.builder = context_builder
        self._retrieval_cache: Dict[str, Tuple[str, List[Any], float]] = {}
        self._cache_ttl = 60

    @property
    def name(self) -> str:
        return "MemoryRetrievalMiddleware"

    def before_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any]:
        messages = state.get("messages", [])
        query = ""
        for msg in reversed(messages):
            if hasattr(msg, "content") and hasattr(msg, "type"):
                if msg.type == "human":
                    query = msg.content
                    break

        if not query:
            return state

        # 从 runtime 获取 thread_id
        thread_id = "default"
        if hasattr(runtime, 'config') and runtime.config:
            thread_id = runtime.config.get("configurable", {}).get("thread_id", "default")

        memories = self._get_cached_memories(thread_id, query)

        if memories is None:
            if hasattr(self.builder, 'memory_manager') and self.builder.memory_manager:
                memories = self.builder.memory_manager.recall(query, top_k=5)
            else:
                memories = []
            self._cache_retrieval(thread_id, query, memories)

        state["retrieved_memories"] = memories

        return state

    def _get_cached_memories(
        self,
        thread_id: str,
        query: str
    ) -> Optional[List[Any]]:
        if thread_id not in self._retrieval_cache:
            return None

        cached_query, memories, timestamp = self._retrieval_cache[thread_id]

        if cached_query != query:
            return None

        if time.time() - timestamp > self._cache_ttl:
            del self._retrieval_cache[thread_id]
            return None

        return memories

    def _cache_retrieval(
        self,
        thread_id: str,
        query: str,
        memories: List[Any]
    ):
        self._retrieval_cache[thread_id] = (query, memories, time.time())

    def clear_cache(self, thread_id: Optional[str] = None):
        if thread_id is None:
            self._retrieval_cache.clear()
        elif thread_id in self._retrieval_cache:
            del self._retrieval_cache[thread_id]
