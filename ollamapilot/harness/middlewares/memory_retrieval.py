"""
MemoryRetrievalMiddleware - 记忆检索中间件

包装现有的 MemoryManager，在模型调用前检索相关记忆。
继承 LangChain AgentMiddleware
"""

import time
from typing import Any, Dict, List, Optional, Tuple

from langchain.agents.middleware import AgentMiddleware


class MemoryRetrievalMiddleware(AgentMiddleware):
    """
    记忆检索中间件

    包装现有的 MemoryManager，支持缓存避免重复检索。
    
    优化点：
    1. 每轮对话只检索一次
    2. 缓存检索结果
    3. 线程安全：使用 thread_id 作为缓存键
    
    继承 LangChain AgentMiddleware，统一使用 LangChain 中间件机制
    """

    def __init__(self, memory_manager: Any, cache_ttl: int = 60):
        self.memory_manager = memory_manager
        self._cache_ttl = cache_ttl
        self._retrieval_cache: Dict[str, Tuple[str, List[Any], float]] = {}

    @property
    def name(self) -> str:
        return "MemoryRetrievalMiddleware"

    def before_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any]:
        """在模型调用前检索记忆"""
        messages = state.get("messages", [])
        
        # 提取用户查询（支持对象和字典两种格式）
        query = ""
        for msg in reversed(messages):
            # 支持对象格式
            if hasattr(msg, "content") and hasattr(msg, "type"):
                if msg.type == "human":
                    query = msg.content
                    break
            # 支持字典格式
            elif isinstance(msg, dict):
                if msg.get("type") == "human":
                    query = msg.get("content", "")
                    break

        if not query or not self.memory_manager:
            return state

        # 从 runtime 或 state 获取 thread_id
        thread_id = "default"
        if runtime and hasattr(runtime, "config") and runtime.config:
            thread_id = runtime.config.get("configurable", {}).get("thread_id", "default")
        elif "thread_id" in state:
            thread_id = state["thread_id"]

        # 检查缓存
        memories = self._get_cached_memories(thread_id, query)

        if memories is None:
            # 调用现有 MemoryManager 检索
            try:
                memories = self.memory_manager.recall(query, top_k=5)
            except Exception:
                memories = []
            self._cache_retrieval(thread_id, query, memories)

        # 注入到 state
        state["retrieved_memories"] = memories

        return state

    def _get_cached_memories(
        self,
        thread_id: str,
        query: str
    ) -> Optional[List[Any]]:
        """获取缓存的记忆"""
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
        """缓存检索结果"""
        self._retrieval_cache[thread_id] = (query, memories, time.time())

    def clear_cache(self, thread_id: Optional[str] = None):
        """清除缓存"""
        if thread_id is None:
            self._retrieval_cache.clear()
        elif thread_id in self._retrieval_cache:
            del self._retrieval_cache[thread_id]
