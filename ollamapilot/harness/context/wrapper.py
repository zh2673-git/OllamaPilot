"""
ContextBuilderWrapper - ContextBuilder 包装器

包装现有的 ContextBuilder，适配 Harness 架构
"""

from typing import Any, Dict, List, Optional


class ContextBuilderWrapper:
    """
    ContextBuilder 包装器
    
    包装现有的 ContextBuilder，提供统一的接口。
    """
    
    def __init__(self, context_builder: Any):
        self._builder = context_builder
    
    def build_four_layer(
        self,
        query: str,
        history: Optional[List[Any]] = None,
        thread_id: str = "default",
        **kwargs
    ) -> Any:
        """
        构建四层 Context
        
        Args:
            query: 用户查询
            history: 对话历史
            thread_id: 会话 ID
            
        Returns:
            Context 对象
        """
        if hasattr(self._builder, 'build_four_layer'):
            return self._builder.build_four_layer(
                query=query,
                history=history,
                thread_id=thread_id,
                **kwargs
            )
        elif hasattr(self._builder, 'build'):
            return self._builder.build(query, history=history)
        else:
            raise AttributeError("ContextBuilder 没有 build 或 build_four_layer 方法")
    
    def __getattr__(self, name: str) -> Any:
        """代理到原始 builder"""
        return getattr(self._builder, name)
