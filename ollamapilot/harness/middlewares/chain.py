"""
中间件链 - 链式编排中间件执行

借鉴 DeerFlow 的 MiddlewareChain 设计
支持条件中断，小模型友好

注意：现在中间件继承 LangChain AgentMiddleware，
此类保留用于向后兼容和特殊场景
"""

from typing import Any, Dict, List, Optional
import logging

from ollamapilot.harness.middlewares.base import HarnessMiddleware, MiddlewareResult

logger = logging.getLogger("ollamapilot.harness.middlewares")


class MiddlewareChain:
    """
    中间件链
    
    职责：
    1. 按顺序执行中间件
    2. 支持条件中断
    3. 错误处理和恢复
    
    注意：现在 Harness 中间件继承 LangChain AgentMiddleware，
    此类主要用于向后兼容和手动调用场景。
    LangChain 的 Agent 会自动调用中间件的 before_model/after_model 方法。
    
    使用示例：
        chain = MiddlewareChain()
        chain.add(ContextInjectionMiddleware(context_builder))
        chain.add(MemoryRetrievalMiddleware(memory_manager))
        chain.add(ClarificationMiddleware())  # 支持中断
        
        result = await chain.process(state)
        if result.interrupted:
            return result.interrupt_message
    """
    
    def __init__(self):
        self.middlewares: List[HarnessMiddleware] = []
        self._enabled = True
    
    def add(self, middleware: HarnessMiddleware) -> "MiddlewareChain":
        """添加中间件到链尾"""
        self.middlewares.append(middleware)
        return self
    
    def insert(self, index: int, middleware: HarnessMiddleware) -> "MiddlewareChain":
        """在指定位置插入中间件"""
        self.middlewares.insert(index, middleware)
        return self
    
    def remove(self, name: str) -> bool:
        """移除指定名称的中间件"""
        for i, m in enumerate(self.middlewares):
            if m.name == name:
                self.middlewares.pop(i)
                return True
        return False
    
    def get(self, name: str) -> Optional[HarnessMiddleware]:
        """获取指定名称的中间件"""
        for m in self.middlewares:
            if m.name == name:
                return m
        return None
    
    def enable(self):
        """启用中间件链"""
        self._enabled = True
    
    def disable(self):
        """禁用中间件链"""
        self._enabled = False
    
    async def process_before_model(self, state: Dict[str, Any]) -> MiddlewareResult:
        """
        执行 before_model 链
        
        Args:
            state: 当前状态
            
        Returns:
            MiddlewareResult: 最终结果，可能中断
        """
        if not self._enabled:
            return MiddlewareResult.continue_(state)
        
        current_state = state.copy()
        
        for middleware in self.middlewares:
            try:
                # 调用 LangChain 风格的 before_model
                if hasattr(middleware, 'before_model'):
                    result_state = middleware.before_model(current_state, None)
                    current_state = result_state
                
                # 检查是否需要中断（通过 state 标记）
                if current_state.get('needs_clarification'):
                    return MiddlewareResult.interrupt(
                        current_state, 
                        current_state.get('clarification_message', '需要澄清')
                    )
                    
            except Exception as e:
                logger.warning(f"中间件 {middleware.name} 执行失败: {e}")
                # 继续执行，不中断
                continue
        
        return MiddlewareResult.continue_(current_state)
    
    async def process_after_model(
        self, 
        state: Dict[str, Any], 
        response: Any
    ) -> MiddlewareResult:
        """
        执行 after_model 链
        
        Args:
            state: 当前状态
            response: 模型响应
            
        Returns:
            MiddlewareResult: 最终结果
        """
        if not self._enabled:
            return MiddlewareResult.continue_(state)
        
        current_state = state.copy()
        
        for middleware in self.middlewares:
            try:
                # 调用 LangChain 风格的 after_model
                if hasattr(middleware, 'after_model'):
                    result_state = middleware.after_model(current_state, None)
                    current_state = result_state
                    
            except Exception as e:
                logger.warning(f"中间件 {middleware.name} after_model 失败: {e}")
                continue
        
        return MiddlewareResult.continue_(current_state)
    
    def __len__(self) -> int:
        """返回中间件数量"""
        return len(self.middlewares)
    
    def __iter__(self):
        """迭代中间件"""
        return iter(self.middlewares)
