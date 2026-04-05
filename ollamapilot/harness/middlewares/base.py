"""
中间件基类 - 适配 LangChain AgentMiddleware

统一使用 LangChain 的中间件机制
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from langchain.agents.middleware import AgentMiddleware


class HarnessMiddleware(AgentMiddleware):
    """
    Harness 中间件基类
    
    继承 LangChain 的 AgentMiddleware，统一使用 LangChain 中间件机制。
    
    执行时机：
    - before_model: 模型调用前
    - after_model: 模型调用后
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """中间件名称"""
        pass
    
    def before_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any]:
        """
        在模型调用前执行
        
        Args:
            state: 当前状态，包含 messages 等
            runtime: LangChain 运行时
            
        Returns:
            Dict: 修改后的状态
        """
        return state
    
    def after_model(self, state: Dict[str, Any], runtime: Any) -> Dict[str, Any]:
        """
        在模型调用后执行
        
        Args:
            state: 当前状态
            runtime: LangChain 运行时
            
        Returns:
            Dict: 修改后的状态
        """
        return state


# 保留 MiddlewareResult 用于兼容
class MiddlewareResult:
    """中间件执行结果（兼容类）"""
    state: Dict[str, Any]
    interrupted: bool = False
    interrupt_message: Optional[str] = None
    
    def __init__(self, state: Dict[str, Any], interrupted: bool = False, interrupt_message: Optional[str] = None):
        self.state = state
        self.interrupted = interrupted
        self.interrupt_message = interrupt_message
    
    @classmethod
    def continue_(cls, state: Dict[str, Any]) -> "MiddlewareResult":
        """继续执行"""
        return cls(state=state, interrupted=False)
    
    @classmethod
    def interrupt(cls, state: Dict[str, Any], message: str) -> "MiddlewareResult":
        """中断执行"""
        return cls(state=state, interrupted=True, interrupt_message=message)
