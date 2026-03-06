"""
LangChain v1+ 兼容的中间件基类

基于 langchain.agents.middleware.AgentMiddleware 设计
提供生命周期钩子方法
"""

from typing import Any, Optional, Callable
from abc import ABC, abstractmethod
from langchain_core.messages import AnyMessage


class AgentState(dict):
    """
    Agent 状态类
    
    包含消息列表和其他运行时状态
    """
    def __init__(self, messages: list[AnyMessage] = None, **kwargs):
        super().__init__()
        self["messages"] = messages or []
        self.update(kwargs)
    
    @property
    def messages(self) -> list[AnyMessage]:
        return self.get("messages", [])
    
    @messages.setter
    def messages(self, value: list[AnyMessage]):
        self["messages"] = value


class AgentMiddleware(ABC):
    """
    Agent 中间件基类
    
    所有中间件必须继承此类，实现相应的生命周期钩子方法。
    
    生命周期：
    1. before_agent()    → Agent 执行前
    2. before_model()    → 模型调用前
    3. wrap_model_call() → 包装模型调用（可选）
    4. after_model()     → 模型调用后
    5. wrap_tool_call()  → 包装工具调用（可选，每个工具一次）
    6. after_agent()     → Agent 执行后
    
    示例：
        class MyMiddleware(AgentMiddleware):
            def before_model(self, state: AgentState, config: dict) -> dict:
                # 修改消息
                state["messages"].append(SystemMessage(content="提示词"))
                return {"messages": state["messages"]}
    """
    
    def before_agent(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        Agent 执行前调用
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典或 None
        """
        return None
    
    def before_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        模型调用前调用
        
        可以修改消息列表、注入提示词等
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典或 None
        """
        return None
    
    def wrap_model_call(
        self,
        state: AgentState,
        config: Optional[dict] = None
    ) -> Optional[Callable]:
        """
        包装模型调用
        
        返回一个包装函数，可以控制模型调用的整个过程。
        如果不返回，则使用默认的模型调用。
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            包装函数或 None
        """
        return None
    
    def after_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        模型调用后调用
        
        可以处理响应、触发人工审核等
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典或 None
        """
        return None
    
    def wrap_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        state: AgentState,
        config: Optional[dict] = None
    ) -> Optional[Callable]:
        """
        包装工具调用
        
        返回一个包装函数，可以控制工具调用的整个过程。
        每个工具调用都会触发一次。
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            state: 当前状态
            config: 运行配置
            
        Returns:
            包装函数或 None
        """
        return None
    
    def after_tool_call(
        self,
        tool_name: str,
        tool_result: Any,
        state: AgentState,
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        工具调用后调用
        
        Args:
            tool_name: 工具名称
            tool_result: 工具执行结果
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典或 None
        """
        return None
    
    def after_agent(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        Agent 执行后调用
        
        可以清理资源、保存结果等
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典或 None
        """
        return None


class MiddlewareChain:
    """
    中间件链
    
    管理多个中间件的执行顺序
    """
    
    def __init__(self, middlewares: Optional[list[AgentMiddleware]] = None):
        self.middlewares = middlewares or []
    
    def add(self, middleware: AgentMiddleware):
        """添加中间件"""
        self.middlewares.append(middleware)
    
    def execute_before_agent(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> AgentState:
        """执行所有 before_agent 钩子"""
        for middleware in self.middlewares:
            update = middleware.before_agent(state, config)
            if update:
                state.update(update)
        return state
    
    def execute_before_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> AgentState:
        """执行所有 before_model 钩子"""
        for middleware in self.middlewares:
            update = middleware.before_model(state, config)
            if update:
                state.update(update)
        return state
    
    def execute_after_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> AgentState:
        """执行所有 after_model 钩子"""
        for middleware in self.middlewares:
            update = middleware.after_model(state, config)
            if update:
                state.update(update)
        return state
    
    def execute_after_agent(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> AgentState:
        """执行所有 after_agent 钩子"""
        for middleware in self.middlewares:
            update = middleware.after_agent(state, config)
            if update:
                state.update(update)
        return state
