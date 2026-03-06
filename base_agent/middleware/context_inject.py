"""
Context Inject 中间件 - LangChain v1+ 兼容版本

在模型调用前注入上下文信息
"""

from typing import Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from .base import AgentMiddleware, AgentState


class ContextInjectMiddleware(AgentMiddleware):
    """
    上下文注入中间件
    
    在模型调用前注入额外的上下文信息，如：
    - 当前时间
    - 用户信息
    - 会话上下文
    
    示例:
        middleware = ContextInjectMiddleware(
            context_provider=lambda: {
                "current_time": datetime.now().isoformat(),
                "user_name": "张三"
            }
        )
    """
    
    def __init__(
        self, 
        context_provider: Optional[callable] = None,
        inject_format: str = "system"
    ):
        """
        初始化中间件
        
        Args:
            context_provider: 上下文提供者函数，返回 dict
            inject_format: 注入格式，"system" 或 "user_prefix"
        """
        self.context_provider = context_provider
        self.inject_format = inject_format
    
    def before_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        在模型调用前注入上下文
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新或 None
        """
        if self.context_provider is None:
            return None
        
        try:
            context = self.context_provider()
        except Exception:
            return None
        
        if not context:
            return None
        
        messages = list(state.messages)
        
        if self.inject_format == "system":
            # 格式化为系统消息追加
            context_text = self._format_context(context)
            
            # 查找现有系统消息
            system_idx = None
            for i, msg in enumerate(messages):
                if isinstance(msg, SystemMessage):
                    system_idx = i
                    break
            
            if system_idx is not None:
                # 追加到现有系统消息
                existing_msg = messages[system_idx]
                if hasattr(existing_msg, 'content'):
                    existing_msg.content += f"\n\n{context_text}"
            else:
                # 添加新的系统消息
                messages.insert(0, SystemMessage(content=context_text))
        
        elif self.inject_format == "user_prefix":
            # 作为用户消息的前缀
            context_text = self._format_context(context)
            
            # 查找最后一条用户消息
            for i in range(len(messages) - 1, -1, -1):
                msg = messages[i]
                is_human = isinstance(msg, HumanMessage)
                
                if is_human:
                    # 在消息前添加上下文
                    if hasattr(msg, 'content'):
                        msg.content = f"[{context_text}]\n{msg.content}"
                    break
        
        return {"messages": messages}
    
    def _format_context(self, context: dict) -> str:
        """
        格式化上下文为文本
        
        Args:
            context: 上下文字典
            
        Returns:
            格式化后的文本
        """
        lines = ["【上下文信息】"]
        for key, value in context.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
