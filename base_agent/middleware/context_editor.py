"""
Context Editor 中间件 - LangChain v1+ 兼容版本

动态编辑上下文，如摘要生成、历史消息裁剪等
"""

from typing import Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from .base import AgentMiddleware, AgentState


class ContextEditorMiddleware(AgentMiddleware):
    """
    上下文编辑中间件
    
    在模型调用前对消息列表进行处理，如：
    - 限制消息数量
    - 生成历史摘要
    - 移除敏感信息
    
    示例:
        middleware = ContextEditorMiddleware(max_messages=10)
        
        # 只保留最近的10条消息
    """
    
    def __init__(
        self,
        max_messages: Optional[int] = None,
        max_tokens: Optional[int] = None,
        summarize_threshold: int = 20
    ):
        """
        初始化中间件
        
        Args:
            max_messages: 最大保留消息数
            max_tokens: 最大 token 数（估算）
            summarize_threshold: 触发摘要的消息数阈值
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold
    
    def before_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        在模型调用前编辑上下文
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典
        """
        messages = state.messages.copy()
        
        # 限制消息数量
        if self.max_messages and len(messages) > self.max_messages:
            # 保留系统消息和最近的消息
            system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
            other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
            
            # 保留最近的消息
            kept_msgs = other_msgs[-(self.max_messages - len(system_msgs)):]
            messages = system_msgs + kept_msgs
        
        # 如果消息过多，生成摘要
        if len(messages) > self.summarize_threshold:
            messages = self._summarize_history(messages)
        
        return {"messages": messages}
    
    def _summarize_history(self, messages: list) -> list:
        """
        生成历史消息摘要
        
        Args:
            messages: 消息列表
            
        Returns:
            摘要后的消息列表
        """
        # 保留系统消息
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        
        # 保留最近的几条完整消息
        recent_msgs = messages[-5:] if len(messages) > 5 else messages
        
        # 中间的消息生成摘要
        if len(messages) > len(system_msgs) + 5:
            middle_msgs = messages[len(system_msgs):-5]
            
            # 简单摘要：统计对话轮数
            human_count = sum(1 for m in middle_msgs if isinstance(m, HumanMessage))
            ai_count = sum(1 for m in middle_msgs if isinstance(m, AIMessage))
            
            summary = SystemMessage(
                content=f"[历史对话摘要] 之前进行了 {human_count} 轮用户提问和 {ai_count} 轮助手回复"
            )
            
            return system_msgs + [summary] + recent_msgs
        
        return messages


class SensitiveInfoFilterMiddleware(AgentMiddleware):
    """
    敏感信息过滤中间件
    
    在模型调用前过滤敏感信息，如 API 密钥、密码等
    
    示例:
        middleware = SensitiveInfoFilterMiddleware(
            patterns=[r'sk-[a-zA-Z0-9]{20,}']
        )
    """
    
    def __init__(self, patterns: Optional[list[str]] = None):
        """
        初始化中间件
        
        Args:
            patterns: 敏感信息正则模式列表
        """
        import re
        
        self.patterns = patterns or [
            r'sk-[a-zA-Z0-9]{20,}',  # OpenAI API Key
            r'[a-zA-Z0-9]{32,}',      # 一般长密钥
        ]
        self.replacements = {pattern: "[FILTERED]" for pattern in self.patterns}
    
    def before_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        过滤敏感信息
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典
        """
        import re
        
        messages = state.messages.copy()
        
        for msg in messages:
            if hasattr(msg, 'content') and isinstance(msg.content, str):
                for pattern in self.patterns:
                    msg.content = re.sub(pattern, "[FILTERED]", msg.content)
        
        return {"messages": messages}
