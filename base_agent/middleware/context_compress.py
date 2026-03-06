"""
ContextCompressionMiddleware - 上下文压缩

当对话历史过长时，自动压缩上下文以适应小模型的上下文限制
提供摘要生成、消息裁剪等功能
"""

from typing import Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from .base import AgentMiddleware, AgentState


class ContextCompressionMiddleware(AgentMiddleware):
    """
    上下文压缩中间件
    
    问题场景：
    - 小模型上下文窗口有限（如 4k、8k）
    - 长对话历史导致超出限制
    - 需要保留重要信息同时减少 Token 使用
    
    解决方案：
    - 监控 Token 使用量
    - 当接近限制时压缩历史消息
    - 保留最近消息，摘要旧消息
    
    示例:
        middleware = ContextCompressionMiddleware(
            max_messages=20,      # 最多保留20条消息
            max_tokens=3000,      # Token限制
            summarize_threshold=10  # 超过10条时触发摘要
        )
    """
    
    def __init__(
        self,
        max_messages: Optional[int] = 20,
        max_tokens: Optional[int] = 3000,
        summarize_threshold: int = 10,
        keep_recent: int = 4,
        compression_ratio: float = 0.3
    ):
        """
        初始化中间件
        
        Args:
            max_messages: 最大保留消息数
            max_tokens: 最大 Token 数（估算）
            summarize_threshold: 触发摘要的消息数阈值
            keep_recent: 始终保留的最近消息数
            compression_ratio: 摘要压缩比例
        """
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.summarize_threshold = summarize_threshold
        self.keep_recent = keep_recent
        self.compression_ratio = compression_ratio
        self._compression_stats = {
            "times_compressed": 0,
            "messages_removed": 0,
            "messages_summarized": 0
        }
    
    def before_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        在模型调用前压缩上下文
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典
        """
        messages = list(state.messages)
        if not messages:
            return None
        
        original_count = len(messages)
        
        # 策略1: 限制消息数量
        if self.max_messages and len(messages) > self.max_messages:
            messages = self._limit_messages(messages)
        
        # 策略2: 摘要旧消息
        if len(messages) > self.summarize_threshold:
            messages = self._summarize_old_messages(messages)
        
        # 策略3: Token 估算和裁剪
        if self.max_tokens:
            messages = self._limit_by_tokens(messages)
        
        # 更新统计
        if len(messages) < original_count:
            self._compression_stats["times_compressed"] += 1
            self._compression_stats["messages_removed"] += (original_count - len(messages))
        
        if len(messages) < original_count:
            return {"messages": messages}
        
        return None
    
    def _limit_messages(self, messages: list) -> list:
        """
        限制消息数量
        
        保留系统消息和最近的消息
        
        Args:
            messages: 原始消息列表
            
        Returns:
            限制后的消息列表
        """
        # 分离系统消息和其他消息
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        
        # 计算可以保留的其他消息数
        available_slots = self.max_messages - len(system_msgs)
        
        if available_slots <= 0:
            # 只保留系统消息和最后一条
            return system_msgs + other_msgs[-1:] if other_msgs else system_msgs
        
        # 保留最近的消息
        kept_msgs = other_msgs[-available_slots:]
        
        return system_msgs + kept_msgs
    
    def _summarize_old_messages(self, messages: list) -> list:
        """
        摘要旧消息
        
        保留系统消息、最近消息，将中间消息替换为摘要
        
        Args:
            messages: 原始消息列表
            
        Returns:
            摘要后的消息列表
        """
        # 分离系统消息
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]
        
        # 如果消息不够多，不需要摘要
        if len(other_msgs) <= self.keep_recent + 2:
            return messages
        
        # 保留最近的消息
        recent_msgs = other_msgs[-self.keep_recent:]
        
        # 中间的消息生成摘要
        middle_msgs = other_msgs[:-self.keep_recent]
        
        # 生成简单摘要
        summary = self._generate_summary(middle_msgs)
        
        if summary:
            summary_msg = SystemMessage(content=f"[历史对话摘要] {summary}")
            return system_msgs + [summary_msg] + recent_msgs
        
        return system_msgs + recent_msgs
    
    def _generate_summary(self, messages: list) -> str:
        """
        生成消息摘要
        
        Args:
            messages: 需要摘要的消息列表
            
        Returns:
            摘要文本
        """
        # 统计对话轮数
        human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
        ai_count = sum(1 for m in messages if isinstance(m, AIMessage))
        tool_count = sum(1 for m in messages if isinstance(m, ToolMessage))
        
        # 提取关键信息
        topics = []
        for msg in messages:
            if isinstance(msg, HumanMessage) and hasattr(msg, 'content'):
                content = msg.content
                if len(content) > 10:
                    # 提取前30个字符作为主题提示
                    topic = content[:30] + "..." if len(content) > 30 else content
                    topics.append(topic)
        
        # 构建摘要
        summary_parts = []
        
        if human_count > 0:
            summary_parts.append(f"共{human_count}轮对话")
        
        if tool_count > 0:
            summary_parts.append(f"执行了{tool_count}次工具调用")
        
        if topics:
            # 只保留前3个主题
            summary_parts.append(f"讨论主题: {'; '.join(topics[:3])}")
        
        return " | ".join(summary_parts) if summary_parts else ""
    
    def _limit_by_tokens(self, messages: list) -> list:
        """
        基于 Token 估算限制消息
        
        简单的启发式估算：
        - 英文: 1 token ≈ 4 characters
        - 中文: 1 token ≈ 1 character
        
        Args:
            messages: 原始消息列表
            
        Returns:
            限制后的消息列表
        """
        total_tokens = 0
        kept_indices = []
        
        # 从后向前计算，优先保留最近的消息
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            
            # 估算 Token 数
            token_count = self._estimate_tokens(msg)
            
            # 系统消息必须保留
            if isinstance(msg, SystemMessage):
                kept_indices.append(i)
                total_tokens += token_count
                continue
            
            # 检查是否超出限制
            if total_tokens + token_count > self.max_tokens:
                break
            
            kept_indices.append(i)
            total_tokens += token_count
        
        # 按原始顺序重建消息列表
        kept_indices.sort()
        return [messages[i] for i in kept_indices]
    
    def _estimate_tokens(self, message) -> int:
        """
        估算消息的 Token 数
        
        Args:
            message: 消息对象
            
        Returns:
            估算的 Token 数
        """
        if not hasattr(message, 'content'):
            return 10  # 默认估算
        
        content = message.content
        if not content:
            return 10
        
        # 简单估算：中文字符 ≈ 1 token，其他 ≈ 0.25 token
        chinese_chars = sum(1 for c in content if '\u4e00' <= c <= '\u9fff')
        other_chars = len(content) - chinese_chars
        
        return chinese_chars + int(other_chars / 4) + 10  # +10 for message overhead
    
    def get_stats(self) -> dict:
        """获取压缩统计"""
        return {
            "times_compressed": self._compression_stats["times_compressed"],
            "messages_removed": self._compression_stats["messages_removed"],
            "messages_summarized": self._compression_stats["messages_summarized"],
            "config": {
                "max_messages": self.max_messages,
                "max_tokens": self.max_tokens,
                "summarize_threshold": self.summarize_threshold
            }
        }
