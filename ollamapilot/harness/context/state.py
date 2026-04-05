"""
AgentState - Agent 状态定义

定义 Harness 架构中的状态数据结构
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AgentState:
    """
    Agent 状态
    
    包含：
    - messages: 消息列表
    - context: 四层 Context
    - memories: 检索到的记忆
    - thread_id: 会话 ID
    - metadata: 元数据
    """
    
    messages: List[Any] = field(default_factory=list)
    context: Optional[Any] = None
    memories: List[str] = field(default_factory=list)
    thread_id: str = "default"
    query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "messages": self.messages,
            "context": self.context,
            "memories": self.memories,
            "thread_id": self.thread_id,
            "query": self.query,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentState":
        """从字典创建"""
        return cls(
            messages=data.get("messages", []),
            context=data.get("context"),
            memories=data.get("memories", []),
            thread_id=data.get("thread_id", "default"),
            query=data.get("query", ""),
            metadata=data.get("metadata", {}),
        )
