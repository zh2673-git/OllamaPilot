"""
记忆类型定义
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class MemoryType(Enum):
    """记忆类型"""
    SEMANTIC = "semantic"      # 语义记忆：用户偏好、重要事实
    PROCEDURAL = "procedural"  # 程序记忆：Skill 使用模式
    EPISODIC = "episodic"      # 情景记忆：对话摘要


@dataclass
class MemoryEntry:
    """
    记忆条目
    
    统一存储所有类型的记忆
    """
    id: str
    type: MemoryType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    importance: float = 1.0  # 重要性评分（0-1）
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """从字典创建"""
        return cls(
            id=data["id"],
            type=MemoryType(data["type"]),
            content=data["content"],
            metadata=data.get("metadata", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            importance=data.get("importance", 1.0),
        )
