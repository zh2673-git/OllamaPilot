"""
会话管理模块
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Session:
    """会话对象，保存会话状态"""
    session_id: str
    name: str
    model_name: str
    embedding_model: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    message_count: int = 0
    description: str = ""  # 会话描述/主题
    _source: str = "memory"  # 来源：memory（内存创建）或 database（数据库恢复）
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    @property
    def source(self) -> str:
        """获取会话来源"""
        return self._source
    
    @source.setter
    def source(self, value: str):
        """设置会话来源"""
        self._source = value
    
    @property
    def is_from_database(self) -> bool:
        """是否从数据库恢复"""
        return self._source == "database"
    
    def update(self):
        """更新会话时间"""
        self.updated_at = datetime.now()
    
    def increment_message(self):
        """增加消息计数"""
        self.message_count += 1
        self.update()
    
    def rename(self, new_name: str):
        """重命名会话"""
        self.name = new_name
        self.update()
    
    def set_description(self, description: str):
        """设置会话描述"""
        self.description = description
        self.update()
    
    def get_display_info(self) -> str:
        """获取显示信息"""
        source_marker = "📦" if self.is_from_database else "💾"
        return f"{source_marker} {self.name} ({self.message_count} 条消息)"
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "name": self.name,
            "model_name": self.model_name,
            "embedding_model": self.embedding_model,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "message_count": self.message_count,
            "description": self.description,
            "source": self._source
        }
