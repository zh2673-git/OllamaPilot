"""
会话管理模块
"""

from dataclasses import dataclass
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
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
    
    def update(self):
        """更新会话时间"""
        self.updated_at = datetime.now()
    
    def increment_message(self):
        """增加消息计数"""
        self.message_count += 1
        self.update()
