"""
渠道会话管理器

为 Channels（QQ、飞书、钉钉等）提供用户级别的会话持久化
每个用户有独立的对话历史，保存在 SQLite 数据库中
"""

import os
import sqlite3
import threading
from pathlib import Path
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, asdict
import json
from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver


@dataclass
class UserSession:
    """用户会话信息"""
    user_id: str
    channel_name: str
    session_id: str
    created_at: str
    updated_at: str
    message_count: int = 0
    
    def to_dict(self) -> dict:
        return asdict(self)


class ChannelSessionManager:
    """
    渠道会话管理器
    
    管理多个用户的对话会话，每个用户有独立的：
    - SQLite 数据库连接
    - SqliteSaver checkpointer
    - 会话元数据
    
    使用示例:
        manager = ChannelSessionManager("./data/channel_sessions")
        checkpointer = manager.get_checkpointer("qq", "user_123")
        # 使用 checkpointer 创建 agent
        # ...
        manager.close_user_session("qq", "user_123")
    """
    
    def __init__(self, base_dir: str = "./data/channel_sessions"):
        """
        初始化会话管理器
        
        Args:
            base_dir: 会话数据存储目录
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 用户会话缓存: {(channel, user_id): SqliteSaver}
        self._user_checkpointers: Dict[tuple, SqliteSaver] = {}
        
        # 线程锁，保护并发访问
        self._lock = threading.RLock()
        
        # 初始化元数据数据库
        self._init_metadata_db()
    
    def _init_metadata_db(self):
        """初始化会话元数据数据库"""
        meta_path = self.base_dir / "sessions_meta.db"
        conn = sqlite3.connect(str(meta_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                message_count INTEGER DEFAULT 0,
                UNIQUE(channel_name, user_id, session_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _get_user_db_path(self, channel: str, user_id: str) -> Path:
        """获取用户数据库文件路径"""
        # 按渠道分目录，避免文件名冲突
        channel_dir = self.base_dir / channel
        channel_dir.mkdir(exist_ok=True)
        
        # 使用用户 ID 作为文件名（进行简单编码避免特殊字符）
        safe_user_id = user_id.replace("/", "_").replace("\\", "_")
        return channel_dir / f"{safe_user_id}.db"
    
    def get_checkpointer(self, channel: str, user_id: str) -> SqliteSaver:
        """
        获取用户的 checkpointer
        
        如果用户没有活跃的 checkpointer，会创建新的连接
        
        Args:
            channel: 渠道名称 (qq, feishu, dingtalk)
            user_id: 用户 ID
            
        Returns:
            SqliteSaver 实例
        """
        key = (channel, user_id)
        
        with self._lock:
            # 检查是否已有缓存的 checkpointer
            if key in self._user_checkpointers:
                return self._user_checkpointers[key]
            
            # 创建新的数据库连接
            db_path = self._get_user_db_path(channel, user_id)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            
            # 创建 SqliteSaver
            checkpointer = SqliteSaver(conn)
            
            # 缓存起来
            self._user_checkpointers[key] = checkpointer
            
            # 记录会话元数据
            self._record_session_start(channel, user_id)
            
            return checkpointer
    
    def _record_session_start(self, channel: str, user_id: str):
        """记录会话开始"""
        meta_path = self.base_dir / "sessions_meta.db"
        conn = sqlite3.connect(str(meta_path))
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        session_id = f"{channel}_{user_id}_{now}"
        
        try:
            cursor.execute("""
                INSERT OR REPLACE INTO user_sessions 
                (channel_name, user_id, session_id, created_at, updated_at, message_count)
                VALUES (?, ?, ?, ?, ?, COALESCE(
                    (SELECT message_count FROM user_sessions 
                     WHERE channel_name = ? AND user_id = ? AND session_id = ?), 0
                ))
            """, (channel, user_id, session_id, now, now, 
                  channel, user_id, session_id))
            conn.commit()
        except Exception as e:
            print(f"⚠️ 记录会话元数据失败: {e}")
        finally:
            conn.close()
    
    def update_session_activity(self, channel: str, user_id: str, 
                                message_count: int = 1):
        """
        更新会话活动时间
        
        Args:
            channel: 渠道名称
            user_id: 用户 ID
            message_count: 新增消息数
        """
        meta_path = self.base_dir / "sessions_meta.db"
        conn = sqlite3.connect(str(meta_path))
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        try:
            cursor.execute("""
                UPDATE user_sessions 
                SET updated_at = ?, message_count = message_count + ?
                WHERE channel_name = ? AND user_id = ?
            """, (now, message_count, channel, user_id))
            conn.commit()
        except Exception as e:
            print(f"⚠️ 更新会话活动失败: {e}")
        finally:
            conn.close()
    
    def get_user_sessions(self, channel: str, 
                          user_id: str) -> List[UserSession]:
        """
        获取用户的所有会话历史
        
        Args:
            channel: 渠道名称
            user_id: 用户 ID
            
        Returns:
            会话列表
        """
        meta_path = self.base_dir / "sessions_meta.db"
        conn = sqlite3.connect(str(meta_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT channel_name, user_id, session_id, created_at, 
                   updated_at, message_count
            FROM user_sessions
            WHERE channel_name = ? AND user_id = ?
            ORDER BY updated_at DESC
        """, (channel, user_id))
        
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            sessions.append(UserSession(
                channel_name=row[0],
                user_id=row[1],
                session_id=row[2],
                created_at=row[3],
                updated_at=row[4],
                message_count=row[5]
            ))
        
        return sessions
    
    def list_all_sessions(self, channel: Optional[str] = None) -> List[UserSession]:
        """
        列出所有会话
        
        Args:
            channel: 可选，按渠道筛选
            
        Returns:
            会话列表
        """
        meta_path = self.base_dir / "sessions_meta.db"
        conn = sqlite3.connect(str(meta_path))
        cursor = conn.cursor()
        
        if channel:
            cursor.execute("""
                SELECT channel_name, user_id, session_id, created_at,
                       updated_at, message_count
                FROM user_sessions
                WHERE channel_name = ?
                ORDER BY updated_at DESC
            """, (channel,))
        else:
            cursor.execute("""
                SELECT channel_name, user_id, session_id, created_at,
                       updated_at, message_count
                FROM user_sessions
                ORDER BY updated_at DESC
            """)
        
        rows = cursor.fetchall()
        conn.close()
        
        sessions = []
        for row in rows:
            sessions.append(UserSession(
                channel_name=row[0],
                user_id=row[1],
                session_id=row[2],
                created_at=row[3],
                updated_at=row[4],
                message_count=row[5]
            ))
        
        return sessions
    
    def close_user_session(self, channel: str, user_id: str):
        """
        关闭用户的会话连接
        
        Args:
            channel: 渠道名称
            user_id: 用户 ID
        """
        key = (channel, user_id)
        
        with self._lock:
            if key in self._user_checkpointers:
                checkpointer = self._user_checkpointers[key]
                # 关闭数据库连接
                if hasattr(checkpointer, 'conn'):
                    checkpointer.conn.close()
                del self._user_checkpointers[key]
    
    def close_all_sessions(self):
        """关闭所有会话连接"""
        with self._lock:
            for key, checkpointer in list(self._user_checkpointers.items()):
                if hasattr(checkpointer, 'conn'):
                    checkpointer.conn.close()
            self._user_checkpointers.clear()
    
    def get_session_stats(self) -> Dict[str, Any]:
        """获取会话统计信息"""
        meta_path = self.base_dir / "sessions_meta.db"
        conn = sqlite3.connect(str(meta_path))
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(DISTINCT user_id) FROM user_sessions")
        total_users = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM user_sessions")
        total_sessions = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(message_count) FROM user_sessions")
        total_messages = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_users": total_users,
            "total_sessions": total_sessions,
            "total_messages": total_messages,
            "active_sessions_in_memory": len(self._user_checkpointers)
        }


# 全局会话管理器实例（单例模式）
_session_manager: Optional[ChannelSessionManager] = None


def get_session_manager(base_dir: str = "./data/channel_sessions") -> ChannelSessionManager:
    """
    获取全局会话管理器实例
    
    Args:
        base_dir: 会话数据存储目录
        
    Returns:
        ChannelSessionManager 实例
    """
    global _session_manager
    if _session_manager is None:
        _session_manager = ChannelSessionManager(base_dir)
    return _session_manager
