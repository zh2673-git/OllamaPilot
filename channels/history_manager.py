"""
Channels 历史管理器 V2 - 支持多会话

为 Channels（QQ、飞书、钉钉等）提供对话历史的内存存储和 JSON 持久化。
- 每个用户可以有多个会话
- 内存存储保证读写速度
- 定期保存到 JSON 文件
- 程序退出时自动保存
"""

import atexit
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class SessionInfo:
    """会话信息"""
    session_id: str
    name: str
    created_at: float
    updated_at: float
    message_count: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SessionInfo":
        return cls(**data)


class ChannelHistoryManager:
    """
    Channel 对话历史管理器 V2

    设计原则：
    - 每个渠道用户可以有多个会话
    - 每个会话有独立的 JSON 文件
    - 内存存储保证读写速度
    - 定期保存到 JSON 文件防止丢失
    - 程序退出时自动保存
    """

    def __init__(
        self,
        channel: str,
        user_id: str,
        storage_dir: str = "./data/channel_sessions/history",
        auto_save_interval: int = 30,
    ):
        self.channel = channel
        self.user_id = user_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.auto_save_interval = auto_save_interval

        # 用户目录: {storage_dir}/{channel}/{user_id}/
        self.user_dir = self.storage_dir / channel / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)

        # 当前会话 ID
        self._current_session_id: Optional[str] = None

        # 当前会话的消息
        self._messages: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._save_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._dirty = False

        # 注册退出时保存
        atexit.register(self.save)

        # 加载或创建默认会话
        self._load_or_create_default_session()

    def _get_session_file(self, session_id: str) -> Path:
        """获取会话文件路径"""
        return self.user_dir / f"{session_id}.json"

    def _get_session_meta_file(self) -> Path:
        """获取会话元数据文件路径"""
        return self.user_dir / "_sessions.json"

    def _load_sessions_meta(self) -> List[SessionInfo]:
        """加载所有会话元数据"""
        meta_file = self._get_session_meta_file()
        if not meta_file.exists():
            return []

        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [SessionInfo.from_dict(s) for s in data]
        except Exception:
            pass
        return []

    def _save_sessions_meta(self, sessions: List[SessionInfo]):
        """保存会话元数据"""
        meta_file = self._get_session_meta_file()
        try:
            with open(meta_file, "w", encoding="utf-8") as f:
                json.dump([s.to_dict() for s in sessions], f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_session_messages(self, session_id: str) -> List[Dict[str, Any]]:
        """加载指定会话的消息"""
        session_file = self._get_session_file(session_id)
        if not session_file.exists():
            return []

        try:
            with open(session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return []

    def _load_or_create_default_session(self):
        """加载或创建默认会话"""
        sessions = self._load_sessions_meta()

        if sessions:
            # 使用最新的会话
            sessions.sort(key=lambda s: s.updated_at, reverse=True)
            self._current_session_id = sessions[0].session_id
            self._messages = self._load_session_messages(self._current_session_id)
        else:
            # 创建默认会话
            self.create_session("默认会话")

    def create_session(self, name: str = None) -> str:
        """创建新会话

        Args:
            name: 会话名称，默认为 "会话_{时间戳}"

        Returns:
            新会话的 session_id
        """
        session_id = f"sess_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        if not name:
            name = f"会话_{session_id[:8]}"

        now = time.time()
        new_session = SessionInfo(
            session_id=session_id,
            name=name,
            created_at=now,
            updated_at=now,
            message_count=0
        )

        # 保存到元数据
        sessions = self._load_sessions_meta()
        sessions.append(new_session)
        self._save_sessions_meta(sessions)

        # 切换到新会话
        with self._lock:
            # 保存当前会话
            if self._current_session_id and self._dirty:
                self._save_current_session()

            # 切换到新会话
            self._current_session_id = session_id
            self._messages = []
            self._dirty = False

        return session_id

    def switch_session(self, session_id: str) -> bool:
        """切换到指定会话

        Args:
            session_id: 会话 ID（可以是前几位）

        Returns:
            是否切换成功
        """
        sessions = self._load_sessions_meta()

        # 查找匹配的会话
        matched = None
        for s in sessions:
            if s.session_id.startswith(session_id):
                matched = s
                break

        if not matched:
            return False

        with self._lock:
            # 保存当前会话
            if self._current_session_id and self._dirty:
                self._save_current_session()

            # 切换到新会话
            self._current_session_id = matched.session_id
            self._messages = self._load_session_messages(matched.session_id)
            self._dirty = False

        return True

    def list_sessions(self) -> List[SessionInfo]:
        """列出所有会话"""
        sessions = self._load_sessions_meta()
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        return sessions

    def get_current_session_id(self) -> Optional[str]:
        """获取当前会话 ID"""
        return self._current_session_id

    def get_current_session_name(self) -> str:
        """获取当前会话名称"""
        sessions = self._load_sessions_meta()
        for s in sessions:
            if s.session_id == self._current_session_id:
                return s.name
        return "未知会话"

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """重命名会话"""
        sessions = self._load_sessions_meta()
        for s in sessions:
            if s.session_id.startswith(session_id):
                s.name = new_name
                s.updated_at = time.time()
                self._save_sessions_meta(sessions)
                return True
        return False

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        sessions = self._load_sessions_meta()
        matched = None
        for s in sessions:
            if s.session_id.startswith(session_id):
                matched = s
                break

        if not matched:
            return False

        # 删除文件
        session_file = self._get_session_file(matched.session_id)
        try:
            if session_file.exists():
                session_file.unlink()
        except Exception:
            pass

        # 从元数据中移除
        sessions = [s for s in sessions if s.session_id != matched.session_id]
        self._save_sessions_meta(sessions)

        # 如果删除的是当前会话，切换到默认会话
        if self._current_session_id == matched.session_id:
            if sessions:
                self.switch_session(sessions[0].session_id)
            else:
                self.create_session("默认会话")

        return True

    def _save_current_session(self):
        """保存当前会话到文件"""
        if not self._current_session_id:
            return

        session_file = self._get_session_file(self._current_session_id)
        try:
            with open(session_file, "w", encoding="utf-8") as f:
                json.dump(self._messages, f, ensure_ascii=False, indent=2)

            # 更新元数据
            sessions = self._load_sessions_meta()
            for s in sessions:
                if s.session_id == self._current_session_id:
                    s.updated_at = time.time()
                    s.message_count = len(self._messages)
                    break
            self._save_sessions_meta(sessions)

            self._dirty = False
        except Exception:
            pass

    def save(self) -> bool:
        """保存当前会话"""
        with self._lock:
            if not self._dirty:
                return True
            self._save_current_session()
            return not self._dirty

    def start_auto_save(self):
        """启动定期保存线程"""
        if self._save_thread is not None:
            return

        self._stop_event.clear()
        self._save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self._save_thread.start()

    def stop_auto_save(self):
        """停止定期保存线程"""
        if self._save_thread is None:
            return

        self._stop_event.set()
        self._save_thread.join(timeout=2)
        self._save_thread = None

    def _auto_save_loop(self):
        """定期保存循环"""
        while not self._stop_event.wait(self.auto_save_interval):
            if self._dirty:
                self.save()

    def get_messages(self) -> List[Dict[str, Any]]:
        """获取当前会话的所有消息"""
        with self._lock:
            return list(self._messages)

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """添加消息到当前会话"""
        with self._lock:
            msg = {
                "role": role,
                "content": content,
                "timestamp": time.time(),
            }
            if metadata:
                msg["metadata"] = metadata
            self._messages.append(msg)
            self._dirty = True

    def add_human_message(self, content: str):
        """添加用户消息"""
        self.add_message("human", content)

    def add_ai_message(self, content: str):
        """添加 AI 消息"""
        self.add_message("ai", content)

    def add_tool_message(self, content: str, tool_name: str = ""):
        """添加工具消息"""
        self.add_message("tool", content, {"tool_name": tool_name})

    def clear(self):
        """清空当前会话"""
        with self._lock:
            self._messages.clear()
            self._dirty = True

    def close(self):
        """关闭，保存并清理"""
        self.stop_auto_save()
        self.save()
        atexit.unregister(self.save)


class ChannelHistoryManagerRegistry:
    """
    Channel 历史管理器注册中心

    管理所有渠道用户的历史管理器实例
    """

    def __init__(self, storage_dir: str = "./data/channel_sessions/history"):
        self.storage_dir = Path(storage_dir)
        self._managers: Dict[tuple, ChannelHistoryManager] = {}
        self._lock = threading.RLock()

    def get_manager(self, channel: str, user_id: str) -> ChannelHistoryManager:
        """获取或创建历史管理器"""
        key = (channel, user_id)

        with self._lock:
            if key not in self._managers:
                manager = ChannelHistoryManager(
                    channel=channel,
                    user_id=user_id,
                    storage_dir=str(self.storage_dir),
                )
                manager.start_auto_save()
                self._managers[key] = manager

            return self._managers[key]

    def close_all(self):
        """关闭所有历史管理器"""
        with self._lock:
            for manager in self._managers.values():
                manager.close()
            self._managers.clear()


# 全局注册中心实例
_registry: Optional[ChannelHistoryManagerRegistry] = None


def get_history_manager(channel: str, user_id: str) -> ChannelHistoryManager:
    """获取历史管理器"""
    global _registry
    if _registry is None:
        _registry = ChannelHistoryManagerRegistry()
    return _registry.get_manager(channel, user_id)
