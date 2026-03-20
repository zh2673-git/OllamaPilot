"""
Channels 历史管理器

为 Channels（QQ、飞书、钉钉等）提供对话历史的内存存储和 JSON 持久化。
- 内存存储保证读写速度
- 定期保存到 JSON 文件
- 程序退出时自动保存
"""

import atexit
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class ChannelHistoryManager:
    """
    Channel 对话历史管理器

    设计原则：
    - 每个渠道用户有独立的历史文件
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

        self._messages: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._save_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._dirty = False

        # 文件路径: {storage_dir}/{channel}/{user_id}.json
        user_dir = self.storage_dir / channel
        user_dir.mkdir(parents=True, exist_ok=True)
        safe_user_id = user_id.replace("/", "_").replace("\\", "_")
        self._storage_file = user_dir / f"{safe_user_id}.json"

        # 注册退出时保存
        atexit.register(self.save)

        # 启动时恢复历史
        self.restore()

    def _load(self) -> List[Dict[str, Any]]:
        """从文件加载历史"""
        if not self._storage_file.exists():
            return []

        try:
            with open(self._storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
        return []

    def save(self) -> bool:
        """保存历史到文件"""
        with self._lock:
            if not self._dirty:
                return True

            try:
                with open(self._storage_file, "w", encoding="utf-8") as f:
                    json.dump(self._messages, f, ensure_ascii=False, indent=2)
                self._dirty = False
                return True
            except Exception:
                return False

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
        """获取所有消息"""
        with self._lock:
            return list(self._messages)

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """添加消息"""
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
        """清空历史"""
        with self._lock:
            self._messages.clear()
            self._dirty = True

    def restore(self):
        """从文件恢复历史"""
        messages = self._load()
        with self._lock:
            self._messages = messages
            self._dirty = False

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
