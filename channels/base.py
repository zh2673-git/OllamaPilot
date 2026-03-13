"""
Channels 基类模块

定义渠道的标准接口和消息格式。
所有渠道必须继承 Channel 基类。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable, Awaitable
import asyncio


@dataclass
class ChannelMessage:
    """
    标准化消息格式
    
    所有渠道接收的消息都会转换为这个标准格式，
    便于统一处理。
    
    Attributes:
        message_id: 消息唯一ID
        user_id: 发送者ID（各平台的用户标识）
        user_name: 发送者昵称
        content: 消息内容（纯文本）
        message_type: 消息类型，"private" 或 "group" 或 "channel"
        channel_name: 渠道名称（qq, feishu, dingtalk）
        group_id: 群ID（群聊时）
        group_name: 群名称
        raw_data: 原始消息数据（保留用于调试和扩展）
        timestamp: 消息时间
        images: 图片URL列表
        at_me: 是否@机器人
        reply_to: 回复的消息ID
    """
    message_id: str
    user_id: str
    user_name: str
    content: str
    message_type: str  # "private" | "group" | "channel"
    channel_name: str = "unknown"  # 渠道名称
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    raw_data: Any = None
    timestamp: datetime = field(default_factory=datetime.now)
    images: List[str] = field(default_factory=list)
    at_me: bool = False
    reply_to: Optional[str] = None
    
    def __post_init__(self):
        """验证消息类型"""
        valid_types = ("private", "group", "channel")
        if self.message_type not in valid_types:
            raise ValueError(f"message_type 必须是 {valid_types}， got {self.message_type}")


@dataclass
class ChannelResponse:
    """
    渠道响应格式
    
    用于规范化渠道回复内容的格式。
    """
    content: str
    message_type: str = "text"  # text | markdown | image | card
    buttons: Optional[List[Dict]] = None
    image_url: Optional[str] = None
    
    def __post_init__(self):
        valid_types = ("text", "markdown", "image", "card")
        if self.message_type not in valid_types:
            self.message_type = "text"


# 消息处理器类型别名
MessageHandler = Callable[[ChannelMessage], Awaitable[str]]


class Channel(ABC):
    """
    渠道基类
    
    所有渠道（QQ、飞书、钉钉等）必须继承此类并实现必要的方法。
    
    Example:
        >>> class MyChannel(Channel):
        ...     name = "my_channel"
        ...     description = "我的渠道"
        ...     
        ...     async def start(self):
        ...         # 启动监听
        ...         pass
        ...     
        ...     async def stop(self):
        ...         # 停止监听
        ...         pass
        ...     
        ...     async def send_message(self, user_id: str, content: str, **kwargs) -> bool:
        ...         # 发送消息
        ...         return True
    """
    
    # 渠道元数据（子类必须覆盖）
    name: str = ""
    description: str = ""
    
    def __init__(self, config: Dict[str, Any], message_handler: MessageHandler):
        """
        初始化渠道
        
        Args:
            config: 渠道配置字典
            message_handler: 消息处理回调函数
        """
        self.config = config
        self.message_handler = message_handler
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    @abstractmethod
    async def start(self):
        """
        启动渠道监听
        
        开始接收消息，当收到消息时调用 self.message_handler
        """
        pass
    
    @abstractmethod
    async def stop(self):
        """
        停止渠道监听
        
        清理资源，停止所有后台任务
        """
        pass
    
    @abstractmethod
    async def send_message(self, user_id: str, content: str, **kwargs) -> bool:
        """
        发送私聊消息给用户
        
        Args:
            user_id: 用户ID
            content: 消息内容
            **kwargs: 扩展参数
                - reply_to: 回复的消息ID
                - images: 图片列表
        
        Returns:
            是否发送成功
        """
        pass
    
    @abstractmethod
    async def send_group_message(self, group_id: str, content: str, **kwargs) -> bool:
        """
        发送群消息
        
        Args:
            group_id: 群ID
            content: 消息内容
            **kwargs: 扩展参数
                - reply_to: 回复的消息ID
        
        Returns:
            是否发送成功
        """
        pass
    
    async def handle_message(self, message: ChannelMessage) -> str:
        """
        处理收到的消息
        
        模板方法：检查权限 → 调用处理器 → 返回结果
        
        Args:
            message: 标准化消息
        
        Returns:
            回复内容
        """
        # 权限检查
        if not self.check_permission(message.user_id):
            return "⛔ 您没有使用权限"
        
        # 群聊中检查是否@机器人（如果配置了）
        if (message.message_type == "group" and 
            self.config.get("at_only_in_group", False) and 
            not message.at_me):
            return ""  # 不响应
        
        try:
            # 调用消息处理器
            response = await self.message_handler(message)
            return response
        except Exception as e:
            return f"❌ 处理消息时出错: {str(e)}"
    
    def check_permission(self, user_id: str) -> bool:
        """
        检查用户权限
        
        检查用户是否在白名单中。如果白名单为空，则允许所有用户。
        
        Args:
            user_id: 用户ID
        
        Returns:
            是否有权限
        """
        whitelist = self.config.get("whitelist", [])
        if not whitelist:
            return True
        return str(user_id) in [str(u) for u in whitelist]
    
    def is_admin(self, user_id: str) -> bool:
        """
        检查用户是否是管理员
        
        Args:
            user_id: 用户ID
        
        Returns:
            是否是管理员
        """
        admin_users = self.config.get("admin_users", [])
        return str(user_id) in [str(u) for u in admin_users]
    
    @property
    def is_running(self) -> bool:
        """渠道是否正在运行"""
        return self._running
    
    def _create_task(self, coro) -> asyncio.Task:
        """
        创建后台任务并跟踪
        
        Args:
            coro: 协程对象
        
        Returns:
            Task 对象
        """
        task = asyncio.create_task(coro)
        self._tasks.append(task)
        task.add_done_callback(lambda t: self._tasks.remove(t) if t in self._tasks else None)
        return task
    
    async def _cancel_all_tasks(self):
        """取消所有后台任务"""
        for task in self._tasks[:]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks.clear()


class ChannelError(Exception):
    """渠道相关错误"""
    pass


class ChannelConfigError(ChannelError):
    """渠道配置错误"""
    pass


class ChannelAPIError(ChannelError):
    """渠道 API 调用错误"""
    
    def __init__(self, message: str, status_code: int = None, response: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response = response
