"""
渠道注册表模块

参考 nanobot 的设计，实现渠道的自动发现和注册。
"""

from typing import Dict, Type, Optional, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from .base import Channel

logger = logging.getLogger(__name__)

_channel_registry: Dict[str, Type["Channel"]] = {}


def register_channel(channel_class: Type["Channel"]) -> Type["Channel"]:
    """
    装饰器：注册渠道类
    
    Usage:
        @register_channel
        class MyChannel(Channel):
            name = "my_channel"
            ...
    """
    if not hasattr(channel_class, 'name') or not channel_class.name:
        raise ValueError(f"渠道类 {channel_class.__name__} 必须定义 name 属性")
    
    _channel_registry[channel_class.name] = channel_class
    logger.debug(f"已注册渠道: {channel_class.name}")
    return channel_class


def get_channel(name: str) -> Optional[Type["Channel"]]:
    """
    获取渠道类
    
    Args:
        name: 渠道名称
    
    Returns:
        渠道类，如果不存在返回 None
    """
    return _channel_registry.get(name)


def list_channels() -> Dict[str, Type["Channel"]]:
    """
    列出所有已注册的渠道
    
    Returns:
        渠道名称到渠道类的映射
    """
    return _channel_registry.copy()


def auto_discover_channels():
    """
    自动发现 channels 包中的所有渠道
    
    会自动导入所有渠道模块，触发 @register_channel 装饰器注册。
    """
    from . import qq, feishu, dingtalk
    
    logger.info(f"已自动发现 {len(_channel_registry)} 个渠道: {list(_channel_registry.keys())}")
