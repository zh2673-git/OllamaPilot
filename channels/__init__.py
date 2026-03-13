"""
OllamaPilot Channels - 多渠道远程控制模块

支持通过 QQ、飞书、钉钉等即时通讯工具远程调用 Agent。

使用方式:
    >>> from channels import ChannelRunner
    >>> runner = ChannelRunner("config.yaml")
    >>> runner.start()

示例配置见 config.yaml
"""

from .base import Channel, ChannelMessage, ChannelResponse, ChannelError
from .registry import register_channel, get_channel, list_channels, auto_discover_channels
from .runner import ChannelRunner

__version__ = "1.0.0"
__all__ = [
    "Channel",
    "ChannelMessage",
    "ChannelResponse",
    "ChannelError",
    "register_channel",
    "get_channel",
    "list_channels",
    "auto_discover_channels",
    "ChannelRunner",
]
