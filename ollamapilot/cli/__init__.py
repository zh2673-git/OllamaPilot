"""
OllamaPilot CLI 模块

提供命令行交互界面，包括：
- 会话管理
- 命令自动补全
- 文档索引管理
"""

from .session import Session
from .completer import CommandCompleter
from .chat_manager import OllamaPilotChat

__all__ = ['Session', 'CommandCompleter', 'OllamaPilotChat']
