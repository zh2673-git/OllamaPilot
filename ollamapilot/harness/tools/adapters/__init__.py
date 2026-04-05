"""
工具适配器

将原有工具包装为 Harness Tool 基类
"""

from ollamapilot.harness.tools.adapters.builtin import BuiltinToolAdapter
from ollamapilot.harness.tools.adapters.langchain import LangChainToolAdapter

__all__ = [
    "BuiltinToolAdapter",
    "LangChainToolAdapter",
]
