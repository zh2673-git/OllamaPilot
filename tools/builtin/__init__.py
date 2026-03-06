"""
内置工具包

提供基础工具能力：
- filesystem: 文件读写、目录浏览、文件搜索
- shell: 命令执行、脚本执行
- code: 代码搜索、补丁应用、代码统计

使用方式：
    from tools.builtin import ALL_TOOLS
    from tools.builtin.filesystem import read_file, write_file
"""

from .filesystem import TOOLS as FILESYSTEM_TOOLS
from .shell import TOOLS as SHELL_TOOLS
from .code import TOOLS as CODE_TOOLS

# 所有内置工具
ALL_TOOLS = FILESYSTEM_TOOLS + SHELL_TOOLS + CODE_TOOLS

# 工具名称到工具的映射
TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}

def get_tool(name: str):
    """获取指定名称的工具"""
    return TOOL_MAP.get(name)

def list_tools():
    """列出所有可用工具"""
    return list(ALL_TOOLS)

__all__ = [
    "FILESYSTEM_TOOLS",
    "SHELL_TOOLS",
    "CODE_TOOLS",
    "ALL_TOOLS",
    "TOOL_MAP",
    "get_tool",
    "list_tools",
]
