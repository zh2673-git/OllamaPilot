"""
工具总包

统一管理所有工具：
- builtin: 内置工具（文件、Shell、代码）
- mcp: MCP 工具配置
- registry: 工具注册中心

使用方式：
    from tools import tool_registry
    from tools.builtin import read_file, shell_exec
    
    # 获取工具
    tool = tool_registry.get_tool("read_file")
    
    # 获取所有工具
    all_tools = tool_registry.get_all_tools()
"""

from .registry import ToolRegistry, tool_registry, MCPConfig
from .builtin import (
    FILESYSTEM_TOOLS,
    SHELL_TOOLS,
    CODE_TOOLS,
    ALL_TOOLS,
    TOOL_MAP,
    get_tool,
    list_tools,
)

__all__ = [
    # 注册中心
    "ToolRegistry",
    "tool_registry",
    "MCPConfig",
    # 内置工具
    "FILESYSTEM_TOOLS",
    "SHELL_TOOLS",
    "CODE_TOOLS",
    "ALL_TOOLS",
    "TOOL_MAP",
    "get_tool",
    "list_tools",
]
