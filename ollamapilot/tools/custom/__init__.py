"""
自定义工具目录

放置你的自定义工具文件。
每个工具文件可以包含一个或多个 @tool 装饰的函数。

示例:
    my_tool.py:
        from langchain_core.tools import tool
        
        @tool
        def my_tool(query: str) -> str:
            \"\"\"工具描述\"\"\"
            return f"处理结果: {query}"

然后在 SKILL.md 中使用:
    tools:
      - custom://my_tool.py:my_tool
"""

from pathlib import Path
from typing import List, Optional
from langchain_core.tools import BaseTool


def load_custom_tool(tool_path: str) -> Optional[BaseTool]:
    """
    加载自定义工具
    
    Args:
        tool_path: 工具路径，格式为 "filename.py:function_name"
        
    Returns:
        工具实例或 None
        
    Example:
        >>> tool = load_custom_tool("my_tool.py:my_function")
    """
    if ":" not in tool_path:
        return None
    
    file_name, func_name = tool_path.split(":", 1)
    
    # 构建完整路径
    custom_dir = Path(__file__).parent
    file_path = custom_dir / file_name
    
    if not file_path.exists():
        return None
    
    # 动态导入模块
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"custom.{file_name[:-3]}", 
        file_path
    )
    if not spec or not spec.loader:
        return None
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # 获取工具函数
    tool_func = getattr(module, func_name, None)
    if tool_func is None:
        return None
    
    # 如果是 BaseTool 实例，直接返回
    if isinstance(tool_func, BaseTool):
        return tool_func
    
    # 如果是函数，转换为工具
    from langchain_core.tools import tool
    if callable(tool_func):
        return tool(tool_func)
    
    return None


def discover_custom_tools() -> List[BaseTool]:
    """
    自动发现 custom 目录下的所有工具
    
    Returns:
        工具列表
    """
    tools = []
    custom_dir = Path(__file__).parent
    
    for file_path in custom_dir.glob("*.py"):
        if file_path.name.startswith("_"):
            continue
        
        # 尝试导入模块
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                f"custom.{file_path.stem}",
                file_path
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # 查找所有工具
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, BaseTool):
                        tools.append(attr)
        except Exception:
            continue
    
    return tools
