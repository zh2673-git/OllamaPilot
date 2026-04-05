"""
ToolRegistry - 工具注册表

管理所有工具的统一注册和发现。
支持三层工具共存：
- 现有 Skill（通过适配器）
- 新 Tool 基类
- 内置工具
"""

from typing import Any, Callable, Dict, List, Optional, Type
import logging

from ollamapilot.harness.tools.base import Tool
from ollamapilot.harness.tools.adapter import SkillToolAdapter
from ollamapilot.harness.tools.adapters.builtin import BuiltinToolAdapter
from ollamapilot.harness.tools.adapters.langchain import LangChainToolAdapter

logger = logging.getLogger("ollamapilot.harness.tools.registry")


class ToolRegistry:
    """
    工具注册表

    统一管理所有工具的注册、发现和转换。
    
    功能：
    1. 注册新 Tool 基类工具
    2. 注册现有 Skill（自动适配）
    3. 按名称查找工具
    4. 获取所有工具（LangChain 格式）
    """

    def __init__(self):
        # 新 Tool 基类工具
        self._tools: Dict[str, Tool] = {}
        
        # Skill 适配器
        self._skill_adapters: Dict[str, SkillToolAdapter] = {}
        
        # 缓存的 LangChain 工具
        self._langchain_tools: Dict[str, Any] = {}
        
        # 工具名称映射（别名）
        self._aliases: Dict[str, str] = {}

    def register_tool(self, tool: Tool, aliases: Optional[List[str]] = None) -> None:
        """
        注册新 Tool 基类工具
        
        Args:
            tool: Tool 实例
            aliases: 别名列表
        """
        self._tools[tool.name] = tool
        
        # 注册别名
        if aliases:
            for alias in aliases:
                self._aliases[alias] = tool.name
        
        # 清除缓存
        self._clear_cache()
        
        logger.debug(f"注册工具: {tool.name}")

    def register_tool_class(self, tool_class: Type[Tool], aliases: Optional[List[str]] = None) -> None:
        """
        注册 Tool 类（自动实例化）
        
        Args:
            tool_class: Tool 子类
            aliases: 别名列表
        """
        tool = tool_class()
        self.register_tool(tool, aliases)

    def register_skill(self, skill: Any) -> None:
        """
        注册现有 Skill（自动适配）
        
        Args:
            skill: Skill 实例
        """
        adapter = SkillToolAdapter(skill)
        self._skill_adapters[adapter.name] = adapter
        
        # 清除缓存
        self._clear_cache()
        
        logger.debug(f"注册 Skill 适配器: {adapter.name}")

    def register_skills(self, skills: List[Any]) -> None:
        """
        批量注册 Skill
        
        Args:
            skills: Skill 实例列表
        """
        for skill in skills:
            self.register_skill(skill)

    def register_builtin_tool(self, builtin_tool_func: Callable, aliases: Optional[List[str]] = None) -> None:
        """
        注册原有内置工具（@tool 装饰的函数）
        
        Args:
            builtin_tool_func: 原有工具函数
            aliases: 别名列表
        """
        adapter = BuiltinToolAdapter(builtin_tool_func)
        self._tools[adapter.name] = adapter
        
        # 注册别名
        if aliases:
            for alias in aliases:
                self._aliases[alias] = adapter.name
        
        # 清除缓存
        self._clear_cache()
        
        logger.debug(f"注册内置工具适配器: {adapter.name}")

    def register_langchain_tool(self, langchain_tool: Any, aliases: Optional[List[str]] = None) -> None:
        """
        注册 LangChain 工具
        
        Args:
            langchain_tool: LangChain BaseTool 实例
            aliases: 别名列表
        """
        adapter = LangChainToolAdapter(langchain_tool)
        self._tools[adapter.name] = adapter
        
        # 注册别名
        if aliases:
            for alias in aliases:
                self._aliases[alias] = adapter.name
        
        # 清除缓存
        self._clear_cache()
        
        logger.debug(f"注册 LangChain 工具适配器: {adapter.name}")

    def get_tool(self, name: str) -> Optional[Tool]:
        """
        按名称获取工具
        
        Args:
            name: 工具名称或别名
            
        Returns:
            Tool 实例或 None
        """
        # 解析别名
        if name in self._aliases:
            name = self._aliases[name]
        
        # 先查找新 Tool
        if name in self._tools:
            return self._tools[name]
        
        # 再查找 Skill 适配器
        if name in self._skill_adapters:
            return self._skill_adapters[name]
        
        return None

    def get_all_tools(self) -> List[Tool]:
        """
        获取所有工具
        
        Returns:
            Tool 实例列表
        """
        tools = list(self._tools.values())
        tools.extend(self._skill_adapters.values())
        return tools

    def get_langchain_tools(self) -> List[Any]:
        """
        获取所有工具的 LangChain 格式
        
        Returns:
            LangChain BaseTool 列表
        """
        # 检查缓存
        if self._langchain_tools:
            return list(self._langchain_tools.values())
        
        tools = []
        
        # 转换新 Tool
        for name, tool in self._tools.items():
            try:
                lc_tool = tool.to_langchain_tool()
                tools.append(lc_tool)
                self._langchain_tools[name] = lc_tool
            except Exception as e:
                logger.warning(f"转换工具 {name} 失败: {e}")
        
        # 转换 Skill 适配器
        for name, adapter in self._skill_adapters.items():
            try:
                lc_tool = adapter.to_langchain_tool()
                tools.append(lc_tool)
                self._langchain_tools[name] = lc_tool
            except Exception as e:
                logger.warning(f"转换 Skill {name} 失败: {e}")
        
        return tools

    def remove_tool(self, name: str) -> bool:
        """
        移除工具
        
        Args:
            name: 工具名称
            
        Returns:
            是否成功移除
        """
        removed = False
        
        if name in self._tools:
            del self._tools[name]
            removed = True
        
        if name in self._skill_adapters:
            del self._skill_adapters[name]
            removed = True
        
        if removed:
            self._clear_cache()
        
        return removed

    def clear(self):
        """清空所有工具"""
        self._tools.clear()
        self._skill_adapters.clear()
        self._clear_cache()

    def _clear_cache(self):
        """清除缓存"""
        self._langchain_tools.clear()

    def get_tool_names(self) -> List[str]:
        """
        获取所有工具名称
        
        Returns:
            工具名称列表
        """
        names = set(self._tools.keys())
        names.update(self._skill_adapters.keys())
        return sorted(names)

    def has_tool(self, name: str) -> bool:
        """
        检查是否存在工具
        
        Args:
            name: 工具名称
            
        Returns:
            是否存在
        """
        if name in self._aliases:
            name = self._aliases[name]
        return name in self._tools or name in self._skill_adapters

    def get_stats(self) -> Dict[str, int]:
        """
        获取注册表统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "tools": len(self._tools),
            "skills": len(self._skill_adapters),
            "aliases": len(self._aliases),
            "total": len(self._tools) + len(self._skill_adapters)
        }
