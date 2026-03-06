"""
工具加载器 - 按需加载 Skill 工具

新设计：
1. 所有 Skill 自动加载所有内置工具（基础能力）
2. Skill 只需配置额外工具（local/mcp）
3. 移除 inherit 逻辑，简化配置
"""

import importlib
from typing import List, Optional, Dict, Any
from langchain_core.tools import BaseTool

from ..skill.skill import SkillToolConfig


class ToolLoader:
    """工具加载器 - 简化版"""
    
    def __init__(self):
        """初始化工具加载器"""
        self._tool_cache: Dict[str, BaseTool] = {}
        self._builtin_tools_loaded = False
        self._builtin_tools: List[BaseTool] = []
    
    def load_tools_for_skill(self, extra_tools_config: List[SkillToolConfig] = None) -> List[BaseTool]:
        """
        为 Skill 加载工具
        
        流程：
        1. 自动加载所有内置工具（每个 Skill 都有）
        2. 加载 Skill 配置的额外工具
        
        Args:
            extra_tools_config: 额外工具配置列表（Skill 特有的工具）
            
        Returns:
            加载的工具列表（内置工具 + 额外工具）
        """
        tools = []
        loaded_names = set()
        
        # 1. 自动加载所有内置工具
        builtin_tools = self._get_all_builtin_tools()
        for tool in builtin_tools:
            if tool.name not in loaded_names:
                tools.append(tool)
                loaded_names.add(tool.name)
        
        # 2. 加载额外工具（如果配置了）
        if extra_tools_config:
            for config in extra_tools_config:
                if config.name in loaded_names:
                    continue
                
                tool = self._load_extra_tool(config)
                if tool:
                    tools.append(tool)
                    loaded_names.add(config.name)
        
        return tools
    
    def _get_all_builtin_tools(self) -> List[BaseTool]:
        """获取所有内置工具（自动加载）"""
        if self._builtin_tools_loaded:
            return self._builtin_tools
        
        try:
            from tools import tool_registry
            self._builtin_tools = tool_registry.get_builtin_tools()
            self._builtin_tools_loaded = True
            print(f"✅ 已自动加载 {len(self._builtin_tools)} 个内置工具")
        except ImportError:
            print("⚠️ 无法导入工具注册中心")
            self._builtin_tools = []
        except Exception as e:
            print(f"⚠️ 加载内置工具失败: {e}")
            self._builtin_tools = []
        
        return self._builtin_tools
    
    def _load_extra_tool(self, config: SkillToolConfig) -> Optional[BaseTool]:
        """加载额外工具（local 或 mcp）"""
        cache_key = f"{config.type}:{config.name}"
        
        # 检查缓存
        if cache_key in self._tool_cache:
            return self._tool_cache[cache_key]
        
        try:
            if config.type == "local":
                tool = self._load_local_tool(config)
            elif config.type == "mcp":
                tool = self._load_mcp_tool(config)
            elif config.type == "builtin":
                # 内置工具已经在前面加载了，这里跳过
                return None
            else:
                print(f"⚠️ 未知工具类型: {config.type}")
                return None
            
            # 缓存工具
            if tool:
                self._tool_cache[cache_key] = tool
            
            return tool
            
        except Exception as e:
            print(f"⚠️ 加载工具 {config.name} 失败: {e}")
            return None
    
    def _load_local_tool(self, config: SkillToolConfig) -> Optional[BaseTool]:
        """加载本地工具"""
        if not config.module:
            print(f"⚠️ 本地工具 {config.name} 未指定模块路径")
            return None
        
        try:
            # 动态导入模块
            module = importlib.import_module(config.module)
            
            # 首先尝试从 TOOLS 列表中查找（推荐方式）
            tools_list = getattr(module, 'TOOLS', [])
            for t in tools_list:
                if hasattr(t, 'name') and t.name == config.name:
                    return t
            
            # 如果 TOOLS 列表中没有，尝试直接获取属性
            tool_attr = getattr(module, config.name, None)
            
            if tool_attr is None:
                print(f"⚠️ 模块 {config.module} 中未找到工具 {config.name}")
                return None
            
            # 如果已经是 BaseTool 实例，直接返回
            if isinstance(tool_attr, BaseTool):
                return tool_attr
            
            # 如果是函数，尝试调用它获取工具实例
            # 注意：这要求函数不需要参数就能返回工具实例
            if callable(tool_attr):
                try:
                    return tool_attr()
                except TypeError as e:
                    # 函数需要参数，无法直接调用
                    print(f"⚠️ 加载工具 {config.name} 失败: {e}")
                    print(f"   提示: 请确保工具函数在 TOOLS 列表中导出，或创建一个无参工厂函数")
                    return None
            
            return tool_attr
            
        except ImportError as e:
            print(f"⚠️ 导入模块 {config.module} 失败: {e}")
            return None
    
    def _load_mcp_tool(self, config: SkillToolConfig) -> Optional[BaseTool]:
        """加载 MCP 工具"""
        if not config.server:
            print(f"⚠️ MCP 工具 {config.name} 未指定服务器")
            return None
        
        try:
            from langchain_mcp_adapters.tools import load_mcp_tools
            from tools import tool_registry
            
            # 从注册中心获取 MCP 配置
            mcp_config = tool_registry.get_mcp_config(config.server)
            if not mcp_config:
                print(f"⚠️ 未找到 MCP 服务器配置: {config.server}")
                return None
            
            # TODO: 实现 MCP 客户端连接和工具加载
            print(f"⚠️ MCP 工具加载暂未实现: {config.name}")
            return None
            
        except ImportError:
            print(f"⚠️ 未安装 langchain-mcp-adapters，无法加载 MCP 工具")
            return None
        except Exception as e:
            print(f"⚠️ 加载 MCP 工具 {config.name} 失败: {e}")
            return None
