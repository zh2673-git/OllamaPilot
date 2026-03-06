"""
工具注册中心

自动扫描和管理所有工具：
1. 内置工具 (tools/builtin/)
2. MCP 工具 (tools/mcp/)

使用方式：
    from tools.registry import tool_registry
    
    # 获取所有可用工具
    all_tools = tool_registry.get_all_tools()
    
    # 获取特定工具
    tool = tool_registry.get_tool("read_file")
    
    # 刷新 MCP 工具（添加新 MCP 后调用）
    tool_registry.refresh_mcp_tools()
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from langchain_core.tools import BaseTool

# 导入内置工具
from .builtin import ALL_TOOLS as BUILTIN_TOOLS, TOOL_MAP as BUILTIN_TOOL_MAP


@dataclass
class MCPConfig:
    """MCP 工具配置"""
    name: str
    server: str
    description: str = ""
    args: Dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    """
    工具注册中心
    
    自动扫描 tools/builtin/ 和 tools/mcp/ 目录
    统一管理所有工具的注册和发现
    """
    
    def __init__(self, mcp_dir: str = "tools/mcp"):
        """
        初始化工具注册中心
        
        Args:
            mcp_dir: MCP 配置目录路径
        """
        self.mcp_dir = Path(mcp_dir)
        
        # 内置工具（自动加载）
        self._builtin_tools: Dict[str, BaseTool] = {}
        self._load_builtin_tools()
        
        # MCP 工具配置
        self._mcp_configs: Dict[str, MCPConfig] = {}
        self._mcp_tools: Dict[str, Any] = {}  # 延迟加载
        
        # 扫描 MCP 配置
        self._scan_mcp_configs()
    
    def _load_builtin_tools(self):
        """加载所有内置工具"""
        for tool in BUILTIN_TOOLS:
            self._builtin_tools[tool.name] = tool
        print(f"✅ 已加载 {len(self._builtin_tools)} 个内置工具")
    
    def _scan_mcp_configs(self):
        """扫描 MCP 配置目录"""
        if not self.mcp_dir.exists():
            print(f"⚠️ MCP 配置目录不存在: {self.mcp_dir}")
            return
        
        count = 0
        for config_file in self.mcp_dir.glob("*.yaml"):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                
                if not config:
                    continue
                
                # 解析 MCP 配置
                mcp_config = MCPConfig(
                    name=config.get('name', config_file.stem),
                    server=config.get('server', config_file.stem),
                    description=config.get('description', ''),
                    args=config.get('args', {})
                )
                
                self._mcp_configs[mcp_config.name] = mcp_config
                count += 1
                
            except Exception as e:
                print(f"⚠️ 解析 MCP 配置 {config_file} 失败: {e}")
        
        if count > 0:
            print(f"✅ 已扫描 {count} 个 MCP 配置")
    
    def get_tool(self, name: str) -> Optional[BaseTool]:
        """
        获取指定名称的工具
        
        Args:
            name: 工具名称
            
        Returns:
            工具实例或 None
        """
        # 1. 先查找内置工具
        if name in self._builtin_tools:
            return self._builtin_tools[name]
        
        # 2. 查找 MCP 工具（延迟加载）
        if name in self._mcp_configs:
            return self._load_mcp_tool(name)
        
        return None
    
    def get_builtin_tool(self, name: str) -> Optional[BaseTool]:
        """获取内置工具"""
        return self._builtin_tools.get(name)
    
    def get_all_tools(self) -> List[BaseTool]:
        """
        获取所有可用工具
        
        Returns:
            所有工具列表（内置 + MCP）
        """
        tools = list(self._builtin_tools.values())
        
        # 加载所有 MCP 工具
        for name in self._mcp_configs:
            mcp_tool = self._load_mcp_tool(name)
            if mcp_tool:
                tools.append(mcp_tool)
        
        return tools
    
    def get_builtin_tools(self) -> List[BaseTool]:
        """获取所有内置工具"""
        return list(self._builtin_tools.values())
    
    def get_mcp_tools(self) -> List[Any]:
        """获取所有 MCP 工具"""
        tools = []
        for name in self._mcp_configs:
            tool = self._load_mcp_tool(name)
            if tool:
                tools.append(tool)
        return tools
    
    def _load_mcp_tool(self, name: str) -> Optional[Any]:
        """加载 MCP 工具（延迟加载）"""
        # 检查缓存
        if name in self._mcp_tools:
            return self._mcp_tools[name]
        
        config = self._mcp_configs.get(name)
        if not config:
            return None
        
        try:
            # 尝试导入 MCP 适配器
            from langchain_mcp_adapters.tools import load_mcp_tools
            from mcp import StdioServerParameters
            
            # 创建服务器参数
            server_params = StdioServerParameters(
                command=config.args.get('command', 'npx'),
                args=config.args.get('args', []),
                env=config.args.get('env', {})
            )
            
            # 加载工具
            tools = load_mcp_tools(server_params)
            
            # 找到指定工具
            for tool in tools:
                if tool.name == name:
                    self._mcp_tools[name] = tool
                    return tool
            
            # 如果找不到同名工具，返回第一个
            if tools:
                self._mcp_tools[name] = tools[0]
                return tools[0]
            
            return None
            
        except ImportError:
            print(f"⚠️ 未安装 langchain-mcp-adapters，无法加载 MCP 工具: {name}")
            return None
        except Exception as e:
            print(f"⚠️ 加载 MCP 工具 {name} 失败: {e}")
            return None
    
    def list_builtin_tools(self) -> List[str]:
        """列出所有内置工具名称"""
        return list(self._builtin_tools.keys())
    
    def list_mcp_configs(self) -> List[str]:
        """列出所有 MCP 配置名称"""
        return list(self._mcp_configs.keys())
    
    def get_mcp_config(self, name: str) -> Optional[MCPConfig]:
        """获取 MCP 配置"""
        return self._mcp_configs.get(name)
    
    def refresh_mcp_tools(self):
        """刷新 MCP 工具（添加新 MCP 配置后调用）"""
        self._mcp_configs.clear()
        self._mcp_tools.clear()
        self._scan_mcp_configs()
        print("✅ MCP 工具已刷新")
    
    def get_tool_info(self) -> Dict[str, Dict]:
        """
        获取所有工具信息（用于生成 SKILL.md）
        
        Returns:
            工具信息字典
        """
        info = {
            'builtin': {},
            'mcp': {}
        }
        
        # 内置工具信息
        for name, tool in self._builtin_tools.items():
            info['builtin'][name] = {
                'type': 'builtin',
                'description': getattr(tool, 'description', ''),
                'module': f'tools.builtin'
            }
        
        # MCP 工具信息
        for name, config in self._mcp_configs.items():
            info['mcp'][name] = {
                'type': 'mcp',
                'description': config.description,
                'server': config.server
            }
        
        return info


# 全局工具注册中心实例
tool_registry = ToolRegistry()


def get_registry(mcp_dir: str = "tools/mcp") -> ToolRegistry:
    """
    创建新的工具注册中心实例
    
    Args:
        mcp_dir: MCP 配置目录路径
        
    Returns:
        ToolRegistry 实例
    """
    return ToolRegistry(mcp_dir)
