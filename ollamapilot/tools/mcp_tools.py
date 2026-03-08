"""
MCP 工具支持

MCP (Model Context Protocol) 工具配置和加载。

使用方法:

1. 在 SKILL.md 中配置 MCP 工具:
    ---
    name: my_skill
    tools:
      - mcp://server_name/tool_name
    ---

2. 在代码中使用 MCP 中间件:
    from ollamapilot.tools.mcp_tools import create_mcp_middleware
    
    mcp_middleware = create_mcp_middleware(
        server_url="https://mcp.example.com/mcp",
        server_name="my-server"
    )
    
    agent = create_agent(
        model=model,
        middleware=[mcp_middleware, ...]
    )

注意: MCP 需要 LangChain 版本支持，目前部分功能可能受限。
"""

from typing import Optional, Dict, Any, List
from langchain.agents.middleware import AgentMiddleware
from langchain_core.tools import BaseTool


class MCPMiddleware(AgentMiddleware):
    """
    MCP 服务器中间件
    
    为 Agent 添加 MCP 服务器支持。
    在 before_model 阶段注入 MCP 配置。
    
    示例:
        >>> mcp_mw = MCPMiddleware(
        ...     server_url="https://mcp.example.com/mcp",
        ...     server_name="my-server"
        ... )
        >>> agent = create_agent(model, middleware=[mcp_mw])
    """
    
    def __init__(
        self,
        server_url: str,
        server_name: str,
        allowed_tools: Optional[List[str]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        """
        初始化 MCP 中间件
        
        Args:
            server_url: MCP 服务器 URL
            server_name: 服务器名称标识
            allowed_tools: 允许使用的工具列表，None 表示全部
            headers: 请求头（如认证信息）
        """
        super().__init__()
        self.server_url = server_url
        self.server_name = server_name
        self.allowed_tools = allowed_tools
        self.headers = headers or {}
    
    @property
    def name(self) -> str:
        return f"MCPMiddleware({self.server_name})"
    
    def before_model(self, state: Any, runtime: Any) -> Optional[Dict[str, Any]]:
        """
        在模型调用前注入 MCP 配置
        
        注意: 实际 MCP 工具调用需要模型支持 MCP 协议。
        这里主要是记录和配置。
        """
        # MCP 配置存储在 metadata 中
        mcp_config = {
            "type": "mcp",
            "server_url": self.server_url,
            "server_name": self.server_name,
            "allowed_tools": self.allowed_tools,
        }
        
        # 返回状态更新
        return {
            "metadata": {
                "mcp_servers": [mcp_config]
            }
        }


def create_mcp_middleware(
    server_url: str,
    server_name: str,
    allowed_tools: Optional[List[str]] = None,
    **kwargs
) -> MCPMiddleware:
    """
    创建 MCP 中间件
    
    工厂函数，简化 MCP 中间件创建。
    
    Args:
        server_url: MCP 服务器 URL
        server_name: 服务器名称
        allowed_tools: 允许的工具列表
        **kwargs: 其他参数
        
    Returns:
        MCPMiddleware 实例
        
    Example:
        >>> mcp_mw = create_mcp_middleware(
        ...     "https://mcp.example.com/mcp",
        ...     "my-server",
        ...     allowed_tools=["search", "query"]
        ... )
    """
    return MCPMiddleware(
        server_url=server_url,
        server_name=server_name,
        allowed_tools=allowed_tools,
        **kwargs
    )


def parse_mcp_tool_ref(tool_ref: str) -> Optional[Dict[str, str]]:
    """
    解析 MCP 工具引用
    
    格式: mcp://server_name/tool_name
    
    Args:
        tool_ref: 工具引用字符串
        
    Returns:
        解析结果字典或 None
        
    Example:
        >>> parse_mcp_tool_ref("mcp://my-server/search")
        {"server": "my-server", "tool": "search"}
    """
    if not tool_ref.startswith("mcp://"):
        return None
    
    # 去掉 mcp:// 前缀
    path = tool_ref[6:]
    
    # 分割 server 和 tool
    if "/" not in path:
        return None
    
    server_name, tool_name = path.split("/", 1)
    
    return {
        "server": server_name,
        "tool": tool_name
    }


# MCP 工具注册表（简化版）
_mcp_servers: Dict[str, Dict[str, Any]] = {}


def register_mcp_server(
    name: str,
    url: str,
    headers: Optional[Dict[str, str]] = None
) -> None:
    """
    注册 MCP 服务器
    
    Args:
        name: 服务器名称
        url: 服务器 URL
        headers: 请求头
    """
    _mcp_servers[name] = {
        "url": url,
        "headers": headers or {}
    }


def get_mcp_server(name: str) -> Optional[Dict[str, Any]]:
    """
    获取 MCP 服务器配置
    
    Args:
        name: 服务器名称
        
    Returns:
        服务器配置或 None
    """
    return _mcp_servers.get(name)


def list_mcp_servers() -> List[str]:
    """
    列出所有已注册的 MCP 服务器
    
    Returns:
        服务器名称列表
    """
    return list(_mcp_servers.keys())
