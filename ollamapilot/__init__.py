"""
OllamaPilot - 基于 LangChain 1.0+ 的 Ollama 智能助手

核心特性:
    - 模块化架构: USB 式即插即用设计
    - 小模型优化: 针对本地 Ollama 模型深度优化
    - Skill 系统: 可独立开发、部署的功能模块
    - 工具生态: 内置常用工具，支持自定义扩展

示例:
    >>> from ollamapilot import create_agent, init_ollama_model
    >>> 
    >>> # 初始化模型
    >>> model = init_ollama_model("qwen3.5:4b")
    >>> 
    >>> # 创建 Agent
    >>> agent = create_agent(model, skills_dir="skills")
    >>> 
    >>> # 执行对话
    >>> response = agent.invoke("明天苏州天气怎么样？")
"""

from ollamapilot.models import init_ollama_model, list_ollama_models
from ollamapilot.agent import create_agent, OllamaPilotAgent
from ollamapilot.skills import Skill, SkillRegistry, DefaultSkill, MarkdownSkill
from ollamapilot.skill_middleware import (
    SkillMiddleware,
    SkillSelectorMiddleware,
    ToolLoggingMiddleware,
    create_skill_middlewares,
)
from ollamapilot.tools.mcp_tools import (
    MCPMiddleware,
    create_mcp_middleware,
    register_mcp_server,
    get_mcp_server,
    list_mcp_servers,
)
from ollamapilot.tools.custom import load_custom_tool, discover_custom_tools

__version__ = "1.0.0"
__all__ = [
    # 模型
    "init_ollama_model",
    "list_ollama_models",
    # Agent
    "create_agent",
    "OllamaPilotAgent",
    # Skill
    "Skill",
    "SkillRegistry",
    "DefaultSkill",
    "MarkdownSkill",
    # Skill Middleware
    "SkillMiddleware",
    "SkillSelectorMiddleware",
    "ToolLoggingMiddleware",
    "create_skill_middlewares",
    # MCP 工具
    "MCPMiddleware",
    "create_mcp_middleware",
    "register_mcp_server",
    "get_mcp_server",
    "list_mcp_servers",
    # 自定义工具
    "load_custom_tool",
    "discover_custom_tools",
]
