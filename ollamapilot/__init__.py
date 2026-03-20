"""
OllamaPilot - 基于 LangChain 1.0+ 的 Ollama 智能助手

核心特性:
    - 模块化架构: USB 式即插即用设计
    - 小模型优化: 针对本地 Ollama 模型深度优化
    - Skill 系统: 可独立开发、部署的功能模块
    - 工具生态: 内置常用工具，支持自定义扩展
    - GraphRAG: 基于知识图谱的检索增强（作为独立 Skill）
    - 四层 Context: L3/L2/L1/L0 架构
    - 中间件系统: Context 注入、记忆检索、上下文压缩

V0.5.0 重构：简化架构，移除废弃文件依赖

示例:
    >>> from ollamapilot import create_agent, init_ollama_model
    >>>
    >>> # 初始化模型
    >>> model = init_ollama_model("qwen3.5:4b")
    >>>
    >>> # 创建 Agent（自动加载 skills 目录下的所有 Skill）
    >>> agent = create_agent(model, skills_dir="skills")
    >>>
    >>> # 执行对话
    >>> response = agent.invoke("明天苏州天气怎么样？")

GraphRAG Skill 使用:
    将 GraphRAG Skill 放入 skills/graphrag/ 目录，Agent 会自动加载。
    触发词: "知识图谱", "文档问答", "检索", "添加文档" 等
"""

# 模型管理
from ollamapilot.models import (
    init_ollama_model,
    list_ollama_models,
    list_ollama_chat_models,
    list_ollama_embedding_models,
    select_model_interactive,
    select_models_interactive,
    is_embedding_model,
)

# Agent
from ollamapilot.agent import create_agent, OllamaPilotAgent, create_ollama_agent

# Context
from ollamapilot.context import (
    ContextBuilder,
    Context,
    ContextCompactor,
    CompressionResult,
    TokenOptimizer,
)

# Memory
from ollamapilot.memory import (
    MemoryManager,
    MemoryIndexer,
    MemoryType,
    MemoryEntry,
    SearchResult,
)

# Middlewares
from ollamapilot.middlewares import (
    ContextInjectionMiddleware,
    MemoryRetrievalMiddleware,
    CompactionMiddleware,
)

# Skill
from ollamapilot.skills import Skill, SkillRegistry, DefaultSkill, MarkdownSkill

# MCP 工具
from ollamapilot.tools.mcp_tools import (
    MCPMiddleware,
    create_mcp_middleware,
    register_mcp_server,
    get_mcp_server,
    list_mcp_servers,
)

# 自定义工具
from ollamapilot.tools.custom import load_custom_tool, discover_custom_tools

# 日志配置
from ollamapilot.logging_config import setup_logging, get_logger, set_module_level

__version__ = "1.1.0"
__all__ = [
    # 模型管理
    "init_ollama_model",
    "list_ollama_models",
    "list_ollama_chat_models",
    "list_ollama_embedding_models",
    "select_model_interactive",
    "select_models_interactive",
    "is_embedding_model",
    # Agent
    "create_agent",
    "create_ollama_agent",
    "OllamaPilotAgent",
    # Context
    "ContextBuilder",
    "Context",
    "ContextCompactor",
    "CompressionResult",
    "TokenOptimizer",
    # Memory
    "MemoryManager",
    "MemoryIndexer",
    "MemoryType",
    "MemoryEntry",
    "SearchResult",
    # Middlewares
    "ContextInjectionMiddleware",
    "MemoryRetrievalMiddleware",
    "CompactionMiddleware",
    # Skill
    "Skill",
    "SkillRegistry",
    "DefaultSkill",
    "MarkdownSkill",
    # MCP 工具
    "MCPMiddleware",
    "create_mcp_middleware",
    "register_mcp_server",
    "get_mcp_server",
    "list_mcp_servers",
    # 自定义工具
    "load_custom_tool",
    "discover_custom_tools",
    # 日志配置
    "setup_logging",
    "get_logger",
    "set_module_level",
]