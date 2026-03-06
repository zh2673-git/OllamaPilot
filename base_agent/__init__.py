"""
基座智能体 + Skill 架构

一个基于 LangChain 1.0+ 的模块化智能体框架
"""

# 从子模块导入
from .skill import (
    Skill,
    SkillMetadata,
    SkillToolConfig,
    SkillRegistry,
    simplify_tool_description,
    format_tools_for_small_model,
    auto_register,
    SkillRouter,
    discover_skills_metadata,
    SkillChunker,
    SkillChunk,
    ChunkMatch,
    AdaptiveSkillLoader,
)
from .tool import ToolLoader
from .model import (
    init_ollama_model,
    init_openai_model,
    list_ollama_models,
    get_ollama_models,
    select_model,
    quick_select_model,
    create_ollama_model,
    OllamaClient,
    OllamaLangChainWrapper,
    ChatOptions,
    Message,
    ChatResponse,
    ContentType,
    StreamEvent,
    StreamingProcessor,
    StreamingDisplay,
    create_streaming_handler,
)

# 保留原有的 Agent 导入
from .agent import Agent, create_agent, AgentConfig

__all__ = [
    # Skill 定义
    "Skill",
    "SkillMetadata",
    "SkillToolConfig",
    # Skill 注册中心
    "SkillRegistry",
    "simplify_tool_description",
    "format_tools_for_small_model",
    "auto_register",
    # 工具加载
    "ToolLoader",
    # 模型相关
    "init_ollama_model",
    "init_openai_model",
    "list_ollama_models",
    "get_ollama_models",
    "select_model",
    "quick_select_model",
    "create_ollama_model",
    # Ollama 客户端
    "OllamaClient",
    "OllamaLangChainWrapper",
    "ChatOptions",
    "Message",
    "ChatResponse",
    # 流式处理
    "ContentType",
    "StreamEvent",
    "StreamingProcessor",
    "StreamingDisplay",
    "create_streaming_handler",
    # Skill 路由
    "SkillRouter",
    "discover_skills_metadata",
    # Skill 分块
    "SkillChunker",
    "SkillChunk",
    "ChunkMatch",
    "AdaptiveSkillLoader",
    # Agent
    "Agent",
    "create_agent",
    "AgentConfig",
]

__version__ = "0.1.0"
