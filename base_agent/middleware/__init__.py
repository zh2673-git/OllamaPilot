"""
LangChain v1+ 兼容的中间件系统

提供可插拔的 Agent 增强能力
"""

from .base import AgentMiddleware, AgentState, MiddlewareChain
from .skill_loader import SkillLoaderMiddleware
from .tool_retry import ToolRetryMiddleware
from .context_editor import ContextEditorMiddleware, SensitiveInfoFilterMiddleware
from .context_inject import ContextInjectMiddleware
from .memory import MemoryMiddleware
from .dangling_tool import DanglingToolCallMiddleware
from .tool_format import ToolFormatMiddleware
from .context_compress import ContextCompressionMiddleware
from .react_guide import ReActGuidanceMiddleware

__all__ = [
    # 基类
    "AgentMiddleware",
    "AgentState",
    "MiddlewareChain",
    # 原有中间件实现
    "SkillLoaderMiddleware",
    "ToolRetryMiddleware",
    "ContextEditorMiddleware",
    "SensitiveInfoFilterMiddleware",
    "ContextInjectMiddleware",
    "MemoryMiddleware",
    # 小模型优化中间件
    "DanglingToolCallMiddleware",
    "ToolFormatMiddleware",
    "ContextCompressionMiddleware",
    "ReActGuidanceMiddleware",
]
