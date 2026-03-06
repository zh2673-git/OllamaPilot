"""
模型层 - 负责模型管理、流式输出等
"""

from .ollama_client import (
    get_ollama_models,
    select_model,
    quick_select_model,
    create_ollama_model,
    OllamaClient,
    OllamaLangChainWrapper,
    ChatOptions,
    Message,
    ChatResponse,
)
from .streaming import (
    ContentType,
    StreamEvent,
    StreamingProcessor,
    StreamingDisplay,
    create_streaming_handler,
)
from .utils import (
    init_ollama_model,
    init_openai_model,
    list_ollama_models,
)

__all__ = [
    # Ollama 客户端
    "get_ollama_models",
    "select_model",
    "quick_select_model",
    "create_ollama_model",
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
    # 模型工具
    "init_ollama_model",
    "init_openai_model",
    "list_ollama_models",
]
