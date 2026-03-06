"""
模型工具函数模块

提供模型初始化、配置加载等实用函数
"""

from typing import Optional
from langchain_core.language_models.chat_models import BaseChatModel


def init_ollama_model(
    model: str = "qwen2.5:7b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    **kwargs
) -> BaseChatModel:
    """
    初始化 Ollama 本地模型

    支持多种本地模型，如 qwen2.5, llama3.2, deepseek-r1, mistral 等
    默认使用 qwen2.5:7b（中文场景表现优秀）

    Args:
        model: 模型名称，默认 "qwen2.5:7b"
        base_url: Ollama 服务地址，默认 http://localhost:11434
        temperature: 温度参数，控制生成随机性
        **kwargs: 其他参数传递给 ChatOllama

    Returns:
        BaseChatModel: 聊天模型实例

    Raises:
        ImportError: 如果未安装 langchain-ollama

    Example:
        >>> model = init_ollama_model()  # 使用默认 qwen2.5:7b
        >>> model = init_ollama_model(model="llama3.2")
        >>> model = init_ollama_model(model="qwen2.5:14b", temperature=0.5)
    """
    try:
        from langchain_ollama import ChatOllama
        
        chat_model = ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
        return chat_model
    except ImportError:
        raise ImportError(
            "使用 Ollama 模型需要安装 langchain-ollama。\n"
            "请运行: pip install langchain-ollama"
        )


def init_openai_model(
    model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    **kwargs
) -> BaseChatModel:
    """
    初始化 OpenAI 模型
    
    Args:
        model: 模型名称，如 gpt-4o, gpt-4o-mini, gpt-3.5-turbo
        api_key: API Key，默认从环境变量 OPENAI_API_KEY 读取
        base_url: 自定义 API 地址（用于代理或兼容服务）
        temperature: 温度参数
        **kwargs: 其他参数
        
    Returns:
        BaseChatModel: 聊天模型实例
        
    Example:
        >>> model = init_openai_model(model="gpt-4o-mini")
        >>> model = init_openai_model(model="gpt-4o", temperature=0.5)
    """
    from langchain.chat_models import init_chat_model
    
    chat_model = init_chat_model(
        model=model,
        model_provider="openai",
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        **kwargs
    )
    return chat_model


def init_model(
    provider: str = "ollama",
    model: Optional[str] = None,
    **kwargs
) -> BaseChatModel:
    """
    统一模型初始化接口

    根据 provider 自动选择合适的初始化方式
    默认使用 Ollama + qwen2.5:7b（中文场景优秀）

    Args:
        provider: 模型提供商，支持 "ollama", "openai"
        model: 模型名称，默认根据 provider 选择
        **kwargs: 其他参数

    Returns:
        BaseChatModel: 聊天模型实例

    Example:
        >>> # 使用默认 Ollama + qwen2.5:7b
        >>> model = init_model()
        >>>
        >>> # 使用其他 Ollama 模型
        >>> model = init_model(provider="ollama", model="llama3.2")
        >>>
        >>> # 使用 OpenAI 模型
        >>> model = init_model(provider="openai", model="gpt-4o-mini")
    """
    # 默认模型
    default_models = {
        "ollama": "qwen2.5:7b",  # 默认使用 qwen2.5:7b，中文场景优秀
        "openai": "gpt-4o-mini",
    }

    if model is None:
        model = default_models.get(provider, "qwen2.5:7b")

    if provider == "ollama":
        return init_ollama_model(model=model, **kwargs)
    elif provider == "openai":
        return init_openai_model(model=model, **kwargs)
    else:
        raise ValueError(f"不支持的模型提供商: {provider}。支持: ollama, openai")


def list_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """
    列出本地可用的 Ollama 模型
    
    Args:
        base_url: Ollama 服务地址
        
    Returns:
        list[str]: 可用模型名称列表
        
    Example:
        >>> models = list_ollama_models()
        >>> print(models)
        ['llama3.2', 'qwen2.5', 'deepseek-r1']
    """
    import requests
    
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        
        models = [model["name"] for model in data.get("models", [])]
        return models
    except Exception as e:
        raise ConnectionError(
            f"无法连接到 Ollama 服务 ({base_url})。\n"
            f"请确保 Ollama 已安装并正在运行。\n"
            f"错误: {e}"
        )
