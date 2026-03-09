"""
模型管理模块

提供 Ollama 模型的统一初始化和查询功能。
基于 LangChain 1.0+ 的 init_chat_model 和 ChatOllama。
"""

from typing import Optional, List, Tuple
import requests
from langchain_ollama import ChatOllama


# Embedding 模型特征关键字
EMBEDDING_KEYWORDS = ['embed', 'embedding', 'bge', 'gte', 'm3e']
KNOWN_EMBEDDING_MODELS = [
    'nomic-embed-text',
    'mxbai-embed-large',
    'snowflake-arctic-embed',
    'qwen3-embedding'
]


def list_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """
    获取 Ollama 中已安装的模型列表

    Args:
        base_url: Ollama 服务地址

    Returns:
        模型名称列表

    Example:
        >>> models = list_ollama_models()
        >>> print(models)
        ['qwen3.5:4b', 'llama3.1:8b', ...]
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return sorted(models)
        return []
    except requests.exceptions.ConnectionError:
        return []
    except Exception:
        return []


def is_embedding_model(model_name: str) -> bool:
    """
    判断是否为 Embedding 模型

    Args:
        model_name: 模型名称

    Returns:
        是否为 Embedding 模型
    """
    model_lower = model_name.lower()

    # 检查关键字
    if any(kw in model_lower for kw in EMBEDDING_KEYWORDS):
        return True

    # 检查已知模型
    if any(known in model_lower for known in KNOWN_EMBEDDING_MODELS):
        return True

    return False


def list_ollama_embedding_models(base_url: str = "http://localhost:11434") -> List[str]:
    """
    获取 Ollama 中的 Embedding 模型列表

    Args:
        base_url: Ollama 服务地址

    Returns:
        Embedding 模型名称列表
    """
    all_models = list_ollama_models(base_url)
    embedding_models = [m for m in all_models if is_embedding_model(m)]
    return embedding_models


def list_ollama_chat_models(base_url: str = "http://localhost:11434") -> List[str]:
    """
    获取 Ollama 中的对话模型列表（排除 Embedding 模型）

    Args:
        base_url: Ollama 服务地址

    Returns:
        对话模型名称列表
    """
    all_models = list_ollama_models(base_url)
    chat_models = [m for m in all_models if not is_embedding_model(m)]
    return chat_models


def init_ollama_model(
    model: str,
    temperature: float = 0.7,
    base_url: str = "http://localhost:11434",
    **kwargs
) -> ChatOllama:
    """
    初始化 Ollama 模型
    
    基于 LangChain 1.0+ 的 ChatOllama，针对小模型优化配置。
    
    Args:
        model: 模型名称（如 "qwen3.5:4b"）
        temperature: 温度参数，控制随机性（0-2）
        base_url: Ollama 服务地址
        **kwargs: 其他参数传递给 ChatOllama
        
    Returns:
        ChatOllama 实例
        
    Example:
        >>> llm = init_ollama_model("qwen3.5:4b", temperature=0.7)
        >>> response = llm.invoke("你好")
    """
    return ChatOllama(
        model=model,
        temperature=temperature,
        base_url=base_url,
        # 小模型优化配置
        num_ctx=8192,  # 上下文窗口
        num_predict=2048,  # 最大生成 token
        top_p=0.9,
        top_k=40,
        repeat_penalty=1.1,
        **kwargs
    )


def select_model_interactive(base_url: str = "http://localhost:11434") -> Optional[str]:
    """
    交互式选择模型

    Returns:
        选中的模型名称，如果没有可用模型返回 None
    """
    models = list_ollama_models(base_url)

    if not models:
        print("⚠️ 未检测到 Ollama 模型，请确保 Ollama 服务已启动")
        return None

    print("\n📋 可用模型:")
    print("-" * 40)

    for i, model_name in enumerate(models, 1):
        print(f"  {i}. {model_name}")

    print("-" * 40)

    while True:
        try:
            choice = input(f"\n请选择模型 (1-{len(models)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                print(f"✅ 已选择: {selected}")
                return selected
            print(f"⚠️ 请输入 1-{len(models)} 之间的数字")
        except ValueError:
            print("⚠️ 请输入有效的数字")
        except KeyboardInterrupt:
            print("\n❌ 取消选择")
            return None


def select_models_interactive(base_url: str = "http://localhost:11434") -> Tuple[Optional[str], Optional[str]]:
    """
    交互式选择对话模型和 Embedding 模型

    Returns:
        (chat_model, embedding_model) 元组
    """
    # 选择对话模型
    chat_models = list_ollama_chat_models(base_url)
    if not chat_models:
        print("⚠️ 未检测到对话模型")
        return None, None

    print("\n📋 可用对话模型:")
    for i, model in enumerate(chat_models, 1):
        print(f"  {i}. {model}")

    try:
        choice = input(f"\n请选择对话模型 (1-{len(chat_models)}): ").strip()
        idx = int(choice) - 1
        chat_model = chat_models[idx] if 0 <= idx < len(chat_models) else chat_models[0]
        print(f"✅ 已选择对话模型: {chat_model}")
    except (ValueError, IndexError):
        chat_model = chat_models[0]
        print(f"✅ 默认使用: {chat_model}")

    # 选择 Embedding 模型
    embedding_models = list_ollama_embedding_models(base_url)
    if not embedding_models:
        print("⚠️ 未检测到 Embedding 模型，将使用默认配置")
        return chat_model, None

    print("\n📋 可用 Embedding 模型:")
    for i, model in enumerate(embedding_models, 1):
        print(f"  {i}. {model}")

    try:
        choice = input(f"\n请选择 Embedding 模型 (1-{len(embedding_models)}): ").strip()
        idx = int(choice) - 1
        embedding_model = embedding_models[idx] if 0 <= idx < len(embedding_models) else embedding_models[0]
        print(f"✅ 已选择 Embedding 模型: {embedding_model}")
    except (ValueError, IndexError):
        embedding_model = embedding_models[0]
        print(f"✅ 默认使用: {embedding_model}")

    return chat_model, embedding_model
