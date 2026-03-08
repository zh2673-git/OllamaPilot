"""
模型管理模块

提供 Ollama 模型的统一初始化和查询功能。
基于 LangChain 1.0+ 的 init_chat_model 和 ChatOllama。
"""

from typing import Optional, List
import requests
from langchain_ollama import ChatOllama


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
