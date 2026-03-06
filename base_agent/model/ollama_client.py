"""
Ollama 客户端模块

提供 Ollama 本地模型的统一管理和交互功能
支持思维链流式显示
"""

import requests
import json
from typing import Iterator, Optional, Dict, Any, List
from dataclasses import dataclass


# ==================== 模型管理功能 ====================

def get_ollama_models(base_url: str = "http://localhost:11434") -> List[str]:
    """
    获取 Ollama 中已安装的模型列表
    
    Args:
        base_url: Ollama 服务地址
        
    Returns:
        List[str]: 模型名称列表
    """
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            return sorted(models)
        else:
            print(f"⚠️ 获取模型列表失败: HTTP {response.status_code}")
            return []
    except requests.exceptions.ConnectionError:
        print("⚠️ 无法连接到 Ollama 服务，请确保 Ollama 已启动")
        return []
    except Exception as e:
        print(f"⚠️ 获取模型列表出错: {e}")
        return []


def select_model() -> Optional[str]:
    """
    交互式选择模型
    
    Returns:
        Optional[str]: 选中的模型名称，如果失败返回 None
    """
    print("\n" + "=" * 60)
    print("🤖 模型选择")
    print("=" * 60)
    
    # 获取可用模型
    models = get_ollama_models()
    
    if not models:
        print("\n⚠️ 未检测到 Ollama 模型，使用默认模型 (qwen3.5:4b)")
        return "qwen3.5:4b"
    
    # 显示模型列表
    print("\n📋 可用模型列表:")
    print("-" * 40)
    
    # 分组显示模型
    model_groups = {}
    for model in models:
        # 提取模型系列名称
        base_name = model.split(":")[0]
        if base_name not in model_groups:
            model_groups[base_name] = []
        model_groups[base_name].append(model)
    
    # 显示分组后的模型
    idx = 1
    model_map = {}
    for group_name in sorted(model_groups.keys()):
        group_models = sorted(model_groups[group_name])
        print(f"\n【{group_name}】")
        for model in group_models:
            print(f"  {idx}. {model}")
            model_map[idx] = model
            idx += 1
    
    # 用户选择
    print("\n" + "-" * 40)
    while True:
        try:
            choice = input(f"\n请选择模型 (1-{len(models)}, 直接回车使用默认): ").strip()
            
            # 直接回车使用默认
            if not choice:
                default_model = "qwen3.5:4b"
                if default_model in models:
                    print(f"✅ 使用默认模型: {default_model}")
                    return default_model
                else:
                    # 使用第一个可用模型
                    first_model = models[0]
                    print(f"✅ 使用首个可用模型: {first_model}")
                    return first_model
            
            # 数字选择
            choice_num = int(choice)
            if choice_num in model_map:
                selected = model_map[choice_num]
                print(f"✅ 已选择模型: {selected}")
                return selected
            else:
                print(f"⚠️ 无效选择，请输入 1-{len(models)} 之间的数字")
                
        except ValueError:
            # 直接输入模型名称
            if choice in models:
                print(f"✅ 已选择模型: {choice}")
                return choice
            else:
                print(f"⚠️ 模型 '{choice}' 不在列表中，请重新选择")
        except KeyboardInterrupt:
            print("\n\n使用默认模型")
            return "qwen3.5:4b"


def quick_select_model(model_name: str) -> str:
    """
    快速选择指定模型
    
    Args:
        model_name: 模型名称
        
    Returns:
        str: 模型名称
    """
    models = get_ollama_models()
    
    if model_name in models:
        print(f"✅ 使用模型: {model_name}")
        return model_name
    elif models:
        print(f"⚠️ 模型 '{model_name}' 不可用，使用默认模型: {models[0]}")
        return models[0]
    else:
        print(f"⚠️ 使用指定模型: {model_name}")
        return model_name


# 向后兼容别名
get_models = get_ollama_models


# ==================== 数据类定义 ====================

@dataclass
class Message:
    """消息"""
    role: str
    content: str


@dataclass
class ChatOptions:
    """聊天选项"""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None


@dataclass
class ChatResponse:
    """聊天响应"""
    content: str
    thinking: str = ""
    done: bool = False


# 向后兼容
OllamaMessage = Message
OllamaChunk = ChatResponse


# ==================== 客户端类 ====================

class OllamaClient:
    """
    Ollama 原生客户端
    
    直接使用 Ollama HTTP API，保留完整的模型输出（包括思维链）
    """
    
    def __init__(
        self,
        model: str = "qwen3.5:4b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        **options
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.options = options
    
    def stream(
        self,
        messages: List[Message],
        system: Optional[str] = None
    ) -> Iterator[ChatResponse]:
        """
        流式生成回复
        
        Args:
            messages: 对话历史
            system: 系统提示词
            
        Yields:
            ChatResponse: 输出块
        """
        url = f"{self.base_url}/api/chat"
        
        # 构建消息列表
        ollama_messages = []
        
        # 如果有系统提示词，作为第一条消息
        if system:
            ollama_messages.append({"role": "system", "content": system})
        
        # 添加其他消息
        for msg in messages:
            ollama_messages.append({"role": msg.role, "content": msg.content})
        
        # 构建请求体
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": True,
            "options": {
                "temperature": self.temperature,
                **self.options
            }
        }
        
        # 发送请求
        response = requests.post(url, json=payload, stream=True)
        response.raise_for_status()
        
        # 解析流式响应
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line)
                    
                    # 提取内容和思维链
                    if "message" in data:
                        msg = data["message"]
                        content = msg.get("content", "")
                        thinking = msg.get("thinking", "")
                        
                        # 检查是否完成
                        is_done = data.get("done", False)
                        yield ChatResponse(content=content, thinking=thinking, done=is_done)
                        
                        if is_done:
                            break
                        
                except json.JSONDecodeError:
                    continue
    
    def invoke(
        self,
        messages: List[Message],
        system: Optional[str] = None
    ) -> str:
        """
        非流式生成回复
        
        Args:
            messages: 对话历史
            system: 系统提示词
            
        Returns:
            str: 完整回复
        """
        url = f"{self.base_url}/api/chat"
        
        # 构建消息列表
        ollama_messages = []
        
        # 如果有系统提示词，作为第一条消息
        if system:
            ollama_messages.append({"role": "system", "content": system})
        
        # 添加其他消息
        for msg in messages:
            ollama_messages.append({"role": msg.role, "content": msg.content})
        
        payload = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                **self.options
            }
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        data = response.json()
        if "message" in data and "content" in data["message"]:
            return data["message"]["content"]
        
        return ""


class OllamaLangChainWrapper:
    """
    Ollama 客户端的 LangChain 风格包装
    
    兼容 LangChain 接口，但使用原生 Ollama API
    """
    
    def __init__(self, client: OllamaClient):
        self.client = client
    
    def stream(self, messages):
        """
        流式输出（兼容 LangChain）
        
        messages 可以是 LangChain 的 Message 对象列表
        """
        # 转换消息格式，提取系统提示词
        ollama_messages = []
        system_prompt = None
        
        for msg in messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                if msg.type == "system":
                    system_prompt = msg.content
                else:
                    role = "user" if msg.type == "human" else "assistant"
                    ollama_messages.append(Message(role=role, content=msg.content))
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_prompt = content
                else:
                    ollama_messages.append(Message(role=role, content=content))
        
        # 流式生成（传递系统提示词）
        # 流式发送思维链和内容，用标记区分
        in_thinking_phase = True
        
        for chunk in self.client.stream(ollama_messages, system=system_prompt):
            if chunk.done:
                break
            
            # 发送思维链（如果有且还在思维链阶段）
            if chunk.thinking and in_thinking_phase:
                class FakeChunk:
                    def __init__(self, content, chunk_type='thinking'):
                        self.content = content
                        self.chunk_type = chunk_type
                yield FakeChunk(chunk.thinking, 'thinking')
            
            # 发送内容（如果有）
            if chunk.content:
                if in_thinking_phase:
                    in_thinking_phase = False  # 进入内容阶段
                class FakeChunk:
                    def __init__(self, content, chunk_type='content'):
                        self.content = content
                        self.chunk_type = chunk_type
                yield FakeChunk(chunk.content, 'content')
    
    def invoke(self, messages):
        """非流式调用（兼容 LangChain）"""
        ollama_messages = []
        system_prompt = None
        
        for msg in messages:
            if hasattr(msg, 'type') and hasattr(msg, 'content'):
                if msg.type == "system":
                    system_prompt = msg.content
                else:
                    role = "user" if msg.type == "human" else "assistant"
                    ollama_messages.append(Message(role=role, content=msg.content))
            elif isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    system_prompt = content
                else:
                    ollama_messages.append(Message(role=role, content=content))
        
        result = self.client.invoke(ollama_messages, system=system_prompt)
        
        # 返回类似 LangChain 的对象
        class FakeResult:
            def __init__(self, content):
                self.content = content
        
        return FakeResult(result)


def create_ollama_model(
    model: str = "qwen3.5:4b",
    base_url: str = "http://localhost:11434",
    temperature: float = 0.7,
    use_native: bool = True,
    **kwargs
):
    """
    创建 Ollama 模型
    
    Args:
        model: 模型名称
        base_url: Ollama 服务地址
        temperature: 温度参数
        use_native: 是否使用原生 API（保留思维链）
        **kwargs: 其他参数
        
    Returns:
        模型实例（兼容 LangChain 接口）
    """
    if use_native:
        client = OllamaClient(
            model=model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
        return OllamaLangChainWrapper(client)
    else:
        # 使用 LangChain 的 ChatOllama
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=temperature,
            **kwargs
        )
