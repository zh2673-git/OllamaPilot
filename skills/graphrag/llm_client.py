"""
LLM 客户端包装器

用于实体抽取的轻量级LLM调用
"""

import requests
import json
from typing import Optional


class SimpleLLMClient:
    """
    简单的LLM客户端
    
    用于实体抽取等轻量级任务
    """
    
    def __init__(self, model_name: str = "qwen3.5:4b", base_url: str = "http://localhost:11434"):
        """
        初始化LLM客户端
        
        Args:
            model_name: Ollama模型名称
            base_url: Ollama服务地址
        """
        self.model_name = model_name
        self.base_url = base_url
        self.generate_url = f"{base_url}/api/generate"
    
    def generate(self, prompt: str, timeout: int = 120, silent: bool = False) -> str:
        """
        生成文本
        
        Args:
            prompt: 提示词
            timeout: 超时时间（秒），默认120秒
            silent: 是否静默模式（不打印错误信息）
            
        Returns:
            生成的文本
        """
        try:
            response = requests.post(
                self.generate_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                if not silent:
                    print(f"⚠️ LLM调用失败: {response.status_code}")
                return ""
                
        except requests.exceptions.Timeout:
            if not silent:
                print(f"⏳ LLM响应较慢，使用备用方案...")
            return ""
        except Exception as e:
            if not silent:
                print(f"⚠️ LLM调用错误: {e}")
            return ""
    
    def is_available(self) -> bool:
        """检查LLM服务是否可用"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
