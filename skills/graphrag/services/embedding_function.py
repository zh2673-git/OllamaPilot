"""
自定义 Embedding 函数

绕过 ChromaDB 的默认 embedding 函数，直接使用 Ollama API，
避免 onnxruntime 等依赖导致的内存访问冲突。
"""

import requests
import time
from typing import List, Optional
import numpy as np


class OllamaEmbeddingFunction:
    """
    自定义 Ollama Embedding 函数

    直接使用 HTTP API 调用 Ollama，避免 ChromaDB 内置函数的问题。
    """

    def __init__(
        self,
        url: str = "http://localhost:11434/api/embeddings",
        model_name: str = "qwen3-embedding:4b",
        timeout: int = 120,
        max_retries: int = 3
    ):
        self.url = url
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        生成文本的 embedding 向量

        Args:
            input: 文本列表

        Returns:
            embedding 向量列表
        """
        results = []

        for text in input:
            embedding = self._get_embedding_with_retry(text)
            results.append(embedding)

        return results

    def _get_embedding_with_retry(self, text: str) -> List[float]:
        """带重试的 embedding 获取"""
        for attempt in range(self.max_retries):
            try:
                return self._get_embedding(text)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避
                    print(f"      ⚠️ Embedding 请求失败，{wait_time}s 后重试...")
                    time.sleep(wait_time)
                else:
                    raise e

        return []

    def _get_embedding(self, text: str) -> List[float]:
        """获取单个文本的 embedding"""
        # 截断过长的文本
        max_length = 8000
        if len(text) > max_length:
            text = text[:max_length]

        try:
            response = requests.post(
                self.url,
                json={"model": self.model_name, "prompt": text},
                timeout=self.timeout
            )

            if response.status_code != 200:
                # 提供更详细的错误信息
                error_detail = ""
                try:
                    error_data = response.json()
                    error_detail = f" - {error_data.get('error', '')}"
                except:
                    error_detail = f" - {response.text[:100]}"
                raise Exception(f"Embedding API 错误 {response.status_code}{error_detail}")

            data = response.json()
            embedding = data.get("embedding", [])

            if not embedding:
                raise Exception("Embedding 返回为空")

            return embedding

        except requests.exceptions.ConnectionError:
            raise Exception(f"无法连接到 Ollama 服务 ({self.url})，请确认 Ollama 是否已启动")
        except requests.exceptions.Timeout:
            raise Exception(f"Embedding 请求超时 ({self.timeout}s)")


class SafeEmbeddingFunction:
    """
    安全的 Embedding 函数包装器

    捕获所有异常，避免程序崩溃。
    支持重试机制，确保 embedding 成功。
    """

    def __init__(self, embedding_fn, raise_on_error: bool = False, max_retries: int = 3):
        self.embedding_fn = embedding_fn
        self.fallback_used = False
        self.raise_on_error = raise_on_error
        self.max_retries = max_retries

    def __call__(self, input: List[str]) -> List[List[float]]:
        """安全调用 embedding 函数（带重试）"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return self.embedding_fn(input)
            except Exception as e:
                last_error = e
                if self.raise_on_error:
                    # 迁移模式下直接抛出，让上层处理
                    raise
                
                # 普通模式下重试
                if attempt < self.max_retries - 1:
                    wait_time = 0.5 * (attempt + 1)  # 递增延迟：0.5s, 1s, 1.5s
                    print(f"      ⚠️ Embedding 失败 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                    print(f"         等待 {wait_time}s 后重试...")
                    time.sleep(wait_time)
                else:
                    # 所有重试都失败
                    print(f"      ❌ Embedding 最终失败: {e}")
                    raise  # 抛出异常让上层处理
        
        # 不应该执行到这里
        raise last_error if last_error else Exception("Embedding 调用失败")
