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

        response = requests.post(
            self.url,
            json={"model": self.model_name, "prompt": text},
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise Exception(f"Embedding API 错误: {response.status_code}")

        data = response.json()
        embedding = data.get("embedding", [])

        if not embedding:
            raise Exception("Embedding 返回为空")

        return embedding


class SafeEmbeddingFunction:
    """
    安全的 Embedding 函数包装器

    捕获所有异常，避免程序崩溃。
    """

    def __init__(self, embedding_fn):
        self.embedding_fn = embedding_fn
        self.fallback_used = False

    def __call__(self, input: List[str]) -> List[List[float]]:
        """安全调用 embedding 函数"""
        try:
            return self.embedding_fn(input)
        except Exception as e:
            print(f"      ⚠️ Embedding 调用失败: {e}")
            # 返回零向量作为 fallback
            # 假设 embedding 维度为 4096 (qwen3-embedding:8b)
            dim = 4096 if "8b" in str(self.embedding_fn.model_name) else 3584
            return [[0.0] * dim for _ in input]
