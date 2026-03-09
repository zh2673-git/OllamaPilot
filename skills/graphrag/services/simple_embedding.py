"""
简单 Embedding 实现

纯 Python 实现，不依赖 onnxruntime，作为 fallback 使用。
使用简单的哈希 + 随机投影方法生成固定维度的向量。
"""

import hashlib
import random
from typing import List
import math


class SimpleEmbeddingFunction:
    """
    简单 Embedding 函数

    使用基于哈希的方法生成固定维度的向量，不依赖任何外部库。
    这不是真正的语义 embedding，但可以作为 fallback 使用。
    """

    def __init__(self, dim: int = 384, seed: int = 42):
        """
        初始化

        Args:
            dim: 输出向量维度
            seed: 随机种子，保证可重复性
        """
        self.dim = dim
        self.seed = seed
        self._projections = {}

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        生成文本的 embedding 向量

        Args:
            input: 文本列表

        Returns:
            embedding 向量列表
        """
        return [self._embed_text(text) for text in input]

    def _embed_text(self, text: str) -> List[float]:
        """为单个文本生成 embedding"""
        # 使用文本的哈希作为随机种子
        text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
        rng = random.Random(self.seed + text_hash)

        # 生成随机向量
        vector = [rng.uniform(-1, 1) for _ in range(self.dim)]

        # 归一化
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector


class HashEmbeddingFunction:
    """
    基于哈希的 Embedding 函数

    使用字符 n-gram 的哈希来生成向量，有一定的语义保持能力。
    """

    def __init__(self, dim: int = 384, ngram_range: tuple = (1, 2)):
        """
        初始化

        Args:
            dim: 输出向量维度
            ngram_range: n-gram 范围
        """
        self.dim = dim
        self.ngram_range = ngram_range

    def __call__(self, input: List[str]) -> List[List[float]]:
        """生成 embedding"""
        return [self._embed_text(text) for text in input]

    def _embed_text(self, text: str) -> List[float]:
        """为单个文本生成 embedding"""
        vector = [0.0] * self.dim

        # 生成 n-grams
        ngrams = self._get_ngrams(text.lower())

        for ngram in ngrams:
            # 使用哈希确定位置
            hash_val = int(hashlib.md5(ngram.encode()).hexdigest(), 16)
            idx = hash_val % self.dim

            # 使用哈希的另一部分确定值
            value = ((hash_val >> 16) % 1000) / 1000.0

            vector[idx] += value

        # 归一化
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    def _get_ngrams(self, text: str) -> List[str]:
        """获取 n-grams"""
        ngrams = []

        for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
            for i in range(len(text) - n + 1):
                ngrams.append(text[i:i + n])

        return ngrams
