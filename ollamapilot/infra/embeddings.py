"""
EmbeddingManager - 嵌入模型管理

统一管理嵌入模型的加载和使用。
"""

from typing import Optional


class EmbeddingManager:
    """
    嵌入模型管理器
    
    提供统一的嵌入模型接口。
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        初始化 EmbeddingManager
        
        Args:
            model_name: 模型名称
        """
        self.model_name = model_name or "qwen3-embedding:0.6b"
        self._embedding_model = None
    
    def get_embedding_model(self):
        """
        获取嵌入模型
        
        Returns:
            嵌入模型实例
        """
        if self._embedding_model is None:
            # 尝试导入 OllamaEmbeddings
            try:
                from langchain_ollama import OllamaEmbeddings
                self._embedding_model = OllamaEmbeddings(model=self.model_name)
            except ImportError:
                # 回退到简单实现
                self._embedding_model = SimpleEmbeddingModel(self.model_name)
        
        return self._embedding_model
    
    def embed_text(self, text: str) -> list:
        """
        嵌入单个文本
        
        Args:
            text: 文本
            
        Returns:
            向量
        """
        model = self.get_embedding_model()
        return model.embed_query(text)
    
    def embed_texts(self, texts: list) -> list:
        """
        嵌入多个文本
        
        Args:
            texts: 文本列表
            
        Returns:
            向量列表
        """
        model = self.get_embedding_model()
        return model.embed_documents(texts)


class SimpleEmbeddingModel:
    """
    简单嵌入模型（回退实现）
    
    当无法加载 OllamaEmbeddings 时使用。
    """
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._dimension = 768  # 默认维度
    
    def embed_query(self, text: str) -> list:
        """嵌入查询文本"""
        # 简单实现：返回随机向量（仅用于测试）
        import hashlib
        import random
        
        # 使用文本的哈希作为种子，确保相同文本产生相同向量
        seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
        random.seed(seed)
        
        return [random.uniform(-1, 1) for _ in range(self._dimension)]
    
    def embed_documents(self, texts: list) -> list:
        """嵌入文档列表"""
        return [self.embed_query(text) for text in texts]
