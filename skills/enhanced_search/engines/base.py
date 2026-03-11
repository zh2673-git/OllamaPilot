"""
搜索引擎基类定义

参考 Local Deep Research 的搜索引擎架构设计
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class SearchResult(BaseModel):
    """
    搜索结果模型
    
    Attributes:
        title: 结果标题
        url: 结果链接
        snippet: 内容摘要
        source: 来源引擎名称
        score: 相关性分数 (0-1)
        category: 结果类别
        published_date: 发布日期
        author: 作者
        metadata: 额外元数据
    """
    title: str = Field(..., description="结果标题")
    url: str = Field(..., description="结果链接")
    snippet: str = Field(..., description="内容摘要")
    source: str = Field(..., description="来源引擎")
    score: Optional[float] = Field(None, description="相关性分数")
    category: str = Field("general", description="类别: general/academic/code/encyclopedia")
    published_date: Optional[str] = Field(None, description="发布日期")
    author: Optional[str] = Field(None, description="作者")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    
    def __hash__(self):
        """用于去重"""
        return hash((self.url, self.title))
    
    def __eq__(self, other):
        """基于 URL 和标题判断是否相同"""
        if not isinstance(other, SearchResult):
            return False
        return self.url == other.url and self.title == other.title


class SearchEngineBase(ABC):
    """
    搜索引擎基类
    
    所有搜索引擎必须继承此类并实现必要的方法。
    参考 Local Deep Research 的设计模式。
    
    Example:
        class ArXivSearchEngine(SearchEngineBase):
            name = "arxiv"
            description = "arXiv 学术论文搜索"
            category = "academic"
            
            async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
                # 实现搜索逻辑
                pass
            
            def is_available(self) -> bool:
                # 检查可用性
                return True
    """
    
    # 引擎元数据（类属性，子类应覆盖）
    name: str = ""
    description: str = ""
    category: str = "general"  # general/academic/code/encyclopedia
    
    def __init__(self):
        """初始化搜索引擎"""
        if not self.name:
            raise ValueError(f"{self.__class__.__name__} 必须设置 name 属性")
        if not self.description:
            raise ValueError(f"{self.__class__.__name__} 必须设置 description 属性")
    
    @abstractmethod
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查引擎是否可用
        
        Returns:
            bool: 是否可用
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            健康状态信息
        """
        return {
            "name": self.name,
            "available": self.is_available(),
            "category": self.category,
            "timestamp": datetime.now().isoformat()
        }
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', category='{self.category}')>"


class SearchEngineFactory:
    """
    搜索引擎工厂
    
    管理所有搜索引擎的注册和创建。
    """
    
    _engines: Dict[str, type] = {}
    
    @classmethod
    def register(cls, engine_class: type):
        """
        注册搜索引擎
        
        Args:
            engine_class: 搜索引擎类
        """
        if not issubclass(engine_class, SearchEngineBase):
            raise ValueError(f"{engine_class} 必须继承 SearchEngineBase")
        cls._engines[engine_class.name] = engine_class
        return engine_class
    
    @classmethod
    def create(cls, name: str) -> Optional[SearchEngineBase]:
        """
        创建搜索引擎实例
        
        Args:
            name: 引擎名称
            
        Returns:
            SearchEngineBase 实例或 None
        """
        engine_class = cls._engines.get(name)
        if engine_class:
            return engine_class()
        return None
    
    @classmethod
    def list_engines(cls, category: Optional[str] = None) -> List[str]:
        """
        列出所有可用引擎
        
        Args:
            category: 按类别过滤
            
        Returns:
            引擎名称列表
        """
        if category:
            return [
                name for name, engine_class in cls._engines.items()
                if engine_class.category == category
            ]
        return list(cls._engines.keys())
    
    @classmethod
    def get_categories(cls) -> List[str]:
        """获取所有类别"""
        categories = set()
        for engine_class in cls._engines.values():
            categories.add(engine_class.category)
        return list(categories)


# 装饰器用于自动注册

def register_engine(engine_class: type) -> type:
    """
    搜索引擎注册装饰器
    
    Example:
        @register_engine
        class ArXivSearchEngine(SearchEngineBase):
            name = "arxiv"
            ...
    """
    SearchEngineFactory.register(engine_class)
    return engine_class
