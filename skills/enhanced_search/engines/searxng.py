"""
SearXNG 搜索引擎

复用内置的 web_search 工具实现
"""

import os
import json
import urllib.request
import urllib.parse
from typing import List

from .base import SearchEngineBase, SearchResult, register_engine


@register_engine
class SearXNGSearchEngine(SearchEngineBase):
    """
    SearXNG 元搜索引擎
    
    本地部署的聚合搜索引擎，无需 API Key，完全免费。
    复用 ollamapilot/tools/builtin.py 中的 SearXNG 实现。
    
    特点:
    - 本地部署，数据隐私
    - 聚合多源结果
    - 无需 API Key
    - 国内可用
    """
    
    name = "searxng"
    description = "SearXNG 本地聚合搜索引擎"
    category = "general"
    
    def __init__(self, base_url: str = None):
        """
        初始化 SearXNG 引擎
        
        Args:
            base_url: SearXNG 服务地址，默认从环境变量读取
        """
        super().__init__()
        self.base_url = base_url or os.environ.get("SEARXNG_URL", "http://localhost:8080")
    
    def is_available(self) -> bool:
        """
        检查 SearXNG 是否可用
        
        Returns:
            bool: 是否可用
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/healthz",
                headers={"User-Agent": "OllamaPilot/1.0"},
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        执行搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        if not self.is_available():
            raise RuntimeError(
                f"SearXNG 服务未运行 ({self.base_url})\n"
                f"请运行: docker run -d --name searxng -p 8080:8080 searxng/searxng"
            )
        
        try:
            # 限制结果数量
            num_results = max(1, min(20, num_results))
            
            # 构建 SearXNG 请求
            params = {
                "q": query,
                "format": "json",
                "language": "zh-CN",
                "safesearch": "0",
            }
            
            url = f"{self.base_url}/search?{urllib.parse.urlencode(params)}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/json",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            
            # 解析结果
            results = data.get("results", [])
            results = results[:num_results]
            
            search_results = []
            for result in results:
                search_results.append(SearchResult(
                    title=result.get("title", "无标题"),
                    url=result.get("url", ""),
                    snippet=result.get("content", "无描述"),
                    source=self.name,
                    category=self.category,
                    metadata={
                        "engine": result.get("engine", "unknown"),
                        "score": result.get("score"),
                    }
                ))
            
            return search_results
            
        except Exception as e:
            raise RuntimeError(f"SearXNG 搜索失败: {e}")
