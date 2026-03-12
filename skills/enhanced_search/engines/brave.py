"""
Brave 搜索引擎

Brave Search API，国内可用
免费额度：2000次/月
申请地址：https://brave.com/search/api/
"""

import os
import asyncio
import aiohttp
from typing import List, Optional

from skills.enhanced_search.engines.base import SearchEngineBase, SearchResult, register_engine


@register_engine
class BraveSearchEngine(SearchEngineBase):
    """
    Brave Search API
    
    特点:
    - 国内可直接访问
    - 独立的搜索引擎，不依赖 Google/Bing
    - 隐私保护，不追踪用户
    
    限制:
    - 需要 API Key
    - 免费额度：2000次/月
    """
    
    name = "brave"
    description = "Brave Search API"
    category = "general"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("BRAVE_API_KEY")
        self.base_url = "https://api.search.brave.com/res/v1/web/search"
    
    def is_available(self) -> bool:
        """
        检查引擎是否可用
        
        Returns:
            bool: 是否配置了 API Key
        """
        return bool(self.api_key)
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        执行 Brave 搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        if not self.api_key:
            raise ValueError("BRAVE_API_KEY not configured")
        
        results = []
        
        try:
            headers = {
                "X-Subscription-Token": self.api_key,
                "Accept": "application/json"
            }
            
            params = {
                "q": query,
                "count": min(num_results, 20),  # Brave 最大返回20条
                "offset": 0,
                "mkt": "zh-CN",  # 中文市场
                "safesearch": "off"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.base_url,
                    headers=headers,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Brave API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    
                    # 解析网页搜索结果
                    web_results = data.get("web", {}).get("results", [])
                    
                    for i, item in enumerate(web_results[:num_results]):
                        result = SearchResult(
                            title=item.get("title", "无标题"),
                            url=item.get("url", ""),
                            snippet=item.get("description", ""),
                            source=self.name,
                            score=1.0 - (i * 0.1),
                            category=self.category
                        )
                        results.append(result)
        
        except Exception as e:
            print(f"❌ Brave 搜索失败: {e}")
            raise
        
        return results


if __name__ == "__main__":
    # 测试
    async def test():
        engine = BraveSearchEngine()
        print(f"引擎可用性: {engine.is_available()}")
        
        if engine.is_available():
            results = await engine.search("Python 教程", num_results=5)
            print(f"\n找到 {len(results)} 条结果:\n")
            
            for i, r in enumerate(results, 1):
                print(f"{i}. {r.title}")
                print(f"   URL: {r.url}")
                print(f"   摘要: {r.snippet[:100]}...")
                print()
    
    asyncio.run(test())
