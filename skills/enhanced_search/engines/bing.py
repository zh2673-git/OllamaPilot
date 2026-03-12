"""
Bing 搜索引擎

微软必应搜索 API，国内可用
免费额度：1000次/月
申请地址：https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
"""

import os
import asyncio
import aiohttp
from typing import List, Optional

from skills.enhanced_search.engines.base import SearchEngineBase, SearchResult, register_engine


@register_engine
class BingSearchEngine(SearchEngineBase):
    """
    微软必应搜索 API
    
    特点:
    - 国内可直接访问
    - 返回必应搜索结果
    - 支持网页、图片、新闻等
    
    限制:
    - 需要 Azure API Key
    - 免费额度：1000次/月
    """
    
    name = "bing"
    description = "微软必应搜索 API"
    category = "general"
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.bing.microsoft.com/v7.0/search"

    def is_available(self) -> bool:
        """
        检查引擎是否可用

        Returns:
            bool: 是否配置了 API Key
        """
        # 实时检查环境变量，确保 .env 加载后生效
        api_key = os.environ.get("BING_API_KEY")
        return bool(api_key)

    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        执行 Bing 搜索

        Args:
            query: 搜索查询
            num_results: 返回结果数量

        Returns:
            List[SearchResult]: 搜索结果列表
        """
        # 实时获取 API Key
        api_key = os.environ.get("BING_API_KEY")
        if not api_key:
            raise ValueError("BING_API_KEY not configured")

        results = []

        try:
            headers = {
                "Ocp-Apim-Subscription-Key": api_key
            }
            
            params = {
                "q": query,
                "count": min(num_results, 50),  # Bing 最大返回50条
                "mkt": "zh-CN",  # 中文市场
                "setLang": "zh"
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
                        raise Exception(f"Bing API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    
                    # 解析网页搜索结果
                    web_pages = data.get("webPages", {})
                    web_results = web_pages.get("value", [])
                    
                    for i, item in enumerate(web_results[:num_results]):
                        result = SearchResult(
                            title=item.get("name", "无标题"),
                            url=item.get("url", ""),
                            snippet=item.get("snippet", ""),
                            source=self.name,
                            score=1.0 - (i * 0.1),
                            category=self.category
                        )
                        results.append(result)
        
        except Exception as e:
            print(f"❌ Bing 搜索失败: {e}")
            raise
        
        return results


if __name__ == "__main__":
    # 测试
    async def test():
        engine = BingSearchEngine()
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
