"""
Serper.dev 搜索引擎

Google 搜索 API，国内可用
免费额度：2500次/月
申请地址：https://serper.dev
"""

import os
import asyncio
import aiohttp
from typing import List, Optional
from urllib.parse import quote

from skills.enhanced_search.engines.base import SearchEngineBase, SearchResult, register_engine


@register_engine
class SerperSearchEngine(SearchEngineBase):
    """
    Serper.dev Google 搜索 API
    
    特点:
    - 国内可直接访问
    - 返回 Google 搜索结果
    - 支持搜索建议、图片、新闻等
    
    限制:
    - 需要 API Key
    - 免费额度：2500次/月
    """
    
    name = "serper"
    description = "Serper.dev Google 搜索 API"
    category = "general"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("SERPER_API_KEY")
        self.base_url = "https://google.serper.dev/search"
    
    def is_available(self) -> bool:
        """
        检查引擎是否可用
        
        Returns:
            bool: 是否配置了 API Key
        """
        return bool(self.api_key)
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        执行 Serper 搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not configured")
        
        results = []
        
        try:
            headers = {
                "X-API-KEY": self.api_key,
                "Content-Type": "application/json"
            }
            
            payload = {
                "q": query,
                "num": min(num_results, 10)  # Serper 最大返回10条
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Serper API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    
                    # 解析有机搜索结果
                    organic_results = data.get("organic", [])
                    
                    for i, item in enumerate(organic_results[:num_results]):
                        result = SearchResult(
                            title=item.get("title", "无标题"),
                            url=item.get("link", ""),
                            snippet=item.get("snippet", ""),
                            source=self.name,
                            score=1.0 - (i * 0.1),
                            category=self.category
                        )
                        results.append(result)
                    
                    # 如果有知识图谱，也加入结果
                    knowledge_graph = data.get("knowledgeGraph", {})
                    if knowledge_graph:
                        kg_result = SearchResult(
                            title=knowledge_graph.get("title", "知识图谱"),
                            url=knowledge_graph.get("website", ""),
                            snippet=knowledge_graph.get("description", ""),
                            source=f"{self.name}_kg",
                            score=1.0,
                            category=self.category
                        )
                        results.insert(0, kg_result)  # 知识图谱优先级最高
        
        except Exception as e:
            print(f"❌ Serper 搜索失败: {e}")
            raise
        
        return results


if __name__ == "__main__":
    # 测试
    async def test():
        engine = SerperSearchEngine()
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
