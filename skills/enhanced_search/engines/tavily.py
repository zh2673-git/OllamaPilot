"""
Tavily 搜索引擎

专为 AI 应用设计的搜索 API，国内可用
免费额度：1000次/月
申请地址：https://tavily.com
"""

import os
import asyncio
import aiohttp
from typing import List, Optional

from skills.enhanced_search.engines.base import SearchEngineBase, SearchResult, register_engine


@register_engine
class TavilySearchEngine(SearchEngineBase):
    """
    Tavily Search API - 专为 AI 应用设计的搜索
    
    特点:
    - 国内可直接访问
    - 专为 AI/LLM 应用优化
    - 返回结构化的搜索结果，包含摘要
    - 支持搜索深度控制
    
    限制:
    - 需要 API Key
    - 免费额度：1000次/月
    """
    
    name = "tavily"
    description = "Tavily Search API - 专为AI应用设计"
    category = "general"
    
    def __init__(self):
        super().__init__()
        self.api_key = os.environ.get("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com/search"
    
    def is_available(self) -> bool:
        """
        检查引擎是否可用
        
        Returns:
            bool: 是否配置了 API Key
        """
        return bool(self.api_key)
    
    async def search(
        self, 
        query: str, 
        num_results: int = 10,
        search_depth: str = "basic"  # basic 或 advanced
    ) -> List[SearchResult]:
        """
        执行 Tavily 搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            search_depth: 搜索深度 (basic/advanced)
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY not configured")
        
        results = []
        
        try:
            payload = {
                "api_key": self.api_key,
                "query": query,
                "max_results": min(num_results, 20),  # Tavily 最大返回20条
                "search_depth": search_depth,
                "include_answer": True,  # 包含AI生成的答案摘要
                "include_images": False,
                "include_raw_content": False,
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15)  # Tavily 可能需要更长时间
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise Exception(f"Tavily API error: {response.status} - {error_text}")
                    
                    data = await response.json()
                    
                    # 如果有AI生成的答案，作为第一个结果
                    answer = data.get("answer", "")
                    if answer:
                        answer_result = SearchResult(
                            title="AI 智能摘要",
                            url="",
                            snippet=answer,
                            source=f"{self.name}_answer",
                            score=2.0,  # 最高优先级
                            category=self.category
                        )
                        results.append(answer_result)
                    
                    # 解析搜索结果
                    search_results = data.get("results", [])
                    
                    for i, item in enumerate(search_results[:num_results]):
                        result = SearchResult(
                            title=item.get("title", "无标题"),
                            url=item.get("url", ""),
                            snippet=item.get("content", ""),  # Tavily 提供内容摘要
                            source=self.name,
                            score=1.0 - (i * 0.1),
                            category=self.category
                        )
                        results.append(result)
        
        except Exception as e:
            print(f"❌ Tavily 搜索失败: {e}")
            raise
        
        return results


if __name__ == "__main__":
    # 测试
    async def test():
        engine = TavilySearchEngine()
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
