"""
DuckDuckGo 搜索引擎

完全免费的搜索引擎，无需API Key，国内可访问
作为 SearXNG 的备用方案
"""

import asyncio
import aiohttp
from typing import List, Optional
from urllib.parse import quote

from skills.enhanced_search.engines.base import SearchEngineBase, SearchResult, register_engine


@register_engine
class DuckDuckGoSearchEngine(SearchEngineBase):
    """
    DuckDuckGo 搜索引擎
    
    特点:
    - 完全免费，无需API Key
    - 隐私保护，不追踪用户
    - 国内可访问
    - 可作为 SearXNG 的备用方案
    
    限制:
    - 需要解析HTML，可能受页面结构变化影响
    - 速率限制：建议每秒不超过1次请求
    """
    
    name = "duckduckgo"
    description = "DuckDuckGo 隐私搜索引擎"
    category = "general"
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://html.duckduckgo.com/html"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "DNT": "1",
            "Connection": "keep-alive",
        }
    
    def is_available(self) -> bool:
        """
        检查引擎是否可用
        
        Returns:
            bool: 是否可用（无需API Key，始终可用）
        """
        return True
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        执行 DuckDuckGo 搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("⚠️ 未安装 BeautifulSoup，DuckDuckGo 搜索需要: pip install beautifulsoup4")
            return []
        
        results = []
        
        try:
            # DuckDuckGo HTML 版本
            params = {
                "q": query,
                "kl": "zh-cn",  # 中文区域
            }
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(
                    self.base_url, 
                    params=params, 
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        print(f"❌ DuckDuckGo 请求失败: {response.status}")
                        return []
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 解析搜索结果
                    # DuckDuckGo HTML 版本的搜索结果在 .result 类中
                    result_divs = soup.find_all('div', class_='result')
                    
                    for i, div in enumerate(result_divs[:num_results]):
                        try:
                            # 提取标题和链接
                            title_a = div.find('a', class_='result__a')
                            if not title_a:
                                continue
                            
                            title = title_a.get_text(strip=True)
                            url = title_a.get('href', '')
                            
                            # 提取摘要
                            snippet_div = div.find('a', class_='result__snippet')
                            snippet = snippet_div.get_text(strip=True) if snippet_div else ""
                            
                            # 创建结果对象
                            result = SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source=self.name,
                                score=1.0 - (i * 0.1),  # 按排名递减
                                category=self.category
                            )
                            results.append(result)
                            
                        except Exception as e:
                            print(f"⚠️ 解析结果失败: {e}")
                            continue
                    
                    # 如果没有找到结果，尝试备选解析方式
                    if not results:
                        results = await self._fallback_parse(soup, num_results)
        
        except asyncio.TimeoutError:
            print("❌ DuckDuckGo 请求超时")
        except Exception as e:
            print(f"❌ DuckDuckGo 搜索失败: {e}")
        
        return results
    
    async def _fallback_parse(self, soup, num_results: int) -> List[SearchResult]:
        """
        备选解析方式
        
        当主解析方式失败时尝试其他选择器
        """
        results = []
        
        try:
            # 尝试其他可能的选择器
            selectors = [
                ('div', 'web-result'),
                ('div', 'result__body'),
                ('article', None),
            ]
            
            for tag, class_name in selectors:
                if class_name:
                    elements = soup.find_all(tag, class_=class_name)
                else:
                    elements = soup.find_all(tag)
                
                if elements:
                    for i, elem in enumerate(elements[:num_results]):
                        try:
                            # 尝试提取链接
                            link = elem.find('a')
                            if not link:
                                continue
                            
                            title = link.get_text(strip=True)
                            url = link.get('href', '')
                            
                            # 尝试提取摘要
                            snippet_elem = elem.find(['p', 'span', 'div'], class_=lambda x: x and 'snippet' in x.lower())
                            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                            
                            result = SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source=self.name,
                                score=1.0 - (i * 0.1),
                                category=self.category
                            )
                            results.append(result)
                            
                        except Exception:
                            continue
                    
                    if results:
                        break
        
        except Exception as e:
            print(f"⚠️ 备选解析失败: {e}")
        
        return results


# 同步包装函数（供工具调用）
def duckduckgo_search(query: str, num_results: int = 10) -> List[SearchResult]:
    """
    同步方式的 DuckDuckGo 搜索
    
    Args:
        query: 搜索查询
        num_results: 返回结果数量
        
    Returns:
        List[SearchResult]: 搜索结果列表
    """
    engine = DuckDuckGoSearchEngine()
    
    try:
        # 使用 asyncio.run 运行异步方法
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # 如果事件循环已在运行，创建新任务
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(engine.search(query, num_results))
    except Exception as e:
        print(f"❌ DuckDuckGo 搜索失败: {e}")
        return []


if __name__ == "__main__":
    # 测试
    async def test():
        engine = DuckDuckGoSearchEngine()
        print(f"引擎可用性: {engine.is_available()}")
        
        results = await engine.search("Python 教程", num_results=5)
        print(f"\n找到 {len(results)} 条结果:\n")
        
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.title}")
            print(f"   URL: {r.url}")
            print(f"   摘要: {r.snippet[:100]}...")
            print()
    
    asyncio.run(test())
