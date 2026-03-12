"""
百度搜索引擎

国内最常用的搜索引擎，无需API Key
通过网页爬虫方式获取搜索结果
"""

import asyncio
import aiohttp
from typing import List, Optional
from urllib.parse import quote, unquote
import re

from skills.enhanced_search.engines.base import SearchEngineBase, SearchResult, register_engine


@register_engine
class BaiduSearchEngine(SearchEngineBase):
    """
    百度搜索引擎
    
    特点:
    - 完全免费，无需API Key
    - 中文搜索结果质量高
    - 国内访问速度快
    
    限制:
    - 需要解析HTML，可能受页面结构变化影响
    - 有反爬虫机制，需要控制请求频率
    """
    
    name = "baidu"
    description = "百度搜索"
    category = "general"
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.baidu.com/s"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Cookie": "BAIDUID=1234567890:FG=1",
        }
    
    def is_available(self) -> bool:
        """百度搜索永远可用（无需配置）"""
        return True
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        执行百度搜索
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            print("⚠️ 未安装 BeautifulSoup，百度搜索需要: pip install beautifulsoup4")
            return []
        
        results = []
        
        try:
            # 百度搜索参数
            params = {
                "wd": query,
                "pn": 0,  # 起始位置
                "rn": min(num_results, 50),  # 每页结果数
                "ie": "utf-8",
            }
            
            async with aiohttp.ClientSession(headers=self.headers) as session:
                async with session.get(
                    self.base_url, 
                    params=params, 
                    timeout=aiohttp.ClientTimeout(total=10),
                    ssl=False  # 百度有时SSL证书有问题
                ) as response:
                    if response.status != 200:
                        print(f"❌ 百度搜索请求失败: {response.status}")
                        return []
                    
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # 解析搜索结果
                    # 百度的搜索结果在 .result 或 .c-container 类中
                    result_divs = soup.find_all(['div'], class_=['result', 'c-container'])
                    
                    for i, div in enumerate(result_divs[:num_results]):
                        try:
                            # 提取标题和链接
                            title_tag = div.find('h3')
                            if not title_tag:
                                continue
                            
                            title = title_tag.get_text(strip=True)
                            
                            # 提取链接
                            link_tag = title_tag.find('a')
                            if not link_tag:
                                continue
                            
                            url = link_tag.get('href', '')
                            # 百度链接是跳转链接，需要解析真实URL
                            if url.startswith('/link'):
                                url = f"https://www.baidu.com{url}"
                            
                            # 提取摘要
                            content_div = div.find(['div', 'span'], class_=re.compile('content|abstract'))
                            if not content_div:
                                # 尝试其他选择器
                                content_div = div.find('span', class_='content-right_8Zs40')
                            
                            snippet = content_div.get_text(strip=True) if content_div else ""
                            
                            # 创建结果对象
                            result = SearchResult(
                                title=title,
                                url=url,
                                snippet=snippet,
                                source=self.name,
                                score=1.0 - (i * 0.1),
                                category=self.category
                            )
                            results.append(result)
                            
                        except Exception as e:
                            print(f"⚠️ 解析百度结果失败: {e}")
                            continue
                    
                    if not results:
                        # 尝试备选解析方式
                        results = await self._fallback_parse(soup, num_results)
        
        except asyncio.TimeoutError:
            print("❌ 百度搜索请求超时")
        except Exception as e:
            print(f"❌ 百度搜索失败: {e}")
        
        return results
    
    async def _fallback_parse(self, soup, num_results: int) -> List[SearchResult]:
        """备选解析方式"""
        results = []
        
        try:
            # 尝试其他可能的选择器
            selectors = [
                ('div', {'class': 'result'}),
                ('div', {'class': 'c-container'}),
            ]
            
            for tag, attrs in selectors:
                elements = soup.find_all(tag, attrs)
                
                if elements:
                    for i, elem in enumerate(elements[:num_results]):
                        try:
                            # 尝试提取标题
                            title_elem = elem.find('h3') or elem.find('a')
                            if not title_elem:
                                continue
                            
                            title = title_elem.get_text(strip=True)
                            url = title_elem.get('href', '')
                            
                            # 尝试提取摘要
                            snippet_elem = elem.find(['div', 'span', 'p'])
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
            print(f"⚠️ 百度备选解析失败: {e}")
        
        return results


# 同步包装函数
def baidu_search(query: str, num_results: int = 10) -> List[SearchResult]:
    """同步方式的百度搜索"""
    engine = BaiduSearchEngine()
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
        return loop.run_until_complete(engine.search(query, num_results))
    except Exception as e:
        print(f"❌ 百度搜索失败: {e}")
        return []


if __name__ == "__main__":
    # 测试
    async def test():
        engine = BaiduSearchEngine()
        print(f"引擎可用性: {engine.is_available()}")
        
        results = await engine.search("Python 教程", num_results=5)
        print(f"\n找到 {len(results)} 条结果:\n")
        
        for i, r in enumerate(results, 1):
            print(f"{i}. {r.title}")
            print(f"   URL: {r.url}")
            print(f"   摘要: {r.snippet[:100]}...")
            print()
    
    asyncio.run(test())
