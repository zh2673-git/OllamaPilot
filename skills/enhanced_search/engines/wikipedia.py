"""
Wikipedia 百科搜索引擎

使用 Wikipedia API 搜索百科条目，完全免费，无需 API Key。
"""

import urllib.request
import urllib.parse
import json
from typing import List, Optional

from .base import SearchEngineBase, SearchResult, register_engine


@register_engine
class WikipediaSearchEngine(SearchEngineBase):
    """
    Wikipedia 百科搜索引擎
    
    特点:
    - 完全免费，无需 API Key
    - 多语言支持 (默认中文)
    - 国内可用
    - 丰富的元数据
    
    API 文档: https://www.mediawiki.org/wiki/API:Search
    """
    
    name = "wikipedia"
    description = "Wikipedia 百科搜索"
    category = "encyclopedia"
    
    # Wikipedia API 基础 URL
    API_URL = "https://zh.wikipedia.org/w/api.php"
    
    def __init__(self, language: str = "zh"):
        """
        初始化 Wikipedia 引擎
        
        Args:
            language: 语言代码 (zh, en, ja 等)
        """
        super().__init__()
        self.language = language
        self.api_url = f"https://{language}.wikipedia.org/w/api.php"
    
    def is_available(self) -> bool:
        """
        检查 Wikipedia API 是否可用
        
        Returns:
            bool: 是否可用
        """
        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": "test",
                "srlimit": 1
            }
            url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
            
            headers = {
                "User-Agent": "OllamaPilot/1.0 (https://github.com/your-repo)",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        搜索 Wikipedia
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量 (最大 50)
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        try:
            # 限制结果数量
            num_results = max(1, min(50, num_results))
            
            # 构建查询参数
            params = {
                "action": "query",
                "format": "json",
                "list": "search",
                "srsearch": query,
                "srlimit": num_results,
                "srprop": "timestamp|wordcount|snippet|titlesnippet|redirecttitle",
                "utf8": 1,
                "formatversion": 2
            }
            
            url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
            
            headers = {
                "User-Agent": "OllamaPilot/1.0 (https://github.com/your-repo)",
                "Accept": "application/json",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            
            search_results = []
            
            # 解析搜索结果
            search_list = data.get("query", {}).get("search", [])
            
            for item in search_list:
                title = item.get("title", "")
                page_id = item.get("pageid", 0)
                snippet = item.get("snippet", "")
                wordcount = item.get("wordcount", 0)
                timestamp = item.get("timestamp", "")
                
                # 构建页面 URL
                page_url = f"https://{self.language}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                
                # 清理 HTML 标签
                snippet = self._clean_html(snippet)
                
                search_results.append(SearchResult(
                    title=title,
                    url=page_url,
                    snippet=snippet if snippet else f"Wikipedia 条目: {title}",
                    source=self.name,
                    category=self.category,
                    published_date=timestamp.split("T")[0] if timestamp else None,
                    metadata={
                        "page_id": page_id,
                        "wordcount": wordcount,
                        "language": self.language,
                    }
                ))
            
            return search_results
            
        except Exception as e:
            raise RuntimeError(f"Wikipedia 搜索失败: {e}")
    
    async def get_page_summary(self, title: str) -> Optional[str]:
        """
        获取 Wikipedia 页面摘要
        
        Args:
            title: 页面标题
            
        Returns:
            页面摘要或 None
        """
        try:
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "exsentences": 3,
                "utf8": 1,
                "formatversion": 2
            }
            
            url = f"{self.api_url}?{urllib.parse.urlencode(params)}"
            
            headers = {
                "User-Agent": "OllamaPilot/1.0",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            
            pages = data.get("query", {}).get("pages", [])
            if pages and len(pages) > 0:
                extract = pages[0].get("extract", "")
                return extract
            
            return None
            
        except Exception:
            return None
    
    def _clean_html(self, text: str) -> str:
        """
        清理 HTML 标签
        
        Args:
            text: 包含 HTML 的文本
            
        Returns:
            清理后的文本
        """
        import re
        # 移除 HTML 标签
        text = re.sub(r'<[^>]+>', '', text)
        # 解码 HTML 实体
        text = text.replace('&quot;', '"')
        text = text.replace('&amp;', '&')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&#39;', "'")
        return text.strip()


@register_engine
class WikipediaENSearchEngine(WikipediaSearchEngine):
    """
    英文 Wikipedia 搜索引擎
    """
    
    name = "wikipedia_en"
    description = "Wikipedia English 百科搜索"
    category = "encyclopedia"
    
    def __init__(self):
        super().__init__(language="en")
