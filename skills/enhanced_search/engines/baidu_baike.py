"""
百度百科搜索引擎

使用百度百科搜索百科条目，完全免费，无需 API Key。
"""

import urllib.request
import urllib.parse
import json
import re
from typing import List

from .base import SearchEngineBase, SearchResult, register_engine


@register_engine
class BaiduBaikeSearchEngine(SearchEngineBase):
    """
    百度百科搜索引擎
    
    特点:
    - 完全免费，无需 API Key
    - 中文百科，国内优化
    - 国内访问速度快
    - 丰富的中文词条内容
    
    使用百度搜索建议 API 和百度百科页面
    """
    
    name = "baidu_baike"
    description = "百度百科搜索"
    category = "encyclopedia"
    
    # 百度搜索建议 API
    SUGGEST_API = "https://sp0.baidu.com/5a1Fazu8AA54nxGko9WTAnF6hhy/su"
    # 百度百科搜索页面
    BAIKE_URL = "https://baike.baidu.com/item/"
    
    def __init__(self):
        super().__init__()
    
    def is_available(self) -> bool:
        """
        检查百度百科是否可用
        
        Returns:
            bool: 是否可用
        """
        try:
            url = "https://baike.baidu.com"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        搜索百度百科
        
        由于百度百科没有公开的搜索 API，我们使用以下策略:
        1. 尝试直接访问词条页面
        2. 使用百度搜索建议获取相关词条
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        search_results = []
        
        # 策略 1: 尝试直接访问词条页面
        direct_result = await self._try_direct_access(query)
        if direct_result:
            search_results.append(direct_result)
        
        # 策略 2: 获取搜索建议
        suggestions = await self._get_suggestions(query)
        
        for suggestion in suggestions[:num_results - len(search_results)]:
            if suggestion != query:  # 避免重复
                result = await self._try_direct_access(suggestion)
                if result:
                    search_results.append(result)
        
        return search_results
    
    async def _try_direct_access(self, query: str) -> SearchResult:
        """
        尝试直接访问百度百科词条
        
        Args:
            query: 查询词
            
        Returns:
            SearchResult 或 None
        """
        try:
            # 构建词条 URL
            encoded_query = urllib.parse.quote(query)
            url = f"{self.BAIKE_URL}{encoded_query}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                html = response.read().decode("utf-8")
            
            # 提取标题
            title_match = re.search(r'<title>(.*?)_百度百科</title>', html)
            if not title_match:
                return None
            
            title = title_match.group(1).strip()
            
            # 提取摘要 (从 meta description)
            desc_match = re.search(r'<meta name="description" content="(.*?)"', html)
            snippet = desc_match.group(1) if desc_match else f"百度百科词条: {title}"
            
            # 清理摘要
            snippet = snippet.replace("百度百科", "").strip()
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            
            return SearchResult(
                title=title,
                url=url,
                snippet=snippet,
                source=self.name,
                category=self.category,
                metadata={
                    "query": query,
                    "is_direct_match": True,
                }
            )
            
        except Exception:
            return None
    
    async def _get_suggestions(self, query: str) -> List[str]:
        """
        获取百度搜索建议
        
        Args:
            query: 查询词
            
        Returns:
            建议列表
        """
        try:
            params = {
                "wd": query,
                "cb": "callback",
            }
            
            url = f"{self.SUGGEST_API}?{urllib.parse.urlencode(params)}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Referer": "https://www.baidu.com/",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=10) as response:
                data = response.read().decode("gbk")  # 百度返回 GBK 编码
            
            # 解析 JSONP 响应
            # 格式: callback({"q":"query","p":false,"s":["suggestion1","suggestion2",...]})
            json_match = re.search(r'callback\((.*)\)', data)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return result.get("s", [])
            
            return []
            
        except Exception:
            return []
    
    async def get_summary(self, title: str) -> str:
        """
        获取百度百科词条摘要
        
        Args:
            title: 词条标题
            
        Returns:
            摘要文本
        """
        try:
            encoded_title = urllib.parse.quote(title)
            url = f"{self.BAIKE_URL}{encoded_title}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                html = response.read().decode("utf-8")
            
            # 提取摘要
            desc_match = re.search(r'<meta name="description" content="(.*?)"', html)
            if desc_match:
                summary = desc_match.group(1)
                summary = summary.replace("百度百科", "").strip()
                return summary
            
            return ""
            
        except Exception:
            return ""
