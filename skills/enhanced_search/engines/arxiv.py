"""
arXiv 学术论文搜索引擎

使用 arXiv API 搜索学术论文，完全免费，无需 API Key。
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List
from datetime import datetime

from .base import SearchEngineBase, SearchResult, register_engine


@register_engine
class ArXivSearchEngine(SearchEngineBase):
    """
    arXiv 学术论文搜索引擎
    
    特点:
    - 完全免费，无需 API Key
    - 覆盖物理、数学、计算机科学等领域
    - 国内可用
    - 支持按日期、相关性排序
    
    API 文档: https://arxiv.org/help/api
    """
    
    name = "arxiv"
    description = "arXiv 学术论文搜索"
    category = "academic"
    
    # arXiv API 基础 URL
    API_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        super().__init__()
    
    def is_available(self) -> bool:
        """
        检查 arXiv API 是否可用
        
        Returns:
            bool: 是否可用
        """
        try:
            req = urllib.request.Request(
                "http://export.arxiv.org/api/query?search_query=all:quantum&max_results=1",
                headers={"User-Agent": "OllamaPilot/1.0"},
                timeout=10
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        搜索 arXiv 论文
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量 (最大 50)
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        try:
            # 限制结果数量 (arXiv API 建议不要一次请求太多)
            num_results = max(1, min(50, num_results))
            
            # 构建查询参数
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": num_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
            
            headers = {
                "User-Agent": "OllamaPilot/1.0 (https://github.com/your-repo)",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode("utf-8")
            
            # 解析 XML
            root = ET.fromstring(data)
            
            # arXiv Atom 命名空间
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"
            }
            
            search_results = []
            
            for entry in root.findall("atom:entry", ns):
                # 跳过 arXiv 的说明条目
                if entry.find("atom:title", ns) is None:
                    continue
                    
                title_elem = entry.find("atom:title", ns)
                title = title_elem.text if title_elem is not None else "无标题"
                
                url_elem = entry.find("atom:id", ns)
                url = url_elem.text if url_elem is not None else ""
                
                summary_elem = entry.find("atom:summary", ns)
                snippet = summary_elem.text if summary_elem is not None else ""
                # 限制摘要长度
                snippet = snippet[:500] + "..." if len(snippet) > 500 else snippet
                
                # 获取作者
                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None:
                        authors.append(name_elem.text)
                author_str = ", ".join(authors[:3])  # 最多显示3个作者
                if len(authors) > 3:
                    author_str += " et al."
                
                # 获取发布日期
                published_elem = entry.find("atom:published", ns)
                published_date = None
                if published_elem is not None:
                    try:
                        dt = datetime.fromisoformat(published_elem.text.replace("Z", "+00:00"))
                        published_date = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        pass
                
                # 获取 PDF 链接
                pdf_url = ""
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                        break
                
                # 获取分类
                categories = []
                for cat in entry.findall("arxiv:primary_category", ns):
                    categories.append(cat.get("term", ""))
                
                search_results.append(SearchResult(
                    title=title.strip(),
                    url=url,
                    snippet=snippet.strip(),
                    source=self.name,
                    category=self.category,
                    published_date=published_date,
                    author=author_str,
                    metadata={
                        "pdf_url": pdf_url,
                        "categories": categories,
                        "authors": authors,
                    }
                ))
            
            return search_results
            
        except Exception as e:
            raise RuntimeError(f"arXiv 搜索失败: {e}")
    
    async def search_by_category(self, query: str, category: str, num_results: int = 10) -> List[SearchResult]:
        """
        按类别搜索 arXiv 论文
        
        Args:
            query: 搜索查询
            category: arXiv 类别 (如 cs.AI, cs.CL, physics.quant-ph)
            num_results: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        # 构建类别过滤查询
        category_query = f"cat:{category} AND all:{query}"
        
        try:
            num_results = max(1, min(50, num_results))
            
            params = {
                "search_query": category_query,
                "start": 0,
                "max_results": num_results,
                "sortBy": "relevance",
                "sortOrder": "descending"
            }
            
            url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
            
            headers = {
                "User-Agent": "OllamaPilot/1.0",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode("utf-8")
            
            # 解析 XML (复用 search 方法的逻辑)
            root = ET.fromstring(data)
            ns = {
                "atom": "http://www.w3.org/2005/Atom",
                "arxiv": "http://arxiv.org/schemas/atom"
            }
            
            search_results = []
            
            for entry in root.findall("atom:entry", ns):
                if entry.find("atom:title", ns) is None:
                    continue
                    
                title_elem = entry.find("atom:title", ns)
                title = title_elem.text if title_elem is not None else "无标题"
                
                url_elem = entry.find("atom:id", ns)
                url = url_elem.text if url_elem is not None else ""
                
                summary_elem = entry.find("atom:summary", ns)
                snippet = summary_elem.text if summary_elem is not None else ""
                snippet = snippet[:500] + "..." if len(snippet) > 500 else snippet
                
                authors = []
                for author in entry.findall("atom:author", ns):
                    name_elem = author.find("atom:name", ns)
                    if name_elem is not None:
                        authors.append(name_elem.text)
                author_str = ", ".join(authors[:3])
                if len(authors) > 3:
                    author_str += " et al."
                
                published_elem = entry.find("atom:published", ns)
                published_date = None
                if published_elem is not None:
                    try:
                        dt = datetime.fromisoformat(published_elem.text.replace("Z", "+00:00"))
                        published_date = dt.strftime("%Y-%m-%d")
                    except ValueError:
                        pass

                pdf_url = ""
                for link in entry.findall("atom:link", ns):
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                        break
                
                categories = []
                for cat in entry.findall("arxiv:primary_category", ns):
                    categories.append(cat.get("term", ""))
                
                search_results.append(SearchResult(
                    title=title.strip(),
                    url=url,
                    snippet=snippet.strip(),
                    source=self.name,
                    category=self.category,
                    published_date=published_date,
                    author=author_str,
                    metadata={
                        "pdf_url": pdf_url,
                        "categories": categories,
                        "authors": authors,
                        "search_category": category,
                    }
                ))
            
            return search_results
            
        except Exception as e:
            raise RuntimeError(f"arXiv 分类搜索失败: {e}")
