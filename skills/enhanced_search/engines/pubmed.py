"""
PubMed 医学文献搜索引擎

使用 PubMed E-utilities API 搜索医学文献，完全免费，无需 API Key。
"""

import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Optional
from datetime import datetime

from .base import SearchEngineBase, SearchResult, register_engine


@register_engine
class PubMedSearchEngine(SearchEngineBase):
    """
    PubMed 医学文献搜索引擎
    
    特点:
    - 完全免费，无需 API Key
    - 覆盖生物医学、生命科学文献
    - 国内可用
    - 提供摘要和元数据
    
    API 文档: https://www.ncbi.nlm.nih.gov/home/documentation/
    """
    
    name = "pubmed"
    description = "PubMed 医学文献搜索"
    category = "academic"
    
    # PubMed E-utilities API
    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    
    def __init__(self):
        super().__init__()
    
    def is_available(self) -> bool:
        """
        检查 PubMed API 是否可用
        
        Returns:
            bool: 是否可用
        """
        try:
            params = {
                "db": "pubmed",
                "term": "test",
                "retmax": 1,
                "retmode": "json"
            }
            url = f"{self.ESEARCH_URL}?{urllib.parse.urlencode(params)}"
            
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
        搜索 PubMed 文献
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量 (最大 100)
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        try:
            # 限制结果数量
            num_results = max(1, min(100, num_results))
            
            # 步骤 1: 搜索获取 ID 列表
            id_list = await self._search_ids(query, num_results)
            
            if not id_list:
                return []
            
            # 步骤 2: 获取文献详情
            results = await self._fetch_summaries(id_list)
            
            return results
            
        except Exception as e:
            raise RuntimeError(f"PubMed 搜索失败: {e}")
    
    async def _search_ids(self, query: str, max_results: int) -> List[str]:
        """
        搜索获取文献 ID 列表
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            
        Returns:
            ID 列表
        """
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance"
        }
        
        url = f"{self.ESEARCH_URL}?{urllib.parse.urlencode(params)}"
        
        headers = {
            "User-Agent": "OllamaPilot/1.0",
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        import json
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        return data.get("esearchresult", {}).get("idlist", [])
    
    async def _fetch_summaries(self, id_list: List[str]) -> List[SearchResult]:
        """
        获取文献摘要
        
        Args:
            id_list: 文献 ID 列表
            
        Returns:
            搜索结果列表
        """
        if not id_list:
            return []
        
        params = {
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "xml"
        }
        
        url = f"{self.EFETCH_URL}?{urllib.parse.urlencode(params)}"
        
        headers = {
            "User-Agent": "OllamaPilot/1.0",
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read().decode("utf-8")
        
        # 解析 XML
        root = ET.fromstring(data)
        
        search_results = []
        
        for article in root.findall(".//PubmedArticle"):
            try:
                # 获取 PMID
                pmid_elem = article.find(".//PMID")
                pmid = pmid_elem.text if pmid_elem is not None else ""
                
                # 获取标题
                title_elem = article.find(".//ArticleTitle")
                title = title_elem.text if title_elem is not None else "无标题"
                
                # 获取摘要
                abstract_elem = article.find(".//Abstract/AbstractText")
                snippet = abstract_elem.text if abstract_elem is not None else ""
                if not snippet:
                    # 尝试获取其他摘要字段
                    abstract_elems = article.findall(".//AbstractText")
                    snippets = [elem.text for elem in abstract_elems if elem.text]
                    snippet = " ".join(snippets)
                
                # 限制摘要长度
                if snippet:
                    snippet = snippet[:500] + "..." if len(snippet) > 500 else snippet
                else:
                    snippet = "无摘要"
                
                # 获取作者
                authors = []
                author_list = article.findall(".//Author")
                for author in author_list[:5]:  # 最多取 5 个作者
                    last_name = author.find("LastName")
                    fore_name = author.find("ForeName")
                    if last_name is not None:
                        name = last_name.text
                        if fore_name is not None:
                            name = f"{fore_name.text} {name}"
                        authors.append(name)
                
                author_str = ", ".join(authors) if authors else "未知作者"
                if len(author_list) > 5:
                    author_str += " et al."
                
                # 获取发布日期
                pub_date = None
                date_elem = article.find(".//PubDate")
                if date_elem is not None:
                    year = date_elem.find("Year")
                    month = date_elem.find("Month")
                    day = date_elem.find("Day")
                    
                    date_parts = []
                    if year is not None:
                        date_parts.append(year.text)
                    if month is not None:
                        date_parts.append(month.text)
                    if day is not None:
                        date_parts.append(day.text)
                    
                    pub_date = "-".join(date_parts) if date_parts else None
                
                # 构建 URL
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                
                # 获取期刊信息
                journal_elem = article.find(".//Journal/Title")
                journal = journal_elem.text if journal_elem is not None else ""
                
                search_results.append(SearchResult(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=self.name,
                    category=self.category,
                    published_date=pub_date,
                    author=author_str,
                    metadata={
                        "pmid": pmid,
                        "journal": journal,
                        "authors": authors,
                    }
                ))
                
            except Exception:
                continue
        
        return search_results
    
    async def get_abstract(self, pmid: str) -> Optional[str]:
        """
        获取文献完整摘要
        
        Args:
            pmid: PubMed ID
            
        Returns:
            摘要文本或 None
        """
        try:
            params = {
                "db": "pubmed",
                "id": pmid,
                "retmode": "xml"
            }
            
            url = f"{self.EFETCH_URL}?{urllib.parse.urlencode(params)}"
            
            headers = {
                "User-Agent": "OllamaPilot/1.0",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = response.read().decode("utf-8")
            
            root = ET.fromstring(data)
            
            abstract_elems = root.findall(".//AbstractText")
            abstracts = [elem.text for elem in abstract_elems if elem.text]
            
            return "\n\n".join(abstracts) if abstracts else None
            
        except Exception:
            return None
