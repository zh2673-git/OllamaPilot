"""
GitHub 代码搜索引擎

使用 GitHub Search API 搜索代码仓库，无需 API Key（有速率限制）。
"""

import urllib.request
import urllib.parse
import json
from typing import List
from datetime import datetime

from .base import SearchEngineBase, SearchResult, register_engine


@register_engine
class GitHubSearchEngine(SearchEngineBase):
    """
    GitHub 代码搜索引擎
    
    特点:
    - 无需 API Key（有速率限制）
    - 搜索开源代码仓库
    - 国内可用
    - 丰富的仓库元数据
    
    API 文档: https://docs.github.com/en/rest/search
    
    注意: 未认证请求限制为每分钟 10 次
    """
    
    name = "github"
    description = "GitHub 开源代码搜索"
    category = "code"
    
    # GitHub API 基础 URL
    API_URL = "https://api.github.com/search/repositories"
    
    def __init__(self):
        super().__init__()
    
    def is_available(self) -> bool:
        """
        检查 GitHub API 是否可用
        
        Returns:
            bool: 是否可用
        """
        try:
            headers = {
                "User-Agent": "OllamaPilot/1.0",
                "Accept": "application/vnd.github.v3+json"
            }
            
            req = urllib.request.Request(
                "https://api.github.com",
                headers=headers
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        搜索 GitHub 仓库
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量 (最大 100)
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        try:
            # 限制结果数量
            num_results = max(1, min(100, num_results))
            
            # 构建查询参数
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": num_results
            }
            
            url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
            
            headers = {
                "User-Agent": "OllamaPilot/1.0 (https://github.com/your-repo)",
                "Accept": "application/vnd.github.v3+json"
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            
            search_results = []
            
            items = data.get("items", [])
            
            for item in items:
                name = item.get("full_name", "")
                url = item.get("html_url", "")
                description = item.get("description", "") or "无描述"
                
                # 构建详细摘要
                stars = item.get("stargazers_count", 0)
                language = item.get("language", "未知")
                updated_at = item.get("updated_at", "")
                
                snippet = f"{description}\n"
                snippet += f"⭐ {stars} stars | 语言: {language}"
                
                # 解析更新日期
                published_date = None
                if updated_at:
                    try:
                        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        published_date = dt.strftime("%Y-%m-%d")
                    except:
                        pass
                
                search_results.append(SearchResult(
                    title=name,
                    url=url,
                    snippet=snippet,
                    source=self.name,
                    category=self.category,
                    published_date=published_date,
                    metadata={
                        "stars": stars,
                        "language": language,
                        "forks": item.get("forks_count", 0),
                        "open_issues": item.get("open_issues_count", 0),
                        "topics": item.get("topics", []),
                        "license": item.get("license", {}).get("name") if item.get("license") else None,
                    }
                ))
            
            return search_results
            
        except urllib.error.HTTPError as e:
            if e.code == 403:
                raise RuntimeError(
                    "GitHub API 速率限制 exceeded. "
                    "未认证请求限制为每分钟 10 次。请稍后再试。"
                )
            raise RuntimeError(f"GitHub 搜索失败: {e}")
        except Exception as e:
            raise RuntimeError(f"GitHub 搜索失败: {e}")
    
    async def search_by_language(self, query: str, language: str, num_results: int = 10) -> List[SearchResult]:
        """
        按编程语言搜索 GitHub 仓库
        
        Args:
            query: 搜索查询
            language: 编程语言 (如 python, javascript, go)
            num_results: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        # 构建语言过滤查询
        filtered_query = f"{query} language:{language}"
        return await self.search(filtered_query, num_results)


@register_engine
class GiteeSearchEngine(SearchEngineBase):
    """
    Gitee 代码搜索引擎
    
    特点:
    - 国内代码托管平台，访问速度快
    - 无需 API Key（有速率限制）
    - 中文支持好
    
    API 文档: https://gitee.com/api/v5/swagger
    """
    
    name = "gitee"
    description = "Gitee 开源代码搜索"
    category = "code"
    
    # Gitee API 基础 URL
    API_URL = "https://gitee.com/api/v5/search/repositories"
    
    def __init__(self):
        super().__init__()
    
    def is_available(self) -> bool:
        """
        检查 Gitee API 是否可用
        
        Returns:
            bool: 是否可用
        """
        try:
            headers = {
                "User-Agent": "OllamaPilot/1.0",
            }
            
            req = urllib.request.Request(
                "https://gitee.com/api/v5",
                headers=headers
            )
            
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """
        搜索 Gitee 仓库
        
        Args:
            query: 搜索查询
            num_results: 返回结果数量 (最大 100)
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        try:
            # 限制结果数量
            num_results = max(1, min(100, num_results))
            
            # 构建查询参数
            params = {
                "q": query,
                "sort": "stars_count",
                "order": "desc",
                "per_page": num_results,
                "page": 1
            }
            
            url = f"{self.API_URL}?{urllib.parse.urlencode(params)}"
            
            headers = {
                "User-Agent": "OllamaPilot/1.0 (https://github.com/your-repo)",
            }
            
            req = urllib.request.Request(url, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
            
            search_results = []
            
            # Gitee 返回的是列表
            items = data if isinstance(data, list) else []
            
            for item in items:
                name = item.get("full_name", "")
                url = item.get("html_url", "")
                description = item.get("description", "") or "无描述"
                
                # 构建详细摘要
                stars = item.get("stargazers_count", 0)
                language = item.get("language", "未知")
                updated_at = item.get("updated_at", "")
                
                snippet = f"{description}\n"
                snippet += f"⭐ {stars} stars | 语言: {language}"
                
                # 解析更新日期
                published_date = None
                if updated_at:
                    try:
                        dt = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                        published_date = dt.strftime("%Y-%m-%d")
                    except:
                        pass
                
                search_results.append(SearchResult(
                    title=name,
                    url=url,
                    snippet=snippet,
                    source=self.name,
                    category=self.category,
                    published_date=published_date,
                    metadata={
                        "stars": stars,
                        "language": language,
                        "forks": item.get("forks_count", 0),
                        "open_issues": item.get("open_issues_count", 0),
                        "license": item.get("license", {}).get("name") if item.get("license") else None,
                        "is_gitee": True,
                    }
                ))
            
            return search_results
            
        except Exception as e:
            raise RuntimeError(f"Gitee 搜索失败: {e}")
