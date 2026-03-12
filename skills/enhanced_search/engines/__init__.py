"""
搜索引擎模块
"""

from .base import SearchEngineBase, SearchResult, SearchEngineFactory, register_engine
from .searxng import SearXNGSearchEngine
from .arxiv import ArXivSearchEngine
from .wikipedia import WikipediaSearchEngine, WikipediaENSearchEngine
from .baidu_baike import BaiduBaikeSearchEngine
from .pubmed import PubMedSearchEngine
from .github import GitHubSearchEngine, GiteeSearchEngine
from .duckduckgo import DuckDuckGoSearchEngine

__all__ = [
    "SearchEngineBase",
    "SearchResult",
    "SearchEngineFactory",
    "register_engine",
    "SearXNGSearchEngine",
    "ArXivSearchEngine",
    "WikipediaSearchEngine",
    "WikipediaENSearchEngine",
    "BaiduBaikeSearchEngine",
    "PubMedSearchEngine",
    "GitHubSearchEngine",
    "GiteeSearchEngine",
    "DuckDuckGoSearchEngine",
]
