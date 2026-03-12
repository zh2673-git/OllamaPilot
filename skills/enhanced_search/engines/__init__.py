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
from .serper import SerperSearchEngine
from .bing import BingSearchEngine
from .brave import BraveSearchEngine
from .tavily import TavilySearchEngine

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
    "SerperSearchEngine",
    "BingSearchEngine",
    "BraveSearchEngine",
    "TavilySearchEngine",
]
