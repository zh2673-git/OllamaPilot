"""
EnhancedSearchSkill - 增强搜索模块

提供多引擎专业搜索能力
"""

from .engines.base import SearchEngineBase, SearchResult, SearchEngineFactory
from .engines import (
    SearXNGSearchEngine,
    DuckDuckGoSearchEngine,
    SerperSearchEngine,
    BingSearchEngine,
    BraveSearchEngine,
    TavilySearchEngine,
    PubMedSearchEngine,
    ArXivSearchEngine,
    GitHubSearchEngine,
    GiteeSearchEngine,
    BaiduBaikeSearchEngine,
    WikipediaSearchEngine,
)
from .aggregator import ResultsAggregator
from .engine_router import SearchEngineRouter
from .quota_manager import APIQuotaManager, get_quota_manager

__all__ = [
    # 基础类
    "SearchEngineBase",
    "SearchResult",
    "SearchEngineFactory",
    # 引擎
    "SearXNGSearchEngine",
    "DuckDuckGoSearchEngine",
    "SerperSearchEngine",
    "BingSearchEngine",
    "BraveSearchEngine",
    "TavilySearchEngine",
    "PubMedSearchEngine",
    "ArXivSearchEngine",
    "GitHubSearchEngine",
    "GiteeSearchEngine",
    "BaiduBaikeSearchEngine",
    "WikipediaSearchEngine",
    # 工具
    "ResultsAggregator",
    "SearchEngineRouter",
    "APIQuotaManager",
    "get_quota_manager",
]
