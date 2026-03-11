"""
增强搜索工具函数

为 SKILL.md 提供实际的工具实现
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from skills.enhanced_search.engines import (
    ArXivSearchEngine,
    WikipediaSearchEngine,
    BaiduBaikeSearchEngine,
    PubMedSearchEngine,
    GitHubSearchEngine,
    GiteeSearchEngine,
    SearXNGSearchEngine,
)
from skills.enhanced_search.aggregator import ResultsAggregator


# 全局聚合器实例
_aggregator = None


def _get_aggregator():
    """获取聚合器实例"""
    global _aggregator
    if _aggregator is None:
        _aggregator = ResultsAggregator()
    return _aggregator


def academic_search(query: str, num_results: int = 10) -> str:
    """
    学术搜索 - 搜索学术论文和文献（来源：PubMed）
    
    参数:
        query: 搜索查询
        num_results: 返回结果数量（默认10）
        
    返回:
        格式化搜索结果
    """
    import asyncio
    
    async def _search():
        results = []
        
        # 使用 PubMed
        engine = PubMedSearchEngine()
        if engine.is_available():
            try:
                engine_results = await engine.search(query, num_results)
                results.extend(engine_results)
            except Exception as e:
                print(f"PubMed 搜索失败: {e}")
        
        return results
    
    try:
        results = asyncio.run(_search())
        
        if not results:
            return "未找到相关学术文献。"
        
        # 聚合和排序
        aggregator = _get_aggregator()
        aggregated = aggregator.aggregate(results)
        ranked = aggregator.rank(aggregated, query, boost_categories=["academic"])
        
        return aggregator.format_results(ranked)
        
    except Exception as e:
        return f"学术搜索失败: {e}"


def code_search(query: str, num_results: int = 10, language: str = None) -> str:
    """
    代码搜索 - 搜索开源代码仓库（来源：GitHub）
    
    参数:
        query: 搜索查询
        num_results: 返回结果数量（默认10）
        language: 编程语言过滤（可选）
        
    返回:
        格式化搜索结果
    """
    import asyncio
    
    async def _search():
        results = []
        
        # 使用 GitHub
        engine = GitHubSearchEngine()
        if engine.is_available():
            try:
                if language:
                    engine_results = await engine.search_by_language(query, language, num_results)
                else:
                    engine_results = await engine.search(query, num_results)
                results.extend(engine_results)
            except Exception as e:
                print(f"GitHub 搜索失败: {e}")
        
        return results
    
    try:
        results = asyncio.run(_search())
        
        if not results:
            return "未找到相关代码仓库。"
        
        # 聚合和排序
        aggregator = _get_aggregator()
        aggregated = aggregator.aggregate(results)
        ranked = aggregator.rank(aggregated, query, boost_categories=["code"])
        
        return aggregator.format_results(ranked)
        
    except Exception as e:
        return f"代码搜索失败: {e}"


def encyclopedia_search(query: str, num_results: int = 10) -> str:
    """
    百科搜索 - 搜索百科知识（来源：百度百科）
    
    参数:
        query: 搜索查询
        num_results: 返回结果数量（默认10）
        
    返回:
        格式化搜索结果
    """
    import asyncio
    
    async def _search():
        results = []
        
        # 使用百度百科
        engine = BaiduBaikeSearchEngine()
        if engine.is_available():
            try:
                engine_results = await engine.search(query, num_results)
                results.extend(engine_results)
            except Exception as e:
                print(f"百度百科搜索失败: {e}")
        
        return results
    
    try:
        results = asyncio.run(_search())
        
        if not results:
            return "未找到相关百科条目。"
        
        # 聚合和排序
        aggregator = _get_aggregator()
        aggregated = aggregator.aggregate(results)
        ranked = aggregator.rank(aggregated, query, boost_categories=["encyclopedia"])
        
        return aggregator.format_results(ranked)
        
    except Exception as e:
        return f"百科搜索失败: {e}"


def multi_engine_search(query: str, num_results: int = 10, engines: list = None) -> str:
    """
    多引擎聚合搜索 - 同时使用多个搜索引擎
    
    参数:
        query: 搜索查询
        num_results: 返回结果数量（默认10）
        engines: 指定搜索引擎列表（可选，默认自动选择）
        
    返回:
        格式化搜索结果
    """
    import asyncio
    
    async def _search():
        results = []
        
        # 默认使用所有可用引擎
        if engines is None:
            engines_to_use = [
                ("searxng", SearXNGSearchEngine()),
                ("baidu_baike", BaiduBaikeSearchEngine()),
                ("pubmed", PubMedSearchEngine()),
            ]
        else:
            engine_map = {
                "searxng": SearXNGSearchEngine(),
                "baidu_baike": BaiduBaikeSearchEngine(),
                "pubmed": PubMedSearchEngine(),
                "github": GitHubSearchEngine(),
            }
            engines_to_use = [(name, engine_map.get(name)) for name in engines if name in engine_map]
        
        per_engine = max(3, num_results // len(engines_to_use)) if engines_to_use else num_results
        
        for name, engine in engines_to_use:
            if engine and engine.is_available():
                try:
                    engine_results = await engine.search(query, per_engine)
                    results.extend(engine_results)
                except Exception as e:
                    print(f"引擎 {name} 搜索失败: {e}")
        
        return results
    
    try:
        results = asyncio.run(_search())
        
        if not results:
            return "未找到相关结果。"
        
        # 聚合和排序
        aggregator = _get_aggregator()
        aggregated = aggregator.aggregate(results)
        ranked = aggregator.rank(aggregated, query)
        
        # 添加统计信息
        stats = aggregator.get_stats(ranked)
        stats_str = f"\n统计: 共 {stats['total']} 条结果"
        if stats.get('by_source'):
            sources_str = ", ".join(f"{k}: {v}" for k, v in stats['by_source'].items())
            stats_str += f" | 来源: {sources_str}"
        
        return stats_str + "\n\n" + aggregator.format_results(ranked)
        
    except Exception as e:
        return f"多引擎搜索失败: {e}"
