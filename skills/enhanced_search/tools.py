"""
增强搜索工具函数

为 SKILL.md 提供实际的工具实现
支持智能引擎选择和自动降级
"""

import sys
import asyncio
from pathlib import Path
from typing import List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from skills.enhanced_search.engines import (
    SearchResult,
    SearXNGSearchEngine,
)
from skills.enhanced_search.aggregator import ResultsAggregator
from skills.enhanced_search.engine_router import SearchEngineRouter, smart_search
from skills.enhanced_search.quota_manager import get_quota_manager


# 全局实例
_aggregator = None
_router = None


def _get_aggregator():
    """获取聚合器实例"""
    global _aggregator
    if _aggregator is None:
        _aggregator = ResultsAggregator()
    return _aggregator


def _get_router():
    """获取路由器实例"""
    global _router
    if _router is None:
        _router = SearchEngineRouter()
    return _router


def _format_results(results: List[SearchResult], query: str = "") -> str:
    """格式化搜索结果"""
    if not results:
        return "未找到相关结果。"
    
    # 聚合和排序
    aggregator = _get_aggregator()
    aggregated = aggregator.aggregate(results)
    ranked = aggregator.rank(aggregated, query)
    
    # 添加统计信息
    stats = aggregator.get_stats(ranked)
    stats_str = f"\n📊 统计: 共 {stats['total']} 条结果"
    if stats.get('by_source'):
        sources_str = ", ".join(f"{k}: {v}" for k, v in stats['by_source'].items())
        stats_str += f" | 来源: {sources_str}"
    
    return stats_str + "\n\n" + aggregator.format_results(ranked)


def academic_search(query: str, num_results: int = 10) -> str:
    """
    学术搜索 - 搜索学术论文和文献
    
    使用策略:
    1. 优先使用 PubMed（医学文献，无限额度）
    2. 其次使用 arXiv（学术论文，无限额度）
    3. 自动降级处理
    
    参数:
        query: 搜索查询
        num_results: 返回结果数量（默认10）
        
    返回:
        格式化搜索结果
    """
    try:
        # 使用路由器进行学术类别搜索
        results = asyncio.run(smart_search(query, category="academic", num_results=num_results))
        return _format_results(results, query)
        
    except Exception as e:
        return f"学术搜索失败: {e}"


def code_search(query: str, num_results: int = 10, language: str = None) -> str:
    """
    代码搜索 - 搜索开源代码仓库
    
    使用策略:
    1. 优先使用 GitHub（500次/小时，需配置Token）
    2. GitHub配额用完时自动降级到 Gitee（无限额度）
    
    参数:
        query: 搜索查询
        num_results: 返回结果数量（默认10）
        language: 编程语言过滤（可选）
        
    返回:
        格式化搜索结果
    """
    try:
        router = _get_router()
        
        # 如果有语言过滤，添加语言到查询
        search_query = f"{query} language:{language}" if language else query
        
        # 使用路由器进行代码类别搜索
        results = asyncio.run(router.search(search_query, category="code", num_results=num_results))
        return _format_results(results, query)
        
    except Exception as e:
        return f"代码搜索失败: {e}"


def encyclopedia_search(query: str, num_results: int = 10) -> str:
    """
    百科搜索 - 搜索百科知识
    
    使用策略:
    1. 优先使用百度百科（中文优化，无限额度）
    2. 其次使用 Wikipedia（无限额度）
    
    参数:
        query: 搜索查询
        num_results: 返回结果数量（默认10）
        
    返回:
        格式化搜索结果
    """
    try:
        # 使用路由器进行百科类别搜索
        results = asyncio.run(smart_search(query, category="encyclopedia", num_results=num_results))
        return _format_results(results, query)
        
    except Exception as e:
        return f"百科搜索失败: {e}"


def multi_engine_search(query: str, num_results: int = 10, engines: list = None) -> str:
    """
    多引擎聚合搜索 - 同时使用多个搜索引擎
    
    使用策略:
    1. 优先使用 SearXNG（本地聚合，无限额度）
    2. SearXNG不可用时降级到 DuckDuckGo（免费，无需API Key）
    3. 智能配额管理，自动切换引擎
    
    参数:
        query: 搜索查询
        num_results: 返回结果数量（默认10）
        engines: 指定搜索引擎列表（可选，默认自动选择）
        
    返回:
        格式化搜索结果
    """
    try:
        router = _get_router()
        
        # 如果指定了引擎列表，手动调用
        if engines:
            results = []
            for engine_name in engines:
                engine_results = asyncio.run(router.search(query, category="general", num_results=num_results))
                results.extend(engine_results)
        else:
            # 使用智能路由
            results = asyncio.run(smart_search(query, category="general", num_results=num_results))
        
        return _format_results(results, query)
        
    except Exception as e:
        return f"多引擎搜索失败: {e}"


def get_search_quota_report() -> str:
    """
    获取搜索API配额使用报告
    
    返回:
        配额使用报告文本
    """
    try:
        manager = get_quota_manager()
        report = manager.get_usage_report()
        
        output = ["📊 API配额使用报告", "=" * 50, ""]
        
        for engine_name, info in report['engines'].items():
            limit = info['limit']
            used = info['used']
            remaining = info['remaining']
            percent = info['usage_percent']
            
            if limit == 'unlimited':
                output.append(f"✅ {engine_name}: 无限额度")
            else:
                status_icon = "🟢" if float(percent.rstrip('%')) < 80 else "🟡" if float(percent.rstrip('%')) < 100 else "🔴"
                output.append(f"{status_icon} {engine_name}:")
                output.append(f"   限额: {limit} | 已用: {used} | 剩余: {remaining} | 使用率: {percent}")
            
            output.append("")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"获取配额报告失败: {e}"


def check_search_engine_availability() -> str:
    """
    检查各搜索引擎可用性
    
    返回:
        可用性报告文本
    """
    try:
        router = _get_router()
        available = router.get_available_engines()
        
        output = ["🔍 搜索引擎可用性检查", "=" * 50, ""]
        
        category_names = {
            "general": "通用搜索",
            "academic": "学术搜索",
            "code": "代码搜索",
            "encyclopedia": "百科搜索",
        }
        
        for category, engines in available.items():
            name = category_names.get(category, category)
            if engines:
                output.append(f"✅ {name}: {', '.join(engines)}")
            else:
                output.append(f"❌ {name}: 无可用引擎")
        
        output.append("")
        output.append("💡 提示: 使用 .env 文件配置API Key可解锁更多引擎")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"检查可用性失败: {e}"


if __name__ == "__main__":
    # 测试
    print("=== 增强搜索工具测试 ===\n")
    
    # 测试配额报告
    print("1. 配额报告:")
    print(get_search_quota_report())
    print()
    
    # 测试可用性检查
    print("2. 可用性检查:")
    print(check_search_engine_availability())
    print()
    
    # 测试搜索
    print("3. 多引擎搜索测试:")
    result = multi_engine_search("Python 教程", num_results=5)
    print(result[:500] + "..." if len(result) > 500 else result)
