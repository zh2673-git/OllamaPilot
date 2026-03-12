"""
搜索引擎路由器

实现智能引擎选择和自动降级策略
"""

import asyncio
from typing import List, Dict, Optional
from skills.enhanced_search.engines import (
    SearchEngineFactory, 
    SearchResult,
    SearXNGSearchEngine,
    DuckDuckGoSearchEngine,
)
from skills.enhanced_search.quota_manager import get_quota_manager


class SearchEngineRouter:
    """
    搜索引擎路由器
    
    根据查询类别智能选择引擎，实现自动降级：
    1. 优先使用指定类别的最佳引擎
    2. 配额用完或失败时自动降级
    3. 所有引擎失败时使用兜底方案
    
    使用示例:
        router = SearchEngineRouter()
        results = await router.search("Python 教程", category="general")
    """
    
    # 引擎优先级配置
    ENGINE_PRIORITY = {
        "academic": ["pubmed", "arxiv"],  # 学术领域
        "code": ["github", "gitee"],  # 代码领域
        "encyclopedia": ["baidu_baike", "wikipedia"],  # 百科领域
        "general": [
            "tavily",       # 专为AI设计，质量最佳（需配置API Key，1000次/月）
            "searxng",      # 本地聚合搜索（无限额度）
            "serper",       # Google搜索（需配置API Key，2500次/月）
            "bing",         # 必应搜索（需配置API Key，1000次/月）
            "brave",        # Brave搜索（需配置API Key，2000次/月）
            "duckduckgo",   # 免费备用（无需配置）
        ],  # 通用领域
        "research": [
            "tavily",       # 专为AI研究设计（需配置API Key，1000次/月）
            "searxng",      # 本地聚合搜索
            "serper",       # Google搜索
            "duckduckgo",   # 免费备用
        ],  # 深度研究领域
    }
    
    def __init__(self):
        self.quota_manager = get_quota_manager()
        self._searxng_engine: Optional[SearXNGSearchEngine] = None
        self._duckduckgo_engine: Optional[DuckDuckGoSearchEngine] = None
    
    def _get_searxng(self) -> SearXNGSearchEngine:
        """获取或创建 SearXNG 引擎"""
        if self._searxng_engine is None:
            self._searxng_engine = SearXNGSearchEngine()
        return self._searxng_engine
    
    def _get_duckduckgo(self) -> DuckDuckGoSearchEngine:
        """获取或创建 DuckDuckGo 引擎"""
        if self._duckduckgo_engine is None:
            self._duckduckgo_engine = DuckDuckGoSearchEngine()
        return self._duckduckgo_engine
    
    async def search(
        self, 
        query: str, 
        category: str = "general",
        num_results: int = 10
    ) -> List[SearchResult]:
        """
        智能搜索 - 自动选择引擎并降级
        
        Args:
            query: 搜索查询
            category: 搜索类别 (academic/code/encyclopedia/general)
            num_results: 返回结果数量
            
        Returns:
            List[SearchResult]: 搜索结果列表
        """
        # 获取该类别下的引擎列表
        engine_list = self.ENGINE_PRIORITY.get(category, self.ENGINE_PRIORITY["general"])
        
        all_results = []
        errors = []
        
        for engine_name in engine_list:
            # 检查配额
            if not self.quota_manager.can_use(engine_name):
                print(f"⚠️ {engine_name} 配额已用完，尝试下一个引擎")
                continue
            
            # 获取引擎实例
            engine = self._get_engine(engine_name)
            if not engine:
                continue
            
            # 检查引擎可用性
            if not engine.is_available():
                print(f"⚠️ {engine_name} 不可用，尝试下一个引擎")
                continue
            
            try:
                # 执行搜索
                print(f"🔍 使用 {engine_name} 搜索...")
                results = await engine.search(query, num_results)
                
                if results:
                    # 记录使用
                    self.quota_manager.use(engine_name)
                    all_results.extend(results)
                    print(f"✅ {engine_name} 返回 {len(results)} 条结果")
                    
                    # 如果已有足够结果，可以提前返回
                    if len(all_results) >= num_results:
                        break
                else:
                    print(f"⚠️ {engine_name} 未返回结果")
                    
            except Exception as e:
                error_msg = f"❌ {engine_name} 搜索失败: {e}"
                print(error_msg)
                errors.append(error_msg)
                continue
        
        # 如果所有引擎都失败，使用兜底方案
        if not all_results:
            print("🔄 所有引擎失败，使用兜底方案...")
            all_results = await self._fallback_search(query, num_results)
        
        return all_results[:num_results]
    
    def _get_engine(self, engine_name: str):
        """获取引擎实例"""
        # 特殊处理需要单独实例化的引擎
        if engine_name == "searxng":
            return self._get_searxng()
        elif engine_name == "duckduckgo":
            return self._get_duckduckgo()
        else:
            # 其他引擎通过工厂创建
            return SearchEngineFactory.create(engine_name)
    
    async def _fallback_search(self, query: str, num_results: int) -> List[SearchResult]:
        """
        兜底搜索方案
        
        当所有主要引擎失败时使用
        """
        results = []
        
        # 尝试 DuckDuckGo 作为最终兜底
        try:
            ddg = self._get_duckduckgo()
            if ddg.is_available():
                print("🔍 使用 DuckDuckGo 兜底搜索...")
                results = await ddg.search(query, num_results)
                if results:
                    print(f"✅ DuckDuckGo 兜底成功，返回 {len(results)} 条结果")
                    return results
        except Exception as e:
            print(f"❌ DuckDuckGo 兜底也失败: {e}")
        
        return results
    
    async def multi_category_search(
        self,
        query: str,
        categories: List[str],
        num_results: int = 10
    ) -> Dict[str, List[SearchResult]]:
        """
        多类别搜索
        
        同时在多个类别中搜索，适用于深度研究
        
        Args:
            query: 搜索查询
            categories: 类别列表
            num_results: 每个类别返回结果数
            
        Returns:
            Dict[str, List[SearchResult]]: 按类别分组的结果
        """
        results = {}
        
        for category in categories:
            category_results = await self.search(query, category, num_results)
            results[category] = category_results
        
        return results
    
    def get_available_engines(self) -> Dict[str, List[str]]:
        """
        获取当前可用的引擎列表
        
        Returns:
            Dict[str, List[str]]: 按类别分组的可用引擎
        """
        available = {}
        
        for category, engines in self.ENGINE_PRIORITY.items():
            available[category] = []
            for engine_name in engines:
                # 检查配额
                if not self.quota_manager.can_use(engine_name):
                    continue
                
                # 检查引擎可用性
                engine = self._get_engine(engine_name)
                if engine and engine.is_available():
                    available[category].append(engine_name)
        
        return available
    
    def get_quota_report(self) -> Dict:
        """获取配额使用报告"""
        return self.quota_manager.get_usage_report()


# 便捷函数
async def smart_search(
    query: str,
    category: str = "general",
    num_results: int = 10
) -> List[SearchResult]:
    """
    智能搜索便捷函数
    
    Args:
        query: 搜索查询
        category: 搜索类别
        num_results: 返回结果数量
        
    Returns:
        List[SearchResult]: 搜索结果
    """
    router = SearchEngineRouter()
    return await router.search(query, category, num_results)


if __name__ == "__main__":
    # 测试
    async def test():
        router = SearchEngineRouter()
        
        print("=== 搜索引擎路由器测试 ===\n")
        
        # 测试通用搜索
        print("1. 测试通用搜索:")
        results = await router.search("Python 教程", category="general", num_results=5)
        print(f"   找到 {len(results)} 条结果\n")
        
        # 测试学术搜索
        print("2. 测试学术搜索:")
        results = await router.search("machine learning", category="academic", num_results=3)
        print(f"   找到 {len(results)} 条结果\n")
        
        # 显示可用引擎
        print("3. 可用引擎:")
        available = router.get_available_engines()
        for category, engines in available.items():
            print(f"   {category}: {engines}")
        
        # 显示配额报告
        print("\n4. 配额报告:")
        report = router.get_quota_report()
        for engine, info in report['engines'].items():
            print(f"   {engine}: {info.get('usage_percent', 'N/A')}")
    
    asyncio.run(test())
