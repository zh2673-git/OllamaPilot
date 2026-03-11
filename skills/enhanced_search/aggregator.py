"""
搜索结果聚合器

参考 Local Deep Research 的结果聚合逻辑
"""

from typing import List, Dict, Set, Tuple
from collections import defaultdict
import re

from .engines.base import SearchResult


class ResultsAggregator:
    """
    搜索结果聚合器
    
    负责:
    1. 多引擎结果聚合
    2. 去重
    3. 相关性排序
    4. 格式化输出
    """
    
    def __init__(self):
        """初始化聚合器"""
        self.seen_urls: Set[str] = set()
        self.seen_titles: Set[str] = set()
    
    def aggregate(
        self,
        results: List[SearchResult],
        deduplicate: bool = True,
        min_score: float = 0.0
    ) -> List[SearchResult]:
        """
        聚合搜索结果
        
        Args:
            results: 原始结果列表
            deduplicate: 是否去重
            min_score: 最小相关性分数
            
        Returns:
            聚合后的结果列表
        """
        if not deduplicate:
            return [r for r in results if (r.score or 0.5) >= min_score]
        
        aggregated = []
        self.seen_urls.clear()
        self.seen_titles.clear()
        
        for result in results:
            # 检查分数
            if (result.score or 0.5) < min_score:
                continue
            
            # 检查 URL 重复
            normalized_url = self._normalize_url(result.url)
            if normalized_url in self.seen_urls:
                continue
            
            # 检查标题重复
            normalized_title = self._normalize_title(result.title)
            if normalized_title in self.seen_titles:
                continue
            
            self.seen_urls.add(normalized_url)
            self.seen_titles.add(normalized_title)
            aggregated.append(result)
        
        return aggregated
    
    def rank(
        self,
        results: List[SearchResult],
        query: str,
        boost_categories: List[str] = None
    ) -> List[SearchResult]:
        """
        按相关性排序结果
        
        Args:
            results: 结果列表
            query: 搜索查询
            boost_categories: 优先类别列表
            
        Returns:
            排序后的结果列表
        """
        if not results:
            return []
        
        # 计算每个结果的相关性分数
        scored_results: List[Tuple[float, SearchResult]] = []
        
        for result in results:
            score = self._calculate_relevance_score(result, query, boost_categories)
            scored_results.append((score, result))
        
        # 按分数降序排序
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        # 更新结果的分数
        for score, result in scored_results:
            result.score = score
        
        return [result for _, result in scored_results]
    
    def merge_by_source(
        self,
        results_by_engine: Dict[str, List[SearchResult]],
        max_per_engine: int = 5
    ) -> List[SearchResult]:
        """
        按来源合并结果（轮询方式）
        
        确保每个引擎的结果都有展示机会
        
        Args:
            results_by_engine: 按引擎分组的结果
            max_per_engine: 每个引擎最大结果数
            
        Returns:
            合并后的结果列表
        """
        merged = []
        engine_names = list(results_by_engine.keys())
        engine_indices = {name: 0 for name in engine_names}
        
        # 轮询各引擎
        while True:
            added = False
            for engine_name in engine_names:
                idx = engine_indices[engine_name]
                results = results_by_engine[engine_name]
                
                if idx < len(results) and idx < max_per_engine:
                    result = results[idx]
                    # 检查是否重复
                    normalized_url = self._normalize_url(result.url)
                    if normalized_url not in self.seen_urls:
                        self.seen_urls.add(normalized_url)
                        merged.append(result)
                        added = True
                    
                    engine_indices[engine_name] = idx + 1
            
            if not added:
                break
        
        return merged
    
    def format_results(
        self,
        results: List[SearchResult],
        include_metadata: bool = False
    ) -> str:
        """
        格式化结果为文本
        
        Args:
            results: 结果列表
            include_metadata: 是否包含元数据
            
        Returns:
            格式化后的文本
        """
        if not results:
            return "未找到相关结果。"
        
        lines = [f"找到 {len(results)} 个相关结果:\n"]
        
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. **{result.title}** [{result.source}]")
            lines.append(f"   {result.url}")
            
            if result.author:
                lines.append(f"   作者: {result.author}")
            
            if result.published_date:
                lines.append(f"   日期: {result.published_date}")
            
            # 摘要
            snippet = result.snippet.replace('\n', ' ')
            lines.append(f"   {snippet}")
            
            if include_metadata and result.metadata:
                meta_str = " | ".join(
                    f"{k}: {v}"
                    for k, v in result.metadata.items()
                    if v and k not in ['authors']
                )
                if meta_str:
                    lines.append(f"   元数据: {meta_str}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _calculate_relevance_score(
        self,
        result: SearchResult,
        query: str,
        boost_categories: List[str] = None
    ) -> float:
        """
        计算相关性分数
        
        Args:
            result: 搜索结果
            query: 搜索查询
            boost_categories: 优先类别
            
        Returns:
            相关性分数 (0-1)
        """
        score = result.score or 0.5
        query_lower = query.lower()
        
        # 1. 标题匹配度
        title_lower = result.title.lower()
        if query_lower in title_lower:
            score += 0.3
            # 完全匹配加分更多
            if title_lower == query_lower:
                score += 0.2
        
        # 2. 摘要匹配度
        snippet_lower = result.snippet.lower()
        if query_lower in snippet_lower:
            score += 0.1
        
        # 3. 类别优先
        if boost_categories and result.category in boost_categories:
            score += 0.15
        
        # 4. 来源可信度
        trusted_sources = {
            "arxiv": 0.1,
            "pubmed": 0.1,
            "wikipedia": 0.05,
            "baidu_baike": 0.05,
            "github": 0.05,
        }
        score += trusted_sources.get(result.source, 0)
        
        # 5. 有作者加分（学术性）
        if result.author:
            score += 0.05
        
        # 6. 有日期加分（时效性）
        if result.published_date:
            score += 0.03
        
        return min(1.0, score)
    
    def _normalize_url(self, url: str) -> str:
        """
        标准化 URL 用于去重
        
        Args:
            url: 原始 URL
            
        Returns:
            标准化后的 URL
        """
        # 移除协议
        url = re.sub(r'^https?://', '', url)
        # 移除末尾斜杠
        url = url.rstrip('/')
        # 移除 www 前缀
        url = re.sub(r'^www\.', '', url)
        # 转为小写
        return url.lower()
    
    def _normalize_title(self, title: str) -> str:
        """
        标准化标题用于去重
        
        Args:
            title: 原始标题
            
        Returns:
            标准化后的标题
        """
        # 转为小写
        title = title.lower()
        # 移除多余空格
        title = ' '.join(title.split())
        # 移除常见后缀
        title = re.sub(r'\s*[-|]\s*(wikipedia|百度百科|github|arxiv).*$', '', title)
        return title
    
    def get_stats(self, results: List[SearchResult]) -> Dict:
        """
        获取结果统计信息
        
        Args:
            results: 结果列表
            
        Returns:
            统计信息字典
        """
        stats = {
            "total": len(results),
            "by_source": defaultdict(int),
            "by_category": defaultdict(int),
        }
        
        for result in results:
            stats["by_source"][result.source] += 1
            stats["by_category"][result.category] += 1
        
        # 转换为普通 dict
        stats["by_source"] = dict(stats["by_source"])
        stats["by_category"] = dict(stats["by_category"])
        
        return stats
