"""
API配额管理器

管理各搜索引擎的API配额使用情况，实现智能降级
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


@dataclass
class QuotaInfo:
    """配额信息"""
    limit: int  # 配额上限
    window: int  # 时间窗口（秒）
    used: int  # 已使用
    reset_time: float  # 重置时间戳
    
    def is_expired(self) -> bool:
        """检查配额是否已过期（需要重置）"""
        return time.time() > self.reset_time
    
    def remaining(self) -> int:
        """剩余配额"""
        return max(0, self.limit - self.used)
    
    def usage_percent(self) -> float:
        """使用百分比"""
        if self.limit == 0:
            return 0.0
        return (self.used / self.limit) * 100


class APIQuotaManager:
    """
    API配额管理器
    
    管理各搜索引擎的API配额使用情况：
    - 跟踪API使用次数
    - 自动重置配额（按时间窗口）
    - 提供配额警告
    - 支持持久化存储
    
    示例:
        manager = APIQuotaManager()
        
        # 检查是否可以使用
        if manager.can_use("github"):
            result = github_search("query")
            manager.use("github")
        else:
            # 降级到其他引擎
            result = gitee_search("query")
    """
    
    # 默认配额配置
    DEFAULT_QUOTAS = {
        # 引擎名称: (限额, 时间窗口秒数)
        # 时间窗口: 3600=1小时, 86400=1天, 2592000=30天
        "github": (500, 3600),  # 500次/小时
        "serper": (2500, 2592000),  # 2500次/月
        "bing": (1000, 2592000),  # 1000次/月
        "brave": (2000, 2592000),  # 2000次/月
    }
    
    # 无限额度的引擎
    UNLIMITED_ENGINES = [
        "searxng",
        "pubmed",
        "baidu_baike",
        "duckduckgo",
        "gitee",
        "arxiv",
        "wikipedia",
    ]
    
    def __init__(self, persist_file: Optional[str] = None):
        """
        初始化配额管理器
        
        Args:
            persist_file: 配额持久化文件路径，默认保存在数据目录
        """
        self.quotas: Dict[str, QuotaInfo] = {}
        
        # 设置持久化文件路径
        if persist_file:
            self.persist_file = Path(persist_file)
        else:
            data_dir = Path("./data")
            data_dir.mkdir(exist_ok=True)
            self.persist_file = data_dir / "api_quotas.json"
        
        # 加载已保存的配额信息
        self._load_quotas()
    
    def _load_quotas(self):
        """从文件加载配额信息"""
        if not self.persist_file.exists():
            return
        
        try:
            with open(self.persist_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for engine_name, quota_data in data.items():
                # 检查是否过期
                reset_time = quota_data.get('reset_time', 0)
                if time.time() > reset_time:
                    # 已过期，重置配额
                    self._init_quota(engine_name)
                else:
                    # 恢复配额信息
                    self.quotas[engine_name] = QuotaInfo(**quota_data)
                    
        except Exception as e:
            print(f"⚠️ 加载配额信息失败: {e}")
    
    def _save_quotas(self):
        """保存配额信息到文件"""
        try:
            data = {
                name: asdict(quota)
                for name, quota in self.quotas.items()
            }
            
            with open(self.persist_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"⚠️ 保存配额信息失败: {e}")
    
    def _init_quota(self, engine_name: str):
        """初始化引擎配额"""
        if engine_name in self.UNLIMITED_ENGINES:
            # 无限额度
            self.quotas[engine_name] = QuotaInfo(
                limit=0,
                window=0,
                used=0,
                reset_time=float('inf')
            )
        elif engine_name in self.DEFAULT_QUOTAS:
            # 有配额的引擎
            limit, window = self.DEFAULT_QUOTAS[engine_name]
            self.quotas[engine_name] = QuotaInfo(
                limit=limit,
                window=window,
                used=0,
                reset_time=time.time() + window
            )
        else:
            # 未知引擎，假设无限额度
            self.quotas[engine_name] = QuotaInfo(
                limit=0,
                window=0,
                used=0,
                reset_time=float('inf')
            )
    
    def can_use(self, engine_name: str) -> bool:
        """
        检查引擎是否还有配额
        
        Args:
            engine_name: 引擎名称
            
        Returns:
            bool: 是否可以使用
        """
        # 确保配额信息已初始化
        if engine_name not in self.quotas:
            self._init_quota(engine_name)
        
        quota = self.quotas[engine_name]
        
        # 检查是否过期
        if quota.is_expired():
            self._init_quota(engine_name)
            quota = self.quotas[engine_name]
        
        # 无限额度
        if quota.limit == 0:
            return True
        
        # 检查剩余配额
        return quota.remaining() > 0
    
    def use(self, engine_name: str) -> bool:
        """
        记录引擎使用
        
        Args:
            engine_name: 引擎名称
            
        Returns:
            bool: 是否记录成功
        """
        if not self.can_use(engine_name):
            return False
        
        quota = self.quotas[engine_name]
        
        # 无限额度不需要记录
        if quota.limit == 0:
            return True
        
        # 增加使用计数
        quota.used += 1
        
        # 保存到文件
        self._save_quotas()
        
        return True
    
    def get_quota(self, engine_name: str) -> Optional[QuotaInfo]:
        """
        获取引擎配额信息
        
        Args:
            engine_name: 引擎名称
            
        Returns:
            QuotaInfo: 配额信息，如果不存在返回None
        """
        if engine_name not in self.quotas:
            self._init_quota(engine_name)
        
        return self.quotas.get(engine_name)
    
    def get_usage_report(self) -> Dict[str, Any]:
        """
        获取所有引擎的配额使用报告
        
        Returns:
            Dict: 使用报告
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "engines": {}
        }
        
        for engine_name in set(list(self.DEFAULT_QUOTAS.keys()) + self.UNLIMITED_ENGINES):
            quota = self.get_quota(engine_name)
            if quota:
                report["engines"][engine_name] = {
                    "limit": quota.limit if quota.limit > 0 else "unlimited",
                    "used": quota.used,
                    "remaining": quota.remaining() if quota.limit > 0 else "unlimited",
                    "usage_percent": f"{quota.usage_percent():.1f}%" if quota.limit > 0 else "N/A",
                    "reset_time": datetime.fromtimestamp(quota.reset_time).isoformat() if quota.reset_time < float('inf') else "never",
                }
        
        return report
    
    def reset_quota(self, engine_name: str):
        """
        手动重置引擎配额
        
        Args:
            engine_name: 引擎名称
        """
        self._init_quota(engine_name)
        self._save_quotas()
        print(f"✅ {engine_name} 配额已重置")
    
    def reset_all_quotas(self):
        """重置所有引擎配额"""
        for engine_name in list(self.quotas.keys()):
            self._init_quota(engine_name)
        self._save_quotas()
        print("✅ 所有配额已重置")


# 全局配额管理器实例
_quota_manager: Optional[APIQuotaManager] = None


def get_quota_manager() -> APIQuotaManager:
    """获取全局配额管理器实例"""
    global _quota_manager
    if _quota_manager is None:
        _quota_manager = APIQuotaManager()
    return _quota_manager


if __name__ == "__main__":
    # 测试
    manager = APIQuotaManager()
    
    print("=== API配额管理器测试 ===\n")
    
    # 测试无限额度引擎
    print("1. 测试无限额度引擎 (SearXNG):")
    print(f"   可用: {manager.can_use('searxng')}")
    manager.use('searxng')
    print(f"   使用后: {manager.can_use('searxng')}")
    
    # 测试有限额度引擎
    print("\n2. 测试有限额度引擎 (GitHub):")
    quota = manager.get_quota('github')
    print(f"   限额: {quota.limit}次/小时")
    print(f"   已用: {quota.used}")
    print(f"   剩余: {quota.remaining()}")
    print(f"   可用: {manager.can_use('github')}")
    
    # 模拟使用
    for i in range(3):
        if manager.can_use('github'):
            manager.use('github')
            print(f"   使用 {i+1} 次后剩余: {manager.get_quota('github').remaining()}")
    
    # 显示报告
    print("\n3. 配额使用报告:")
    report = manager.get_usage_report()
    for engine, info in report['engines'].items():
        print(f"   {engine}: {info}")
