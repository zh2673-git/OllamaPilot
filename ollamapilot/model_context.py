"""
模型上下文窗口管理模块

提供自动检测和管理 Ollama 模型上下文窗口大小的功能。
支持从 Ollama API 获取模型信息，并根据模型能力动态调整配置。
"""

import requests
import json
import os
from typing import Optional, Dict, List, Union
from functools import lru_cache
from pathlib import Path


class ModelContextManager:
    """
    模型上下文窗口管理器
    
    功能：
    1. 自动扫描 Ollama 中已安装的模型
    2. 自动检测每个模型的 context_length
    3. 根据模型能力计算工具输出截断阈值
    4. 缓存检测结果到文件，避免重复查询
    
    使用示例：
        >>> manager = ModelContextManager()
        >>> # 扫描所有已安装模型
        >>> manager.scan_installed_models()
        >>> ctx_length = manager.get_context_length("qwen3.5:4b")
        >>> print(ctx_length)  # 262144
        >>> 
        >>> threshold = manager.get_truncation_threshold("qwen3.5:4b")
        >>> print(threshold)  # 50000 (根据 256K 计算得出)
    """
    
    # 截断阈值配置（基于 context_length 的百分比和绝对值）
    TRUNCATION_THRESHOLDS = [
        # (max_context_length, threshold_value)
        # context_length <= 8K: 保守截断，使用 25%
        (8192, 2000),
        # 8K < context_length <= 32K: 中等截断，使用 25%
        (32768, 8000),
        # 32K < context_length <= 128K: 宽松截断，使用 15%
        (131072, 20000),
        # context_length > 128K: 最小截断，使用 10%
        (float('inf'), 50000),
    ]
    
    def __init__(self, base_url: str = "http://localhost:11434", cache_dir: Optional[str] = None):
        """
        初始化管理器
        
        Args:
            base_url: Ollama 服务地址
            cache_dir: 缓存目录，默认使用项目数据目录
        """
        self.base_url = base_url.rstrip('/')
        self._cache: Dict[str, int] = {}
        
        # 设置缓存目录
        if cache_dir is None:
            project_root = Path(__file__).parent.parent
            cache_dir = project_root / "data" / "model_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "context_windows.json"
        
        # 加载缓存
        self._load_cache()
    
    def _load_cache(self):
        """从文件加载缓存"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self._cache = json.load(f)
            except Exception:
                self._cache = {}
    
    def _save_cache(self):
        """保存缓存到文件"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    def list_installed_models(self) -> List[str]:
        """
        获取 Ollama 中已安装的模型列表
        
        Returns:
            模型名称列表
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                return sorted(models)
        except Exception:
            pass
        return []
    
    def scan_installed_models(self, verbose: bool = False) -> Dict[str, int]:
        """
        扫描所有已安装的模型并检测它们的上下文窗口
        
        这是主要入口方法，启动时调用一次即可。
        
        Args:
            verbose: 是否显示详细信息
            
        Returns:
            模型名称 -> 上下文窗口大小的映射
            
        Example:
            >>> manager = ModelContextManager()
            >>> results = manager.scan_installed_models(verbose=True)
            >>> print(results)
            {'qwen3.5:4b': 262144, 'llama3.1:8b': 128000}
        """
        models = self.list_installed_models()
        results = {}
        
        if verbose:
            print(f"🔍 发现 {len(models)} 个已安装模型")
            print("-" * 50)
        
        for model_name in models:
            # 检查是否已有缓存
            if model_name in self._cache:
                context_length = self._cache[model_name]
                results[model_name] = context_length
                if verbose:
                    print(f"📦 {model_name}: {context_length:,} tokens (缓存)")
                continue
            
            # 尝试从 Ollama API 获取
            try:
                context_length = self._detect_from_api(model_name)
                if context_length:
                    self._cache[model_name] = context_length
                    results[model_name] = context_length
                    if verbose:
                        print(f"✅ {model_name}: {context_length:,} tokens (检测)")
                    continue
            except Exception as e:
                if verbose:
                    print(f"⚠️ {model_name}: API 检测失败 - {e}")
            
            # 使用默认值
            results[model_name] = 8192
            self._cache[model_name] = 8192
            if verbose:
                print(f"⚠️ {model_name}: 8192 tokens (默认)")
        
        # 保存缓存
        self._save_cache()
        
        if verbose:
            print("-" * 50)
            print(f"✅ 扫描完成，已缓存 {len(results)} 个模型")
        
        return results
    
    def _detect_from_api(self, model_name: str) -> Optional[int]:
        """
        从 Ollama API 检测模型的上下文窗口
        
        Args:
            model_name: 模型名称
            
        Returns:
            上下文窗口大小，失败返回 None
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/show",
                json={"model": model_name},
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                model_info = data.get('model_info', {})
                
                # 查找 context_length 字段
                for key, value in model_info.items():
                    if key.endswith('.context_length') and isinstance(value, int):
                        return value
        except Exception:
            pass
        return None
    
    def get_context_length(
        self, 
        model_name: str, 
        auto_scan: bool = True,
        fallback_to_default: bool = True
    ) -> int:
        """
        获取模型的上下文窗口大小
        
        查找顺序：
        1. 内存缓存
        2. 文件缓存
        3. Ollama API 实时检测
        4. 自动扫描所有模型（如果启用）
        5. 默认值 (8192)
        
        Args:
            model_name: 模型名称
            auto_scan: 如果未找到，是否自动扫描所有模型
            fallback_to_default: 失败时是否返回默认值
            
        Returns:
            上下文窗口大小（tokens）
            
        Example:
            >>> manager = ModelContextManager()
            >>> manager.get_context_length("qwen3.5:4b")
            262144
        """
        # 1. 检查内存缓存
        if model_name in self._cache:
            return self._cache[model_name]
        
        # 2. 尝试实时检测
        context_length = self._detect_from_api(model_name)
        if context_length:
            self._cache[model_name] = context_length
            self._save_cache()
            return context_length
        
        # 3. 自动扫描所有模型（如果启用）
        if auto_scan:
            self.scan_installed_models()
            if model_name in self._cache:
                return self._cache[model_name]
        
        # 4. 返回默认值
        if fallback_to_default:
            default_value = 8192
            self._cache[model_name] = default_value
            return default_value
        
        raise ValueError(f"无法获取模型 {model_name} 的上下文窗口大小")
    
    def get_truncation_threshold(self, model_name: str, num_ctx: int = None) -> int:
        """
        根据模型上下文窗口计算工具输出截断阈值
        
        策略：
        - 基于实际使用的 num_ctx（而非模型最大能力）
        - 工具输出通常占 num_ctx 的 25%-50%
        - 保留足够空间给对话历史、系统提示等
        
        Args:
            model_name: 模型名称
            num_ctx: 实际使用的上下文窗口大小（可选，默认使用推荐值）
            
        Returns:
            建议的截断阈值（字符数）
            
        Example:
            >>> manager = ModelContextManager()
            >>> # 使用默认 num_ctx (32K)
            >>> manager.get_truncation_threshold("qwen3.5:4b")
            8000
            >>> # 指定 num_ctx
            >>> manager.get_truncation_threshold("qwen3.5:4b", num_ctx=131072)
            20000
        """
        # 使用指定的 num_ctx 或获取推荐值
        if num_ctx is None:
            num_ctx = self.get_recommended_num_ctx(model_name)
        
        # 根据 num_ctx 计算截断阈值（字符数 ≈ token 数 × 3）
        # 保守估计：工具输出占用不超过 num_ctx 的 30%
        threshold = int(num_ctx * 0.3 * 3)  # token × 3 ≈ 字符
        
        # 设置上下限
        min_threshold = 2000   # 最少保留 2000 字符
        max_threshold = 50000  # 最多保留 50000 字符
        
        return max(min_threshold, min(threshold, max_threshold))
    
    def get_recommended_num_ctx(self, model_name: str) -> int:
        """
        获取建议的 num_ctx 值
        
        对于大上下文模型，建议设置一个合理的上限，避免：
        1. 内存不足
        2. 推理速度过慢
        3. 实际不需要那么大上下文
        
        Args:
            model_name: 模型名称
            
        Returns:
            建议的 num_ctx 值
        """
        context_length = self.get_context_length(model_name)
        
        # 设置上限，避免过大导致性能问题
        max_recommended = 32768  # 32K 作为实际使用上限
        
        return min(context_length, max_recommended)
    
    def get_all_models_info(self) -> Dict[str, Dict]:
        """
        获取所有已缓存模型的详细信息
        
        Returns:
            模型信息字典
            
        Example:
            >>> manager = ModelContextManager()
            >>> info = manager.get_all_models_info()
            >>> print(info)
            {
                'qwen3.5:4b': {
                    'context_length': 262144,
                    'truncation_threshold': 50000,
                    'recommended_num_ctx': 32768
                }
            }
        """
        return {
            model_name: {
                'context_length': ctx_len,
                'truncation_threshold': self.get_truncation_threshold(model_name),
                'recommended_num_ctx': self.get_recommended_num_ctx(model_name)
            }
            for model_name, ctx_len in self._cache.items()
        }
    
    def clear_cache(self):
        """清除模型信息缓存（内存和文件）"""
        self._cache.clear()
        if self.cache_file.exists():
            try:
                self.cache_file.unlink()
            except Exception:
                pass


# 全局管理器实例
_manager: Optional[ModelContextManager] = None


def get_model_context_manager(base_url: str = "http://localhost:11434") -> ModelContextManager:
    """
    获取全局模型上下文管理器实例
    
    Args:
        base_url: Ollama 服务地址
        
    Returns:
        ModelContextManager 实例
    """
    global _manager
    if _manager is None:
        _manager = ModelContextManager(base_url)
    return _manager


def scan_installed_models(verbose: bool = False, base_url: str = "http://localhost:11434") -> Dict[str, int]:
    """
    便捷函数：扫描所有已安装模型
    
    Args:
        verbose: 是否显示详细信息
        base_url: Ollama 服务地址
        
    Returns:
        模型名称 -> 上下文窗口大小的映射
        
    Example:
        >>> from ollamapilot.model_context import scan_installed_models
        >>> results = scan_installed_models(verbose=True)
    """
    manager = get_model_context_manager(base_url)
    return manager.scan_installed_models(verbose=verbose)


def get_context_length(model_name: str, base_url: str = "http://localhost:11434") -> int:
    """
    便捷函数：获取模型上下文窗口大小
    
    Args:
        model_name: 模型名称
        base_url: Ollama 服务地址
        
    Returns:
        上下文窗口大小
        
    Example:
        >>> from ollamapilot.model_context import get_context_length
        >>> get_context_length("qwen3.5:4b")
        262144
    """
    manager = get_model_context_manager(base_url)
    return manager.get_context_length(model_name)


def get_truncation_threshold(model_name: str, num_ctx: int = None, base_url: str = "http://localhost:11434") -> int:
    """
    便捷函数：获取工具输出截断阈值
    
    基于实际使用的 num_ctx 计算，确保工具输出不会超出上下文限制。
    
    Args:
        model_name: 模型名称
        num_ctx: 实际使用的上下文窗口大小（可选，默认使用推荐值）
        base_url: Ollama 服务地址
        
    Returns:
        截断阈值（字符数）
        
    Example:
        >>> from ollamapilot.model_context import get_truncation_threshold
        >>> # 使用默认 num_ctx (32K)
        >>> get_truncation_threshold("qwen3.5:4b")
        28800
        >>> # 指定 num_ctx
        >>> get_truncation_threshold("qwen3.5:4b", num_ctx=131072)
        39321
    """
    manager = get_model_context_manager(base_url)
    return manager.get_truncation_threshold(model_name, num_ctx=num_ctx)


def get_recommended_num_ctx(model_name: str, base_url: str = "http://localhost:11434") -> int:
    """
    便捷函数：获取建议的 num_ctx 值
    
    Args:
        model_name: 模型名称
        base_url: Ollama 服务地址
        
    Returns:
        建议的 num_ctx 值
        
    Example:
        >>> from ollamapilot.model_context import get_recommended_num_ctx
        >>> get_recommended_num_ctx("qwen3.5:4b")
        32768
    """
    manager = get_model_context_manager(base_url)
    return manager.get_recommended_num_ctx(model_name)


def get_all_models_info(base_url: str = "http://localhost:11434") -> Dict[str, Dict]:
    """
    便捷函数：获取所有已缓存模型的详细信息
    
    Args:
        base_url: Ollama 服务地址
        
    Returns:
        模型信息字典
        
    Example:
        >>> from ollamapilot.model_context import get_all_models_info
        >>> info = get_all_models_info()
        >>> print(info)
    """
    manager = get_model_context_manager(base_url)
    return manager.get_all_models_info()
