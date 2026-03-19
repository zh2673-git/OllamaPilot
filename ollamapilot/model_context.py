"""
模型上下文窗口管理模块

提供自动检测和管理 Ollama 模型上下文窗口大小的功能。
支持从 Ollama API 获取模型信息，并根据模型能力和硬件配置动态调整配置。
"""

import logging
import requests
import json
import os
import platform
import subprocess
import hashlib
from typing import Optional, Dict, List, Union, Tuple
from functools import lru_cache
from pathlib import Path
from datetime import datetime, timedelta

from ollamapilot.logging_config import get_logger

logger = get_logger("model_context")


class ModelContextManager:
    """
    模型上下文窗口管理器
    
    功能：
    1. 自动扫描 Ollama 中已安装的模型
    2. 自动检测每个模型的 context_length
    3. 根据模型能力和硬件配置（显存）计算最优 num_ctx
    4. 缓存检测结果到文件，避免重复查询
    5. 支持硬件指纹，硬件变化时自动重新计算
    
    使用示例：
        >>> manager = ModelContextManager()
        >>> # 扫描所有已安装模型
        >>> manager.scan_installed_models()
        >>> ctx_length = manager.get_context_length("qwen3.5:4b")
        >>> print(ctx_length)  # 262144
        >>> 
        >>> # 获取基于硬件配置的最优 num_ctx
        >>> num_ctx = manager.get_recommended_num_ctx("qwen3.5:4b")
        >>> print(num_ctx)  # 根据显存动态计算，如 131072
    """
    
    # 截断阈值配置（基于 num_ctx 的百分比）
    # 工具输出占用不超过 num_ctx 的 30%
    TOOL_TRUNCATION_RATIO = 0.3
    
    # KV Cache 系数表 (MB per 1K tokens, INT4/Q4_K_M 精度)
    # 基于公式: 2 × 层数 × 隐藏维度 × 精度字节数 / 1024
    KV_CACHE_COEFFICIENTS = {
        # 模型规模 -> MB/1K tokens
        "1B": 18,    # 估算
        "4B": 37,    # qwen3.5:4b 等
        "7B": 64,    # llama3:8b 等
        "8B": 64,    # llama3:8b 等
        "13B": 120,  # 估算
        "70B": 640,  # 需要40GB+显存
    }
    
    # 模型规模映射（从模型名称估算）
    MODEL_SIZE_PATTERNS = {
        "0.5b": "1B", "1b": "1B", "1.5b": "1B",
        "4b": "4B", "3b": "4B", "3.5b": "4B",
        "7b": "7B", "8b": "8B",
        "13b": "13B", "14b": "13B",
        "70b": "70B", "72b": "70B",
    }
    
    # 缓存有效期（天）
    CACHE_EXPIRY_DAYS = 7
    
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
    
    def _get_hardware_fingerprint(self) -> str:
        """
        获取硬件指纹
        
        用于检测硬件是否变化，变化时需要重新计算 num_ctx。
        
        Returns:
            硬件指纹字符串
        """
        try:
            # 获取 GPU 信息
            gpu_info = self._get_gpu_info()
            
            # 组合硬件信息
            hardware_str = f"{platform.system()}|{platform.machine()}|{gpu_info}"
            
            # 计算哈希
            return hashlib.md5(hardware_str.encode()).hexdigest()[:16]
        except Exception:
            return "unknown"
    
    def _get_gpu_info(self) -> str:
        """
        获取 GPU 信息字符串
        
        Returns:
            GPU 型号和驱动信息
        """
        try:
            if platform.system() == "Windows":
                # Windows: 使用 PowerShell
                ps_cmd = "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty Name"
                result = subprocess.run(
                    ["powershell", "-Command", ps_cmd],
                    capture_output=True, text=True, timeout=5
                )
                gpu_names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                return "|".join(gpu_names) if gpu_names else "unknown"
            else:
                # Linux/Mac: 尝试使用 nvidia-smi
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                gpu_names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                return "|".join(gpu_names) if gpu_names else "unknown"
        except Exception:
            return "unknown"
    
    def detect_vram(self) -> Optional[float]:
        """
        检测可用显存（GB）
        
        Returns:
            可用显存（GB），检测失败返回 None
        """
        try:
            if platform.system() == "Windows":
                return self._detect_vram_windows()
            else:
                return self._detect_vram_linux()
        except Exception:
            return None
    
    def _detect_vram_windows(self) -> Optional[float]:
        """Windows 下检测显存"""
        try:
            # 使用 PowerShell 获取显存信息（兼容新版 Windows）
            # 注意：PowerShell 中除法使用 / 但需要确保是数字类型
            ps_cmd = "Get-CimInstance -ClassName Win32_VideoController | Select-Object -ExpandProperty AdapterRAM | ForEach-Object { [math]::Round($_, 2) }"
            result = subprocess.run(
                ["powershell", "-Command", ps_cmd],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            total_vram = 0
            for line in lines:
                line = line.strip()
                if line:
                    try:
                        # AdapterRAM 以字节为单位
                        vram_bytes = int(float(line))
                        vram_gb = vram_bytes / (1024 ** 3)
                        if vram_gb > 0:
                            total_vram += vram_gb
                    except ValueError:
                        continue
            return total_vram if total_vram > 0 else None
        except Exception:
            return None
    
    def _detect_vram_linux(self) -> Optional[float]:
        """Linux 下检测显存"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            total_vram = 0
            for line in lines:
                line = line.strip()
                if line:
                    total_vram += float(line) / 1024  # MB 转 GB
            return total_vram if total_vram > 0 else None
        except Exception:
            return None
    
    def _estimate_model_size(self, model_name: str) -> str:
        """
        从模型名称估算模型规模
        
        Args:
            model_name: 模型名称（如 "qwen3.5:4b"）
            
        Returns:
            模型规模（如 "4B"）
        """
        model_lower = model_name.lower()
        
        # 尝试匹配已知模式
        for pattern, size in self.MODEL_SIZE_PATTERNS.items():
            if pattern in model_lower:
                return size
        
        # 默认返回 7B
        return "7B"
    
    def _get_kv_cache_coefficient(self, model_size: str) -> float:
        """
        获取 KV Cache 系数
        
        Args:
            model_size: 模型规模（如 "4B"）
            
        Returns:
            MB per 1K tokens
        """
        return self.KV_CACHE_COEFFICIENTS.get(model_size, 64)  # 默认 64MB/1K
    
    def _align_context_value(self, ctx: int) -> int:
        """
        对齐到常见的上下文值
        
        策略：向下取整到不超过计算值的最大档位（保守策略）
        例如：138K -> 128K (131072)，而不是 256K
        
        Args:
            ctx: 计算出的上下文值
            
        Returns:
            对齐后的值
        """
        common_values = [8192, 16384, 32768, 65536, 131072, 262144]
        
        # 找到不超过计算值的最大档位（向下取整）
        result = 8192  # 最小保底值
        for val in common_values:
            if ctx >= val:
                result = val
            else:
                break
        
        return result
    
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
            logger.info(f"发现 {len(models)} 个已安装模型")

        for model_name in models:
            if model_name in self._cache:
                context_length = self._cache[model_name]
                results[model_name] = context_length
                if verbose:
                    logger.debug(f"{model_name}: {context_length:,} tokens (缓存)")
                continue

            try:
                context_length = self._detect_from_api(model_name)
                if context_length:
                    self._cache[model_name] = context_length
                    results[model_name] = context_length
                    if verbose:
                        logger.info(f"{model_name}: {context_length:,} tokens (检测)")
                    continue
            except Exception as e:
                if verbose:
                    logger.warning(f"{model_name}: API 检测失败 - {e}")

            results[model_name] = 8192
            self._cache[model_name] = 8192
            if verbose:
                logger.warning(f"{model_name}: 8192 tokens (默认)")

        self._save_cache()

        if verbose:
            logger.info(f"扫描完成，已缓存 {len(results)} 个模型")

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
    
    def get_recommended_num_ctx(self, model_name: str, mode: str = "auto") -> int:
        """
        获取建议的 num_ctx 值
        
        支持三种模式：
        1. "auto" - 自动检测显存并计算最优值（默认）
        2. "max" - 使用模型最大能力
        3. "conservative" - 使用保守值（32K）
        
        对于 "auto" 模式：
        - 检测当前硬件配置（显存）
        - 根据模型规模计算 KV Cache 占用
        - 返回最优的 num_ctx 值
        - 结果会被缓存，硬件变化时自动重新计算
        
        Args:
            model_name: 模型名称
            mode: 计算模式 ("auto", "max", "conservative")
            
        Returns:
            建议的 num_ctx 值
            
        Example:
            >>> manager = ModelContextManager()
            >>> # 自动模式（基于硬件）
            >>> manager.get_recommended_num_ctx("qwen3.5:4b")
            131072  # 根据显存动态计算
            >>> # 最大能力模式
            >>> manager.get_recommended_num_ctx("qwen3.5:4b", mode="max")
            262144
            >>> # 保守模式
            >>> manager.get_recommended_num_ctx("qwen3.5:4b", mode="conservative")
            32768
        """
        context_length = self.get_context_length(model_name)
        
        # max 模式：使用模型最大能力
        if mode == "max":
            return context_length
        
        # conservative 模式：使用保守值
        if mode == "conservative":
            return min(context_length, 32768)
        
        # auto 模式：基于硬件配置智能计算
        return self._calculate_optimal_num_ctx(model_name, context_length)
    
    def _calculate_optimal_num_ctx(self, model_name: str, max_ctx: int) -> int:
        """
        计算最优 num_ctx 值（基于硬件配置）
        
        算法：
        1. 检测当前显存
        2. 估算模型权重占用（INT4 量化）
        3. 计算可用于上下文的显存
        4. 根据 KV Cache 系数计算最大上下文
        5. 对齐到常见值并缓存
        
        Args:
            model_name: 模型名称
            max_ctx: 模型最大上下文能力
            
        Returns:
            最优 num_ctx 值
        """
        # 获取当前硬件指纹
        current_fingerprint = self._get_hardware_fingerprint()
        
        # 检查缓存
        cache_key = f"num_ctx_config_{model_name}"
        cached_config = self._cache.get(cache_key)
        
        if cached_config:
            # 检查缓存是否有效
            cache_time = datetime.fromtimestamp(cached_config.get("timestamp", 0))
            cache_age = datetime.now() - cache_time
            cached_fingerprint = cached_config.get("hardware_fingerprint", "")
            
            # 缓存未过期且硬件未变化
            if (cache_age.days < self.CACHE_EXPIRY_DAYS and 
                cached_fingerprint == current_fingerprint):
                return cached_config["num_ctx"]
        
        # 检测显存
        vram_gb = self.detect_vram()
        
        # 如果自动检测失败，尝试从配置读取
        if vram_gb is None:
            try:
                from ollamapilot.config import get_config
                config_vram = get_config().get_float('CHAT_VRAM_GB', 0)
                if config_vram > 0:
                    vram_gb = config_vram
            except Exception:
                pass
        
        if vram_gb is None:
            # 检测失败且未配置，使用保守值
            result = min(max_ctx, 32768)
            self._cache_num_ctx_config(model_name, result, current_fingerprint)
            return result
        
        # 估算模型规模
        model_size = self._estimate_model_size(model_name)
        
        # 模型权重占用（INT4 量化：每参数 0.5 字节）
        model_size_gb = int(model_size.replace("B", "")) * 0.5
        
        # 计算可用于上下文的显存（保留 1GB 余量）
        available_for_ctx = vram_gb - model_size_gb - 1
        
        if available_for_ctx <= 0:
            # 显存不足，使用最小值
            result = min(max_ctx, 8192)
            self._cache_num_ctx_config(model_name, result, current_fingerprint)
            return result
        
        # 获取 KV Cache 系数
        kv_coeff = self._get_kv_cache_coefficient(model_size)
        
        # 计算最大上下文（KB）
        # available_for_ctx (GB) * 1024 (MB/GB) / kv_coeff (MB/1K) = max_ctx_k
        max_ctx_k = available_for_ctx * 1024 / kv_coeff
        
        # 转换为 tokens 并对齐
        optimal_ctx = self._align_context_value(int(max_ctx_k * 1024))
        
        # 限制在模型最大能力内
        optimal_ctx = min(optimal_ctx, max_ctx)
        
        # 缓存结果
        self._cache_num_ctx_config(model_name, optimal_ctx, current_fingerprint)
        
        return optimal_ctx
    
    def _cache_num_ctx_config(self, model_name: str, num_ctx: int, fingerprint: str):
        """
        缓存 num_ctx 配置
        
        Args:
            model_name: 模型名称
            num_ctx: 计算出的 num_ctx 值
            fingerprint: 硬件指纹
        """
        cache_key = f"num_ctx_config_{model_name}"
        self._cache[cache_key] = {
            "num_ctx": num_ctx,
            "hardware_fingerprint": fingerprint,
            "timestamp": datetime.now().timestamp(),
            "vram_detected": self.detect_vram(),
        }
        self._save_cache()
    
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
