"""
Ollama 并发锁模块

防止生成模型和向量模型同时调用Ollama导致的冲突
"""

import threading
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# 全局Ollama锁 - 使用普通Lock确保严格的互斥
_ollama_lock = threading.Lock()
_lock_owner: Optional[str] = None
_lock_acquired_time: Optional[float] = None


def acquire_ollama_lock(owner: str = "unknown", timeout: float = 60.0) -> bool:
    """
    获取Ollama锁
    
    Args:
        owner: 锁的持有者标识（如"chat", "index"）
        timeout: 超时时间（秒）
        
    Returns:
        是否成功获取锁
    """
    global _lock_owner, _lock_acquired_time
    
    logger.debug(f"[{owner}] 尝试获取Ollama锁...")
    
    # 使用带超时的acquire
    acquired = _ollama_lock.acquire(timeout=timeout)
    
    if acquired:
        _lock_owner = owner
        _lock_acquired_time = time.time()
        logger.debug(f"[{owner}] 成功获取Ollama锁")
        return True
    else:
        logger.warning(f"[{owner}] 获取Ollama锁超时，当前持有者: {_lock_owner}")
        return False


def release_ollama_lock():
    """释放Ollama锁"""
    global _lock_owner, _lock_acquired_time
    
    try:
        _ollama_lock.release()
        logger.debug(f"[{_lock_owner}] 释放Ollama锁")
        _lock_owner = None
        _lock_acquired_time = None
    except RuntimeError:
        pass  # 锁未被持有


def get_lock_status() -> dict:
    """获取锁状态"""
    locked = _ollama_lock.locked()
    hold_time = None
    if locked and _lock_acquired_time:
        hold_time = time.time() - _lock_acquired_time
    
    return {
        "locked": locked,
        "owner": _lock_owner,
        "hold_time": hold_time
    }


class OllamaLockContext:
    """Ollama锁上下文管理器"""
    
    def __init__(self, owner: str = "unknown", timeout: float = 60.0):
        self.owner = owner
        self.timeout = timeout
        self.acquired = False
    
    def __enter__(self):
        self.acquired = acquire_ollama_lock(self.owner, self.timeout)
        if not self.acquired:
            raise TimeoutError(f"无法获取Ollama锁，当前持有者: {_lock_owner}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.acquired:
            release_ollama_lock()
        return False
