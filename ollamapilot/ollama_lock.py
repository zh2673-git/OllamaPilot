"""
Ollama 并发锁模块

防止生成模型和向量模型同时调用Ollama导致的冲突
"""

import threading
import time
from typing import Optional

# 全局Ollama锁
_ollama_lock = threading.RLock()
_lock_owner: Optional[str] = None
_lock_count = 0


def acquire_ollama_lock(owner: str = "unknown", timeout: float = 60.0) -> bool:
    """
    获取Ollama锁
    
    Args:
        owner: 锁的持有者标识（如"chat", "index"）
        timeout: 超时时间（秒）
        
    Returns:
        是否成功获取锁
    """
    global _lock_owner, _lock_count
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        if _ollama_lock.acquire(blocking=False):
            _lock_owner = owner
            _lock_count += 1
            return True
        time.sleep(0.1)
    
    return False


def release_ollama_lock():
    """释放Ollama锁"""
    global _lock_owner, _lock_count
    
    try:
        _ollama_lock.release()
        _lock_count = max(0, _lock_count - 1)
        if _lock_count == 0:
            _lock_owner = None
    except RuntimeError:
        pass  # 锁未被持有


def get_lock_status() -> dict:
    """获取锁状态"""
    return {
        "locked": _lock_count > 0,
        "owner": _lock_owner,
        "count": _lock_count
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
