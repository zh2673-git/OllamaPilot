"""
EnhancedMemoryManager - 增强型记忆管理器

在现有 MemoryManager 基础上叠加 LLM 事实提取
支持异步更新队列和防抖机制
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Callable
import logging

from ollamapilot.harness.memory.extractor import FactExtractor

logger = logging.getLogger("ollamapilot.harness.memory")


class EnhancedMemoryManager:
    """
    增强型记忆管理器
    
    功能：
    1. 包装现有的 MemoryManager
    2. 添加 LLM 事实提取
    3. 记忆整合与去重
    4. 异步更新队列（防抖机制）
    
    设计原则：
    - 增强而非替换
    - 可选启用
    - 向后兼容
    """
    
    def __init__(
        self, 
        base_manager: Any,
        model: Any,
        enable_extraction: bool = True,
        extraction_threshold: float = 0.7,
        debounce_interval: float = 5.0,  # 防抖间隔（秒）
        max_queue_size: int = 100  # 最大队列大小
    ):
        """
        初始化增强型记忆管理器
        
        Args:
            base_manager: 现有的 MemoryManager
            model: 用于事实提取的 LLM
            enable_extraction: 是否启用事实提取
            extraction_threshold: 提取置信度阈值
            debounce_interval: 防抖间隔（秒）
            max_queue_size: 最大队列大小
        """
        self.base_manager = base_manager
        self.extractor = FactExtractor(model) if enable_extraction else None
        self.extraction_threshold = extraction_threshold
        self.debounce_interval = debounce_interval
        self.max_queue_size = max_queue_size
        
        # 待处理事实队列
        self._pending_facts: List[Dict[str, Any]] = []
        
        # 防抖控制
        self._last_extract_time: float = 0
        self._extract_timer: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # 统计
        self._stats = {
            "total_extracted": 0,
            "total_committed": 0,
            "total_deduped": 0
        }
    
    def recall(self, query: str, top_k: int = 5) -> List[str]:
        """
        检索相关记忆
        
        先调用基础管理器，然后整合增强记忆。
        """
        # 基础检索
        base_memories = []
        if hasattr(self.base_manager, 'recall'):
            try:
                base_memories = self.base_manager.recall(query, top_k=top_k)
            except Exception as e:
                logger.warning(f"基础记忆检索失败: {e}")
        
        # 添加待处理事实（标记为待确认）
        pending = [f"[待确认] {f['content']}" for f in self._pending_facts]
        
        return base_memories + pending
    
    async def remember(self, conversation: str, thread_id: str = "default") -> bool:
        """
        记忆对话（带防抖）
        
        1. 调用基础管理器
        2. 防抖提取事实（可选）
        3. 异步保存事实
        
        Args:
            conversation: 对话内容
            thread_id: 会话ID
            
        Returns:
            bool: 是否成功
        """
        success = True
        
        # 基础记忆（立即执行）
        if hasattr(self.base_manager, 'remember'):
            try:
                self.base_manager.remember(conversation, thread_id)
            except Exception as e:
                logger.warning(f"基础记忆保存失败: {e}")
                success = False
        
        # 防抖事实提取
        if self.extractor:
            await self._debounced_extract(conversation)
        
        return success
    
    async def _debounced_extract(self, conversation: str):
        """
        防抖提取事实
        
        实现机制：
        1. 每次调用时取消之前的定时器
        2. 设置新的定时器，在 debounce_interval 后执行
        3. 如果队列已满，立即执行提取
        """
        async with self._lock:
            # 取消之前的定时器
            if self._extract_timer and not self._extract_timer.done():
                self._extract_timer.cancel()
            
            # 检查队列是否已满
            if len(self._pending_facts) >= self.max_queue_size:
                # 立即执行提取
                await self._do_extract(conversation)
                return
            
            # 设置新的防抖定时器
            self._extract_timer = asyncio.create_task(
                self._extract_after_delay(conversation, self.debounce_interval)
            )
    
    async def _extract_after_delay(self, conversation: str, delay: float):
        """延迟后执行提取"""
        try:
            await asyncio.sleep(delay)
            await self._do_extract(conversation)
        except asyncio.CancelledError:
            # 正常取消，不报错
            pass
        except Exception as e:
            logger.error(f"延迟提取失败: {e}")
    
    async def _do_extract(self, conversation: str):
        """实际执行事实提取"""
        try:
            facts = await self.extractor.extract(conversation)
            
            async with self._lock:
                for fact in facts:
                    if fact.get('confidence', 0) >= self.extraction_threshold:
                        # 去重检查
                        if not self._is_duplicate(fact):
                            self._pending_facts.append(fact)
                            self._stats["total_extracted"] += 1
                        else:
                            self._stats["total_deduped"] += 1
                
                # 更新最后提取时间
                self._last_extract_time = time.time()
                
        except Exception as e:
            logger.warning(f"事实提取失败: {e}")
    
    def _is_duplicate(self, new_fact: Dict[str, Any]) -> bool:
        """检查事实是否重复"""
        new_content = new_fact.get('content', '').lower().strip()
        new_type = new_fact.get('type', '')
        
        for existing in self._pending_facts:
            existing_content = existing.get('content', '').lower().strip()
            existing_type = existing.get('type', '')
            
            # 类型相同且内容相似度超过80%认为是重复
            if existing_type == new_type:
                # 简单相似度检查（包含关系）
                if new_content in existing_content or existing_content in new_content:
                    return True
                
                # 编辑距离检查（可选）
                # if self._similarity(new_content, existing_content) > 0.8:
                #     return True
        
        return False
    
    def commit_facts(self) -> int:
        """
        提交待处理的事实到长期记忆
        
        Returns:
            提交的事实数量
        """
        count = 0
        
        for fact in self._pending_facts:
            try:
                # 保存到基础管理器
                if hasattr(self.base_manager, 'add_memory'):
                    content = f"[{fact['type']}] {fact['content']}"
                    self.base_manager.add_memory(content)
                    count += 1
                    self._stats["total_committed"] += 1
            except Exception as e:
                logger.warning(f"保存事实失败: {e}")
        
        # 清空待处理
        self._pending_facts.clear()
        
        return count
    
    async def commit_facts_async(self) -> int:
        """异步提交事实"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.commit_facts
        )
    
    def get_pending_facts(self) -> List[Dict[str, Any]]:
        """获取待处理的事实"""
        return self._pending_facts.copy()
    
    def clear_pending_facts(self):
        """清空待处理的事实"""
        self._pending_facts.clear()
    
    def get_stats(self) -> Dict[str, int]:
        """获取统计信息"""
        return self._stats.copy()
    
    async def flush(self):
        """
        立即刷新所有待处理事实
        
        取消定时器并立即提交
        """
        async with self._lock:
            if self._extract_timer and not self._extract_timer.done():
                self._extract_timer.cancel()
                try:
                    await self._extract_timer
                except asyncio.CancelledError:
                    pass
        
        # 提交所有待处理事实
        return self.commit_facts()
    
    # 代理基础管理器的方法
    def __getattr__(self, name):
        """代理到基础管理器"""
        return getattr(self.base_manager, name)
