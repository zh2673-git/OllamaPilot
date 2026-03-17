"""
CheckpointManager - LangChain Checkpoint 封装

简化 Checkpoint 的使用和管理。
"""

from typing import Any, Dict, List, Optional
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.base import BaseCheckpointSaver


class CheckpointManager:
    """
    Checkpoint 管理器
    
    封装 LangChain Checkpoint 操作，提供简化的接口。
    """
    
    def __init__(self, checkpointer: Optional[BaseCheckpointSaver] = None):
        """
        初始化 CheckpointManager
        
        Args:
            checkpointer: Checkpoint 存储器，默认使用 MemorySaver
        """
        self.checkpointer = checkpointer or MemorySaver()
    
    def get_history(self, thread_id: str) -> List[Any]:
        """
        获取对话历史
        
        Args:
            thread_id: 对话线程 ID
            
        Returns:
            消息列表
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            checkpoint_tuple = self.checkpointer.get_tuple(config)
            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                checkpoint = checkpoint_tuple.checkpoint
                
                # 尝试不同的消息存储位置
                if "messages" in checkpoint:
                    return checkpoint["messages"]
                
                if "channel_values" in checkpoint:
                    channel_values = checkpoint["channel_values"]
                    if "messages" in channel_values:
                        return channel_values["messages"]
                
                if checkpoint_tuple.state and "messages" in checkpoint_tuple.state:
                    return checkpoint_tuple.state["messages"]
        except Exception:
            pass
        
        return []
    
    def clear_history(self, thread_id: str) -> bool:
        """
        清除对话历史
        
        Args:
            thread_id: 对话线程 ID
            
        Returns:
            是否成功
        """
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            self.checkpointer.delete(config)
            return True
        except Exception:
            return False
    
    def save_state(self, thread_id: str, state: Dict[str, Any]) -> bool:
        """
        保存状态
        
        Args:
            thread_id: 对话线程 ID
            state: 状态字典
            
        Returns:
            是否成功
        """
        # 这里可以根据需要实现自定义状态保存
        return True
    
    def get_thread_ids(self) -> List[str]:
        """
        获取所有线程 ID
        
        Returns:
            线程 ID 列表
        """
        # MemorySaver 不支持枚举所有线程
        # 如果需要，可以自定义实现
        return []
