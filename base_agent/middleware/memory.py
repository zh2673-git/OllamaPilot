"""
Memory Middleware - 轻量级记忆系统

为本地小模型设计的轻量级记忆系统，支持：
1. 跨对话保持关键信息
2. 自动提取用户偏好
3. 简单的记忆注入

特点：
- 轻量级：使用 JSON 文件存储，无需数据库
- 智能提取：自动从对话中提取关键信息
- 可配置：支持启用/禁用，记忆数量限制
"""

import json
import os
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime
from .base import AgentMiddleware, AgentState


class MemoryMiddleware(AgentMiddleware):
    """
    记忆中间件（轻量级）
    
    自动记录和注入跨对话的记忆，包括：
    - 用户偏好（编程语言、编辑器等）
    - 工作上下文（项目名称、技术栈等）
    - 重要事实（关键决策、配置等）
    
    示例:
        middleware = MemoryMiddleware(
            memory_file="~/.agent_memory.json",
            max_memories=20,  # 最多保留20条记忆
            auto_extract=True  # 自动提取记忆
        )
    """
    
    def __init__(
        self,
        memory_file: str = "~/.agent_memory.json",
        max_memories: int = 20,
        auto_extract: bool = True,
        extract_threshold: int = 5  # 每5轮对话提取一次
    ):
        """
        初始化记忆中间件
        
        Args:
            memory_file: 记忆文件路径
            max_memories: 最大记忆数量，超出时移除最旧的
            auto_extract: 是否自动从对话中提取记忆
            extract_threshold: 提取记忆的对话轮数阈值
        """
        self.memory_file = Path(memory_file).expanduser()
        self.max_memories = max_memories
        self.auto_extract = auto_extract
        self.extract_threshold = extract_threshold
        self._conversation_count = 0
        
        # 确保目录存在
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载现有记忆
        self._memories = self._load_memories()
    
    def _load_memories(self) -> List[Dict[str, Any]]:
        """从文件加载记忆"""
        if not self.memory_file.exists():
            return []
        
        try:
            with open(self.memory_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('memories', [])
        except Exception:
            return []
    
    def _save_memories(self) -> None:
        """保存记忆到文件"""
        try:
            data = {
                'memories': self._memories,
                'updated_at': datetime.now().isoformat()
            }
            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存记忆失败: {e}")
    
    def _add_memory(self, content: str, category: str = "general", confidence: float = 0.8) -> None:
        """
        添加一条记忆
        
        Args:
            content: 记忆内容
            category: 记忆类别（preference, context, fact, goal）
            confidence: 置信度 0-1
        """
        # 检查是否已存在相似记忆
        for mem in self._memories:
            if mem['content'] == content:
                mem['confidence'] = max(mem['confidence'], confidence)
                mem['updated_at'] = datetime.now().isoformat()
                self._save_memories()
                return
        
        # 添加新记忆
        memory = {
            'id': f"mem_{len(self._memories)}",
            'content': content,
            'category': category,
            'confidence': confidence,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        self._memories.append(memory)
        
        # 限制记忆数量
        if len(self._memories) > self.max_memories:
            # 移除置信度最低的最旧记忆
            self._memories.sort(key=lambda x: (x['confidence'], x['created_at']))
            self._memories = self._memories[-self.max_memories:]
        
        self._save_memories()
    
    def _extract_memories(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        从对话消息中提取记忆（简化版，无需 LLM）
        
        使用简单的规则提取：
        - 包含"喜欢"、"偏好"、"习惯"的句子 -> preference
        - 包含"项目"、"工作"、"正在"的句子 -> context
        - 包含"决定"、"选择"、"使用"的句子 -> fact
        
        Args:
            messages: 对话消息列表
            
        Returns:
            提取到的记忆列表
        """
        extracted = []
        
        # 关键词映射
        keywords = {
            'preference': ['喜欢', '偏好', '习惯', '常用', '爱用', '不喜欢'],
            'context': ['项目', '工作', '正在做', '负责', '开发', '维护'],
            'fact': ['决定', '选择', '使用', '采用', '配置为', '设置为'],
            'goal': ['目标', '计划', '想要', '希望', '打算']
        }
        
        for msg in messages:
            if msg.get('role') != 'user':
                continue
            
            content = msg.get('content', '')
            
            # 检查每类关键词
            for category, words in keywords.items():
                for word in words:
                    if word in content:
                        # 提取包含关键词的句子
                        sentences = content.split('。')
                        for sent in sentences:
                            if word in sent and len(sent) > 5:
                                # 清理句子
                                clean_sent = sent.strip()
                                if len(clean_sent) < 200:  # 限制长度
                                    extracted.append({
                                        'content': clean_sent,
                                        'category': category,
                                        'confidence': 0.7
                                    })
                                break
        
        return extracted
    
    def _format_memories_for_prompt(self) -> str:
        """将记忆格式化为提示词"""
        if not self._memories:
            return ""
        
        # 按类别分组
        by_category = {}
        for mem in self._memories:
            cat = mem['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(mem)
        
        # 生成提示词
        lines = ["\n【记忆】"]
        
        category_names = {
            'preference': '用户偏好',
            'context': '工作上下文',
            'fact': '重要事实',
            'goal': '当前目标'
        }
        
        for cat, memories in by_category.items():
            cat_name = category_names.get(cat, cat)
            lines.append(f"\n{cat_name}:")
            
            # 按置信度排序，只显示高置信度的
            memories.sort(key=lambda x: x['confidence'], reverse=True)
            for mem in memories[:3]:  # 每类最多3条
                confidence = mem['confidence']
                content = mem['content']
                lines.append(f"  • {content} (置信度: {confidence:.0%})")
        
        lines.append("\n" + "-" * 40)
        
        return "\n".join(lines)
    
    def before_model(
        self,
        state: AgentState,
        config: Optional[dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用前注入记忆
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新
        """
        if not self._memories:
            return None
        
        # 格式化记忆
        memory_prompt = self._format_memories_for_prompt()
        
        if not memory_prompt:
            return None
        
        # 注入到系统消息或用户消息前
        messages = state.get('messages', [])
        
        # 找到第一条用户消息，在其前添加记忆
        for i, msg in enumerate(messages):
            if hasattr(msg, 'type') and msg.type == 'human':
                # 在消息内容前添加记忆
                original_content = msg.content if hasattr(msg, 'content') else str(msg)
                new_content = f"{memory_prompt}\n\n用户输入: {original_content}"
                
                # 更新消息
                from langchain_core.messages import HumanMessage
                messages[i] = HumanMessage(content=new_content)
                break
        
        return {'messages': messages}
    
    def after_model(
        self,
        response: Any,
        state: AgentState,
        config: Optional[dict] = None
    ) -> Optional[Dict[str, Any]]:
        """
        模型调用后提取记忆
        
        Args:
            response: 模型响应
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新
        """
        if not self.auto_extract:
            return None
        
        self._conversation_count += 1
        
        # 达到阈值时提取记忆
        if self._conversation_count >= self.extract_threshold:
            self._conversation_count = 0
            
            messages = state.get('messages', [])
            extracted = self._extract_memories(messages)
            
            for mem in extracted:
                self._add_memory(
                    content=mem['content'],
                    category=mem['category'],
                    confidence=mem['confidence']
                )
            
            if extracted and config and config.get('verbose'):
                print(f"💾 提取了 {len(extracted)} 条记忆")
        
        return None
    
    def get_memories(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        获取记忆列表
        
        Args:
            category: 按类别筛选，None 表示所有
            
        Returns:
            记忆列表
        """
        if category:
            return [m for m in self._memories if m['category'] == category]
        return self._memories.copy()
    
    def clear_memories(self) -> None:
        """清空所有记忆"""
        self._memories = []
        self._save_memories()
    
    def remove_memory(self, memory_id: str) -> bool:
        """
        删除指定记忆
        
        Args:
            memory_id: 记忆 ID
            
        Returns:
            是否成功删除
        """
        for i, mem in enumerate(self._memories):
            if mem['id'] == memory_id:
                self._memories.pop(i)
                self._save_memories()
                return True
        return False
