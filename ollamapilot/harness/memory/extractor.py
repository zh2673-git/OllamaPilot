"""
FactExtractor - LLM 事实提取器

借鉴 DeerFlow 的 LLM 事实提取设计
从对话中提取关键事实用于记忆
"""

from typing import Any, Dict, List, Optional
import json
import logging

logger = logging.getLogger("ollamapilot.harness.memory")


class FactExtractor:
    """
    事实提取器
    
    使用 LLM 从对话中提取结构化事实。
    
    提取类型：
    - user_preference: 用户偏好
    - user_fact: 用户相关事实
    - task_info: 任务信息
    - context: 上下文信息
    """
    
    EXTRACTION_PROMPT = """从以下对话中提取关键事实。

对话：
{conversation}

请提取以下类型的事实：
1. user_preference - 用户偏好（如"我喜欢Python"）
2. user_fact - 用户相关事实（如"我是软件工程师"）
3. task_info - 任务信息（如"正在开发Web应用"）
4. context - 上下文信息（如"使用FastAPI框架"）

以JSON格式返回，格式如下：
{{
    "facts": [
        {{
            "type": "user_preference",
            "content": "用户喜欢Python",
            "confidence": 0.9
        }}
    ]
}}

只返回JSON，不要有其他内容。"""
    
    def __init__(self, model: Any):
        """
        初始化提取器
        
        Args:
            model: 用于提取的 LLM
        """
        self.model = model
    
    async def extract(self, conversation: str) -> List[Dict[str, Any]]:
        """
        从对话中提取事实
        
        Args:
            conversation: 对话内容
            
        Returns:
            提取的事实列表
        """
        try:
            prompt = self.EXTRACTION_PROMPT.format(conversation=conversation)
            
            # 调用 LLM
            from langchain_core.messages import HumanMessage
            response = self.model.invoke([HumanMessage(content=prompt)])
            
            # 解析 JSON
            content = response.content if hasattr(response, 'content') else str(response)
            
            # 提取 JSON 部分
            json_str = self._extract_json(content)
            data = json.loads(json_str)
            
            facts = data.get('facts', [])
            
            # 过滤低置信度
            facts = [f for f in facts if f.get('confidence', 0) > 0.7]
            
            return facts
            
        except Exception as e:
            logger.warning(f"事实提取失败: {e}")
            return []
    
    def _extract_json(self, text: str) -> str:
        """从文本中提取 JSON"""
        # 尝试直接解析
        text = text.strip()
        
        # 查找 JSON 代码块
        if '```json' in text:
            start = text.find('```json') + 7
            end = text.find('```', start)
            if end > start:
                return text[start:end].strip()
        
        if '```' in text:
            start = text.find('```') + 3
            end = text.find('```', start)
            if end > start:
                return text[start:end].strip()
        
        # 查找花括号
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > start:
            return text[start:end+1]
        
        return text
