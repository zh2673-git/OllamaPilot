"""
深度研究状态定义

参考 Open Deep Research 的状态管理
"""

from typing import TypedDict, List, Optional, Annotated
from operator import add
from langchain_core.messages import BaseMessage


class ResearchFinding(TypedDict):
    """研究发现"""
    topic: str
    content: str
    sources: List[str]
    confidence: float


class ResearchBrief(TypedDict):
    """研究简报"""
    topic: str
    objectives: List[str]
    subtopics: List[str]
    key_questions: List[str]


class ResearchState(TypedDict):
    """
    研究状态
    
    用于 LangGraph 状态管理
    """
    # 消息历史
    messages: Annotated[List[BaseMessage], add]
    
    # 原始查询
    original_query: str
    
    # 研究简报
    research_brief: Optional[ResearchBrief]
    
    # 研究发现列表
    research_findings: Annotated[List[ResearchFinding], add]
    
    # 当前迭代次数
    research_iterations: int
    
    # 最终报告
    final_report: Optional[str]
    
    # 是否需要澄清
    needs_clarification: bool
    
    # 澄清问题
    clarification_questions: List[str]
    
    # 用户澄清回答
    clarification_answers: Optional[str]
    
    # 研究完成标志
    is_complete: bool
    
    # 错误信息
    error: Optional[str]
