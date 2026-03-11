"""
DeepResearchSkill - 深度研究模块

提供多轮深度研究能力
"""

from .skill import DeepResearchSkill
from .state import ResearchState, ResearchFinding, ResearchBrief

__all__ = [
    "DeepResearchSkill",
    "ResearchState",
    "ResearchFinding",
    "ResearchBrief",
]
