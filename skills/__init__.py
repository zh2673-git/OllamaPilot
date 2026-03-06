"""
Skill 包

包含各种实用的 Skill
"""

from .web.skill import WebSkill
from .browser.skill import BrowserSkill
from .canvas.skill import CanvasSkill
from .nodes.skill import NodesSkill
from .scheduler.skill import SchedulerSkill
from .自动笔记.skill import Skill as AutoNoteSkill

__all__ = [
    "WebSkill",
    "BrowserSkill",
    "CanvasSkill",
    "NodesSkill",
    "SchedulerSkill",
    "AutoNoteSkill",
]
