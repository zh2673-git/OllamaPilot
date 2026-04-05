"""
ResearchSubAgent - 研究型子 Agent

专门处理研究类任务，支持多步骤信息收集
"""

from typing import Any, Dict, List, Optional

from ollamapilot.harness.subagents.base import SubAgent, SubAgentResult


class ResearchSubAgent(SubAgent):
    """
    研究型子 Agent
    
    专门处理研究类任务，包括：
    - 信息收集
    - 资料整理
    - 研究报告生成
    
    特点：
    - 结构化输出
    - 多角度分析
    - 引用来源
    """
    
    name = "research"
    description = "研究型子 Agent，处理信息收集和研究任务"
    
    def __init__(self, model: Any, **kwargs):
        super().__init__(model, **kwargs)
        self.max_tokens = kwargs.get('max_tokens', 4000)
        self.search_tool = kwargs.get('search_tool')
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> SubAgentResult:
        """
        执行研究任务
        
        Args:
            task: 研究主题或问题
            context: 上下文信息
            
        Returns:
            SubAgentResult: 研究结果
        """
        try:
            # 步骤1：信息收集（如果有搜索工具）
            search_results = []
            if self.search_tool:
                search_results = await self._search_info(task)
            
            # 步骤2：分析和整理
            analysis = await self._analyze(task, search_results)
            
            # 步骤3：生成报告
            report = await self._generate_report(task, analysis, search_results)
            
            return SubAgentResult.success_result(
                output=report,
                data={
                    "task_type": "research",
                    "task": task,
                    "search_results_count": len(search_results)
                }
            )
            
        except Exception as e:
            return SubAgentResult.error_result(
                error=str(e),
                output=f"研究任务执行失败: {e}"
            )
    
    async def _search_info(self, query: str) -> List[Dict[str, str]]:
        """搜索相关信息"""
        try:
            if hasattr(self.search_tool, 'invoke'):
                result = self.search_tool.invoke(query)
                return [{"source": "search", "content": str(result)}]
            return []
        except Exception:
            return []
    
    async def _analyze(self, task: str, search_results: List[Dict]) -> str:
        """分析信息"""
        from langchain_core.messages import HumanMessage
        
        prompt = f"""请对以下研究主题进行分析：

主题：{task}

{'参考信息：' + chr(10).join([r.get('content', '') for r in search_results]) if search_results else '请基于你的知识进行分析。'}

请从以下几个角度分析：
1. 核心概念解释
2. 主要观点和论据
3. 相关背景和上下文
4. 潜在影响和意义

以结构化方式输出分析结果。"""
        
        response = self.model.invoke(
            [HumanMessage(content=prompt)],
            max_tokens=self.max_tokens
        )
        
        return response.content if hasattr(response, 'content') else str(response)
    
    async def _generate_report(
        self, 
        task: str, 
        analysis: str, 
        search_results: List[Dict]
    ) -> str:
        """生成研究报告"""
        report = []
        report.append(f"# 研究报告：{task}\n")
        report.append("## 摘要\n")
        report.append(f"本报告对「{task}」进行了深入研究。\n")
        
        report.append("## 详细分析\n")
        report.append(analysis)
        
        if search_results:
            report.append("\n## 参考来源\n")
            for i, result in enumerate(search_results, 1):
                report.append(f"{i}. {result.get('source', '未知来源')}")
        
        report.append("\n## 结论\n")
        report.append("基于以上分析，该主题涉及多个层面，需要综合考虑各方面因素。")
        
        return "\n".join(report)
    
    def can_handle(self, task: str) -> bool:
        """判断是否适合处理研究类任务"""
        research_keywords = [
            '研究', '分析', '调查', '报告', '综述', '调研',
            'research', 'analyze', 'investigate', 'report', 'survey'
        ]
        task_lower = task.lower()
        return any(kw in task_lower for kw in research_keywords)
