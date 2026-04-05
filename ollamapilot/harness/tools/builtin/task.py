"""
TaskTool - 任务分解工具

借鉴 DeerFlow 的 Task 工具设计
支持将复杂任务分解为子任务并委托给子 Agent
"""

import asyncio
from typing import Any, Callable, Dict, List, Optional
import logging

from ollamapilot.harness.tools.base import (
    Tool, ToolContext, ToolResult, ValidationResult, PermissionResult
)
from ollamapilot.harness.subagents.factory import SubAgentFactory

logger = logging.getLogger("ollamapilot.harness.tools.task")


class TaskTool(Tool):
    """
    任务分解工具
    
    将复杂任务分解为多个子任务，并委托给子 Agent 并行执行。
    
    使用场景：
    1. 复杂问题需要分解处理
    2. 多维度信息需要并行收集
    3. 任务需要专业化分工
    
    特点：
    - 自动任务分解
    - 支持并行执行（max 3 并发）
    - 结果自动整合
    """
    
    name = "task"
    description = "将复杂任务分解为子任务并委托执行，支持并行处理（最多3个并发）"
    
    # 最大并发数
    MAX_CONCURRENT = 3
    
    def __init__(self, subagent_factory: Optional[SubAgentFactory] = None):
        super().__init__()
        self.subagent_factory = subagent_factory or SubAgentFactory()
        self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT)
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "description": {
                    "type": "string",
                    "description": "任务描述"
                },
                "subtasks": {
                    "type": "array",
                    "description": "子任务列表（可选，如不提供则自动分解）",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string", "description": "子任务名称"},
                            "description": {"type": "string", "description": "子任务描述"},
                            "agent_type": {"type": "string", "description": "使用的子Agent类型（可选）"}
                        },
                        "required": ["name", "description"]
                    }
                },
                "context": {
                    "type": "object",
                    "description": "任务上下文信息（可选）"
                }
            },
            "required": ["description"]
        }
    
    async def validate(self, input_data: Dict[str, Any]) -> ValidationResult:
        """验证输入"""
        base_result = await super().validate(input_data)
        if not base_result.valid:
            return base_result
        
        description = input_data.get('description', '')
        if len(description) < 5:
            return ValidationResult.failure("任务描述太短，请提供更详细的描述")
        
        return ValidationResult.success()
    
    async def execute(
        self, 
        input_data: Dict[str, Any], 
        context: ToolContext,
        on_progress: Optional[Callable[[str, float], None]] = None
    ) -> ToolResult:
        """
        执行任务分解和委托
        
        Args:
            input_data: 包含 description, subtasks(可选), context(可选)
            context: 执行上下文
            on_progress: 进度回调
        """
        description = input_data.get('description', '')
        subtasks = input_data.get('subtasks', [])
        task_context = input_data.get('context', {})
        
        # 如果没有提供子任务，自动分解
        if not subtasks:
            if on_progress:
                on_progress("正在分解任务...", 0.1)
            
            subtasks = await self._decompose_task(description, context)
        
        if not subtasks:
            return ToolResult.error_result("无法分解任务，请提供更详细的描述")
        
        # 限制子任务数量
        if len(subtasks) > self.MAX_CONCURRENT:
            logger.warning(f"子任务数量({len(subtasks)})超过最大并发数，只处理前{self.MAX_CONCURRENT}个")
            subtasks = subtasks[:self.MAX_CONCURRENT]
        
        if on_progress:
            on_progress(f"开始执行 {len(subtasks)} 个子任务...", 0.2)
        
        # 并行执行子任务（带并发限制）
        results = await self._execute_subtasks(
            subtasks, 
            task_context, 
            context,
            on_progress
        )
        
        if on_progress:
            on_progress("整合结果...", 0.9)
        
        # 整合结果
        output = self._merge_results(description, subtasks, results)
        
        if on_progress:
            on_progress("完成", 1.0)
        
        return ToolResult.success_result(
            output,
            {
                "total_subtasks": len(subtasks),
                "successful": sum(1 for r in results if r.get('success')),
                "failed": sum(1 for r in results if not r.get('success'))
            }
        )
    
    async def _decompose_task(
        self, 
        description: str, 
        context: ToolContext
    ) -> List[Dict[str, Any]]:
        """
        自动分解任务
        
        使用 LLM 将复杂任务分解为子任务
        """
        try:
            # 这里简化实现，实际可以使用 LLM 进行智能分解
            # 根据任务描述提取关键词，生成子任务
            
            # 示例：如果是研究类任务，分解为收集、分析、总结
            if any(kw in description for kw in ['研究', '分析', '调查', '报告']):
                return [
                    {"name": "信息收集", "description": f"收集关于'{description}'的基础信息"},
                    {"name": "深入分析", "description": f"分析'{description}'的关键要点"},
                    {"name": "总结归纳", "description": f"总结'{description}'的核心结论"}
                ]
            
            # 示例：如果是比较类任务，分解为各个方面
            if any(kw in description for kw in ['比较', '对比', 'vs', '区别']):
                return [
                    {"name": "收集A方信息", "description": f"收集第一个比较对象的信息"},
                    {"name": "收集B方信息", "description": f"收集第二个比较对象的信息"},
                    {"name": "对比分析", "description": f"对比分析两者的差异"}
                ]
            
            # 默认分解：直接执行
            return [
                {"name": "主任务", "description": description}
            ]
            
        except Exception as e:
            logger.error(f"任务分解失败: {e}")
            return [{"name": "主任务", "description": description}]
    
    async def _execute_subtasks(
        self,
        subtasks: List[Dict[str, Any]],
        task_context: Dict[str, Any],
        tool_context: ToolContext,
        on_progress: Optional[Callable[[str, float], None]]
    ) -> List[Dict[str, Any]]:
        """
        执行子任务（带并发限制）
        
        使用信号量控制并发数为 max 3
        """
        async def execute_single(idx: int, subtask: Dict[str, Any]) -> Dict[str, Any]:
            async with self._semaphore:  # 限制并发
                name = subtask.get('name', f'子任务{idx+1}')
                description = subtask.get('description', '')
                agent_type = subtask.get('agent_type', 'general')
                
                if on_progress:
                    on_progress(f"执行 {name}...", 0.2 + 0.6 * (idx / len(subtasks)))
                
                try:
                    # 查找或创建子 Agent
                    agent = self.subagent_factory.get(agent_type)
                    if not agent:
                        # 创建通用子 Agent
                        from ollamapilot.harness.subagents.builtin.general import GeneralSubAgent
                        agent = GeneralSubAgent(tool_context)
                        self.subagent_factory.register(agent_type, GeneralSubAgent)
                        self.subagent_factory._instances[agent_type] = agent
                    
                    # 执行子任务
                    result = await agent.execute(
                        description,
                        {**task_context, "parent_task": task_context.get('parent_task')}
                    )
                    
                    return {
                        "name": name,
                        "success": result.success,
                        "output": result.output,
                        "error": result.error
                    }
                    
                except Exception as e:
                    logger.error(f"子任务 {name} 执行失败: {e}")
                    return {
                        "name": name,
                        "success": False,
                        "output": "",
                        "error": str(e)
                    }
        
        # 并行执行所有子任务
        tasks = [execute_single(i, subtask) for i, subtask in enumerate(subtasks)]
        return await asyncio.gather(*tasks)
    
    def _merge_results(
        self,
        parent_description: str,
        subtasks: List[Dict[str, Any]],
        results: List[Dict[str, Any]]
    ) -> str:
        """整合子任务结果"""
        output = []
        output.append(f"# 任务执行结果: {parent_description}\n")
        output.append(f"共分解为 {len(subtasks)} 个子任务，并发执行（max {self.MAX_CONCURRENT}）\n")
        
        for i, (subtask, result) in enumerate(zip(subtasks, results), 1):
            name = subtask.get('name', f'子任务{i}')
            status = "✅" if result.get('success') else "❌"
            
            output.append(f"\n## {i}. {name} {status}")
            output.append(f"**描述**: {subtask.get('description', '')}")
            
            if result.get('success'):
                output.append(f"\n**结果**:\n{result.get('output', '')}")
            else:
                output.append(f"\n**错误**: {result.get('error', '未知错误')}")
        
        # 统计
        successful = sum(1 for r in results if r.get('success'))
        output.append(f"\n\n---\n**统计**: {successful}/{len(results)} 个子任务成功")
        
        return "\n".join(output)
