"""
CodeSubAgent - 代码型子 Agent

专门处理代码相关任务
"""

from typing import Any, Dict, Optional
import subprocess
import tempfile
import os

from ollamapilot.harness.subagents.base import SubAgent, SubAgentResult


class CodeSubAgent(SubAgent):
    """
    代码型子 Agent
    
    专门处理代码相关任务，包括：
    - 代码生成
    - 代码解释
    - 代码调试
    - 代码优化
    
    特点：
    - 支持多种编程语言
    - 可执行生成的代码
    - 提供详细注释
    """
    
    name = "code"
    description = "代码型子 Agent，处理代码生成、解释和调试"
    
    def __init__(self, model: Any, **kwargs):
        super().__init__(model, **kwargs)
        self.max_tokens = kwargs.get('max_tokens', 4000)
        self.supported_languages = ['python', 'javascript', 'java', 'cpp', 'go', 'rust']
    
    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> SubAgentResult:
        """
        执行代码任务
        
        Args:
            task: 代码任务描述
            context: 上下文信息
            
        Returns:
            SubAgentResult: 执行结果
        """
        try:
            # 判断任务类型
            task_type = self._detect_task_type(task)
            
            if task_type == 'generate':
                return await self._generate_code(task, context)
            elif task_type == 'explain':
                return await self._explain_code(task, context)
            elif task_type == 'debug':
                return await self._debug_code(task, context)
            else:
                return await self._general_code_help(task, context)
                
        except Exception as e:
            return SubAgentResult.error_result(
                error=str(e),
                output=f"代码任务执行失败: {e}"
            )
    
    def _detect_task_type(self, task: str) -> str:
        """检测任务类型"""
        task_lower = task.lower()
        
        if any(kw in task_lower for kw in ['生成', '编写', '写', 'create', 'write', 'generate']):
            return 'generate'
        elif any(kw in task_lower for kw in ['解释', '说明', 'explain', 'describe']):
            return 'explain'
        elif any(kw in task_lower for kw in ['调试', '修复', 'bug', 'debug', 'fix']):
            return 'debug'
        else:
            return 'general'
    
    async def _generate_code(self, task: str, context: Optional[Dict]) -> SubAgentResult:
        """生成代码"""
        from langchain_core.messages import HumanMessage
        
        # 提取编程语言
        language = self._extract_language(task) or 'python'
        
        prompt = f"""请生成 {language} 代码来完成以下任务：

任务：{task}

要求：
1. 代码要完整、可运行
2. 添加必要的注释
3. 包含错误处理
4. 提供使用示例

请直接输出代码，使用 markdown 代码块格式。"""
        
        response = self.model.invoke(
            [HumanMessage(content=prompt)],
            max_tokens=self.max_tokens
        )
        
        code = response.content if hasattr(response, 'content') else str(response)
        
        # 尝试执行代码（如果是 Python）
        execution_result = None
        if language == 'python':
            execution_result = await self._execute_python_code(code)
        
        output = f"生成的代码：\n\n{code}"
        if execution_result:
            output += f"\n\n执行结果：\n{execution_result}"
        
        return SubAgentResult.success_result(
            output=output,
            data={
                "task_type": "code_generation",
                "language": language,
                "executed": language == 'python'
            }
        )
    
    async def _explain_code(self, task: str, context: Optional[Dict]) -> SubAgentResult:
        """解释代码"""
        from langchain_core.messages import HumanMessage
        
        prompt = f"""请详细解释以下代码：

{task}

请从以下几个方面解释：
1. 代码功能概述
2. 关键逻辑分析
3. 使用的算法或数据结构
4. 潜在问题或优化建议
5. 使用示例

请用中文回答。"""
        
        response = self.model.invoke(
            [HumanMessage(content=prompt)],
            max_tokens=self.max_tokens
        )
        
        explanation = response.content if hasattr(response, 'content') else str(response)
        
        return SubAgentResult.success_result(
            output=explanation,
            data={"task_type": "code_explanation"}
        )
    
    async def _debug_code(self, task: str, context: Optional[Dict]) -> SubAgentResult:
        """调试代码"""
        from langchain_core.messages import HumanMessage
        
        prompt = f"""请帮助调试以下代码：

{task}

请：
1. 分析可能的错误原因
2. 提供修复后的代码
3. 解释修复方案
4. 提供测试建议

请用中文回答。"""
        
        response = self.model.invoke(
            [HumanMessage(content=prompt)],
            max_tokens=self.max_tokens
        )
        
        debug_result = response.content if hasattr(response, 'content') else str(response)
        
        return SubAgentResult.success_result(
            output=debug_result,
            data={"task_type": "code_debugging"}
        )
    
    async def _general_code_help(self, task: str, context: Optional[Dict]) -> SubAgentResult:
        """一般性代码帮助"""
        from langchain_core.messages import HumanMessage
        
        prompt = f"""请帮助解决以下编程问题：

{task}

请提供：
1. 问题分析
2. 解决方案
3. 代码示例（如适用）
4. 最佳实践建议

请用中文回答。"""
        
        response = self.model.invoke(
            [HumanMessage(content=prompt)],
            max_tokens=self.max_tokens
        )
        
        help_content = response.content if hasattr(response, 'content') else str(response)
        
        return SubAgentResult.success_result(
            output=help_content,
            data={"task_type": "code_help"}
        )
    
    def _extract_language(self, task: str) -> Optional[str]:
        """提取编程语言"""
        task_lower = task.lower()
        for lang in self.supported_languages:
            if lang in task_lower:
                return lang
        return None
    
    async def _execute_python_code(self, code: str) -> Optional[str]:
        """执行 Python 代码并返回结果"""
        try:
            # 提取代码块
            import re
            code_match = re.search(r'```python\n(.*?)\n```', code, re.DOTALL)
            if code_match:
                code = code_match.group(1)
            
            # 创建临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # 执行代码
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # 清理临时文件
            os.unlink(temp_file)
            
            output = []
            if result.stdout:
                output.append(f"输出：\n{result.stdout}")
            if result.stderr:
                output.append(f"错误：\n{result.stderr}")
            if result.returncode != 0:
                output.append(f"退出码：{result.returncode}")
            
            return "\n".join(output) if output else "代码执行成功，无输出"
            
        except Exception as e:
            return f"代码执行失败：{e}"
    
    def can_handle(self, task: str) -> bool:
        """判断是否适合处理代码类任务"""
        code_keywords = [
            '代码', '编程', '程序', '函数', '类', '算法',
            'python', 'java', 'javascript', 'cpp', 'go', 'rust',
            'code', 'program', 'function', 'class', 'algorithm',
            'bug', 'debug', 'error', 'fix'
        ]
        task_lower = task.lower()
        return any(kw in task_lower for kw in code_keywords)
