"""
ReActGuidanceMiddleware - ReAct 流程引导

为小模型提供清晰的 ReAct (Reasoning + Acting) 流程指导
帮助模型理解何时思考、何时调用工具、如何总结结果
"""

from typing import Any, Optional
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from .base import AgentMiddleware, AgentState


class ReActGuidanceMiddleware(AgentMiddleware):
    """
    ReAct 流程引导中间件
    
    ReAct 模式：
    1. Reason (推理): 分析用户需求，制定计划
    2. Act (行动): 调用工具执行任务
    3. Observe (观察): 查看工具执行结果
    4. 循环: 重复直到完成任务
    
    小模型常见问题：
    - 不清楚何时应该调用工具
    - 工具调用后不知道如何总结结果
    - 缺乏清晰的思考流程
    
    解决方案：
    - 在系统提示词中注入 ReAct 流程指导
    - 为每个步骤提供明确的指令
    - 添加 Few-shot 示例
    
    示例:
        middleware = ReActGuidanceMiddleware(
            add_examples=True,
            enforce_structure=True
        )
    """
    
    # ReAct 流程指导模板
    REACT_GUIDE = """
## ReAct 工作流程指导

你必须按照以下步骤处理用户请求：

### 步骤 1: 分析需求 (Reason)
- 理解用户想要什么
- 判断是否需要调用工具
- 如果需要工具，规划调用顺序

### 步骤 2: 调用工具 (Act)
- 使用正确的格式调用工具
- 等待工具执行结果
- 不要在一次回复中调用多个工具（除非必要）

### 步骤 3: 观察结果 (Observe)
- 分析工具返回的结果
- 判断任务是否完成
- 如果未完成，回到步骤1继续

### 步骤 4: 总结回复
- 当任务完成时，给用户清晰的总结
- 不要只返回原始数据，要解释含义

---

## 决策规则

**何时调用工具？**
- 用户需要获取实时信息（天气、新闻等）
- 用户需要操作文件或执行命令
- 用户需要搜索网络信息
- 用户需要计算或数据处理

**何时直接回复？**
- 用户只是问候或闲聊
- 用户询问你的能力
- 问题不需要外部信息即可回答

---

## 工具调用格式（关键！）

当你决定调用工具时，**必须严格按照以下格式，只输出 JSON，不要有任何其他文字**：

```json
{"name": "工具名称", "arguments": {"参数名": "参数值"}}
```

⚠️ **严格要求**：
1. **只输出 JSON**，不要添加"我将..."、"首先..."等计划性文字
2. **不要输出 markdown 代码块标记**（如 ```json），直接输出纯 JSON
3. **确保 JSON 格式正确**，使用双引号
4. 如果你需要调用工具，**整个回复必须且只能是 JSON 格式**

✅ **正确示例**：
```
{"name": "web_search", "arguments": {"query": "GitHub原理"}}
```

❌ **错误示例**：
```
我将搜索 GitHub 的原理信息。
{"name": "web_search", ...}
```

**判断标准**：
- 如果需要获取外部信息或执行操作 → **输出纯 JSON**
- 如果只是回答已知信息 → **直接回复文字**
"""
    
    # Few-shot 示例
    FEW_SHOT_EXAMPLES = """
---

## 示例对话

### 示例 1: 需要调用工具

用户：北京今天天气怎么样？

分析：用户询问天气，需要获取实时信息，需要调用工具。

你的输出（纯 JSON，不要其他文字）：
```
{"name": "get_weather", "arguments": {"city": "北京"}}
```

工具返回：{"temperature": "25°C", "condition": "晴"}

你的回复：
北京今天天气晴朗，气温25°C，适合外出！

### 示例 2: 不需要调用工具

用户：你好

分析：这是问候语，不需要调用工具，直接回复。

你的回复：
你好！有什么我可以帮助你的吗？

### 示例 3: 多步骤任务（关键示例）

用户：帮我搜索 Python 教程，然后保存到文件

分析：这是一个多步骤任务，需要先搜索，再保存。

步骤 1（调用搜索工具，只输出 JSON）：
```
{"name": "web_search", "arguments": {"query": "Python 教程"}}
```

工具返回：[搜索结果...]

步骤 2（调用保存工具，只输出 JSON）：
```
{"name": "write_file", "arguments": {"file_path": "python_tutorial.md", "content": "# Python 教程\n\n[搜索到的内容...]"}}
```

工具返回：文件写入成功

你的回复：
我已经帮你搜索了 Python 教程并保存到 python_tutorial.md 文件中。

---

## 重要提醒

❌ **绝对不要这样做**：
```
我将帮你搜索 Python 教程。
{"name": "web_search", ...}
```

✅ **必须这样做**：
```
{"name": "web_search", "arguments": {"query": "Python 教程"}}
```

**记住**：需要调用工具时，整个回复只能是纯 JSON，不要有任何解释性文字！
"""
    
    def __init__(
        self,
        add_guide: bool = True,
        add_examples: bool = True,
        enforce_structure: bool = False,
        step_by_step: bool = True
    ):
        """
        初始化中间件
        
        Args:
            add_guide: 是否添加 ReAct 流程指导
            add_examples: 是否添加 Few-shot 示例
            enforce_structure: 是否强制要求结构化输出
            step_by_step: 是否要求模型分步骤思考
        """
        self.add_guide = add_guide
        self.add_examples = add_examples
        self.enforce_structure = enforce_structure
        self.step_by_step = step_by_step
        self._guide_injected = False
    
    def before_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        在模型调用前注入 ReAct 指导
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典
        """
        if not self.add_guide:
            return None
        
        messages = list(state.messages)
        if not messages:
            return None
        
        # 只在第一次调用时注入指导
        if self._guide_injected:
            # 后续调用只添加步骤提示
            return self._add_step_hint(messages)
        
        # 构建指导内容
        guide_content = self.REACT_GUIDE
        
        if self.add_examples:
            guide_content += self.FEW_SHOT_EXAMPLES
        
        if self.step_by_step:
            guide_content += """
---

## 当前任务

请按照 ReAct 流程处理用户的请求。
记住：先思考，再行动，观察结果，最后总结。
"""
        
        # 查找系统消息
        system_idx = None
        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                system_idx = i
                break
        
        if system_idx is not None:
            # 追加到现有系统消息
            existing_msg = messages[system_idx]
            if hasattr(existing_msg, 'content'):
                existing_msg.content = f"{existing_msg.content}\n\n{guide_content}"
        else:
            # 添加新的系统消息
            messages.insert(0, SystemMessage(content=guide_content))
        
        self._guide_injected = True
        
        return {"messages": messages}
    
    def _add_step_hint(self, messages: list) -> Optional[dict[str, Any]]:
        """
        根据当前状态添加步骤提示
        
        Args:
            messages: 消息列表
            
        Returns:
            状态更新或 None
        """
        if not messages:
            return None
        
        # 分析最后一条消息的类型
        last_msg = messages[-1]
        
        # 如果最后一条是工具返回，提示模型总结
        if hasattr(last_msg, 'type') and last_msg.type == 'tool':
            hint = SystemMessage(content="""
[系统提示] 工具执行完成。请根据结果：
1. 分析工具返回的数据
2. 判断任务是否完成
3. 如果完成，给用户清晰的总结
4. 如果未完成，继续调用需要的工具
""")
            messages.append(hint)
            return {"messages": messages}
        
        # 如果最后一条是 AI 的工具调用，提示等待结果
        if isinstance(last_msg, AIMessage):
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                hint = SystemMessage(content="""
[系统提示] 你已调用工具，请等待工具执行结果，然后根据结果决定下一步。
""")
                messages.append(hint)
                return {"messages": messages}
        
        return None
    
    def after_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        在模型调用后检查输出结构
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典
        """
        if not self.enforce_structure:
            return None
        
        messages = list(state.messages)
        if not messages:
            return None
        
        # 获取最后一条 AI 消息
        last_ai_idx = -1
        for i in range(len(messages) - 1, -1, -1):
            if isinstance(messages[i], AIMessage):
                last_ai_idx = i
                break
        
        if last_ai_idx < 0:
            return None
        
        ai_msg = messages[last_ai_idx]
        if not hasattr(ai_msg, 'content'):
            return None
        
        content = ai_msg.content.strip()
        
        # 检查是否符合 ReAct 结构
        if self._should_call_tool(content):
            # 模型应该调用工具但没有正确格式
            if not self._is_valid_tool_format(content):
                # 尝试修复格式
                fixed_content = self._fix_tool_format(content)
                if fixed_content != content:
                    messages[last_ai_idx].content = fixed_content
                    return {"messages": messages}
        
        return None
    
    def _should_call_tool(self, content: str) -> bool:
        """
        判断模型是否应该调用工具
        
        Args:
            content: 模型输出内容
            
        Returns:
            是否应该调用工具
        """
        # 简单启发式：如果包含某些关键词，可能需要调用工具
        tool_keywords = [
            "调用", "使用工具", "搜索", "查询", "获取",
            "call", "use tool", "search", "query", "fetch"
        ]
        
        content_lower = content.lower()
        return any(kw in content_lower for kw in tool_keywords)
    
    def _is_valid_tool_format(self, content: str) -> bool:
        """
        检查是否是有效的工具调用格式
        
        Args:
            content: 模型输出内容
            
        Returns:
            是否有效
        """
        # 检查是否包含 JSON 格式的工具调用
        import json
        import re
        
        # 尝试提取 JSON
        json_patterns = [
            r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
            r'\{\s*"name"\s*:\s*"[^"]+"[\s\S]*\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    data = json.loads(match) if isinstance(match, str) else json.loads(match[0])
                    if "name" in data:
                        return True
                except json.JSONDecodeError:
                    continue
        
        return False
    
    def _fix_tool_format(self, content: str) -> str:
        """
        尝试修复工具调用格式
        
        Args:
            content: 原始内容
            
        Returns:
            修复后的内容
        """
        # 如果内容包含自然语言描述和工具调用意图
        # 尝试提取并包装成正确格式
        
        # 简单策略：如果内容没有 JSON 格式，提示模型重新输出
        if "```json" not in content and "{\"name\":" not in content:
            return content + """

[系统提示] 你似乎想调用工具，但格式不正确。请使用以下格式：
```json
{
  "name": "工具名称",
  "arguments": {
    "参数": "值"
  }
}
```
"""
        
        return content
