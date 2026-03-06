"""
ToolFormatMiddleware - 强制工具调用格式

针对小模型工具调用格式不稳定的问题，提供：
1. 强制格式示例（Few-shot）
2. 格式验证和修复
3. 多格式兼容解析
"""

import json
import re
from typing import Any, Optional
from langchain_core.messages import SystemMessage, AIMessage
from .base import AgentMiddleware, AgentState


class ToolFormatMiddleware(AgentMiddleware):
    """
    工具格式强制中间件
    
    小模型常见问题：
    - 输出格式不统一：{"name": "xxx"} vs tool_name(args)
    - JSON 格式错误：缺少引号、括号不匹配
    - 参数类型错误：字符串 vs 数字
    
    解决方案：
    - 在系统提示词中注入格式示例
    - 验证和修复模型输出的格式
    - 支持多种格式的解析
    
    示例:
        middleware = ToolFormatMiddleware(
            format_style="json",  # 强制使用 JSON 格式
            add_examples=True     # 添加 Few-shot 示例
        )
    """
    
    # 格式示例模板
    FORMAT_EXAMPLES = {
        "json": """
## 工具调用格式（JSON）- 严格要求

当你需要调用工具时，**必须只输出纯 JSON，不要有任何其他文字**。

### 正确格式：
{"name": "工具名称", "arguments": {"参数1": "值1", "参数2": "值2"}}

### 示例：
用户：查一下北京天气
你的输出（只能是这行 JSON）：
{"name": "get_weather", "arguments": {"city": "北京"}}

### ⚠️ 重要规则：
1. **只输出 JSON**，不要添加"我将..."、"首先..."等文字
2. **不要输出 markdown 代码块**（如 ```json）
3. **整个回复必须且只能是 JSON**
4. 使用紧凑格式，不要换行和缩进（除非必要）

### ❌ 错误示例：
```
我将帮你查询北京天气。
{"name": "get_weather", ...}
```

### ✅ 正确示例：
```
{"name": "get_weather", "arguments": {"city": "北京"}}
```
""",
        "function": """
## 工具调用格式（函数调用）

当你需要调用工具时，必须严格按照以下格式输出：

工具名称(参数1="值1", 参数2="值2")

示例：
用户：查一下北京天气
你的输出：
get_weather(city="北京")

重要规则：
1. 使用工具名称作为函数名
2. 参数使用 key=value 格式
3. 字符串值用双引号包裹
4. 不要添加任何其他说明文字
""",
        "xml": """
## 工具调用格式（XML）

当你需要调用工具时，必须严格按照以下格式输出：

<tool_call>
  <name>工具名称</name>
  <arguments>
    <参数1>值1</参数1>
    <参数2>值2</参数2>
  </arguments>
</tool_call>

示例：
用户：查一下北京天气
你的输出：
<tool_call>
  <name>get_weather</name>
  <arguments>
    <city>北京</city>
  </arguments>
</tool_call>
"""
    }
    
    def __init__(
        self,
        format_style: str = "json",
        add_examples: bool = True,
        auto_fix: bool = True,
        strict_mode: bool = False
    ):
        """
        初始化中间件
        
        Args:
            format_style: 格式风格 - "json" | "function" | "xml" | "auto"
            add_examples: 是否在系统提示词中添加格式示例
            auto_fix: 是否自动修复格式错误
            strict_mode: 严格模式（只接受指定格式）
        """
        self.format_style = format_style
        self.add_examples = add_examples
        self.auto_fix = auto_fix
        self.strict_mode = strict_mode
        self._fix_stats = {"total": 0, "fixed": 0, "failed": 0}
    
    def before_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        在模型调用前注入格式示例
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典
        """
        if not self.add_examples:
            return None
        
        messages = list(state.messages)
        if not messages:
            return None
        
        # 查找系统消息
        system_idx = None
        for i, msg in enumerate(messages):
            if isinstance(msg, SystemMessage):
                system_idx = i
                break
        
        # 获取格式示例
        format_guide = self.FORMAT_EXAMPLES.get(
            self.format_style, 
            self.FORMAT_EXAMPLES["json"]
        )
        
        if system_idx is not None:
            # 追加到现有系统消息
            existing_msg = messages[system_idx]
            if hasattr(existing_msg, 'content'):
                existing_msg.content = f"{existing_msg.content}\n\n{format_guide}"
        else:
            # 添加新的系统消息
            messages.insert(0, SystemMessage(content=format_guide))
        
        return {"messages": messages}
    
    def after_model(
        self, 
        state: AgentState, 
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        在模型调用后修复格式错误
        
        Args:
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新字典
        """
        if not self.auto_fix:
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
        
        content = ai_msg.content
        
        # 尝试修复格式
        fixed_content = self._fix_format(content)
        
        if fixed_content != content:
            self._fix_stats["total"] += 1
            self._fix_stats["fixed"] += 1
            messages[last_ai_idx].content = fixed_content
            return {"messages": messages}
        
        return None
    
    def _fix_format(self, content: str) -> str:
        """
        修复格式错误
        
        Args:
            content: 原始内容
            
        Returns:
            修复后的内容
        """
        # 修复1: 单引号转双引号
        content = self._fix_quotes(content)
        
        # 修复2: 补齐缺失的括号
        content = self._fix_brackets(content)
        
        # 修复3: 修复常见的 JSON 格式错误
        content = self._fix_json_errors(content)
        
        return content
    
    def _fix_quotes(self, content: str) -> str:
        """修复引号问题"""
        # 将单引号替换为双引号（但保留已在双引号内的单引号）
        # 这是一个简化版本，实际可能需要更复杂的处理
        return content.replace("'", '"')
    
    def _fix_brackets(self, content: str) -> str:
        """修复括号不匹配"""
        # 计算括号数量
        open_braces = content.count('{')
        close_braces = content.count('}')
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        
        # 补齐缺失的右括号
        if open_braces > close_braces:
            content += '}' * (open_braces - close_braces)
        if open_brackets > close_brackets:
            content += ']' * (open_brackets - close_brackets)
        
        return content
    
    def _fix_json_errors(self, content: str) -> str:
        """修复常见的 JSON 错误"""
        # 修复末尾逗号
        content = re.sub(r',(\s*[}\]])', r'\1', content)
        
        # 修复缺少的逗号（在两个值之间）
        content = re.sub(r'(\w+)\s+(\w+):', r'\1, \2:', content)
        
        return content
    
    def parse_tool_call(self, content: str) -> Optional[dict]:
        """
        解析工具调用（支持多种格式）
        
        Args:
            content: 模型输出内容
            
        Returns:
            解析结果 {"name": str, "arguments": dict} 或 None
        """
        # 尝试 JSON 格式
        result = self._parse_json_format(content)
        if result:
            return result
        
        # 尝试函数格式
        result = self._parse_function_format(content)
        if result:
            return result
        
        # 尝试 XML 格式
        result = self._parse_xml_format(content)
        if result:
            return result
        
        return None
    
    def _parse_json_format(self, content: str) -> Optional[dict]:
        """解析 JSON 格式"""
        # 匹配 ```json {...} ```
        patterns = [
            r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
            r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[\s\S]*?\})\s*\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        tool_name = match[0]
                        args_str = match[1]
                        args = json.loads(args_str)
                        return {"name": tool_name, "arguments": args}
                    else:
                        data = json.loads(match)
                        if "name" in data:
                            return {
                                "name": data["name"],
                                "arguments": data.get("arguments", {})
                            }
                except json.JSONDecodeError:
                    continue
        
        return None
    
    def _parse_function_format(self, content: str) -> Optional[dict]:
        """解析函数调用格式"""
        pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        match = re.search(pattern, content)
        
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            
            # 解析参数
            args = {}
            if args_str.strip():
                kv_pattern = r'(\w+)\s*=\s*["\']([^"\']+)["\']'
                kv_matches = re.findall(kv_pattern, args_str)
                for key, value in kv_matches:
                    args[key] = value
            
            return {"name": tool_name, "arguments": args}
        
        return None
    
    def _parse_xml_format(self, content: str) -> Optional[dict]:
        """解析 XML 格式"""
        name_match = re.search(r'<name>([^<]+)</name>', content)
        if name_match:
            tool_name = name_match.group(1)
            
            args = {}
            args_match = re.search(r'<arguments>([\s\S]*?)</arguments>', content)
            if args_match:
                args_content = args_match.group(1)
                # 解析 <key>value</key> 格式
                arg_matches = re.findall(r'<(\w+)>([^<]+)</\1>', args_content)
                for key, value in arg_matches:
                    if key != 'name':
                        args[key] = value
            
            return {"name": tool_name, "arguments": args}
        
        return None
    
    def get_stats(self) -> dict:
        """获取修复统计"""
        return {
            "format_style": self.format_style,
            "total_checked": self._fix_stats["total"],
            "fixed": self._fix_stats["fixed"],
            "failed": self._fix_stats["failed"]
        }
