"""
Agent - 智能体（小模型优化版）

针对本地小模型优化的智能体架构：
- 简化决策流程，减少模型负担
- 支持手工工具调用解析（小模型 bind_tools 不稳定）
- 可配置的提示词模板
- 渐进式 Skill 加载
"""

import re
import json
import time
from typing import List, Optional, Dict, Any, Callable
from datetime import datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import BaseTool

from .skill import SkillRouter, format_tools_for_small_model, SkillRegistry, discover_skills_metadata
from .middleware import AgentMiddleware, MiddlewareChain, AgentState
from .skill import AdaptiveSkillLoader, SkillChunker
from .model import create_streaming_handler, StreamingProcessor
from .tool import ToolLoader


class ToolCallParser:
    """
    工具调用解析器（针对小模型优化）
    
    支持多种工具调用格式：
    1. JSON 格式：{"name": "tool_name", "arguments": {...}}
    2. 函数格式：tool_name(arg1=value1, arg2=value2)
    3. 代码块格式：```tool\n{...}\n```
    4. 自然语言中的工具调用
    """
    
    @staticmethod
    def parse(response: str, available_tools: List[str]) -> Optional[Dict[str, Any]]:
        """
        从模型响应中解析工具调用
        
        Args:
            response: 模型响应文本
            available_tools: 可用工具名称列表
            
        Returns:
            解析结果 {"tool": str, "arguments": dict} 或 None
        """
        # 尝试多种解析策略
        parsers = [
            ToolCallParser._parse_json_format,
            ToolCallParser._parse_function_format,
            ToolCallParser._parse_codeblock_format,
            ToolCallParser._parse_inline_format,
        ]
        
        for parser in parsers:
            result = parser(response, available_tools)
            if result:
                return result
        
        return None
    
    @staticmethod
    def _parse_json_format(response: str, available_tools: List[str]) -> Optional[Dict[str, Any]]:
        """解析 JSON 格式"""
        # 首先尝试直接解析整个响应（紧凑格式）
        response = response.strip()
        
        # 尝试直接解析纯 JSON（整个响应就是一个 JSON 对象）
        try:
            data = json.loads(response)
            if isinstance(data, dict):
                tool_name = data.get("name") or data.get("tool")
                args = data.get("arguments") or data.get("args") or {}
                if tool_name and tool_name in available_tools:
                    return {"tool": tool_name, "arguments": args}
        except json.JSONDecodeError:
            pass
        
        # 匹配 ```json {...} ``` 或直接的 JSON
        patterns = [
            r'```(?:json)?\s*(\{[\s\S]*?\})\s*```',
            r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[\s\S]*?\})\s*\}',
            r'\{\s*"tool"\s*:\s*"([^"]+)"\s*,\s*"args"\s*:\s*(\{[\s\S]*?\})\s*\}',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                try:
                    if isinstance(match, tuple):
                        tool_name = match[0]
                        args_str = match[1]
                        if tool_name in available_tools:
                            return {"tool": tool_name, "arguments": json.loads(args_str)}
                    else:
                        data = json.loads(match)
                        tool_name = data.get("name") or data.get("tool")
                        args = data.get("arguments") or data.get("args") or {}
                        if tool_name in available_tools:
                            return {"tool": tool_name, "arguments": args}
                except json.JSONDecodeError:
                    continue
        
        return None
    
    @staticmethod
    def _parse_function_format(response: str, available_tools: List[str]) -> Optional[Dict[str, Any]]:
        """解析函数调用格式：tool_name(key=value)"""
        pattern = r'(\w+)\s*\(\s*([^)]*)\s*\)'
        matches = re.findall(pattern, response)
        
        for tool_name, args_str in matches:
            if tool_name not in available_tools:
                continue
            
            # 解析参数
            args = {}
            if args_str.strip():
                # 尝试解析 key=value 格式
                kv_pattern = r'(\w+)\s*=\s*["\']?([^,"\']+)["\']?'
                kv_matches = re.findall(kv_pattern, args_str)
                for key, value in kv_matches:
                    args[key] = value.strip()
            
            return {"tool": tool_name, "arguments": args}
        
        return None
    
    @staticmethod
    def _parse_codeblock_format(response: str, available_tools: List[str]) -> Optional[Dict[str, Any]]:
        """解析代码块格式"""
        # 匹配 ```tool\nname: tool_name\nargs: {...}\n```
        pattern = r'```tool\s*\nname:\s*(\w+)\s*\nargs:\s*(\{[^}]*\})\s*\n```'
        match = re.search(pattern, response)
        
        if match:
            tool_name = match.group(1)
            args_str = match.group(2)
            
            if tool_name in available_tools:
                try:
                    args = json.loads(args_str)
                    return {"tool": tool_name, "arguments": args}
                except json.JSONDecodeError:
                    pass
        
        return None
    
    @staticmethod
    def _parse_inline_format(response: str, available_tools: List[str]) -> Optional[Dict[str, Any]]:
        """解析内联格式（自然语言中的工具调用）"""
        # 匹配 "使用 tool_name 工具，参数是 {...}"
        for tool_name in available_tools:
            patterns = [
                rf'使用\s*{tool_name}\s*工具\s*[,，]?\s*参数[是为:]+\s*(\{{[^}}]*\}})',
                rf'调用\s*{tool_name}\s*[,，]?\s*(\{{[^}}]*\}})',
                rf'{tool_name}\s*[:：]\s*(\{{[^}}]*\}})',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE)
                if match:
                    try:
                        args_str = match.group(1)
                        args = json.loads(args_str)
                        return {"tool": tool_name, "arguments": args}
                    except json.JSONDecodeError:
                        continue
        
        return None


class AgentConfig:
    """智能体配置"""
    
    def __init__(
        self,
        # 执行策略
        use_manual_tool_parsing: bool = True,  # 使用手工工具解析（小模型推荐）
        max_iterations: int = 10,  # 最大迭代次数
        max_no_output_seconds: int = 60,  # 卡住检测时间
        
        # Skill 策略
        skill_load_strategy: str = "gradual",  # gradual（渐进）/ full（完整）
        enable_self_assessment: bool = False,  # 小模型关闭自评估，减少负担
        
        # 提示词模板（可配置）
        system_prompt_template: Optional[str] = None,
        tool_prompt_template: Optional[str] = None,
        skill_prompt_template: Optional[str] = None,
    ):
        self.use_manual_tool_parsing = use_manual_tool_parsing
        self.max_iterations = max_iterations
        self.max_no_output_seconds = max_no_output_seconds
        self.skill_load_strategy = skill_load_strategy
        self.enable_self_assessment = enable_self_assessment
        
        # 默认提示词模板
        self.system_prompt_template = system_prompt_template or self._default_system_prompt()
        self.tool_prompt_template = tool_prompt_template or self._default_tool_prompt()
        self.skill_prompt_template = skill_prompt_template or self._default_skill_prompt()
    
    def _default_system_prompt(self) -> str:
        return """你是一个智能助手，帮助用户完成各种任务。

重要规则：
1. 理解用户需求后，直接调用合适的工具
2. 如果不需要工具，直接回答
3. 完成任务后，给出简洁的结果说明

请直接思考并回答，不需要使用特殊格式。
"""
    
    def _default_tool_prompt(self) -> str:
        return """可用工具：
{tools_desc}

## 工具调用规则（重要！）

**当你需要调用工具时，必须只输出 JSON 格式，不要有任何其他文字。**

### 正确格式（纯 JSON）：
{{"name": "工具名", "arguments": {{"参数": "值"}}}}

### 严格要求：
1. **只输出 JSON**，不要添加"我将..."、"首先..."等计划性文字
2. **不要输出 markdown 代码块标记**（如 ```json）
3. **整个回复必须且只能是 JSON 格式**
4. 如果不需要调用工具，直接回复文字即可

### 示例：
✅ 正确：{{"name": "web_search", "arguments": {{"query": "Python"}}}}
❌ 错误：我将搜索 Python 的信息。{{"name": "web_search", ...}}

**记住：需要工具时，只输出 JSON！**"""
    
    def _default_skill_prompt(self) -> str:
        return """任务指南：
{skill_content}

请按照以上指南完成任务。
直接执行，不要过多解释。"""


class Agent:
    """
    智能体（小模型优化版）
    
    核心设计原则：
    1. 简化流程：减少模型决策负担
    2. 灵活解析：支持多种工具调用格式
    3. 渐进加载：根据模型能力加载 Skill 内容
    4. 可配置：提示词和策略可自定义
    """
    
    def __init__(
        self,
        model,
        skill_router: SkillRouter,
        skill_registry: SkillRegistry,
        middleware: Optional[List[AgentMiddleware]] = None,
        config: Optional[AgentConfig] = None,
        mcp_config_path: Optional[str] = None,
    ):
        self.model = model
        self.skill_router = skill_router
        self.skill_registry = skill_registry
        self.middleware_chain = MiddlewareChain(middleware or [])
        self.config = config or AgentConfig()
        self.conversation_history: List[Dict[str, Any]] = []
        # 工具加载器
        self.tool_loader = ToolLoader()
    
    def invoke(self, user_input: str, verbose: bool = True) -> str:
        """
        执行用户请求
        
        简化流程：
        1. 路由决策（是否需要 Skill）
        2. 加载 Skill（根据策略）
        3. 执行（支持手工工具解析）
        4. 返回结果
        """
        if verbose:
            print(f"\n{'='*50}")
            print(f"📝 用户: {user_input}")
            print(f"{'='*50}")
        
        # 步骤1: 路由决策
        skill_name = self._route(user_input, verbose)
        
        if not skill_name:
            # 普通对话
            return self._chat(user_input, verbose)
        
        # 步骤2: 加载 Skill（传入用户输入用于智能检索）
        skill_content = self._load_skill(skill_name, user_input, verbose)
        
        # 步骤3: 执行
        return self._execute(user_input, skill_name, skill_content, verbose)
    
    def _route(self, user_input: str, verbose: bool) -> Optional[str]:
        """
        路由决策 - 统一由模型决策
        
        将触发词匹配信息作为参考，让模型一次性做出最优决策
        """
        if verbose:
            print("\n🔍 分析意图...")
        
        # 找出所有匹配的触发词（作为参考信息）
        matched_triggers = self._find_matched_triggers(user_input)
        
        # 让模型统一决策
        skill_name = self._model_decide_skill(user_input, matched_triggers, verbose)
        
        if verbose:
            if skill_name:
                print(f"✅ 决策 Skill: {skill_name}")
            else:
                print("💬 普通对话")
        
        return skill_name
    
    def _find_matched_triggers(self, user_input: str) -> Dict[str, List[str]]:
        """找出用户输入中匹配的所有 Skill 触发词"""
        user_input_lower = user_input.lower()
        matched = {}
        
        for metadata in self.skill_router.list_metadata():
            if metadata.triggers:
                matched_triggers = []
                for trigger in metadata.triggers:
                    if trigger.lower() in user_input_lower:
                        matched_triggers.append(trigger)
                if matched_triggers:
                    matched[metadata.name] = matched_triggers
        
        return matched
    
    def _model_decide_skill(self, user_input: str, matched_triggers: Dict[str, List[str]], verbose: bool) -> Optional[str]:
        """
        让模型根据 Skill 描述和触发词匹配情况统一决策
        """
        # 构建触发词匹配信息
        triggers_info = ""
        if matched_triggers:
            triggers_lines = []
            for skill_name, triggers in matched_triggers.items():
                triggers_lines.append(f"  - {skill_name}: 匹配触发词 {triggers}")
            triggers_info = f"\n## 触发词匹配情况\n\n用户输入中包含以下 Skill 的触发词：\n{chr(10).join(triggers_lines)}\n\n注意：触发词匹配仅供参考，请根据用户真实意图选择最合适的 Skill。"
        
        # 从 Skill 元数据动态构建能力说明
        skill_details = []
        for meta in self.skill_router.list_metadata():
            # 标记是否有触发词匹配
            match_marker = " ★" if meta.name in matched_triggers else ""
            
            # 构建工具信息
            tools_info = ""
            if meta.tools:
                tool_names = [t.name for t in meta.tools]
                tools_info = f"\n  可用工具: {', '.join(tool_names)}"
            
            # 构建触发词信息
            triggers_list = ""
            if meta.triggers:
                triggers_list = f"\n  触发关键词: {', '.join(meta.triggers[:5])}"
            
            skill_details.append(
                f"【{meta.name}{match_marker}】\n"
                f"  描述: {meta.description}\n"
                f"  标签: {', '.join(meta.tags) if meta.tags else '无'}"
                f"{tools_info}{triggers_list}"
            )
        
        decision_prompt = f"""你是 Skill 路由决策专家。分析用户需求，选择最合适的 Skill 来完成任务。{triggers_info}

## 可用 Skill 列表（★表示触发词匹配）

{chr(10).join(skill_details)}

## 决策规则

1. **理解用户意图**: 用户真正想要完成什么任务？
2. **评估 Skill 能力**: 哪个 Skill 的功能最能满足这个需求？
3. **参考触发词**: 触发词匹配是重要参考，但不是唯一标准
4. **做出决策**:
   - 选择最能完成用户需求的 Skill → 回复 Skill 名称
   - 如果只是闲聊、问候、不需要工具 → 回复 NONE

## 示例

用户: "搜索一下今天的天气"
触发词匹配: web: ['搜索']
思考: 用户需要搜索信息，web Skill 有 web_search 工具，虽然触发词匹配 web，但确实需要搜索功能
回复: web

用户: "当前有哪些热点新闻？"
触发词匹配: 无
思考: 用户需要获取最新信息，web Skill 有搜索能力可以查找新闻
回复: web

用户: "你好"
触发词匹配: 无
思考: 普通问候，不需要任何 Skill
回复: NONE

## 现在分析

用户: {user_input}
触发词匹配: {list(matched_triggers.keys()) if matched_triggers else '无'}

请按以下格式回复:
思考: [你的分析过程，说明为什么选择这个 Skill]
回复: [Skill名称 或 NONE]"""

        try:
            # 调用模型进行决策（支持流式输出）
            messages = [HumanMessage(content=decision_prompt)]
            
            if verbose:
                print(f"📝 模型决策过程:\n{'-'*40}")
            
            # 使用流式输出显示思维过程
            content = self._stream_decision(messages, verbose)
            
            if verbose:
                print(f"\n{'-'*40}")
            
            # 解析决策结果
            decision = None
            
            # 尝试提取 "回复:" 后的内容
            if "回复:" in content:
                decision = content.split("回复:")[-1].strip()
            else:
                # 尝试最后一行
                decision = content.split('\n')[-1].strip()
            
            # 清理
            decision = decision.strip('[]"\'').split()[0] if decision else None
            
            # 验证
            if decision and decision.upper() != "NONE":
                if decision in self.skill_router._skills_metadata:
                    return decision
                # 尝试模糊匹配
                for skill_name in self.skill_router._skills_metadata.keys():
                    if skill_name in decision or decision in skill_name:
                        return skill_name
            
            return None
            
        except Exception as e:
            if verbose:
                print(f"⚠️ 模型决策失败: {e}")
            return None
    
    def _stream_decision(self, messages: List, verbose: bool) -> str:
        """
        流式输出决策过程
        
        实时显示模型的思考过程，让用户看到决策是如何做出的
        """
        response = ""
        
        if hasattr(self.model, 'stream'):
            # 流式输出
            for chunk in self.model.stream(messages):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    response += content
                    
                    if verbose:
                        print(content, end="", flush=True)
        else:
            # 非流式
            result = self.model.invoke(messages)
            response = result.content if hasattr(result, 'content') else str(result)
            
            if verbose:
                print(response)
        
        return response.strip()
    
    def _load_skill(self, skill_name: str, user_input: str, verbose: bool) -> Optional[str]:
        """加载 Skill 内容（支持自适应加载）"""
        # 获取完整内容
        full_content = self.skill_router.load_skill_content(skill_name, level="full")
        
        if not full_content:
            return None
        
        # 使用自适应加载器
        loader = AdaptiveSkillLoader()
        content = loader.load(
            skill_name=skill_name,
            content=full_content,
            query=user_input,
            strategy=self.config.skill_load_strategy if self.config.skill_load_strategy != "gradual" else None
        )
        
        if verbose and content:
            info = loader.get_loading_info(skill_name, full_content)
            print(f"📚 加载 Skill: {skill_name}")
            print(f"   策略: {info['strategy']} | 原始: {info['total_chars']}字 | 估计: ~{info['estimated_tokens']} tokens")
        
        return content
    
    def _execute(
        self, 
        user_input: str, 
        skill_name: str, 
        skill_content: Optional[str],
        verbose: bool
    ) -> str:
        """执行任务（按需加载工具）"""
        
        # 构建提示词
        system_prompt = self.config.system_prompt_template
        
        if skill_content:
            skill_prompt = self.config.skill_prompt_template.format(
                skill_content=skill_content[:3000]  # 限制长度
            )
            system_prompt += "\n\n" + skill_prompt
        
        # ===== 按需加载工具 =====
        tools = []
        
        # 1. 获取 Skill 元数据中的工具配置
        metadata = self.skill_router.get_metadata(skill_name)
        if metadata and metadata.tools:
            if verbose:
                print(f"🔧 加载工具 ({len(metadata.tools)}个):")
                for tool_config in metadata.tools:
                    print(f"   - {tool_config.name} ({tool_config.type})")
            
            # 按需加载工具
            tools = self.tool_loader.load_tools_for_skill(metadata.tools)
            
            if verbose and tools:
                print(f"✅ 成功加载 {len(tools)} 个工具")
        
        # 2. 如果没有配置工具，使用注册表中的所有工具（向后兼容）
        if not tools:
            tools = self.skill_registry.get_all_tools()
            if verbose and tools:
                print(f"📦 使用全局工具 ({len(tools)}个)")
        
        # 添加工具描述到提示词
        if tools:
            tools_desc = format_tools_for_small_model(tools)
            tool_prompt = self.config.tool_prompt_template.format(
                tools_desc=tools_desc
            )
            system_prompt += "\n\n" + tool_prompt
        
        # 执行循环
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        
        iteration = 0
        while iteration < self.config.max_iterations:
            iteration += 1
            
            if verbose:
                print(f"\n🔄 迭代 {iteration}/{self.config.max_iterations}")
            
            try:
                response = self._call_model(messages, verbose)
                
                # 检查是否包含工具调用
                tool_call = self._extract_tool_call(response, tools)
                
                if tool_call:
                    # 执行工具
                    if verbose:
                        print(f"🔧 调用工具: {tool_call['tool']}")
                    
                    result = self._execute_tool(tool_call, verbose)
                    
                    # 添加工具结果到对话
                    tool_result_msg = f"\n[工具 {tool_call['tool']} 结果]: {result}\n"
                    messages.append(AIMessage(content=response))
                    messages.append(HumanMessage(content=tool_result_msg))
                    
                    # 继续迭代，让模型处理工具结果
                    continue
                else:
                    # 没有工具调用，任务完成
                    self.conversation_history.append({"role": "user", "content": user_input})
                    self.conversation_history.append({"role": "assistant", "content": response})
                    
                    if verbose:
                        print(f"\n✅ 完成")
                    
                    return response
                    
            except Exception as e:
                if verbose:
                    print(f"\n❌ 错误: {e}")
                
                # 添加错误信息，让模型重试
                error_msg = f"执行出错: {str(e)}。请重试或换一种方式。"
                messages.append(HumanMessage(content=error_msg))
        
        return "达到最大迭代次数，任务未完成。"
    
    def _call_model(self, messages: List, verbose: bool, stream_handler: Optional[StreamingProcessor] = None) -> str:
        """调用模型（支持流式处理和思维链显示）"""
        try:
            if hasattr(self.model, 'stream'):
                # 流式输出
                response = ""
                thinking_started = False
                
                # 创建流式处理器
                if stream_handler is None and verbose:
                    stream_handler = create_streaming_handler(
                        verbose=True,
                        show_thinking=True,
                        show_tools=True
                    )
                
                # 流式输出，同时区分思维链和回答
                in_thinking_phase = True
                thinking_header_printed = False
                content_header_printed = False
                
                for chunk in self.model.stream(messages):
                    if hasattr(chunk, 'content'):
                        content = chunk.content
                        response += content
                        
                        if verbose:
                            # 获取 chunk 类型（thinking 或 content）
                            chunk_type = getattr(chunk, 'chunk_type', 'content')
                            
                            if chunk_type == 'thinking' and not thinking_header_printed:
                                # 开始显示思维链
                                print("\n💭 思维链:")
                                print("-" * 50)
                                thinking_header_printed = True
                            
                            if chunk_type == 'content' and in_thinking_phase:
                                # 从思维链切换到内容
                                in_thinking_phase = False
                                if thinking_header_printed:
                                    print("\n" + "-" * 50)
                                print("\n🤖 回答:")
                                print("-" * 50)
                                content_header_printed = True
                            
                            print(content, end="", flush=True)
                
                # 如果一直在思维链阶段，结束时添加分隔
                if in_thinking_phase and thinking_header_printed and verbose:
                    print("\n" + "-" * 50)
                
                # 刷新处理器
                if stream_handler:
                    stream_handler.flush()
                
                if verbose:
                    print()  # 换行
                
                return response
            else:
                # 非流式
                result = self.model.invoke(messages)
                content = result.content if hasattr(result, 'content') else str(result)
                
                if verbose:
                    print(f"\n🤖 {content}\n")
                
                return content
        except Exception as e:
            raise Exception(f"模型调用失败: {e}")
    
    def _extract_tool_call(
        self, 
        response: str, 
        tools: List[BaseTool]
    ) -> Optional[Dict[str, Any]]:
        """提取工具调用"""
        if not tools:
            return None
        
        available_tools = [t.name for t in tools]
        
        if self.config.use_manual_tool_parsing:
            # 使用手工解析
            return ToolCallParser.parse(response, available_tools)
        else:
            # 依赖模型的 bind_tools（大模型推荐）
            return None
    
    def _execute_tool(self, tool_call: Dict[str, Any], verbose: bool) -> str:
        """执行工具"""
        tool_name = tool_call["tool"]
        arguments = tool_call["arguments"]
        
        # 查找工具
        tools = self.skill_registry.get_all_tools()
        tool = None
        for t in tools:
            if t.name == tool_name:
                tool = t
                break
        
        if not tool:
            return f"错误: 工具 '{tool_name}' 不存在"
        
        try:
            # 执行工具
            result = tool.invoke(arguments)
            return str(result)
        except Exception as e:
            return f"工具执行错误: {e}"
    
    def _chat(self, user_input: str, verbose: bool) -> str:
        """普通对话"""
        # 使用系统提示词（包含思维链要求）
        system_prompt = self.config.system_prompt_template
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input)
        ]
        
        try:
            response = self._call_model(messages, verbose)
            
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return response
        except Exception as e:
            return f"对话出错: {e}"


def create_agent(
    model,
    skills_dir: str | Path = "skills",
    middleware: Optional[List[AgentMiddleware]] = None,
    config: Optional[AgentConfig] = None,
    enable_small_model_middlewares: bool = True,
) -> Agent:
    """
    创建智能体
    
    Args:
        model: 语言模型
        skills_dir: Skill 目录
        middleware: 中间件列表
        config: 配置（可选）
        enable_small_model_middlewares: 是否启用小模型优化中间件
        
    Returns:
        Agent 实例
    """
    # 默认配置（小模型优化）
    if config is None:
        config = AgentConfig(
            use_manual_tool_parsing=True,  # 小模型使用手工解析
            max_iterations=5,  # 减少迭代次数
            skill_load_strategy="gradual",  # 渐进加载
            enable_self_assessment=False,  # 关闭自评估
        )
    
    # 创建路由器和注册表
    router = SkillRouter()
    metadata_list = discover_skills_metadata(skills_dir)
    for metadata in metadata_list:
        router.register_metadata(metadata)
    
    registry = SkillRegistry()
    _load_skills(registry, skills_dir)
    
    # 构建中间件列表
    final_middleware = list(middleware) if middleware else []
    
    # 启用小模型优化中间件
    if enable_small_model_middlewares:
        from .middleware import (
            ReActGuidanceMiddleware,
            ToolFormatMiddleware,
            ContextCompressionMiddleware,
            DanglingToolCallMiddleware,
            ToolRetryMiddleware,
        )
        
        # 按顺序添加小模型优化中间件
        small_model_middlewares = [
            ReActGuidanceMiddleware(add_guide=True, add_examples=True),
            ToolFormatMiddleware(format_style="json", add_examples=True, auto_fix=True),
            ContextCompressionMiddleware(max_messages=20, max_tokens=3000),
            DanglingToolCallMiddleware(auto_summarize=True),
            ToolRetryMiddleware(max_retries=3, auto_fix=True),
        ]
        
        # 将用户自定义中间件放在小模型中间件之后
        final_middleware = small_model_middlewares + final_middleware
    
    # 创建 Agent
    agent = Agent(
        model=model,
        skill_router=router,
        skill_registry=registry,
        middleware=final_middleware,
        config=config,
    )
    
    return agent


def _load_skills(registry: SkillRegistry, skills_dir: str | Path) -> int:
    """加载所有 Skill"""
    import importlib.util
    
    skills_dir = Path(skills_dir)
    count = 0
    
    if not skills_dir.exists():
        return count
    
    for skill_dir in skills_dir.iterdir():
        if not skill_dir.is_dir():
            continue
        
        py_file = skill_dir / "skill.py"
        if not py_file.exists():
            continue
        
        try:
            spec = importlib.util.spec_from_file_location(
                f"skills.{skill_dir.name}", 
                py_file
            )
            if not spec or not spec.loader:
                continue
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type) 
                    and hasattr(attr, 'name')
                    and hasattr(attr, 'description')
                ):
                    try:
                        instance = attr()
                        registry.register(instance)
                        count += 1
                        break
                    except Exception:
                        pass
        except Exception:
            pass
    
    return count
