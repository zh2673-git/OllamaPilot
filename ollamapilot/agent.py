"""
Agent 核心模块 V3 - 基于 LangChain v1 + 中间件架构

核心理念：Agent + Skill + Context + Middleware 管理一切

采用 LangChain 1.0+ 最佳实践：
- init_chat_model() 统一模型初始化
- create_agent() Agent 工厂模式
- Middleware 中间件链
- 文件驱动记忆 (OpenClaw 风格)

V0.5.0 重构：简化架构，移除废弃文件依赖
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware, ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from ollamapilot.config import get_config
from ollamapilot.skills import SkillRegistry
from ollamapilot.tools.builtin import (
    read_file, write_file, list_directory, search_files,
    shell_exec, shell_script, python_exec, web_search, web_fetch
)
from ollamapilot.context.builder import ContextBuilder
from ollamapilot.context.compactor import ContextCompactor
from ollamapilot.memory.manager import MemoryManager
from ollamapilot.middlewares import (
    ContextInjectionMiddleware,
    MemoryRetrievalMiddleware,
    CompactionMiddleware,
)

logger = logging.getLogger("ollamapilot.agent")


def _detect_model_name(model: BaseChatModel) -> Optional[str]:
    """从模型实例中检测模型名称"""
    for attr in ['model', 'model_name', 'deployment_name']:
        if hasattr(model, attr):
            value = getattr(model, attr)
            if value:
                return str(value)

    if hasattr(model, 'model_kwargs'):
        kwargs = getattr(model, 'model_kwargs', {})
        if isinstance(kwargs, dict):
            for key in ['model', 'model_name']:
                if key in kwargs:
                    return str(kwargs[key])

    return None


BUILTIN_TOOLS = {
    "read_file",
    "write_file",
    "list_directory",
    "search_files",
    "shell_exec",
    "shell_script",
    "python_exec",
    "web_search",
    "web_fetch",
}


class OllamaPilotAgent:
    """
    OllamaPilot Agent - V3 版本

    基于 LangChain v1 中间件架构 + 四层 Context

    核心架构：
    - Agent 层：LangChain create_agent
    - Middleware 层：Context 注入 → 记忆检索 → 上下文压缩
    - Context 层：四层架构（L3/L2/L1/L0）
    - Memory 层：文件驱动，被 Context 统管
    """

    def __init__(
        self,
        model: BaseChatModel,
        skills_dir: Optional[str] = None,
        workspace_dir: Optional[str] = None,
        enable_memory: bool = True,
        max_tool_calls: int = 50,
        verbose: bool = True,
        checkpointer=None,
        embedding_model: Optional[str] = None,
        enable_system_memory: bool = True,
        memory_embedding_model: Optional[Any] = None,
        max_context_tokens: Optional[int] = None,
        auto_detect_context: bool = True,
        memory_storage: str = "memory",
        memory_db_path: str = "./data/sessions/conversations.db",
        **kwargs,
    ):
        self.model = model
        self.verbose = verbose
        if workspace_dir:
            self.workspace_dir = Path(workspace_dir)
        else:
            default_workspace = Path(__file__).parent / "workspace"
            if default_workspace.exists():
                self.workspace_dir = default_workspace
            else:
                self.workspace_dir = Path("./workspace")
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        # 首次启动时自动生成 workspace 模板文件
        self._ensure_workspace_files()

        detected_tokens = None
        model_name = _detect_model_name(model)
        if auto_detect_context and max_context_tokens is None:
            if model_name:
                try:
                    from ollamapilot.model_context import get_recommended_num_ctx
                    detected_tokens = get_recommended_num_ctx(model_name)
                    if verbose:
                        print(f"🎯 自动检测模型上下文: {model_name} -> {detected_tokens:,} tokens")
                except Exception:
                    if verbose:
                        print(f"⚠️ 上下文检测失败，使用默认值 8192")
                    detected_tokens = 8192
            else:
                detected_tokens = 8192

        final_max_tokens = max_context_tokens if max_context_tokens is not None else (detected_tokens or 8192)
        self.max_context_tokens = final_max_tokens

        skill_config = {}
        if embedding_model:
            skill_config["graphrag"] = {"embedding_model": embedding_model}
        if model_name:
            skill_config["graphrag"] = skill_config.get("graphrag", {})
            skill_config["graphrag"]["model_name"] = model_name

        self.skill_registry = SkillRegistry(skill_config=skill_config)

        skill_count = 0
        if skills_dir:
            skill_count = self.skill_registry.discover_skills(skills_dir)

        self.all_tools = self._get_all_tools()

        if checkpointer:
            self.checkpointer = checkpointer
        elif enable_memory:
            self.checkpointer = MemorySaver()
            if verbose:
                print(f"💾 对话记忆已启用 (内存模式)")
        else:
            self.checkpointer = None

        self.max_tool_calls = max_tool_calls

        self.memory_manager = None
        if enable_system_memory:
            memory_emb = embedding_model or memory_embedding_model
            self.memory_manager = MemoryManager(
                workspace_dir=self.workspace_dir,
                embedding_model=memory_emb,
                enable_vector_search=True,
            )
            if verbose:
                print(f"🧠 记忆管理器已启用")

        self.context_builder = ContextBuilder(
            workspace_dir=self.workspace_dir,
            enable_cache=True,
            cache_ttl=300,
        )
        self.context_builder.memory_manager = self.memory_manager

        self.compactor = ContextCompactor(
            max_tokens=final_max_tokens,
            threshold=0.8,
            preserve_recent=10,
        )

        self._active_skill_name: Optional[str] = None

        if self.verbose:
            memory_status = " | 🧠 记忆管理器已启用" if enable_system_memory else ""
            print(f"📦 已加载 {skill_count} 个 Skill | 🔧 {len(self.all_tools)} 个工具{memory_status}")

    def _get_all_tools(self) -> List[BaseTool]:
        tools = [
            read_file,
            write_file,
            list_directory,
            search_files,
            shell_exec,
            shell_script,
            python_exec,
            web_search,
            web_fetch,
        ]

        skill_tools = self.skill_registry.get_all_tools()
        tools.extend(skill_tools)

        return tools

    def _select_skill(self, query: str) -> Optional[Any]:
        """根据查询选择 Skill"""
        matches = self.skill_registry.find_skill_by_trigger(query)

        if matches:
            return self.skill_registry.get_skill(matches[0])

        default_skill = self.skill_registry.get_default_skill()
        if default_skill:
            return default_skill

        return None

    def _ensure_workspace_files(self):
        """首次启动时自动生成 workspace 模板文件"""
        templates = {
            "SOUL.md": """# SOUL - AI 的灵魂定义

## 核心使命
你是 OllamaPilot，一个智能助手，旨在帮助用户完成各种任务。

## 价值观
- 诚实：不知道就承认，不编造信息
-  helpful：尽力帮助用户解决问题
-  尊重：尊重用户的隐私和选择

## 性格特点
- 友好且专业
- 耐心细致
- 善于倾听

## 能力边界
- 可以调用工具获取信息
- 可以记住重要的事实
- 不能访问互联网（除非有搜索工具）
""",
            "IDENTITY.md": """# IDENTITY - AI 的身份定义

## 基本信息
- 名称：OllamaPilot
- 版本：v0.5.0
- 类型：智能助手

## 技术架构
- 基于 LangGraph 的 Agent 架构
- 支持多 Skill 系统
- 具备四层 Context 管理

## 运行环境
- 本地 Ollama 模型驱动
- 支持多种开源模型
""",
            "USER.md": """# USER - 用户画像

## 用户信息
- 首次使用日期：{date}

## 使用偏好
- 待添加...

## 重要记忆
- 待添加...

*此文件由 AI 自动更新，记录用户的重要信息*
""".format(date=__import__('datetime').datetime.now().strftime("%Y-%m-%d")),
            "AGENTS.md": """# AGENTS - 操作指南

## 可用工具
- 查看当前已加载的工具列表

## Skill 使用
- 根据用户查询自动选择合适的 Skill
- 支持手动切换 Skill

## 记忆管理
- 自动提取重要信息保存到记忆
- 对话中自动检索相关记忆

## 最佳实践
1. 先理解用户意图
2. 选择合适的工具
3. 基于结果回答
""",
            "MEMORY.md": """# MEMORY - 主记忆文件

## 用户基本信息

## 重要事件

## 常用偏好

*此文件由系统自动更新*
"""
        }

        for filename, content in templates.items():
            file_path = self.workspace_dir / filename
            if not file_path.exists():
                try:
                    file_path.write_text(content, encoding='utf-8')
                    if self.verbose:
                        print(f"📝 创建模板文件: {filename}")
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️ 创建 {filename} 失败: {e}")

    def _get_allowed_tools(self) -> set:
        """获取当前允许的工具集合"""
        allowed = set(BUILTIN_TOOLS)

        if self._active_skill_name:
            skill = self.skill_registry.get_skill(self._active_skill_name)
            if skill and hasattr(skill, 'get_required_tools'):
                skill_tool_names = skill.get_required_tools()
                if skill_tool_names:
                    allowed.update(skill_tool_names)

        return allowed

    def _filter_tool_calls(self, messages: List[Any]) -> List[Any]:
        """过滤不允许的工具调用"""
        allowed = self._get_allowed_tools()
        filtered = []

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                filtered_calls = []
                for call in msg.tool_calls:
                    tool_name = call.get('name', '') if isinstance(call, dict) else getattr(call, 'name', '')
                    if tool_name in allowed:
                        filtered_calls.append(call)
                    elif self.verbose:
                        print(f"🔇 过滤不允许的工具: {tool_name}")

                if filtered_calls:
                    msg = AIMessage(
                        content=msg.content,
                        tool_calls=filtered_calls,
                        **({k: v for k, v in msg.additional_kwargs.items()} if hasattr(msg, 'additional_kwargs') else {})
                    )
                else:
                    msg = AIMessage(content=msg.content or "（无可用工具）")

            filtered.append(msg)

        return filtered

    def invoke(self, query: str, thread_id: Optional[str] = None) -> str:
        """执行用户查询（中间件自动处理 Context）"""
        if self.verbose:
            print(f"🤖 用户: {query}")

        skill = self._select_skill(query)

        if skill and self.memory_manager:
            self.memory_manager.record_skill_usage(skill.name, context=query[:100])

        config = {
            "configurable": {"thread_id": thread_id or "default"},
            "recursion_limit": self.max_tool_calls + 10
        }

        messages = [HumanMessage(content=query)]

        if not hasattr(self, '_agent') or self._agent is None:
            self._agent = self._create_agent()

        result = self._agent.invoke(
            {"messages": messages},
            config
        )

        result_messages = result.get("messages", [])
        result_messages = self._filter_tool_calls(result_messages)

        response = ""
        has_tool_calls = False

        if result_messages:
            for msg in result_messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    has_tool_calls = True
                    break

            last_message = result_messages[-1]
            if hasattr(last_message, "content"):
                response = last_message.content or ""

        if has_tool_calls and not response.strip():
            if self.verbose:
                print("🔄 检测到工具调用后无回复，强制生成回复...")
            response = self._force_response(result_messages, config)

        if self.verbose and response:
            print(f"🤖 AI: {response[:200]}{'...' if len(response) > 200 else ''}")

        if self.memory_manager and hasattr(self, '_extract_and_save_memory'):
            self._extract_and_save_memory(query, response)

        return response

    def _create_agent(self):
        """创建 Agent，启用中间件链"""
        from langchain.agents import create_agent
        try:
            middlewares = []

            if hasattr(self, 'context_builder') and self.context_builder:
                middlewares.append(ContextInjectionMiddleware(self.context_builder))

            if hasattr(self, 'memory_manager') and self.memory_manager:
                middlewares.append(MemoryRetrievalMiddleware(self.context_builder))

            if hasattr(self, 'compactor') and self.compactor:
                middlewares.append(CompactionMiddleware(self.compactor))

            middlewares.append(ToolRetryMiddleware(max_retries=2))
            middlewares.append(ToolCallLimitMiddleware(run_limit=self.max_tool_calls))

            return create_agent(
                model=self.model.bind_tools(self.all_tools),
                tools=self.all_tools,
                checkpointer=self.checkpointer,
                middleware=middlewares,
            )
        except Exception as e:
            logger.warning(f"创建带中间件的 Agent 失败: {e}，回退到基础模式")
            try:
                return create_agent(
                    model=self.model.bind_tools(self.all_tools),
                    tools=self.all_tools,
                    checkpointer=self.checkpointer,
                )
            except Exception:
                return None

    async def ainvoke(self, query: str, thread_id: Optional[str] = None) -> str:
        """异步执行用户查询"""
        return await asyncio.to_thread(self.invoke, query, thread_id)

    def _force_response(self, messages: List[Any], config: Dict[str, Any]) -> str:
        """强制生成回复"""
        last_human_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg
                break

        if not last_human_msg:
            return "（无法找到用户查询）"

        force_messages = [last_human_msg]
        tool_call_found = False
        tool_results = []

        for msg in messages:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_call_found = True
                tool_names = [tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown') for tc in msg.tool_calls]
                explanation = f"我需要调用工具来获取信息：{', '.join(tool_names)}"
                force_messages.append(AIMessage(content=explanation))

            elif isinstance(msg, ToolMessage):
                content = msg.content
                if len(content) > 2000:
                    content = content[:2000] + f"\n... (内容已截断)"
                tool_results.append(ToolMessage(
                    content=content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name or "unknown"
                ))

        force_messages.extend(tool_results)

        if not tool_call_found:
            return "（工具执行完成）"

        force_messages.append(
            HumanMessage(content="请基于上述工具执行结果，直接回答用户的问题。")
        )

        if self.verbose:
            print(f"🔄 强制生成回复...")

        try:
            result = self._agent.invoke(
                {"messages": force_messages},
                config
            )

            result_messages = result.get("messages", [])
            result_messages = self._filter_tool_calls(result_messages)

            if result_messages:
                last_message = result_messages[-1]
                if hasattr(last_message, "content") and last_message.content:
                    return last_message.content
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 强制生成回复失败: {e}")

        return "（工具执行完成，但生成回复失败）"

    async def force_response_after_tool(self, thread_id: str):
        """
        工具调用后强制生成回复（异步流式）
        
        小模型有时调用工具完成但无法生成回复，这是保底机制。
        从历史中提取工具执行结果，重新构造消息并生成回复。
        """
        config = {"configurable": {"thread_id": thread_id or "default"}}

        if not hasattr(self, '_agent') or self._agent is None:
            self._agent = self._create_agent()

        history = self.get_history(thread_id) if thread_id else []
        
        last_human_msg = None
        for msg in reversed(history):
            if isinstance(msg, HumanMessage):
                last_human_msg = msg
                break

        if not last_human_msg:
            return

        force_messages = [last_human_msg]
        tool_call_found = False

        for msg in history:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                tool_call_found = True
                tool_names = [tc.get('name', 'unknown') if isinstance(tc, dict) else getattr(tc, 'name', 'unknown') for tc in msg.tool_calls]
                force_messages.append(AIMessage(content=f"我需要调用工具来获取信息：{', '.join(tool_names)}"))

            elif isinstance(msg, ToolMessage):
                content = msg.content
                if len(content) > 2000:
                    content = content[:2000] + f"\n... (内容已截断)"
                force_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=msg.tool_call_id,
                    name=msg.name or "unknown"
                ))

        if not tool_call_found:
            return

        force_messages.append(HumanMessage(content="请基于上述工具执行结果，直接回答用户的问题。"))

        try:
            for chunk in self._agent.stream({"messages": force_messages}, config):
                if chunk:
                    yield chunk
        except Exception as e:
            if self.verbose:
                print(f"⚠️ 强制生成回复失败: {e}")

    def stream(self, query: str, thread_id: Optional[str] = None):
        """流式执行（中间件自动处理 Context）"""
        config = {"configurable": {"thread_id": thread_id or "default"}}

        if not hasattr(self, '_agent') or self._agent is None:
            self._agent = self._create_agent()

        messages = [HumanMessage(content=query)]

        for chunk in self._agent.stream(
            {"messages": messages},
            config
        ):
            yield chunk

    async def astream_events(self, query: str, thread_id: Optional[str] = None):
        """异步流式事件（中间件自动处理 Context）"""
        config = {
            "configurable": {"thread_id": thread_id or "default"},
            "recursion_limit": self.max_tool_calls + 10
        }

        full_response = ""

        if not hasattr(self, '_agent') or self._agent is None:
            self._agent = self._create_agent()

        messages = [HumanMessage(content=query)]

        async for event in self._agent.astream_events(
            {"messages": messages},
            config,
            version="v1"
        ):
            if event.get("event") == "on_chat_model_stream":
                data = event.get("data", {})
                chunk = data.get("chunk", None)
                if chunk and hasattr(chunk, "content"):
                    content = chunk.content
                    if content:
                        full_response += content

            yield event

        if self.memory_manager:
            self._extract_and_save_memory(query, full_response)

    def get_history(self, thread_id: Optional[str] = None) -> List[Any]:
        """获取对话历史"""
        if not self.checkpointer:
            return []

        config = {"configurable": {"thread_id": thread_id or "default"}}

        try:
            checkpoint_tuple = self.checkpointer.get_tuple(config)
            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                checkpoint = checkpoint_tuple.checkpoint

                if "messages" in checkpoint:
                    return checkpoint["messages"]

                if "channel_values" in checkpoint:
                    channel_values = checkpoint["channel_values"]
                    if "messages" in channel_values:
                        return channel_values["messages"]

                if checkpoint_tuple.state:
                    state = checkpoint_tuple.state
                    if "messages" in state:
                        return state["messages"]
        except Exception:
            pass

        return []

    def clear_history(self, thread_id: Optional[str] = None) -> None:
        """清除对话历史"""
        if not self.checkpointer:
            return

        config = {"configurable": {"thread_id": thread_id or "default"}}

        try:
            self.checkpointer.delete(config)
        except Exception:
            pass

    def _extract_and_save_memory(self, query: str, response: str):
        """从对话中提取重要信息并保存"""
        if not self.memory_manager:
            return

        try:
            import re

            name_match = re.search(r'我叫\s*([^，。！？\n]{2,20})(?=[，。！？\n]|$)', query)
            if name_match:
                name = name_match.group(1).strip()
                if name and len(name) >= 2:
                    self._save_memory_if_new(f"用户姓名是{name}", "identity", 0.95)

            if self._should_extract_with_llm(query):
                self._extract_with_llm_async(query, response)

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 提取记忆时出错: {e}")

    def _save_memory_if_new(self, fact: str, category: str, importance: float):
        """保存记忆（如果之前没有保存过相同内容）"""
        if not self.memory_manager:
            return

        try:
            from ollamapilot.memory.types import MemoryEntry, MemoryType
            import hashlib

            entry_id = hashlib.md5(f"{category}:{fact}".encode()).hexdigest()[:12]

            existing = self.memory_manager.search(fact, top_k=5)
            fact_normalized = self._normalize_fact(fact)

            for result in existing:
                if self._normalize_fact(result.content) == fact_normalized:
                    if self.verbose:
                        print(f"   [Memory] 相似记忆已存在，跳过: {fact[:50]}...")
                    return

            entry = MemoryEntry(
                id=entry_id,
                type=MemoryType.SEMANTIC,
                content=fact,
                metadata={"category": category},
                importance=importance,
            )

            self.memory_manager._save_entry(entry)

            if self.verbose:
                print(f"💾 已保存记忆: {fact[:50]}...")

        except Exception as e:
            if self.verbose:
                print(f"   [Memory] 保存记忆时出错: {e}")

    def _normalize_fact(self, fact: str) -> str:
        """标准化事实文本"""
        import re
        prefixes = ['用户', '姓名', '名字', '是', '：', ':', ' ']
        normalized = fact.strip()
        for prefix in prefixes:
            normalized = normalized.replace(prefix, '')
        normalized = normalized.lower()
        normalized = re.sub(r'[^\w\u4e00-\u9fff]', '', normalized)
        return normalized

    def _should_extract_with_llm(self, query: str) -> bool:
        """判断是否需要使用LLM提取记忆"""
        keywords = [
            '喜欢', '爱好', '兴趣', '讨厌', '不喜欢',
            '我是', '我是做', '我的工作', '我的职业',
            '项目', '产品', '公司', '团队',
            '目标', '计划', '梦想', '愿望',
            '地址', '城市', '住在', '来自',
            '年龄', '生日', '出生',
            '学历', '学校', '专业',
            '联系方式', '邮箱', '电话',
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in keywords)

    def _extract_with_llm_async(self, query: str, response: str):
        """使用LLM异步提取记忆"""
        import threading

        def extract():
            try:
                extract_prompt = f"""分析以下对话，提取关于用户的重要事实信息。

用户说: {query}

请提取以下类型的信息（如果有）：
1. 身份信息（姓名、职业、身份等）
2. 偏好信息（喜欢什么、讨厌什么、兴趣爱好等）
3. 背景信息（所在地、公司、项目等）
4. 目标信息（计划、目标、愿望等）

只返回提取到的事实，每行一个，格式为：
- [类别] 具体事实

如果没有重要信息，返回"无"。
"""

                from langchain_core.messages import HumanMessage
                result = self.model.invoke([HumanMessage(content=extract_prompt)])

                if result and result.content:
                    content = result.content.strip()
                    if content and content != '无':
                        for line in content.split('\n'):
                            line = line.strip()
                            if line.startswith('-') or line.startswith('•'):
                                fact_text = line.lstrip('- •').strip()
                                if ']' in fact_text:
                                    category_part, fact = fact_text.split(']', 1)
                                    category = category_part.strip('[').lower()
                                    fact = fact.strip()

                                    category_map = {
                                        '身份': 'identity',
                                        '职业': 'identity',
                                        '偏好': 'preference',
                                        '兴趣': 'preference',
                                        '背景': 'background',
                                        '目标': 'goal',
                                    }
                                    std_category = category_map.get(category, 'general')

                                    self._save_memory_if_new(fact, std_category, 0.8)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ LLM提取记忆失败: {e}")

        thread = threading.Thread(target=extract, daemon=True)
        thread.start()


def create_ollama_agent(
    model: BaseChatModel,
    skills_dir: Optional[str] = None,
    workspace_dir: Optional[str] = None,
    enable_memory: bool = True,
    max_tool_calls: Optional[int] = None,
    verbose: bool = True,
    max_context_tokens: Optional[int] = None,
    auto_detect_context: bool = True,
    enable_system_memory: bool = True,
    **kwargs
) -> OllamaPilotAgent:
    """
    创建 OllamaPilot Agent - 工厂函数

    Args:
        model: 聊天模型实例
        skills_dir: Skill 目录路径
        workspace_dir: Workspace 目录路径（用于文件驱动记忆）
        enable_memory: 是否启用对话记忆
        max_tool_calls: 最大工具调用次数
        verbose: 是否显示详细执行过程
        max_context_tokens: 最大 Context token 数
        auto_detect_context: 是否自动检测模型上下文大小
        enable_system_memory: 是否启用系统记忆

    Returns:
        OllamaPilotAgent 实例
    """
    if max_tool_calls is None:
        config = get_config()
        max_tool_calls = config.get_int('RECURSION_LIMIT', 50)

    return OllamaPilotAgent(
        model=model,
        skills_dir=skills_dir,
        workspace_dir=workspace_dir,
        enable_memory=enable_memory,
        max_tool_calls=max_tool_calls,
        verbose=verbose,
        max_context_tokens=max_context_tokens,
        auto_detect_context=auto_detect_context,
        enable_system_memory=enable_system_memory,
        **kwargs
    )


create_agent = create_ollama_agent