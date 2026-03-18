"""
Agent 核心模块 V2 - 基于 LangChain create_agent

使用 LangChain 原生 create_agent 和 AgentMiddleware 实现。
保持所有现有功能，代码更简洁。

新增 Context 总纲架构支持：
- ContextBuilder: 构建三层 Context（实时层、工作层、知识层）
- SystemMemory: 跨会话长期记忆（可选）
- TokenOptimizer: Token 预算优化
"""

import asyncio
from typing import List, Optional, Dict, Any
from langchain.agents import create_agent as lc_create_agent
from langchain.agents.middleware import (
    ToolRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.tools import BaseTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.memory import MemorySaver

from ollamapilot.infra.optimized_checkpoint import OptimizedCheckpoint
from ollamapilot.skills import SkillRegistry
from ollamapilot.tools.builtin import (
    read_file, write_file, list_directory, search_files,
    shell_exec, shell_script, python_exec, web_search, web_fetch
)
from ollamapilot.skill_middleware import (
    SkillSelectorMiddleware,
    ToolLoggingMiddleware,
    create_skill_middlewares,
)
from ollamapilot.model_context import get_truncation_threshold, get_recommended_num_ctx
from ollamapilot.context.builder import ContextBuilder
from ollamapilot.memory.system_memory import SystemMemory


def _detect_model_name(model: BaseChatModel) -> Optional[str]:
    """
    从模型实例中检测模型名称
    
    Args:
        model: LangChain 模型实例
        
    Returns:
        模型名称或 None
    """
    # 尝试不同的属性获取模型名称
    for attr in ['model', 'model_name', 'deployment_name']:
        if hasattr(model, attr):
            value = getattr(model, attr)
            if value:
                return str(value)
    
    # 尝试从 model_kwargs 获取
    if hasattr(model, 'model_kwargs'):
        kwargs = getattr(model, 'model_kwargs', {})
        if isinstance(kwargs, dict):
            for key in ['model', 'model_name']:
                if key in kwargs:
                    return str(kwargs[key])
    
    return None


class OllamaPilotAgent:
    """
    OllamaPilot Agent - V2 版本

    基于 LangChain create_agent 实现，使用原生 Middleware 机制。

    特性:
    - 使用 create_agent 创建 Agent
    - Skill 通过 AgentMiddleware 实现
    - 自动工具重试和限流
    - 对话记忆持久化
    - 详细执行日志
    - Skill 可注册自定义中间件（如 GraphRAG）

    示例:
        >>> from ollamapilot import init_ollama_model, OllamaPilotAgent
        >>>
        >>> model = init_ollama_model("qwen3.5:4b")
        >>> agent = OllamaPilotAgent(model, skills_dir="skills")
        >>> response = agent.invoke("明天苏州天气怎么样？")
    """

    def __init__(
        self,
        model: BaseChatModel,
        skills_dir: Optional[str] = None,
        enable_memory: bool = True,
        max_tool_calls: int = 50,
        verbose: bool = True,
        checkpointer=None,
        tools: Optional[List[BaseTool]] = None,  # 兼容参数，实际使用内置工具
        embedding_model: Optional[str] = None,  # Embedding 模型名称（用于 GraphRAG Skill）
        enable_system_memory: bool = True,  # 是否启用系统记忆（默认启用）
        system_memory_dir: str = "./data/memories",  # 系统记忆存储目录
        memory_embedding_model: Optional[Any] = None,  # 用于向量检索的嵌入模型
        max_context_tokens: Optional[int] = None,  # 最大 Context token 数（None 则自动检测）
        auto_detect_context: bool = True,  # 是否自动检测模型上下文大小
        memory_storage: str = "sqlite",  # 对话记忆存储方式: "sqlite" 或 "memory"
        memory_db_path: str = "./data/sessions/conversations.db",  # SQLite 存储路径
        **kwargs,  # 忽略其他参数，保持向后兼容
    ):
        """
        初始化 Agent

        Args:
            model: 聊天模型实例
            skills_dir: Skill 目录路径
            enable_memory: 是否启用对话记忆（Checkpoint）
            max_tool_calls: 最大工具调用次数
            verbose: 是否显示详细执行过程
            checkpointer: 自定义 checkpointer
            embedding_model: Embedding 模型名称（传递给 GraphRAG Skill）
            enable_system_memory: 是否启用系统记忆（跨会话长期记忆）
            system_memory_dir: 系统记忆存储目录
            memory_embedding_model: 用于向量检索的嵌入模型（可选）
            max_context_tokens: 最大 Context token 数（None 则根据模型自动检测）
            auto_detect_context: 是否自动检测模型上下文大小
            memory_storage: 对话记忆存储方式: "sqlite" (持久化) 或 "memory" (内存)
            memory_db_path: SQLite 数据库路径（当 memory_storage="sqlite" 时有效）
        """
        self.model = model
        self.verbose = verbose

        # 动态检测模型上下文大小
        detected_tokens = None
        model_name = _detect_model_name(model)
        if auto_detect_context and max_context_tokens is None:
            if model_name:
                try:
                    detected_tokens = get_recommended_num_ctx(model_name)
                    if verbose:
                        print(f"🎯 自动检测模型上下文: {model_name} -> {detected_tokens:,} tokens")
                except Exception as e:
                    if verbose:
                        print(f"⚠️  上下文检测失败: {e}，使用默认值 8192")
                    detected_tokens = 8192
            else:
                if verbose:
                    print(f"⚠️  无法检测模型名称，使用默认上下文 8192 tokens")
                detected_tokens = 8192
        
        # 使用检测值或用户指定值
        final_max_tokens = max_context_tokens if max_context_tokens is not None else (detected_tokens or 8192)
        self.max_context_tokens = final_max_tokens

        # 构建 Skill 配置
        skill_config = {}
        if embedding_model:
            skill_config["graphrag"] = {"embedding_model": embedding_model}
        # 添加主模型名称（用于动态上下文判断）
        if model_name:
            skill_config["graphrag"] = skill_config.get("graphrag", {})
            skill_config["graphrag"]["model_name"] = model_name

        # 初始化 Skill 注册中心
        self.skill_registry = SkillRegistry(skill_config=skill_config)

        # 加载 Skill
        skill_count = 0
        if skills_dir:
            skill_count = self.skill_registry.discover_skills(skills_dir)

        # 收集所有工具（内置 + Skill）
        self.all_tools = self._get_all_tools()

        # 配置 Checkpointer（对话记忆存储）
        if checkpointer:
            self.checkpointer = checkpointer
        elif enable_memory:
            if memory_storage == "sqlite":
                # 使用优化版 Checkpoint：内存优先 + 异步持久化
                import os
                os.makedirs(os.path.dirname(memory_db_path), exist_ok=True)
                try:
                    self.checkpointer = OptimizedCheckpoint(
                        db_path=memory_db_path,
                        save_interval=30,  # 每30秒自动保存一次
                        verbose=verbose
                    )
                    if verbose:
                        print(f"💾 对话记忆已启用 (优化版 SQLite): {memory_db_path}")
                except Exception as e:
                    # 如果初始化失败，回退到内存存储
                    self.checkpointer = MemorySaver()
                    if verbose:
                        print(f"⚠️  优化版 Checkpoint 初始化失败，使用内存模式: {e}")
            else:
                # 使用内存存储（程序重启后丢失）
                self.checkpointer = MemorySaver()
                if verbose:
                    print(f"💾 对话记忆已启用 (内存模式)")
        else:
            self.checkpointer = None

        # 保存最大工具调用次数（用于设置 recursion_limit）
        self.max_tool_calls = max_tool_calls

        # 初始化系统记忆
        self.system_memory = None
        if enable_system_memory:
            # 使用统一版系统记忆（默认启用向量检索）
            self.system_memory = SystemMemory(
                storage_dir=system_memory_dir,
                embedding_model=memory_embedding_model,
                verbose=verbose,
            )
            if verbose:
                print(f"🧠 系统记忆已启用")

        self.context_builder = ContextBuilder(
            max_tokens=final_max_tokens,
            enable_system_memory=enable_system_memory,
            system_memory=self.system_memory,
        )

        # 构建中间件列表
        middleware = self._build_middleware(max_tool_calls)

        # 打印启动摘要
        if self.verbose:
            memory_status = " | 🧠 系统记忆已启用" if enable_system_memory else ""
            print(f"📦 已加载 {skill_count} 个 Skill | 🔧 {len(self.all_tools)} 个工具 | 🔒 工具过滤已启用{memory_status}")

        # 创建 Agent（使用 LangChain 原生 create_agent）
        self.agent = lc_create_agent(
            model=model.bind_tools(self.all_tools),
            tools=self.all_tools,
            middleware=middleware,
            checkpointer=self.checkpointer,
        )

    def _get_all_tools(self) -> List[BaseTool]:
        """获取所有工具（内置工具 + Skill 工具）"""
        # 内置工具
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

        # 收集所有 Skill 提供的工具
        skill_tools = self.skill_registry.get_all_tools()
        tools.extend(skill_tools)

        return tools

    def _build_middleware(self, max_tool_calls: int) -> List[Any]:
        """
        构建中间件列表

        中间件执行顺序（从前到后）:
        1. SkillSelectorMiddleware - Skill 选择
        2. Skill 自定义中间件（如 GraphRAGMiddleware）
        3. ToolFilterMiddleware - 工具过滤（根据 Skill 限制可用工具）
        4. ToolLoggingMiddleware - 工具调用日志
        5. ToolRetryMiddleware - 工具重试
        6. ToolCallLimitMiddleware - 工具调用限制
        """
        middleware = []

        # 1. Skill 选择器（核心）
        self._selector = SkillSelectorMiddleware(
            self.skill_registry,
            default_skill_name="default",
            verbose=self.verbose
        )
        middleware.append(self._selector)

        # 2. Skill 自定义中间件（动态收集）
        skill_middlewares = self.skill_registry.get_all_middlewares()
        for mw in skill_middlewares:
            middleware.append(mw)

        # 3. 工具过滤中间件（根据选中 Skill 限制工具）
        tool_filter = self._selector.get_tool_filter()
        middleware.append(tool_filter)

        # 4. 工具调用日志
        if self.verbose:
            logging_mw = ToolLoggingMiddleware(verbose=True)
            middleware.append(logging_mw)

        # 5. 工具重试
        retry_mw = ToolRetryMiddleware(max_retries=2)
        middleware.append(retry_mw)

        # 6. 工具调用限制
        limit_mw = ToolCallLimitMiddleware(run_limit=max_tool_calls)
        middleware.append(limit_mw)

        return middleware

    def _get_tool_truncation_limit(self) -> int:
        """
        获取工具输出截断阈值
        
        根据当前模型实际使用的 num_ctx 动态计算。
        确保工具输出不会超出模型上下文限制。
        
        Returns:
            截断阈值（字符数）
        """
        try:
            # 尝试从模型获取名称和 num_ctx
            model_name = None
            num_ctx = None
            
            if hasattr(self.model, 'model'):
                model_name = self.model.model
            elif hasattr(self.model, 'model_name'):
                model_name = self.model.model_name
            
            # 获取实际使用的 num_ctx
            if hasattr(self.model, 'num_ctx'):
                num_ctx = self.model.num_ctx
            
            if model_name:
                from ollamapilot.model_context import get_truncation_threshold
                return get_truncation_threshold(model_name, num_ctx=num_ctx)
        except Exception:
            pass
        
        # 默认保守值
        return 2000

    def _build_context(self, query: str, thread_id: Optional[str] = None):
        """
        构建 Context（用于调试和分析）

        Args:
            query: 用户查询
            thread_id: 对话线程 ID

        Returns:
            Context 对象
        """
        # 获取当前激活的 Skill
        skill = self._select_skill_for_query(query)
        if not skill:
            from ollamapilot.skills.default_skill import DefaultSkill
            skill = DefaultSkill()

        # 获取对话历史
        history = self.get_history(thread_id) if thread_id else []

        # 构建 Context
        context = self.context_builder.build(
            query=query,
            skill=skill,
            history=history,
            thread_id=thread_id,
        )

        return context

    def invoke(self, query: str, thread_id: Optional[str] = None) -> str:
        """
        执行用户查询

        Args:
            query: 用户输入
            thread_id: 对话线程 ID（用于持久化记忆）

        Returns:
            模型回复
        """
        if self.verbose:
            print(f"🤖 用户: {query}")

        # 1. 选择 Skill
        skill = self._select_skill_for_query(query)

        # 记录 Skill 使用（用于系统记忆）
        if skill and self.system_memory:
            self.system_memory.record_skill_usage(skill.name, context=query[:100])

        # 2. 获取对话历史
        history = self.get_history(thread_id) if thread_id else []

        # 3. 使用 ContextBuilder 构建完整上下文（三层架构）
        context = self.context_builder.build(
            query=query,
            skill=skill,
            history=history,
            thread_id=thread_id,
        )

        # 4. 将 Context 转换为系统提示词
        system_prompt = self._context_to_prompt(context)

        if self.verbose and self.system_memory:
            # 检查知识层是否有记忆
            from ollamapilot.context.types import Layer
            knowledge = context.get_layer(Layer.KNOWLEDGE)
            if knowledge and knowledge.memories:
                print(f"🧠 已加载 {len(knowledge.memories)} 条系统记忆")

        # 5. 配置（包含 recursion_limit 以防止 LangGraph 默认限制）
        config = {
            "configurable": {"thread_id": thread_id or "default"},
            "recursion_limit": self.max_tool_calls + 10  # 给一些余量
        }

        # 6. 构建消息列表
        messages = []

        # 添加系统提示词（来自 ContextBuilder）
        if system_prompt:
            from langchain_core.messages import SystemMessage
            messages.append(SystemMessage(content=system_prompt))

        # 添加用户查询
        messages.append(HumanMessage(content=query))

        # 7. 执行
        result = self.agent.invoke(
            {"messages": messages},
            config
        )

        # 8. 提取回复
        messages = result.get("messages", [])
        response = ""
        has_tool_calls = False

        if messages:
            # 检查消息历史中是否有工具调用
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    has_tool_calls = True
                    break

            # 获取最后一条消息的内容
            last_message = messages[-1]
            if hasattr(last_message, "content"):
                response = last_message.content or ""

        # 如果有工具调用但回复为空，强制生成回复
        if has_tool_calls and not response.strip():
            if self.verbose:
                print("🔄 检测到工具调用后无回复，强制生成回复...")
            response = self._force_response(messages, config)

        if self.verbose and response:
            print(f"🤖 AI: {response[:200]}{'...' if len(response) > 200 else ''}")

        # 9. 自动提取和保存重要信息到系统记忆
        if self.system_memory:
            self._extract_and_save_memory(query, response)

        return response

    async def ainvoke(self, query: str, thread_id: Optional[str] = None) -> str:
        """
        异步执行用户查询
        
        在事件循环中运行同步的 invoke 方法
        
        Args:
            query: 用户输入
            thread_id: 对话线程 ID（用于持久化记忆）
            
        Returns:
            模型回复
        """
        # 使用 to_thread 在单独的线程中运行同步代码
        import asyncio
        return await asyncio.to_thread(self.invoke, query, thread_id)

    def _context_to_prompt(self, context) -> str:
        """
        将 Context 转换为系统提示词

        Args:
            context: Context 对象

        Returns:
            系统提示词
        """
        if not context or not context.parts:
            return ""

        parts = []

        # 添加系统指令
        parts.append("你是一个智能助手，请根据以下上下文信息回答用户问题。")

        # 添加各层 Context
        context_text = context.to_text()
        if context_text:
            parts.append(context_text)

        # 添加回复要求
        parts.append("请基于以上信息，为用户提供准确、有用的回答。")

        return "\n\n".join(parts)

    def _extract_and_save_memory(self, query: str, response: str):
        """
        从对话中提取重要信息并保存到系统记忆
        使用轻量级LLM分析，无需硬编码规则

        Args:
            query: 用户查询
            response: AI回复
        """
        if not self.system_memory:
            return

        try:
            # 使用轻量级规则快速提取（避免每次调用LLM）
            # 只提取最明确的信息模式
            import re

            # 1. 提取姓名（最明确的模式）
            name_match = re.search(r'我叫\s*([^，。！？\n]{2,20})(?=[，。！？\n]|$)', query)
            if name_match:
                name = name_match.group(1).strip()
                if name and len(name) >= 2:
                    self._save_memory_if_new(f"用户姓名是{name}", "identity", 0.95)

            # 2. 使用LLM进行智能提取（异步，不阻塞回复）
            # 只在查询包含潜在重要信息时触发
            if self._should_extract_with_llm(query):
                self._extract_with_llm_async(query, response)

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 提取记忆时出错: {e}")

    def _save_memory_if_new(self, fact: str, category: str, importance: float):
        """保存记忆（如果之前没有保存过相同内容）"""
        try:
            # 检查是否已存在相似记忆（使用语义检索）
            existing = self.system_memory.recall(fact, top_k=5)

            # 提取关键信息用于比较（如姓名、兴趣爱好）
            fact_normalized = self._normalize_fact(fact)

            for mem in existing:
                mem_normalized = self._normalize_fact(mem)

                # 检查是否是同一类信息（如都是姓名）
                if self._is_same_fact_type(fact_normalized, mem_normalized):
                    # 如果内容相似度很高，不重复保存
                    similarity = self._calculate_similarity(fact_normalized, mem_normalized)
                    if similarity > 0.7:  # 相似度阈值
                        if self.verbose:
                            print(f"   [Memory] 相似记忆已存在，跳过: {fact[:50]}...")
                        return

            self.system_memory.remember_fact(fact, category=category, importance=importance)
            if self.verbose:
                print(f"💾 已保存记忆: {fact[:50]}...")
        except Exception as e:
            if self.verbose:
                print(f"   [Memory] 保存记忆时出错: {e}")

    def _normalize_fact(self, fact: str) -> str:
        """标准化事实文本，用于比较"""
        import re

        # 移除常见前缀
        prefixes = ['用户', '姓名', '名字', '是', '：', ':', ' ']
        normalized = fact.strip()
        for prefix in prefixes:
            normalized = normalized.replace(prefix, '')

        # 转换为小写
        normalized = normalized.lower()

        # 移除标点
        normalized = re.sub(r'[^\w\u4e00-\u9fff]', '', normalized)

        return normalized

    def _is_same_fact_type(self, fact1: str, fact2: str) -> bool:
        """判断两个事实是否是同一类型（如都是姓名）"""
        # 检查是否包含相同的实体类型关键词
        identity_keywords = ['姓名', '名字', '叫', '我是']
        preference_keywords = ['喜欢', '爱好', '兴趣']

        # 检查是否是身份信息
        is_identity1 = any(kw in fact1 for kw in identity_keywords)
        is_identity2 = any(kw in fact2 for kw in identity_keywords)
        if is_identity1 and is_identity2:
            return True

        # 检查是否是偏好信息
        is_pref1 = any(kw in fact1 for kw in preference_keywords)
        is_pref2 = any(kw in fact2 for kw in preference_keywords)
        if is_pref1 and is_pref2:
            return True

        # 检查是否有大量字符重叠
        common_chars = set(fact1) & set(fact2)
        if len(common_chars) >= min(len(fact1), len(fact2)) * 0.5:
            return True

        return False

    def _calculate_similarity(self, s1: str, s2: str) -> float:
        """计算两个字符串的相似度 (0-1)"""
        if not s1 or not s2:
            return 0.0

        # 使用 Jaccard 相似度
        set1 = set(s1)
        set2 = set(s2)

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        if union == 0:
            return 0.0

        return intersection / union

    def _should_extract_with_llm(self, query: str) -> bool:
        """
        判断是否需要使用LLM提取记忆
        基于关键词启发式判断
        """
        # 包含潜在重要信息的关键词
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
        """
        使用LLM异步提取记忆（不阻塞主流程）
        """
        import threading

        def extract():
            try:
                # 构建提取提示词
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

                # 使用简单模型进行提取
                from langchain_core.messages import HumanMessage
                result = self.model.invoke([HumanMessage(content=extract_prompt)])

                if result and result.content:
                    content = result.content.strip()
                    if content and content != '无':
                        # 解析提取结果
                        for line in content.split('\n'):
                            line = line.strip()
                            if line.startswith('-') or line.startswith('•'):
                                # 提取类别和事实
                                fact_text = line.lstrip('- •').strip()
                                if ']' in fact_text:
                                    category_part, fact = fact_text.split(']', 1)
                                    category = category_part.strip('[').lower()
                                    fact = fact.strip()

                                    # 映射到标准类别
                                    category_map = {
                                        '身份': 'identity',
                                        '职业': 'identity',
                                        '偏好': 'preference',
                                        '兴趣': 'preference',
                                        '背景': 'background',
                                        '目标': 'goal',
                                    }
                                    std_category = category_map.get(category, 'general')

                                    # 保存记忆
                                    self._save_memory_if_new(fact, std_category, 0.8)

            except Exception as e:
                if self.verbose:
                    print(f"⚠️ LLM提取记忆失败: {e}")

        # 在后台线程中执行提取
        thread = threading.Thread(target=extract, daemon=True)
        thread.start()

    def _force_response(self, messages: List[Any], config: Dict[str, Any]) -> str:
        """
        强制生成回复（当模型调用工具后没有生成回复时）

        Args:
            messages: 当前消息历史
            config: 配置

        Returns:
            强制生成的回复
        """
        try:
            from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

            # 构建用于强制回复的消息链
            # 包含：HumanMessage -> AIMessage(工具调用说明) -> ToolMessage
            force_messages = []

            # 找到最后一条 HumanMessage
            last_human_msg = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    last_human_msg = msg
                    break

            if not last_human_msg:
                return "（无法找到用户查询）"

            force_messages.append(last_human_msg)

            # 查找工具调用和对应的 ToolMessage
            tool_call_found = False
            tool_results = []

            for msg in messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    tool_call_found = True
                    # 添加工具调用说明
                    tool_names = [tc.get('name', 'unknown') for tc in msg.tool_calls]
                    explanation = f"我需要调用工具来获取信息：{', '.join(tool_names)}"
                    force_messages.append(AIMessage(content=explanation))

                elif isinstance(msg, ToolMessage):
                    # 截断过长的 ToolMessage 内容，避免模型无法处理
                    # 根据模型上下文窗口动态调整截断阈值
                    content = msg.content
                    truncation_limit = self._get_tool_truncation_limit()
                    if len(content) > truncation_limit:
                        original_length = len(content)
                        content = content[:truncation_limit] + f"\n... (内容已截断，原始长度: {original_length} 字符)"
                    tool_results.append(ToolMessage(
                        content=content,
                        tool_call_id=msg.tool_call_id,
                        name=msg.name
                    ))

            # 添加所有 ToolMessage
            force_messages.extend(tool_results)

            if self.verbose:
                print(f"   [ForceResponse] 构建消息链: {len(force_messages)} 条")
                for i, msg in enumerate(force_messages):
                    print(f"      [{i}] {type(msg).__name__}")

            # 如果没有找到工具调用，直接返回提示
            if not tool_call_found:
                return "（工具执行完成）"

            # 添加强制回复提示
            force_messages.append(
                HumanMessage(content="请基于上述工具执行结果，直接回答用户的问题。不要调用任何工具。")
            )

            # 使用 Agent 生成回复（保持与正常流程一致）
            result = self.agent.invoke(
                {"messages": force_messages},
                config
            )

            # 从结果中提取最后一条消息的回复
            result_messages = result.get("messages", [])
            if result_messages:
                last_message = result_messages[-1]
                if hasattr(last_message, "content") and last_message.content:
                    return last_message.content

        except Exception as e:
            if self.verbose:
                print(f"⚠️ 强制生成回复失败: {e}")
                import traceback
                traceback.print_exc()

        return "（工具执行完成，但生成回复失败）"

    def _select_skill_for_query(self, query: str) -> Optional[Any]:
        """
        根据查询选择合适的 Skill

        Args:
            query: 用户查询

        Returns:
            Skill 实例或 None
        """
        # 1. 查找匹配的特定 Skill
        matches = self.skill_registry.find_skill_by_trigger(query)

        if matches:
            return self.skill_registry.get_skill(matches[0])

        # 2. 返回默认 Skill
        default_skill = self.skill_registry.get_default_skill()
        if default_skill:
            return default_skill

        return None

    def stream(self, query: str, thread_id: Optional[str] = None):
        """
        流式执行用户查询

        Args:
            query: 用户输入
            thread_id: 对话线程 ID

        Yields:
            流式输出块
        """
        config = {"configurable": {"thread_id": thread_id or "default"}}

        for chunk in self.agent.stream(
            {"messages": [HumanMessage(content=query)]},
            config
        ):
            yield chunk

    async def astream_events(self, query: str, thread_id: Optional[str] = None):
        """
        异步流式事件输出 - 使用 LangChain 原生 astream_events

        提供结构化的事件流，包括：
        - on_tool_start: 工具开始执行
        - on_tool_end: 工具执行结束
        - on_chat_model_stream: 模型流式输出

        Args:
            query: 用户输入
            thread_id: 对话线程 ID

        Yields:
            事件字典，包含 event, name, data 等字段
        """
        # 配置（包含 recursion_limit 以防止 LangGraph 默认限制）
        config = {
            "configurable": {"thread_id": thread_id or "default"},
            "recursion_limit": self.max_tool_calls + 10  # 给一些余量
        }

        # 收集完整的 AI 回复用于记忆提取
        full_response = ""

        # 只传入当前用户消息，历史消息由 checkpointer 自动管理
        # LangChain Agent 会自动从 checkpointer 加载历史消息并合并
        async for event in self.agent.astream_events(
            {"messages": [HumanMessage(content=query)]},
            config,
            version="v1"
        ):
            # 收集 AI 回复内容
            if event.get("event") == "on_chat_model_stream":
                data = event.get("data", {})
                chunk = data.get("chunk", None)
                if chunk and hasattr(chunk, "content"):
                    content = chunk.content
                    if content:
                        full_response += content

            yield event

        # 对话结束后，自动提取和保存重要信息到系统记忆
        if self.system_memory:
            self._extract_and_save_memory(query, full_response)

    def get_history(self, thread_id: Optional[str] = None) -> List[Any]:
        """
        获取对话历史

        Args:
            thread_id: 对话线程 ID

        Returns:
            消息列表
        """
        if not self.checkpointer:
            return []

        config = {"configurable": {"thread_id": thread_id or "default"}}

        try:
            checkpoint_tuple = self.checkpointer.get_tuple(config)
            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                # 尝试不同的消息存储位置
                checkpoint = checkpoint_tuple.checkpoint

                # 方法1: 直接获取 messages
                if "messages" in checkpoint:
                    return checkpoint["messages"]

                # 方法2: 从 channel_values 获取 (LangGraph 新格式)
                if "channel_values" in checkpoint:
                    channel_values = checkpoint["channel_values"]
                    if "messages" in channel_values:
                        return channel_values["messages"]

                # 方法3: 从完整状态获取
                if checkpoint_tuple.state:
                    state = checkpoint_tuple.state
                    if "messages" in state:
                        return state["messages"]
        except Exception as e:
            if self.verbose:
                print(f"   [GetHistory] 获取历史失败: {e}")

        return []

    def clear_history(self, thread_id: Optional[str] = None) -> None:
        """
        清除对话历史

        Args:
            thread_id: 对话线程 ID
        """
        if not self.checkpointer:
            return

        config = {"configurable": {"thread_id": thread_id or "default"}}

        try:
            self.checkpointer.delete(config)
        except Exception:
            pass

    def get_skill_stats(self, skill_name: str) -> Optional[Dict[str, Any]]:
        """
        获取 Skill 的统计信息

        Args:
            skill_name: Skill 名称

        Returns:
            统计信息字典，如果 Skill 不存在或不支持返回 None
        """
        skill = self.skill_registry.get_skill(skill_name)
        if not skill:
            return None

        # 检查 Skill 是否有 get_stats 方法
        if hasattr(skill, 'get_stats'):
            return skill.get_stats()

        return None

    async def force_response_after_tool(self, thread_id: Optional[str] = None):
        """
        工具调用后强制模型生成回复

        当模型在工具调用后没有生成回复时，使用此方法触发一次额外的模型调用，
        要求模型基于工具结果生成回复。

        注意：此方法会临时清空工具列表，确保只生成文本回复。

        Args:
            thread_id: 对话线程 ID

        Yields:
            模型生成的文本块
        """
        # 临时禁用工具调用，确保只生成文本回复
        # 通过设置一个标志，让 before_model 知道这是强制回复
        if self._selector:
            # 保存原始工具列表
            original_tools = self._selector.tool_filter.allowed_tools.copy()
            # 清空工具列表，禁止工具调用
            self._selector.tool_filter.set_allowed_tools([])
            if self.verbose:
                print("   [ForceResponse] 临时禁用工具调用")

        try:
            # 添加提示消息，要求模型基于工具结果生成回复
            prompt = "基于上述工具执行结果，请为用户提供清晰、有用的回复。总结关键信息并给出实用建议。"

            async for event in self.astream_events(prompt, thread_id=thread_id):
                event_type = event.get("event", "")
                if event_type == "on_chat_model_stream":
                    data = event.get("data", {})
                    chunk = data.get("chunk", None)
                    if chunk and hasattr(chunk, "content"):
                        content = chunk.content
                        if content:
                            yield content
        finally:
            # 恢复原始工具设置
            if self._selector and original_tools:
                self._selector.tool_filter.set_allowed_tools(list(original_tools))
                if self.verbose:
                    print("   [ForceResponse] 恢复工具调用")


def create_ollama_agent(
    model: BaseChatModel,
    skills_dir: Optional[str] = None,
    enable_memory: bool = True,
    max_tool_calls: Optional[int] = None,
    verbose: bool = True,
    max_context_tokens: Optional[int] = None,
    auto_detect_context: bool = True,
    enable_system_memory: bool = True,
    system_memory_dir: str = "./data/memories",
    **kwargs
) -> OllamaPilotAgent:
    """
    创建 OllamaPilot Agent

    工厂函数，快速创建 Agent 实例。
    支持自动检测模型上下文大小，根据模型能力和硬件配置动态调整。

    Args:
        model: 聊天模型实例
        skills_dir: Skill 目录路径
        enable_memory: 是否启用对话记忆
        max_tool_calls: 最大工具调用次数，默认从配置文件读取 RECURSION_LIMIT
        verbose: 是否显示详细执行过程
        max_context_tokens: 最大 Context token 数（None 则自动检测）
        auto_detect_context: 是否自动检测模型上下文大小
        enable_system_memory: 是否启用系统记忆（跨会话长期记忆）
        system_memory_dir: 系统记忆存储目录
        **kwargs: 其他参数传递给 OllamaPilotAgent

    Returns:
        OllamaPilotAgent 实例

    Example:
        >>> from ollamapilot import init_ollama_model, create_ollama_agent
        >>>
        >>> model = init_ollama_model("qwen3.5:4b")
        >>> agent = create_ollama_agent(model, skills_dir="skills")
        >>> response = agent.invoke("明天苏州天气怎么样？")
        
        >>> # 手动指定上下文大小
        >>> agent = create_ollama_agent(model, skills_dir="skills", max_context_tokens=32768)
        
        >>> # 禁用自动检测
        >>> agent = create_ollama_agent(model, skills_dir="skills", auto_detect_context=False)
    """
    # 如果未指定 max_tool_calls，从配置文件读取
    if max_tool_calls is None:
        from ollamapilot.config import get_config
        config = get_config()
        max_tool_calls = config.get_int('RECURSION_LIMIT', 50)

    return OllamaPilotAgent(
        model=model,
        skills_dir=skills_dir,
        enable_memory=enable_memory,
        max_tool_calls=max_tool_calls,
        verbose=verbose,
        max_context_tokens=max_context_tokens,
        auto_detect_context=auto_detect_context,
        enable_system_memory=enable_system_memory,
        system_memory_dir=system_memory_dir,
        **kwargs
    )


# 保持向后兼容的别名
create_agent = create_ollama_agent
