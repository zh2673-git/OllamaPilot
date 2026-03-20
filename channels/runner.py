"""
Channels 运行器

统一运行和管理所有渠道。
作为独立模块运行，通过导入 OllamaPilot 核心来使用 Agent。

使用方式:
    >>> python -m channels.runner
    或
    >>> from channels import ChannelRunner
    >>> runner = ChannelRunner("config.yaml")
    >>> asyncio.run(runner.start())
"""

import argparse
import asyncio
import concurrent.futures
import logging
import signal
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    print("❌ 请先安装 PyYAML: pip install pyyaml")
    sys.exit(1)

try:
    import aiohttp
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    print("❌ 请先安装依赖: pip install aiohttp tenacity")
    sys.exit(1)

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .base import Channel, ChannelMessage, ChannelResponse
from .registry import auto_discover_channels, get_channel, list_channels
from .history_manager import get_history_manager

logger = logging.getLogger(__name__)


class ChannelRunner:
    """
    渠道统一运行器（参考 nanobot + CoPaw 设计）
    
    负责:
    1. 加载配置
    2. 初始化 OllamaPilot Agent
    3. 自动发现并初始化渠道
    4. 处理消息并返回响应
    5. 错误重试和统计
    
    Example:
        >>> runner = ChannelRunner("channels/config.yaml")
        >>> asyncio.run(runner.start())
    """
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
            if not config_path.exists():
                example_path = Path(__file__).parent / "config.example.yaml"
                if example_path.exists():
                    print(f"⚠️ config.yaml 不存在，使用 config.example.yaml")
                    print("   请复制为 config.yaml 并填入你的配置！\n")
                    config_path = example_path
        else:
            config_path = Path(config_path)
        
        self.config = self._load_config(config_path)
        
        # 创建线程池执行器（用于同步 agent 调用）
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        
        # 初始化渠道会话管理器
        from .session_manager import get_session_manager
        self.session_manager = get_session_manager("./data/channel_sessions")
        
        self._setup_logging()
        
        self._init_agent()
        
        auto_discover_channels()
        
        self.channels: Dict[str, Channel] = {}
        self._init_channels()
        
        self._running = False
        self._shutdown_event = asyncio.Event()
        
        self._stats = {
            "total_messages": 0,
            "success": 0,
            "failed": 0,
            "errors_by_channel": {},
            "avg_response_time": 0.0,
        }
        self._response_times: deque = deque(maxlen=100)
    
    def _setup_logging(self):
        """设置日志"""
        log_config = self.config.get("logging", {})
        level = getattr(logging, log_config.get("level", "INFO"))
        
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """加载配置文件"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"已加载配置: {config_path}")
        return config
    
    def _init_agent(self):
        """初始化 OllamaPilot Agent（单用户模式，启动时直接创建）"""
        try:
            from ollamapilot import init_ollama_model, OllamaPilotAgent

            agent_config = self.config.get("agent", {})
            model_name = agent_config.get("model", "qwen3.5:4b")
            skills_dir = agent_config.get("skills_dir", "./skills")
            verbose = agent_config.get("verbose", True)

            logger.info(f"🤖 正在初始化 Agent...")
            logger.info(f"   模型: {model_name}")
            logger.info(f"   Skills 目录: {skills_dir}")
            logger.info(f"   记忆模式: 单用户全局 Agent（启动时加载）")

            model = init_ollama_model(model_name)
            self.model = model
            self.model_name = model_name  # 保存模型名称用于图片分析

            # 单用户模式：直接创建全局 Agent，所有消息共用
            logger.info("🔥 正在加载 Agent（包含所有 Skill 和记忆）...")
            checkpointer = self.session_manager.get_checkpointer("qq", "single_user")
            self.agent = OllamaPilotAgent(
                model=model,
                skills_dir=skills_dir,
                verbose=verbose,
                enable_memory=True,
                checkpointer=checkpointer
            )
            logger.info(f"✅ Agent 加载完成，已加载 {len(self.agent.all_tools)} 个工具")

        except ImportError as e:
            logger.error(f"❌ 导入 OllamaPilot 失败: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 初始化 Agent 失败: {e}")
            raise

    def _get_agent(self) -> Any:
        """
        获取全局 Agent 实例（单用户模式）
        
        Returns:
            OllamaPilotAgent 实例
        """
        return self.agent
    
    def _init_channels(self):
        """初始化所有启用的渠道"""
        channel_configs = self.config.get("channels", {})
        
        for channel_name, channel_config in channel_configs.items():
            if not channel_config.get("enabled", False):
                continue
            
            channel_class = get_channel(channel_name)
            if not channel_class:
                logger.warning(f"未知的渠道类型: {channel_name}")
                continue
            
            try:
                channel = channel_class(
                    config=channel_config,
                    message_handler=self._handle_message
                )
                self.channels[channel_name] = channel
                logger.info(f"✅ {channel_name} 渠道已加载")
            except Exception as e:
                logger.error(f"❌ 初始化 {channel_name} 渠道失败: {e}")
        
        if not self.channels:
            logger.warning("⚠️ 没有启用的渠道，请检查配置")
    
    async def _handle_message(self, message: ChannelMessage) -> ChannelResponse:
        """处理收到的消息（带错误重试和统计）"""
        import time
        start_time = time.time()

        self._stats["total_messages"] += 1
        channel_name = message.channel_name

        if not self._check_global_permission(message.user_id):
            return ChannelResponse(content="⛔ 您没有使用权限")

        if (message.message_type == "group" and
            self.channels.get(channel_name) and
            self.channels[channel_name].config.get("at_only_in_group", False) and
            not message.at_me):
            return ChannelResponse(content="")

        # 检查是否有文件或图片附件
        raw_data = message.raw_data or {}
        files = raw_data.get('files', [])
        images = message.images or []

        if files or images:
            return await self._handle_file_upload(message, files, images)

        # 检查是否是命令
        content = message.content.strip()
        if content.startswith('/'):
            return await self._handle_command(message, content)

        try:
            response = await self._invoke_with_retry(message)
            self._stats["success"] += 1

            duration = time.time() - start_time
            self._record_response_time(duration)

            if isinstance(response, str):
                return ChannelResponse(content=response)
            return response

        except Exception as e:
            self._stats["failed"] += 1
            if channel_name not in self._stats["errors_by_channel"]:
                self._stats["errors_by_channel"][channel_name] = 0
            self._stats["errors_by_channel"][channel_name] += 1

            logger.error(f"处理消息失败 [{channel_name}] (user={message.user_id}): {e}", exc_info=True)
            return ChannelResponse(content=f"❌ 处理出错: {str(e)[:100]}")

    async def _handle_file_upload(self, message: ChannelMessage, files: list, images: list) -> ChannelResponse:
        """处理文件和图片上传"""
        from ollamapilot.utils.file_processor import get_file_processor

        processor = get_file_processor()
        contents = []

        try:
            # 处理文档文件
            for file_info in files:
                try:
                    logger.info(f"📄 下载文件: {file_info['filename']}")
                    file_path = await processor.download_file(
                        file_info['url'],
                        file_info['filename']
                    )

                    logger.info(f"📄 提取内容: {file_info['filename']}")
                    text = processor.extract_text_content(file_path)

                    if text and not text.startswith('['):
                        contents.append(f"📄 文件: {file_info['filename']}\n{text[:5000]}")
                    else:
                        contents.append(f"📄 文件: {file_info['filename']}\n{text}")

                    # 清理临时文件
                    processor.cleanup(file_path)

                except Exception as e:
                    logger.error(f"处理文件失败 {file_info.get('filename')}: {e}")
                    contents.append(f"❌ 无法处理文件: {file_info.get('filename')}")

            # 处理图片
            for i, image_url in enumerate(images):
                try:
                    logger.info(f"🖼️  下载图片 {i+1}/{len(images)}")
                    image_path = await processor.download_file(
                        image_url,
                        f"image_{i+1}.jpg"
                    )

                    logger.info(f"🖼️  分析图片 {i+1}")
                    # 获取模型名称用于图片分析
                    model_name = getattr(self, 'model_name', 'qwen3.5:4b')

                    description = await processor.analyze_image(
                        image_path,
                        query=message.content or "描述这张图片的内容",
                        model_name=model_name
                    )
                    contents.append(f"🖼️  图片 {i+1}:\n{description}")

                    # 清理临时文件
                    processor.cleanup(image_path)

                except Exception as e:
                    logger.error(f"处理图片失败: {e}")
                    contents.append(f"❌ 无法分析图片 {i+1}")

            if not contents:
                return ChannelResponse(content="❌ 无法处理上传的文件/图片")

            # 如果用户只上传文件/图片，没有输入问题，直接返回内容描述
            user_query = message.content or ""
            full_content = "\n\n---\n\n".join(contents)

            if not user_query.strip():
                # 用户没有提问，直接返回文件/图片描述
                return ChannelResponse(content=full_content)

            # 用户有具体问题，才调用 AI 进行回答
            prompt = f"""用户上传了以下文件/图片，请根据内容回答用户问题：

{full_content}

用户问题: {user_query}

请基于以上内容回答。"""

            # 使用线程池调用agent
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._executor,
                self._sync_invoke_with_prompt,
                message,
                prompt
            )

            return ChannelResponse(content=result)

        except Exception as e:
            logger.error(f"处理文件上传失败: {e}", exc_info=True)
            return ChannelResponse(content=f"❌ 处理文件失败: {str(e)[:100]}")

    def _sync_invoke_with_prompt(self, message: ChannelMessage, prompt: str) -> str:
        """同步调用agent，使用自定义提示词"""
        agent = self._get_agent()
        thread_id = f"{message.channel_name}_{message.user_id}_{message.message_type}"

        # 使用agent的invoke，但传入自定义提示
        return agent.invoke(query=prompt, thread_id=thread_id)

    async def _handle_command(self, message: ChannelMessage, content: str) -> ChannelResponse:
        """处理命令"""
        parts = content.split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        channel = message.channel_name
        user_id = message.user_id

        if cmd == '/help':
            return self._cmd_help()

        elif cmd == '/new':
            return self._cmd_new(channel, user_id, args)

        elif cmd == '/sessions':
            return self._cmd_sessions(channel, user_id)

        elif cmd == '/switch':
            return self._cmd_switch(channel, user_id, args)

        elif cmd == '/clear':
            return self._cmd_clear(channel, user_id)

        elif cmd == '/memory':
            return self._cmd_memory(channel, user_id)

        elif cmd == '/stats':
            return self._cmd_stats()

        elif cmd == '/upload':
            return self._cmd_upload()

        else:
            return ChannelResponse(content=f"❌ 未知命令: {cmd}\n输入 /help 查看可用命令")

    def _cmd_upload(self) -> ChannelResponse:
        """上传命令"""
        help_text = """📎 文件上传说明

直接发送文件或图片，我会自动分析内容

支持格式:
  📄 文档: PDF, DOCX, TXT, MD, JSON, CSV, XLSX
  🖼️ 图片: JPG, PNG, GIF, WebP

使用方式:
  1. 直接发送文件 + 问题（可选）
     例: [发送PDF文件] "总结这份报告"

  2. 只发送文件
     我会自动总结文件内容

  3. 发送图片
     我会描述图片内容或回答你的问题

💡 提示:
  • 文件仅用于当前对话，不会保存到知识库
  • 大文件可能需要一些时间处理
  • 可以同时发送多个文件/图片"""
        return ChannelResponse(content=help_text)

    def _cmd_help(self) -> ChannelResponse:
        """帮助命令"""
        help_text = """📋 可用命令列表

会话管理:
  /new [名称]     - 创建新会话
  /sessions       - 列出所有会话
  /switch <ID>    - 切换到指定会话
  /clear          - 清空当前会话

文件/图片:
  /upload         - 文件上传说明

记忆管理:
  /memory         - 查看系统记忆

其他:
  /stats          - 查看运行统计
  /help           - 显示此帮助

💡 提示: 直接输入消息即可与AI对话"""
        return ChannelResponse(content=help_text)

    def _cmd_new(self, channel: str, user_id: str, args: str) -> ChannelResponse:
        """创建新会话"""
        try:
            # 清除当前用户的缓存，强制创建新的Agent
            cache_key = f"{channel}:{user_id}"
            if hasattr(self, '_user_agents') and cache_key in self._user_agents:
                del self._user_agents[cache_key]

            # 创建新的会话
            session_info = self.session_manager.get_or_create_session(channel, user_id)
            session_name = args.strip() if args else f"会话_{session_info['session_id'][:8]}"

            return ChannelResponse(content=f"✅ 已创建新会话: {session_name}\n💡 现在可以开始新的对话了")
        except Exception as e:
            logger.error(f"创建会话失败: {e}")
            return ChannelResponse(content=f"❌ 创建会话失败: {str(e)}")

    def _cmd_sessions(self, channel: str, user_id: str) -> ChannelResponse:
        """列出会话"""
        try:
            sessions = self.session_manager.list_sessions(channel, user_id)
            if not sessions:
                return ChannelResponse(content="📭 暂无会话\n💡 使用 /new 创建新会话")

            lines = ["📋 会话列表:"]
            for s in sessions[:10]:  # 最多显示10个
                msg_count = s.get('message_count', 0)
                last_time = s.get('last_activity', '未知')
                if last_time and last_time != '未知':
                    # 简化时间显示
                    last_time = str(last_time).split('.')[0]  # 去掉微秒
                lines.append(f"  • {s['session_id'][:8]}: {msg_count}条消息 | {last_time}")

            if len(sessions) > 10:
                lines.append(f"  ... 还有 {len(sessions) - 10} 个会话")

            lines.append(f"\n💡 使用 /switch <ID> 切换会话")
            return ChannelResponse(content="\n".join(lines))
        except Exception as e:
            logger.error(f"列出会话失败: {e}")
            return ChannelResponse(content=f"❌ 获取会话列表失败: {str(e)}")

    def _cmd_switch(self, channel: str, user_id: str, args: str) -> ChannelResponse:
        """切换会话"""
        if not args:
            return ChannelResponse(content="❌ 请指定会话ID\n用法: /switch <会话ID前8位>")

        session_id = args.strip()
        try:
            # 查找匹配的会话
            sessions = self.session_manager.list_sessions(channel, user_id)
            matched = None
            for s in sessions:
                if s['session_id'].startswith(session_id):
                    matched = s
                    break

            if not matched:
                return ChannelResponse(content=f"❌ 未找到会话: {session_id}\n使用 /sessions 查看可用会话")

            # 清除缓存，下次会使用新的会话
            cache_key = f"{channel}:{user_id}"
            if hasattr(self, '_user_agents') and cache_key in self._user_agents:
                del self._user_agents[cache_key]

            return ChannelResponse(content=f"✅ 已切换到会话: {matched['session_id'][:8]}\n💡 继续对话将使用此会话的历史")
        except Exception as e:
            logger.error(f"切换会话失败: {e}")
            return ChannelResponse(content=f"❌ 切换会话失败: {str(e)}")

    def _cmd_clear(self, channel: str, user_id: str) -> ChannelResponse:
        """清空当前会话"""
        try:
            # 清除Agent缓存
            cache_key = f"{channel}:{user_id}"
            if hasattr(self, '_user_agents') and cache_key in self._user_agents:
                agent = self._user_agents[cache_key]
                # 获取当前thread_id并清除
                thread_id = f"{channel}_{user_id}_private"
                # 清除 checkpointer
                if agent and agent.checkpointer:
                    try:
                        config = {"configurable": {"thread_id": thread_id}}
                        agent.checkpointer.delete(config)
                    except Exception:
                        pass
                del self._user_agents[cache_key]

            # 清除 JSON 历史文件
            try:
                history_manager = get_history_manager(channel, user_id)
                history_manager.clear()
                history_manager.save()
            except Exception as e:
                logger.warning(f"清除历史文件失败: {e}")

            return ChannelResponse(content="✅ 当前会话已清空\n💡 历史记录已清除，开始新的对话")
        except Exception as e:
            logger.error(f"清空会话失败: {e}")
            return ChannelResponse(content=f"❌ 清空会话失败: {str(e)}")

    def _cmd_memory(self, channel: str, user_id: str) -> ChannelResponse:
        """查看系统记忆"""
        try:
            cache_key = f"{channel}:{user_id}"
            if hasattr(self, '_user_agents') and cache_key in self._user_agents:
                agent = self._user_agents[cache_key]
                if agent.system_memory:
                    stats = agent.system_memory.get_stats()
                    lines = ["🧠 系统记忆统计:"]
                    lines.append(f"  • 语义记忆: {stats.get('semantic', 0)} 条")
                    lines.append(f"  • 程序记忆: {stats.get('procedural', 0)} 条")
                    lines.append(f"  • 情景记忆: {stats.get('episodic', 0)} 条")
                    return ChannelResponse(content="\n".join(lines))

            return ChannelResponse(content="📭 暂无记忆数据\n💡 继续对话后会自动积累记忆")
        except Exception as e:
            logger.error(f"获取记忆失败: {e}")
            return ChannelResponse(content=f"❌ 获取记忆失败: {str(e)}")

    def _cmd_stats(self) -> ChannelResponse:
        """查看运行统计"""
        stats = self.get_stats()
        lines = ["📊 运行统计:"]
        lines.append(f"  • 总消息数: {stats['total_messages']}")
        lines.append(f"  • 成功: {stats['success']} | 失败: {stats['failed']}")
        lines.append(f"  • 成功率: {stats['success_rate']:.1f}%")
        lines.append(f"  • 平均响应: {stats['avg_response_time']}")
        lines.append(f"  • 活跃渠道: {', '.join(stats['active_channels'])}")
        return ChannelResponse(content="\n".join(lines))
    
    async def _invoke_with_retry(self, message: ChannelMessage) -> str:
        """调用 Agent（使用线程池执行器，每个用户独立）"""
        thread_id = f"{message.channel_name}_{message.user_id}_{message.message_type}"

        logger.info(f"🤖 开始调用 Agent，thread_id={thread_id}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"🤖 使用线程池执行器调用 agent.invoke...")
                # 使用线程池执行器在单独线程中运行同步代码
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._executor,
                    self._sync_invoke,
                    message  # 传入完整消息对象
                )
                logger.info(f"✅ Agent 调用完成，结果长度={len(result) if result else 0}")

                # 更新会话活动记录
                self.session_manager.update_session_activity(
                    message.channel_name,
                    message.user_id,
                    message_count=1
                )

                return result
            except (aiohttp.ClientError, asyncio.TimeoutError, ConnectionError) as e:
                logger.warning(f"⚠️ 调用失败: {e}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"第 {attempt + 1} 次重试，等待 {wait_time} 秒...")
                    await asyncio.sleep(wait_time)
                else:
                    raise

    def _sync_invoke(self, message: ChannelMessage) -> str:
        """同步调用 agent（在线程池中运行，单用户模式，带历史持久化）"""
        # 使用全局 Agent
        agent = self._get_agent()

        thread_id = f"{message.channel_name}_{message.user_id}_{message.message_type}"
        channel = message.channel_name
        user_id = message.user_id

        # 获取历史管理器
        history_manager = get_history_manager(channel, user_id)

        # 恢复历史到 agent 的 checkpointer
        self._restore_history_to_agent(agent, thread_id, history_manager)

        # 记录用户消息
        history_manager.add_human_message(message.content)

        # 调用 agent
        response = agent.invoke(query=message.content, thread_id=thread_id)

        # 记录 AI 响应
        if response:
            history_manager.add_ai_message(response)

        # 保存历史
        history_manager.save()

        return response

    def _restore_history_to_agent(self, agent, thread_id: str, history_manager):
        """将历史从 history_manager 恢复到 agent 的 checkpointer"""
        if not agent or not agent.checkpointer:
            return

        from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

        messages = history_manager.get_messages()
        if not messages:
            return

        config = {"configurable": {"thread_id": thread_id}}

        reconstructed = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "human":
                reconstructed.append(HumanMessage(content=content))
            elif role == "ai":
                reconstructed.append(AIMessage(content=content))
            elif role == "tool":
                tool_name = msg.get("metadata", {}).get("tool_name", "unknown")
                reconstructed.append(ToolMessage(content=content, tool_call_id="restored", name=tool_name))

        if reconstructed:
            checkpoint = {"messages": reconstructed}
            try:
                agent.checkpointer.put(
                    config,
                    checkpoint,
                    {"source": "checkpoint", "timestamp": time.time()}
                )
            except Exception:
                pass
    
    def _record_response_time(self, duration: float):
        """记录响应时间"""
        self._response_times.append(duration)
        if self._response_times:
            self._stats["avg_response_time"] = sum(self._response_times) / len(self._response_times)
    
    def _check_global_permission(self, user_id: str) -> bool:
        """检查全局权限"""
        global_config = self.config.get("global", {})
        whitelist = global_config.get("whitelist", [])
        
        if not whitelist:
            return True
        
        return str(user_id) in [str(u) for u in whitelist]
    
    async def start(self):
        """启动所有渠道"""
        if not self.channels:
            logger.error("❌ 没有可用的渠道，无法启动")
            return
        
        self._running = True
        
        self._setup_signal_handlers()
        
        logger.info(f"\n🚀 正在启动 {len(self.channels)} 个渠道...\n")
        
        tasks = []
        for name, channel in self.channels.items():
            try:
                task = asyncio.create_task(channel.start())
                tasks.append(task)
            except Exception as e:
                logger.error(f"❌ 启动 {name} 渠道失败: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("\n✅ 所有渠道已启动，按 Ctrl+C 停止\n")
        
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        
        await self.stop()
    
    async def stop(self):
        """停止所有渠道"""
        if not self._running:
            return
        
        self._running = False
        logger.info("\n⏹️ 正在停止所有渠道...")
        
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info(f"🛑 {name} 渠道已停止")
            except Exception as e:
                logger.error(f"❌ 停止 {name} 渠道失败: {e}")
        
        logger.info("✅ 所有渠道已停止")
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(sig, frame):
            logger.info(f"\n📡 收到信号 {sig}，准备停止...")
            self._shutdown_event.set()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def get_stats(self) -> dict:
        """获取运行统计"""
        total = self._stats["total_messages"]
        return {
            "total_messages": total,
            "success": self._stats["success"],
            "failed": self._stats["failed"],
            "success_rate": (
                (total - self._stats["failed"]) / total * 100
                if total > 0 else 0
            ),
            "avg_response_time": f"{self._stats['avg_response_time']:.2f}s",
            "errors_by_channel": self._stats["errors_by_channel"],
            "active_channels": list(self.channels.keys())
        }


def main():
    """入口函数"""
    parser = argparse.ArgumentParser(description="OllamaPilot Channels 运行器")
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="配置文件路径 (默认: channels/config.yaml)"
    )
    parser.add_argument(
        "--version", "-v",
        action="store_true",
        help="显示版本信息"
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="显示统计信息"
    )
    
    args = parser.parse_args()
    
    if args.version:
        print("OllamaPilot Channels v1.0.0")
        return
    
    try:
        runner = ChannelRunner(args.config)
        
        if args.stats:
            print(runner.get_stats())
            return
        
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        print("\n👋 已取消")
    except Exception as e:
        logger.error(f"\n❌ 运行出错: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
