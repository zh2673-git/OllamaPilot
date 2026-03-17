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
        """初始化 OllamaPilot Agent（Channels 使用用户级 SQLite 持久化）"""
        try:
            from ollamapilot import init_ollama_model, OllamaPilotAgent

            agent_config = self.config.get("agent", {})
            model_name = agent_config.get("model", "qwen3.5:4b")
            skills_dir = agent_config.get("skills_dir", "./skills")
            verbose = agent_config.get("verbose", True)

            logger.info(f"🤖 正在初始化 Agent...")
            logger.info(f"   模型: {model_name}")
            logger.info(f"   Skills 目录: {skills_dir}")
            logger.info(f"   记忆模式: 用户级 SQLite 持久化（Channels 专用）")

            model = init_ollama_model(model_name)
            # 不在这里创建 agent，改为每次消息时创建（使用用户特定的 checkpointer）
            self.model = model
            self.agent_config = {
                "skills_dir": skills_dir,
                "verbose": verbose
            }

            logger.info("✅ Agent 模板初始化完成（每个用户独立）")

        except ImportError as e:
            logger.error(f"❌ 导入 OllamaPilot 失败: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 初始化 Agent 失败: {e}")
            raise

    def _get_or_create_user_agent(self, channel: str, user_id: str) -> Any:
        """
        获取或创建用户专属的 Agent（带缓存）

        每个用户有独立的 SQLite 数据库和 checkpointer
        Agent 实例会被缓存，避免重复加载 Skill

        Args:
            channel: 渠道名称
            user_id: 用户 ID

        Returns:
            OllamaPilotAgent 实例
        """
        from ollamapilot import OllamaPilotAgent

        # 构建缓存键
        cache_key = f"{channel}:{user_id}"

        # 检查缓存
        if hasattr(self, '_user_agents') and cache_key in self._user_agents:
            return self._user_agents[cache_key]

        # 初始化缓存字典
        if not hasattr(self, '_user_agents'):
            self._user_agents = {}

        # 获取用户的 checkpointer
        checkpointer = self.session_manager.get_checkpointer(channel, user_id)

        logger.info(f"🤖 为用户 {user_id} 创建新的 Agent 实例...")

        # 创建用户专属的 Agent
        agent = OllamaPilotAgent(
            model=self.model,
            skills_dir=self.agent_config["skills_dir"],
            verbose=self.agent_config["verbose"],
            enable_memory=True,
            checkpointer=checkpointer  # 使用用户的持久化 checkpointer
        )

        # 缓存起来
        self._user_agents[cache_key] = agent
        logger.info(f"✅ 用户 {user_id} 的 Agent 已创建并缓存")

        return agent
    
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
        """同步调用 agent（在线程池中运行，每个用户独立）"""
        # 获取或创建用户专属的 Agent（带缓存）
        agent = self._get_or_create_user_agent(message.channel_name, message.user_id)

        thread_id = f"{message.channel_name}_{message.user_id}_{message.message_type}"
        return agent.invoke(query=message.content, thread_id=thread_id)
    
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
