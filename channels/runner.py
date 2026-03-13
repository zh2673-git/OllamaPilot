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
        """初始化 OllamaPilot Agent"""
        try:
            from ollamapilot import init_ollama_model, OllamaPilotAgent
            
            agent_config = self.config.get("agent", {})
            model_name = agent_config.get("model", "qwen3.5:4b")
            skills_dir = agent_config.get("skills_dir", "./skills")
            verbose = agent_config.get("verbose", True)
            
            logger.info(f"🤖 正在初始化 Agent...")
            logger.info(f"   模型: {model_name}")
            logger.info(f"   Skills 目录: {skills_dir}")
            
            model = init_ollama_model(model_name)
            self.agent = OllamaPilotAgent(
                model=model,
                skills_dir=skills_dir,
                verbose=verbose
            )
            
            logger.info("✅ Agent 初始化完成")
            
        except ImportError as e:
            logger.error(f"❌ 导入 OllamaPilot 失败: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ 初始化 Agent 失败: {e}")
            raise
    
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
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError, ConnectionError)),
        before_sleep=lambda retry_state: logger.warning(
            f"第 {retry_state.attempt_number} 次重试..."
        )
    )
    async def _invoke_with_retry(self, message: ChannelMessage) -> str:
        """调用 Agent（带重试机制）"""
        thread_id = f"{message.channel_name}_{message.user_id}_{message.message_type}"
        
        if hasattr(self.agent, 'ainvoke'):
            return await self.agent.ainvoke(query=message.content, thread_id=thread_id)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                lambda: self.agent.invoke(query=message.content, thread_id=thread_id)
            )
    
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
