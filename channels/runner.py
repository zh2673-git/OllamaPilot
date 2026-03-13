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
import signal
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    print("❌ 请先安装 PyYAML: pip install pyyaml")
    sys.exit(1)

# 添加项目根目录到路径，确保可以导入 ollamapilot
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from .base import Channel, ChannelMessage
from .qq import QQChannel
from .feishu import FeishuChannel
from .dingtalk import DingTalkChannel


class ChannelRunner:
    """
    渠道统一运行器
    
    负责:
    1. 加载配置
    2. 初始化 OllamaPilot Agent
    3. 初始化并管理所有渠道
    4. 处理消息并返回响应
    
    Example:
        >>> runner = ChannelRunner("channels/config.yaml")
        >>> asyncio.run(runner.start())
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化运行器
        
        Args:
            config_path: 配置文件路径，默认使用 channels/config.yaml
        """
        # 加载配置
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)
        
        self.config = self._load_config(config_path)
        
        # 初始化 Agent
        self.agent = None
        self._init_agent()
        
        # 初始化渠道
        self.channels: List[Channel] = []
        self._init_channels()
        
        # 运行状态
        self._running = False
        self._shutdown_event = asyncio.Event()
    
    def _load_config(self, config_path: Path) -> Dict[str, Any]:
        """加载配置文件"""
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        print(f"✅ 已加载配置: {config_path}")
        return config
    
    def _init_agent(self):
        """初始化 OllamaPilot Agent"""
        try:
            from ollamapilot import init_ollama_model, OllamaPilotAgent
            
            agent_config = self.config.get("agent", {})
            model_name = agent_config.get("model", "qwen3.5:4b")
            skills_dir = agent_config.get("skills_dir", "./skills")
            verbose = agent_config.get("verbose", True)
            
            print(f"🤖 正在初始化 Agent...")
            print(f"   模型: {model_name}")
            print(f"   Skills 目录: {skills_dir}")
            
            model = init_ollama_model(model_name)
            self.agent = OllamaPilotAgent(
                model=model,
                skills_dir=skills_dir,
                verbose=verbose
            )
            
            print("✅ Agent 初始化完成")
            
        except ImportError as e:
            print(f"❌ 导入 OllamaPilot 失败: {e}")
            print(f"   请确保 OllamaPilot 已正确安装")
            raise
        except Exception as e:
            print(f"❌ 初始化 Agent 失败: {e}")
            raise
    
    def _init_channels(self):
        """初始化所有启用的渠道"""
        channel_configs = self.config.get("channels", {})
        
        # QQ
        qq_config = channel_configs.get("qq", {})
        if qq_config.get("enabled", False):
            try:
                channel = QQChannel(qq_config, self._handle_message)
                self.channels.append(channel)
                print("✅ QQ 渠道已加载")
            except Exception as e:
                print(f"⚠️ 加载 QQ 渠道失败: {e}")
        
        # 飞书
        feishu_config = channel_configs.get("feishu", {})
        if feishu_config.get("enabled", False):
            try:
                channel = FeishuChannel(feishu_config, self._handle_message)
                self.channels.append(channel)
                print("✅ 飞书渠道已加载")
            except Exception as e:
                print(f"⚠️ 加载飞书渠道失败: {e}")
        
        # 钉钉
        dingtalk_config = channel_configs.get("dingtalk", {})
        if dingtalk_config.get("enabled", False):
            try:
                channel = DingTalkChannel(dingtalk_config, self._handle_message)
                self.channels.append(channel)
                print("✅ 钉钉渠道已加载")
            except Exception as e:
                print(f"⚠️ 加载钉钉渠道失败: {e}")
        
        if not self.channels:
            print("⚠️ 没有启用的渠道，请检查配置")
    
    async def _handle_message(self, message: ChannelMessage) -> str:
        """
        处理收到的消息（非流式，完整返回）
        
        Args:
            message: 标准化消息
        
        Returns:
            完整的回复内容
        """
        # 全局权限检查
        if not self._check_global_permission(message.user_id):
            return "⛔ 您没有使用权限"
        
        # 构建独立会话ID（每个用户独立上下文）
        thread_id = f"{message.user_id}_{message.message_type}"
        
        try:
            # 调用 Agent（非流式，等待完整结果）
            # TODO: 未来可升级为流式输出
            response = self.agent.invoke(
                query=message.content,
                thread_id=thread_id
            )
            return response
        except Exception as e:
            print(f"❌ Agent 处理消息失败: {e}")
            return f"❌ 处理出错: {str(e)}"
    
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
            print("❌ 没有可用的渠道，无法启动")
            return
        
        self._running = True
        
        # 设置信号处理
        self._setup_signal_handlers()
        
        print(f"\n🚀 正在启动 {len(self.channels)} 个渠道...\n")
        
        # 启动所有渠道
        start_tasks = [channel.start() for channel in self.channels]
        await asyncio.gather(*start_tasks, return_exceptions=True)
        
        print("\n✅ 所有渠道已启动，按 Ctrl+C 停止\n")
        
        # 等待停止信号
        try:
            await self._shutdown_event.wait()
        except asyncio.CancelledError:
            pass
        
        # 停止所有渠道
        await self.stop()
    
    async def stop(self):
        """停止所有渠道"""
        if not self._running:
            return
        
        self._running = False
        print("\n⏹️ 正在停止所有渠道...")
        
        # 停止所有渠道
        stop_tasks = [channel.stop() for channel in self.channels]
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        print("✅ 所有渠道已停止")
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        def signal_handler(sig, frame):
            print(f"\n📡 收到信号 {sig}，准备停止...")
            self._shutdown_event.set()
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
        signal.signal(signal.SIGTERM, signal_handler)  # kill


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
    
    args = parser.parse_args()
    
    if args.version:
        from . import __version__
        print(f"OllamaPilot Channels v{__version__}")
        return
    
    # 创建运行器并启动
    try:
        runner = ChannelRunner(args.config)
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        print("\n👋 已取消")
    except Exception as e:
        print(f"\n❌ 运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
