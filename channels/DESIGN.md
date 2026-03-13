# OllamaPilot Channels 设计方案

## 一、设计目标

为 OllamaPilot 添加多渠道远程控制能力，支持通过 QQ、飞书、钉钉等即时通讯工具远程调用 Agent。

### 核心原则

1. **零侵入**: 不修改现有 `main.py` 和 `ollamapilot/` 任何代码
2. **完全独立**: Channels 作为独立包，与核心解耦
3. **可扩展**: 易于添加新的通讯渠道
4. **个人使用**: 简化设计，专注单人远程控制场景

---

## 二、架构设计

```
┌─────────────────────────────────────────────────────────────────┐
│                        完全解耦架构                              │
│                                                                 │
│   ┌─────────────────┐              ┌─────────────────┐         │
│   │   main.py       │              │  channels/      │         │
│   │   (原有入口)     │              │  (新增独立包)    │         │
│   │                 │              │                 │         │
│   │  python main.py │              │  python -m      │         │
│   │  → 交互式CLI    │   独立运行    │  channels.runner│         │
│   │                 │◄────────────►│  → QQ/飞书/钉钉 │         │
│   │  - 本地开发     │   互不影响    │                 │         │
│   │  - 终端交互     │              │  - 远程控制     │         │
│   │  - 调试测试     │              │  - 消息接收     │         │
│   │                 │              │  - 自动回复     │         │
│   └─────────────────┘              └─────────────────┘         │
│           ↑                                ↑                   │
│           │         共享 Agent 实例         │                   │
│           └────────────────────────────────┘                   │
│                     (共享内存模式)                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 三、目录结构

```
OllamaPilot/
├── main.py                      # ✅ 原有入口，完全不动
├── ollamapilot/                 # ✅ 核心包，完全不动
│   ├── __init__.py
│   ├── agent.py
│   ├── cli/
│   ├── skills/
│   ├── tools/
│   └── ...
│
├── channels/                    # ✅ 新增独立包
│   ├── __init__.py              # 包导出
│   ├── DESIGN.md                # 本文档
│   ├── base.py                  # 渠道基类、消息格式
│   ├── qq.py                    # QQ Bot 实现 (go-cqhttp)
│   ├── feishu.py                # 飞书 Bot 实现
│   ├── dingtalk.py              # 钉钉 Bot 实现
│   ├── runner.py                # 统一运行器入口
│   └── config.yaml              # 渠道配置文件
│
├── skills/                      # 用户自定义 skills
├── .env                         # 原有配置
└── pyproject.toml               # 原有依赖
```

---

## 四、启动方式对比

### 原有使用方式（完全不变）

```bash
# 本地交互式使用
$ python main.py

# 或指定模型
$ python main.py --model qwen3.5:4b

# 或交互式选择模型
$ python main.py --interactive
```

### 新增使用方式

```bash
# 启动渠道服务（QQ/飞书/钉钉）
$ python -m channels.runner

# 输出示例：
# ✅ Agent 初始化完成
# ✅ QQ 渠道已加载 (Bot: 123456789)
# ✅ 飞书渠道已加载 (App: cli_xxx)
# 🤖 等待消息...
```

### 同时使用场景

```bash
# 终端 1：本地开发调试
$ python main.py
🤖 OllamaPilot 已启动
> 你好
你好！有什么可以帮助你的吗？

# 终端 2：远程渠道服务（同时运行）
$ python -m channels.runner
✅ Agent 初始化完成
✅ QQ 渠道已加载
🤖 等待消息...
```

---

## 五、核心模块设计

### 5.1 消息格式 (base.py)

```python
@dataclass
class ChannelMessage:
    """标准化消息格式"""
    message_id: str           # 消息唯一ID
    user_id: str              # 发送者ID
    user_name: str            # 发送者昵称
    content: str              # 消息内容（纯文本）
    message_type: str         # "private" | "group"
    group_id: Optional[str]   # 群ID（群聊时）
    raw_data: Any             # 原始消息数据
    timestamp: datetime       # 消息时间
    images: List[str]         # 图片URL列表
    at_me: bool               # 是否@机器人
```

### 5.2 渠道基类 (base.py)

参考 nanobot 的设计，使用更完善的基类结构：

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ChannelMessage:
    """标准化消息格式（参考 nanobot）"""
    message_id: str
    user_id: str
    user_name: str
    content: str
    message_type: str  # "private" | "group" | "channel"
    channel_name: str  # "qq" | "feishu" | "dingtalk"
    group_id: Optional[str] = None
    group_name: Optional[str] = None
    raw_data: Any = None
    timestamp: datetime = None
    images: List[str] = None
    at_me: bool = False
    reply_to: Optional[str] = None  # 回复的消息ID
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.images is None:
            self.images = []


@dataclass
class ChannelResponse:
    """渠道响应格式"""
    content: str
    message_type: str = "text"  # text | markdown | image | card
    buttons: Optional[List[Dict]] = None
    image_url: Optional[str] = None


class Channel(ABC):
    """渠道基类（参考 nanobot 设计）"""
    
    # 类属性：渠道标识
    name: str = ""           # 渠道名称，如 "qq"
    description: str = ""    # 渠道描述
    
    def __init__(self, config: Dict[str, Any], message_handler: Optional[Callable] = None):
        self.config = config
        self.message_handler = message_handler
        self._running = False
        self._session = None
        self._task = None
    
    @abstractmethod
    async def start(self):
        """启动渠道监听"""
        pass
    
    @abstractmethod
    async def stop(self):
        """停止渠道监听"""
        pass
    
    @abstractmethod
    async def send_message(self, user_id: str, content: str, **kwargs) -> bool:
        """发送消息给用户"""
        pass
    
    @abstractmethod
    async def send_group_message(self, group_id: str, content: str, **kwargs) -> bool:
        """发送群消息"""
        pass
    
    def check_permission(self, user_id: str) -> bool:
        """检查用户权限（白名单）"""
        whitelist = self.config.get("whitelist", [])
        if not whitelist:
            return True
        return user_id in whitelist
    
    async def handle_incoming_message(self, message: ChannelMessage) -> Optional[ChannelResponse]:
        """处理收到的消息（统一入口）"""
        if not self.check_permission(message.user_id):
            logger.warning(f"用户 {message.user_id} 无权限访问")
            return ChannelResponse(content="⛔ 您没有使用权限")
        
        if self.message_handler:
            try:
                response = await self.message_handler(message)
                if isinstance(response, str):
                    return ChannelResponse(content=response)
                return response
            except Exception as e:
                logger.error(f"处理消息失败: {e}", exc_info=True)
                return ChannelResponse(content=f"❌ 处理出错: {str(e)[:100]}")
        return None
```

### 5.3 渠道注册表 (registry.py)

参考 nanobot 的自动发现机制：

```python
"""渠道注册表 - 自动发现和注册渠道（参考 nanobot）"""
import pkgutil
import importlib
from typing import Dict, Type, Optional
from .base import Channel

# 全局注册表
_channel_registry: Dict[str, Type[Channel]] = {}


def register_channel(channel_class: Type[Channel]):
    """注册渠道类"""
    if not hasattr(channel_class, 'name') or not channel_class.name:
        raise ValueError(f"渠道类 {channel_class.__name__} 必须定义 name 属性")
    
    _channel_registry[channel_class.name] = channel_class
    return channel_class


def get_channel(name: str) -> Optional[Type[Channel]]:
    """获取渠道类"""
    return _channel_registry.get(name)


def list_channels() -> Dict[str, Type[Channel]]:
    """列出所有已注册的渠道"""
    return _channel_registry.copy()


def auto_discover_channels():
    """自动发现 channels 包中的所有渠道（参考 nanobot）"""
    from . import qq, feishu, dingtalk  # 显式导入确保注册
    
    # 自动扫描 channels 包
    import channels
    for _, name, _ in pkgutil.iter_modules(channels.__path__):
        if not name.startswith('_'):
            try:
                importlib.import_module(f'channels.{name}')
            except Exception as e:
                print(f"无法导入渠道模块 {name}: {e}")
```

### 5.4 运行器 (runner.py)

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import logging
from .registry import get_channel, auto_discover_channels
from .base import ChannelMessage, ChannelResponse

logger = logging.getLogger(__name__)


class ChannelRunner:
    """渠道统一运行器（参考 nanobot + CoPaw）"""
    
    def __init__(self, config_path: str = None):
        # 加载配置
        self.config = load_config(config_path)
        
        # 初始化 Agent（直接导入核心）
        self._init_agent()
        
        # 自动发现渠道
        auto_discover_channels()
        
        # 初始化启用的渠道
        self.channels: Dict[str, Channel] = {}
        self._init_channels()
        
        # 统计信息
        self._stats = {
            "total_messages": 0,
            "success": 0,
            "failed": 0,
            "errors_by_channel": {},
        }
    
    def _init_agent(self):
        """初始化 OllamaPilot Agent"""
        from ollamapilot import init_ollama_model, OllamaPilotAgent
        
        model = init_ollama_model(self.config["agent"]["model"])
        self.agent = OllamaPilotAgent(
            model=model,
            skills_dir=self.config["agent"].get("skills_dir"),
            verbose=True
        )
    
    def _init_channels(self):
        """初始化启用的渠道（参考 nanobot）"""
        for channel_name, channel_config in self.config.get("channels", {}).items():
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
    
    async def _handle_message(self, message: ChannelMessage) -> ChannelResponse:
        """处理收到的消息（带错误重试）"""
        self._stats["total_messages"] += 1
        channel_name = message.channel_name
        
        try:
            # 使用重试机制调用 Agent
            response = await self._invoke_with_retry(message)
            self._stats["success"] += 1
            
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
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=lambda retry_state: logger.warning(
            f"第 {retry_state.attempt_number} 次重试..."
        )
    )
    async def _invoke_with_retry(self, message: ChannelMessage) -> str:
        """调用 Agent（带重试机制）"""
        # 构建独立会话ID（每个用户每个渠道独立上下文）
        thread_id = f"{message.channel_name}_{message.user_id}_{message.message_type}"
        
        # 调用 Agent（非流式，等待完整结果）
        response = await self._invoke_agent(message.content, thread_id)
        return response
    
    async def _invoke_agent(self, query: str, thread_id: str) -> str:
        """实际调用 Agent（可被子类覆盖）"""
        # 如果 agent 支持异步，使用 ainvoke
        if hasattr(self.agent, 'ainvoke'):
            return await self.agent.ainvoke(query=query, thread_id=thread_id)
        # 否则在线程池中运行同步方法
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.agent.invoke(query=query, thread_id=thread_id)
        )
    
    async def start(self):
        """启动所有渠道"""
        tasks = []
        for name, channel in self.channels.items():
            try:
                task = asyncio.create_task(channel.start())
                tasks.append(task)
                logger.info(f"🚀 {name} 渠道已启动")
            except Exception as e:
                logger.error(f"❌ 启动 {name} 渠道失败: {e}")
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def stop(self):
        """停止所有渠道"""
        for name, channel in self.channels.items():
            try:
                await channel.stop()
                logger.info(f"🛑 {name} 渠道已停止")
            except Exception as e:
                logger.error(f"❌ 停止 {name} 渠道失败: {e}")
    
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
            "errors_by_channel": self._stats["errors_by_channel"],
            "active_channels": list(self.channels.keys())
        }
```

---

## 六、渠道实现详情

### 6.1 QQ 渠道 (qq.py)

**✅ 好消息**: QQ 开放平台已于 2026年3月 向个人用户开放！

**技术方案**: QQ 开放平台官方 Bot API（推荐）

**功能特性**:
- 私聊消息接收与回复
- 群聊消息接收（支持 @触发）
- 频道消息支持
- 富文本消息（Markdown、图片）
- 消息按钮/卡片
- 长消息自动拆分

**配置项**:
```yaml
qq:
  enabled: true
  app_id: "1024xxxxxx"                   # QQ 开放平台 AppID
  app_secret: "xxxxxxxxxx"               # QQ 开放平台 AppSecret
  token: "xxxxxxxxxx"                    # 消息验证 Token
  sandbox: false                         # 是否使用沙箱环境
  whitelist:                             # 用户 ID 白名单
    - "123456789"                        # 你的 QQ 号
  at_only_in_group: true                 # 群聊中只响应 @消息
  intents:                               # 订阅的事件类型
    - "GUILD_MESSAGES"                   # 频道消息
    - "DIRECT_MESSAGE"                   # 私信
    - "GROUP_MESSAGES"                   # 群消息（需申请权限）
```

**部署步骤**:
1. 访问 [QQ 开放平台](https://q.qq.com/)
2. 使用 QQ 扫码登录，创建机器人
3. 获取 AppID、AppSecret、Token
4. 配置事件订阅（WebSocket 或 Webhook）
5. 配置 channels/config.yaml
6. 启动 runner

**参考实现**: [nanobot/channels/qq.py](https://github.com/HKUDS/nanobot/blob/main/nanobot/channels/qq.py)

### 6.2 飞书渠道 (feishu.py)

**技术方案**: 飞书开放平台 Bot API

**功能特性**:
- 私聊消息
- 群聊消息（@触发）
- 富文本消息
- 卡片消息

**配置项**:
```yaml
feishu:
  enabled: true
  app_id: "cli_xxxxxxxxxx"               # 飞书应用 ID
  app_secret: "xxxxxxxxxx"               # 飞书应用密钥
  encrypt_key: ""                         # 消息加密密钥（可选）
  verification_token: ""                  # 验证 Token
  whitelist:                              # 用户 OpenID 白名单
    - "ou_xxxxxxxxxx"
```

**部署步骤**:
1. 登录 [飞书开放平台](https://open.feishu.cn/)
2. 创建企业自建应用
3. 开启机器人能力，获取凭证
4. 配置事件订阅（HTTP 或 WebSocket）
5. 配置 channels/config.yaml

### 6.3 钉钉渠道 (dingtalk.py)

**技术方案**: 钉钉开放平台 Bot API

**功能特性**:
- 企业内部机器人
- 群聊消息（@触发）
- Markdown 消息

**配置项**:
```yaml
dingtalk:
  enabled: true
  app_key: "xxxxxxxxxx"                  # 钉钉 AppKey
  app_secret: "xxxxxxxxxx"               # 钉钉 AppSecret
  robot_code: ""                          # 机器人编码
  whitelist:                              # 用户 UserID 白名单
    - "User123"
```

**部署步骤**:
1. 登录 [钉钉开放平台](https://open.dingtalk.com/)
2. 创建企业内部应用
3. 添加机器人能力
4. 配置事件订阅
5. 配置 channels/config.yaml

---

## 七、配置文件 (config.yaml)

参考 nanobot 的配置结构：

```yaml
# ============================================
# Channels 配置文件
# ============================================

# Agent 配置
agent:
  model: "qwen3.5:4b"                    # 对话模型
  skills_dir: "./skills"                 # Skill 目录
  verbose: true                          # 显示详细日志
  max_workers: 5                         # 消息处理工作者数量

# 全局权限配置
global:
  whitelist: []                           # 全局白名单（所有渠道生效）
  admin_users: []                         # 管理员用户（可使用管理命令）
  max_message_length: 4000                # 最大消息长度

# 渠道配置（统一结构）
channels:
  # QQ 渠道（官方 Bot API）
  qq:
    enabled: true
    app_id: "1024xxxxxx"                 # QQ 开放平台 AppID
    app_secret: "xxxxxxxxxx"             # QQ 开放平台 AppSecret
    token: "xxxxxxxxxx"                  # 消息验证 Token
    sandbox: false                       # 是否使用沙箱环境
    whitelist:
      - "123456789"                      # 你的 QQ 号
    at_only_in_group: true
    intents:
      - "GUILD_MESSAGES"
      - "DIRECT_MESSAGE"
      - "GROUP_MESSAGES"

  # 飞书渠道
  feishu:
    enabled: false
    app_id: "cli_xxxxxxxxxx"
    app_secret: "xxxxxxxxxx"
    encrypt_key: ""                       # 消息加密密钥（可选）
    verification_token: ""                # 验证 Token
    whitelist: []

  # 钉钉渠道
  dingtalk:
    enabled: false
    app_key: "xxxxxxxxxx"
    app_secret: "xxxxxxxxxx"
    robot_code: ""                        # 机器人编码
    whitelist: []

# 日志配置
logging:
  level: "INFO"                          # DEBUG | INFO | WARNING | ERROR
  file: "logs/channels.log"              # 日志文件路径
  max_bytes: 10485760                    # 10MB
  backup_count: 5                        # 保留日志文件数量
```

---

## 八、使用流程

### 8.1 首次部署

```bash
# 1. 确保 OllamaPilot 核心可正常运行
$ python main.py

# 2. 配置 channels/config.yaml
# 编辑配置文件，启用需要的渠道

# 3. 部署对应渠道的 Bot 服务
# - QQ: 启动 go-cqhttp
# - 飞书: 创建应用，配置事件订阅
# - 钉钉: 创建应用，配置机器人

# 4. 启动渠道服务
$ python -m channels.runner
```

### 8.2 日常使用

```bash
# 场景 1：本地开发
$ python main.py

# 场景 2：远程控制（另一个终端）
$ python -m channels.runner

# 场景 3：同时使用
# 终端 1: python main.py
# 终端 2: python -m channels.runner
```

### 8.3 消息交互示例

**QQ 私聊**:
```
用户: 你好
Bot: 你好！有什么可以帮助你的吗？

用户: 查一下明天苏州天气
Bot: 🌤️ 苏州明天天气：晴，15-22℃

用户: 帮我读取文件 D:\test.txt
Bot: 📄 文件内容：
     Hello World!
```

**QQ 群聊**:
```
用户: @小助手 总结一下今天的会议记录
Bot: 📋 会议总结：
     1. xxx
     2. xxx
```

---

## 九、安全设计

### 9.1 权限控制

| 层级 | 控制方式 | 说明 |
|------|---------|------|
| 全局白名单 | `global.whitelist` | 所有渠道生效 |
| 渠道白名单 | `qq.whitelist` 等 | 仅该渠道生效 |
| 管理员 | `global.admin_users` | 可使用管理命令 |

### 9.2 敏感操作保护

```python
# 危险操作列表
DANGEROUS_OPERATIONS = [
    "删除文件", "删除文件夹",
    "执行命令 rm", "执行命令 del",
    "修改系统配置"
]

# 检测到危险操作时，要求二次确认
if is_dangerous_operation(query):
    return "⚠️ 检测到危险操作，请确认：\n回复 '确认执行' 继续"
```

### 9.3 会话隔离

```python
# 每个用户独立会话，互不干扰
thread_id = f"{message.user_id}_{message.message_type}"

# 示例：
# QQ用户A: thread_id = "123456_private"
# QQ用户B: thread_id = "789012_private"
# QQ群C:   thread_id = "345678_group"
```

---

## 十、消息处理流程（参考 CoPaw）

### 10.1 消息队列设计

参考 CoPaw 的异步消息处理机制：

```python
import asyncio
from collections import deque
from typing import Dict, Deque
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MessageTask:
    """消息任务（参考 CoPaw）"""
    message: ChannelMessage
    response_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    created_at: datetime = field(default_factory=datetime.now)
    retry_count: int = 0
    
    async def get_response(self, timeout: float = 300.0) -> ChannelResponse:
        """等待响应（带超时）"""
        try:
            return await asyncio.wait_for(
                self.response_queue.get(),
                timeout=timeout
            )
        except asyncio.TimeoutError:
            return ChannelResponse(content="⏱️ 处理超时，请稍后重试")


class MessageQueue:
    """消息队列管理器（参考 CoPaw 设计）"""
    
    def __init__(self, max_size: int = 100):
        self._queue: asyncio.Queue[MessageTask] = asyncio.Queue(maxsize=max_size)
        self._processing: Dict[str, MessageTask] = {}  # 正在处理的消息
        self._history: Deque[MessageTask] = deque(maxlen=1000)  # 历史记录
    
    async def put(self, message: ChannelMessage) -> MessageTask:
        """添加消息到队列"""
        task = MessageTask(message=message)
        await self._queue.put(task)
        return task
    
    async def get(self) -> MessageTask:
        """获取待处理消息"""
        task = await self._queue.get()
        self._processing[task.message.message_id] = task
        return task
    
    def complete(self, task: MessageTask, response: ChannelResponse):
        """完成消息处理"""
        # 将响应放入队列，通知等待者
        asyncio.create_task(task.response_queue.put(response))
        
        # 从处理中移除
        if task.message.message_id in self._processing:
            del self._processing[task.message.message_id]
        
        # 添加到历史
        self._history.append(task)
    
    def get_queue_status(self) -> dict:
        """获取队列状态"""
        return {
            "pending": self._queue.qsize(),
            "processing": len(self._processing),
            "history": len(self._history)
        }
```

### 10.2 消费者模式（参考 CoPaw）

```python
class ChannelConsumer:
    """消息消费者（参考 CoPaw）"""
    
    def __init__(self, runner: 'ChannelRunner', max_workers: int = 5):
        self.runner = runner
        self.queue = MessageQueue()
        self.max_workers = max_workers
        self._workers: List[asyncio.Task] = []
        self._running = False
    
    async def start(self):
        """启动消费者"""
        self._running = True
        
        # 启动多个工作协程
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self._workers.append(worker)
        
        logger.info(f"✅ 消息消费者已启动，{self.max_workers} 个工作者")
    
    async def stop(self):
        """停止消费者"""
        self._running = False
        
        # 取消所有工作者
        for worker in self._workers:
            worker.cancel()
        
        # 等待取消完成
        await asyncio.gather(*self._workers, return_exceptions=True)
        logger.info("🛑 消息消费者已停止")
    
    async def _worker_loop(self, worker_id: str):
        """工作者循环"""
        while self._running:
            try:
                # 获取待处理消息
                task = await self.queue.get()
                message = task.message
                
                logger.debug(f"[{worker_id}] 处理消息: {message.message_id}")
                
                # 处理消息
                try:
                    response = await self.runner._handle_message(message)
                except Exception as e:
                    logger.error(f"[{worker_id}] 处理失败: {e}")
                    response = ChannelResponse(content=f"❌ 处理失败: {str(e)[:100]}")
                
                # 完成处理
                self.queue.complete(task, response)
                
                # 发送响应到渠道
                await self._send_response(message, response)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[{worker_id}] 工作者异常: {e}", exc_info=True)
    
    async def _send_response(self, message: ChannelMessage, response: ChannelResponse):
        """发送响应到对应渠道"""
        channel = self.runner.channels.get(message.channel_name)
        if not channel:
            logger.error(f"渠道 {message.channel_name} 不存在")
            return
        
        try:
            if message.message_type == "private":
                await channel.send_message(message.user_id, response.content)
            else:
                await channel.send_group_message(
                    message.group_id, 
                    response.content,
                    reply_to=message.message_id
                )
        except Exception as e:
            logger.error(f"发送响应失败: {e}")
    
    async def submit(self, message: ChannelMessage) -> ChannelResponse:
        """提交消息并等待响应"""
        task = await self.queue.put(message)
        return await task.get_response()
```

### 10.3 完整的消息处理流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   渠道收到   │────▶│  创建消息   │────▶│  提交到队列  │────▶│  等待处理   │
│   消息      │     │   任务     │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   发送响应   │◀────│  获取响应   │◀────│  Agent处理  │◀────│  消费者获取 │
│   给用户    │     │   并返回    │     │   消息     │     │   消息     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 十一、错误重试机制

### 11.1 为什么需要错误重试

在实际运行中，可能会遇到以下问题：
- **网络波动**: 与 API 服务的连接不稳定
- **API 限流**: 平台对请求频率有限制
- **服务暂时不可用**: Ollama 或 LLM 服务短暂中断
- **超时**: 复杂查询处理时间较长

### 11.2 重试策略设计

使用 `tenacity` 库实现智能重试：

```python
from tenacity import (
    retry,
    stop_after_attempt,      # 最大重试次数
    wait_exponential,        # 指数退避等待
    retry_if_exception_type, # 按异常类型重试
    before_sleep,            # 重试前回调
    retry_if_result,         # 按结果重试
)

# API 调用重试（带指数退避）
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((
        aiohttp.ClientError,
        asyncio.TimeoutError,
        ConnectionError,
    )),
    before_sleep=lambda retry_state: logger.warning(
        f"第 {retry_state.attempt_number} 次重试，等待 {retry_state.next_action.sleep} 秒..."
    )
)
async def api_call_with_retry(func, *args, **kwargs):
    """带重试的 API 调用"""
    return await func(*args, **kwargs)

# 消息发送重试（更激进）
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_result(lambda result: result is False),  # 发送失败时重试
)
async def send_message_with_retry(channel, user_id: str, content: str) -> bool:
    """带重试的消息发送"""
    return await channel.send_message(user_id, content)
```

### 11.3 不同场景的重试策略

| 场景 | 重试次数 | 等待策略 | 适用异常 |
|-----|---------|---------|---------|
| **Agent 调用** | 3 次 | 指数退避 2-10s | ConnectionError, TimeoutError |
| **发送消息** | 5 次 | 指数退避 1-5s | ClientError, 返回 False |
| **Token 刷新** | 3 次 | 固定 2s | AuthenticationError |
| **文件上传** | 3 次 | 指数退避 5-20s | NetworkError |
| **WebSocket 连接** | 无限 | 指数退避 1-30s | ConnectionError |

### 11.4 渠道级别的重试

```python
# channels/base.py
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_result

class Channel(ABC):
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_result(lambda r: r is False),
        reraise=True
    )
    async def send_message_with_retry(self, user_id: str, content: str, **kwargs) -> bool:
        """带重试的发送消息"""
        return await self.send_message(user_id, content, **kwargs)
    
    async def safe_send_message(self, user_id: str, content: str, **kwargs) -> bool:
        """安全发送消息（带降级处理）"""
        try:
            # 尝试发送
            success = await self.send_message_with_retry(user_id, content, **kwargs)
            if success:
                return True
        except Exception as e:
            logger.error(f"发送消息失败（已重试）: {e}")
        
        # 降级：尝试简化内容发送
        try:
            simplified = content[:100] + "..." if len(content) > 100 else content
            return await self.send_message(user_id, simplified, **kwargs)
        except Exception as e:
            logger.error(f"降级发送也失败: {e}")
            return False
```

### 11.5 错误统计与监控

```python
# 在 runner.py 中
class ChannelRunner:
    def __init__(self, ...):
        self._stats = {
            "total_messages": 0,
            "success": 0,
            "failed": 0,
            "retried": 0,
            "errors_by_type": defaultdict(int),
            "errors_by_channel": defaultdict(int),
            "avg_response_time": 0.0,
        }
        self._response_times: Deque[float] = deque(maxlen=100)
    
    def _record_error(self, error: Exception, channel_name: str = None):
        """记录错误统计"""
        error_type = type(error).__name__
        self._stats["errors_by_type"][error_type] += 1
        
        if channel_name:
            self._stats["errors_by_channel"][channel_name] += 1
        
        # 如果某类错误过多，发送告警
        if self._stats["errors_by_type"][error_type] > 10:
            logger.error(f"⚠️ 错误类型 {error_type} 发生次数过多，请检查！")
    
    def _record_response_time(self, duration: float):
        """记录响应时间"""
        self._response_times.append(duration)
        self._stats["avg_response_time"] = sum(self._response_times) / len(self._response_times)
```

### 11.6 用户友好的错误提示

```python
async def _handle_message(self, message: ChannelMessage) -> ChannelResponse:
    try:
        return await self._invoke_with_retry(message)
    except ConnectionError:
        return ChannelResponse(
            content="❌ 网络连接失败，请检查网络后重试",
            message_type="text"
        )
    except TimeoutError:
        return ChannelResponse(
            content="⏱️ 请求超时，问题可能太复杂，请简化后重试",
            message_type="text"
        )
    except Exception as e:
        logger.error(f"未预期的错误: {e}", exc_info=True)
        return ChannelResponse(
            content=f"❌ 处理出错: {str(e)[:100]}",
            message_type="text"
        )
```

---

## 十一、扩展指南

### 11.1 添加新渠道

```python
# channels/wecom.py (企业微信示例)
from .base import Channel, ChannelMessage

class WeComChannel(Channel):
    name = "wecom"
    description = "企业微信渠道"
    
    async def start(self):
        # 实现启动逻辑
        pass
    
    async def stop(self):
        # 实现停止逻辑
        pass
    
    async def send_message(self, user_id: str, content: str, **kwargs) -> bool:
        # 实现发送消息
        pass
```

### 11.2 注册新渠道

```python
# channels/runner.py
from .wecom import WeComChannel

class ChannelRunner:
    def _init_channels(self):
        # ... 原有渠道 ...
        
        # 添加企业微信
        wecom_config = self.config.get("wecom", {})
        if wecom_config.get("enabled"):
            self.channels.append(WeComChannel(wecom_config, self._handle_message))
```

---

## 十二、依赖说明

### 新增依赖

```toml
# pyproject.toml 新增
[project.optional-dependencies]
channels = [
    "aiohttp>=3.8.0",           # HTTP/WebSocket 客户端
    "pyyaml>=6.0",              # YAML 配置解析
    "tenacity>=8.0.0",          # 错误重试机制
]
```

**tenacity 说明**: 用于实现指数退避重试策略，当网络波动或 API 限流时自动重试，提高稳定性。

### 安装

```bash
# 安装核心 + 渠道支持
pip install -e ".[channels]"

# 或只安装核心
pip install -e .
```

---

## 十三、流式输出方案（未来优化）

### 当前方案（V1.0）：非流式

**实现方式**：
```python
async def _handle_message(self, message: ChannelMessage) -> str:
    # 等待完整结果后一次性返回
    response = self.agent.invoke(query=message.content, thread_id=thread_id)
    return response
```

**特点**：
- ✅ 实现简单，稳定可靠
- ✅ 兼容所有渠道（QQ/飞书/钉钉）
- ⚠️ 用户需要等待完整生成

**适用场景**：
- 短文本回复（< 500 字）
- 工具调用结果
- 快速问答

### 未来方案（V2.0）：伪流式

**实现方式**：
```python
async def _handle_message_stream(self, message: ChannelMessage):
    buffer = ""
    last_send_time = time.time()
    
    async for event in self.agent.astream_events(
        query=message.content, 
        thread_id=thread_id
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            buffer += chunk
            
            # 每 1 秒或每 50 个字符发送一次
            current_time = time.time()
            if current_time - last_send_time > 1.0 or len(buffer) > 50:
                yield buffer
                buffer = ""
                last_send_time = current_time
    
    if buffer:
        yield buffer
```

**特点**：
- ✅ 用户体验接近流式
- ✅ 避免 API 频率限制
- ⚠️ 代码复杂度增加
- ⚠️ 需要渠道支持消息编辑或分段发送

**适用场景**：
- 长文本生成（> 500 字）
- 文章写作
- 代码生成

### 升级计划

| 阶段 | 版本 | 方案 | 说明 |
|------|------|------|------|
| 第一阶段 | V1.0 | 非流式 | 快速实现，验证功能 |
| 第二阶段 | V2.0 | 伪流式 | 优化体验，分段发送 |
| 第三阶段 | V3.0 | 真流式 | 实时推送，WebSocket |

---

## 十四、总结

### 设计亮点

1. **零侵入**: 现有代码 100% 不动
2. **完全独立**: Channels 独立运行，互不影响
3. **即插即用**: 需要时启动，不需要时不影响
4. **个人友好**: 简化设计，专注单人使用
5. **可扩展**: 易于添加新渠道
6. **可演进**: 支持从非流式升级到流式

### 使用方式

| 命令 | 用途 | 是否新增 |
|------|------|---------|
| `python main.py` | 本地交互式使用 | 原有，不变 |
| `python -m channels.runner` | 启动远程渠道 | 新增 |

### 下一步

确认本方案后，开始实现代码（V1.0 非流式版本）：
1. 创建 `channels/` 目录结构
2. 实现 `base.py` 基类
3. 实现 `qq.py` QQ 渠道
4. 实现 `feishu.py` 飞书渠道
5. 实现 `dingtalk.py` 钉钉渠道
6. 实现 `runner.py` 运行器
7. 创建 `config.yaml` 示例配置
