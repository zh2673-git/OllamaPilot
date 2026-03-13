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

```python
class Channel(ABC):
    """渠道基类"""
    
    name: str = ""
    description: str = ""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._running = False
    
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
    
    def check_permission(self, user_id: str) -> bool:
        """检查用户权限（白名单）"""
        whitelist = self.config.get("whitelist", [])
        if not whitelist:
            return True
        return user_id in whitelist
```

### 5.3 运行器 (runner.py)

```python
class ChannelRunner:
    """渠道统一运行器"""
    
    def __init__(self, config_path: str = None):
        # 加载配置
        self.config = load_config(config_path)
        
        # 初始化 Agent（直接导入核心）
        self._init_agent()
        
        # 初始化启用的渠道
        self._init_channels()
    
    def _init_agent(self):
        """初始化 OllamaPilot Agent"""
        from ollamapilot import init_ollama_model, OllamaPilotAgent
        
        model = init_ollama_model(self.config["agent"]["model"])
        self.agent = OllamaPilotAgent(
            model=model,
            skills_dir=self.config["agent"].get("skills_dir"),
            verbose=True
        )
    
    async def _handle_message(self, message: ChannelMessage) -> str:
        """处理收到的消息（非流式，完整返回）"""
        # 权限检查
        if not self._check_permission(message):
            return "⛔ 您没有使用权限"
        
        # 构建独立会话ID（每个用户独立上下文）
        thread_id = f"{message.user_id}_{message.message_type}"
        
        # 调用 Agent（非流式，等待完整结果）
        # TODO: 未来可升级为流式输出（见第十二节）
        response = self.agent.invoke(
            query=message.content,
            thread_id=thread_id
        )
        return response
```

---

## 六、渠道实现详情

### 6.1 QQ 渠道 (qq.py)

**技术方案**: go-cqhttp (开源 QQ Bot 框架)

**功能特性**:
- 私聊消息接收与回复
- 群聊消息接收（支持 @触发）
- 图片消息支持
- 长消息自动拆分

**配置项**:
```yaml
qq:
  enabled: true
  api_url: "http://127.0.0.1:5700"      # go-cqhttp HTTP API 地址
  ws_url: "ws://127.0.0.1:5701"         # go-cqhttp WebSocket 地址
  bot_qq: "123456789"                    # 机器人 QQ 号
  whitelist:                             # 白名单（空则允许所有）
    - "987654321"                        # 你的 QQ 号
  at_only_in_group: true                 # 群聊中只响应 @消息
```

**部署步骤**:
1. 下载 [go-cqhttp](https://github.com/Mrs4s/go-cqhttp)
2. 配置账号密码，启动服务
3. 配置 channels/config.yaml
4. 启动 runner

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

```yaml
# ============================================
# Channels 配置文件
# ============================================

# Agent 配置
agent:
  model: "qwen3.5:4b"                    # 对话模型
  skills_dir: "./skills"                 # Skill 目录
  verbose: true                          # 显示详细日志

# 全局权限配置
global:
  whitelist: []                           # 全局白名单（所有渠道生效）
  admin_users: []                         # 管理员用户（可使用管理命令）

# QQ 渠道配置
qq:
  enabled: true
  api_url: "http://127.0.0.1:5700"
  ws_url: "ws://127.0.0.1:5701"
  bot_qq: "123456789"
  whitelist:
    - "987654321"
  at_only_in_group: true

# 飞书渠道配置
feishu:
  enabled: false
  app_id: "cli_xxxxxxxxxx"
  app_secret: "xxxxxxxxxx"
  whitelist: []

# 钉钉渠道配置
dingtalk:
  enabled: false
  app_key: "xxxxxxxxxx"
  app_secret: "xxxxxxxxxx"
  whitelist: []
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

## 十、扩展指南

### 10.1 添加新渠道

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

### 10.2 注册新渠道

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

## 十一、依赖说明

### 新增依赖

```toml
# pyproject.toml 新增
[project.optional-dependencies]
channels = [
    "aiohttp>=3.8.0",           # HTTP/WebSocket 客户端
    "pyyaml>=6.0",              # YAML 配置解析
]
```

### 安装

```bash
# 安装核心 + 渠道支持
pip install -e ".[channels]"

# 或只安装核心
pip install -e .
```

---

## 十二、流式输出方案（未来优化）

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

## 十三、总结

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
