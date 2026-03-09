# OllamaPilot

<p align="center">
  <strong>基于 LangChain 1.0+ 的本地智能助手框架</strong><br>
  <em>让 4B 小模型也能稳定执行复杂任务</em>
</p>

<p align="center">
  <a href="#特性">特性</a> •
  <a href="#快速开始">快速开始</a> •
  <a href="#skill-系统">Skill 系统</a> •
  <a href="#架构">架构</a> •
  <a href="#测试">测试</a>
</p>

---

## ✨ 为什么选择 OllamaPilot？

### 🎯 三大核心优势

#### 1️⃣ 完全本地运行 - 零云端依赖

- 🔒 **数据隐私**：所有模型运行在本地，数据不出你的电脑
- 💰 **零成本**：无需 API 费用，一次下载永久使用  
- 🚀 **4B 模型可用**：专门优化，让小模型也能干大事

#### 2️⃣ 基于 LangChain 1.0+ 原生实现

- ✅ 使用 `create_agent()` 工厂函数，代码简洁优雅
- ✅ Skill 即 Middleware，符合 LangChain 设计哲学
- ✅ 自动获得内置中间件（重试、限流、日志）
- ✅ 跟随 LangChain 升级，维护简单

#### 3️⃣ 渐进式 Skill 系统

| 复杂度 | 方式 | 代码量 | 适用场景 |
|:------:|------|:------:|----------|
| ⭐ | **SKILL.md** | 0 行 | 简单任务，纯配置 |
| ⭐⭐ | **Python Skill** | 20 行 | 自定义逻辑 |
| ⭐⭐⭐ | **MCP + 自定义** | 50 行 | 复杂集成，外部服务 |

---

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/zh2673-git/OllamaPilot.git
cd OllamaPilot
pip install -e .
```

### 最简单使用（3 行代码）

```python
from ollamapilot import init_ollama_model, create_agent

model = init_ollama_model("qwen3.5:4b")
agent = create_agent(model, skills_dir="skills")
response = agent.invoke("明天苏州天气怎么样？")
print(response)
```

### 运行交互式对话

```bash
python main.py
```

---

## 🧠 核心功能：让智能体"睁眼看世界"

OllamaPilot v0.1.3 引入两大核心能力，让本地模型既能**连接互联网**获取实时信息，又能**深度理解本地文档**进行知识问答。

### 🌐 Web 搜索 - 连接互联网

通过内置的 `web_search` 和 `web_fetch` 工具，智能体可以：
- 🔍 实时搜索网络信息
- 📰 获取最新新闻和动态
- 🌤️ 查询实时天气、股价等信息

```python
# 自动激活 Web 搜索 Skill
agent.invoke("今天比特币价格是多少？")
# AI 会自动搜索网络并返回实时价格
```

### 📚 GraphRAG - 深度理解本地文档

**GraphRAG（Graph Retrieval-Augmented Generation）** 是 v0.1.3 的核心升级：

| 特性 | 说明 |
|------|------|
| 🗂️ **多格式支持** | PDF、TXT、Markdown 等文档自动解析 |
| 🔗 **知识图谱** | 自动提取实体和关系，构建知识网络 |
| 🧩 **智能分块** | 大文档自动切分，保持语义连贯 |
| ⚡ **后台索引** | 文档索引在后台进行，不阻塞对话 |
| 🎯 **精准检索** | 基于实体关系+向量相似度的混合检索 |

```python
# 1. 将文档放入 knowledge_base/ 目录
# 2. 启动程序，自动索引
# 3. 直接提问

agent.invoke("伤寒论中关于太阳病的论述有哪些？")
# AI 会基于本地文档知识回答
```

**工作原理**：

```
┌─────────────────────────────────────────────────────────────┐
│                    GraphRAG 工作流程                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  📄 本地文档 → 🔍 实体抽取 → 🔗 关系构建 → 💾 知识存储    │
│      ↓                                                      │
│  ❓ 用户提问 → 🎯 实体识别 → 🔍 图谱检索 → 📝 生成回答    │
│                                                              │
│  特点：                                                       │
│  • 自动索引 knowledge_base/ 目录下的所有文档                │
│  • 支持多 Embedding 模型隔离存储                            │
│  • 后台异步索引，启动即可对话                                │
│  • 实体关系增强检索，比纯向量检索更精准                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 🎯 双模式协同

Web 搜索和 GraphRAG 可以协同工作：

```
用户: 对比一下伤寒论中的桂枝汤和现代感冒药的用法

🌐 Web 搜索: 获取现代感冒药信息
📚 GraphRAG: 从本地文档获取桂枝汤知识
🤖 AI: 综合分析并给出对比回答
```

---

## 🧩 Skill 系统

### 方式一：SKILL.md（推荐，0 代码）

创建 `skills/weather/SKILL.md`：

```yaml
---
name: weather
description: 查询天气信息
triggers: [天气, 温度, 下雨]
tools: [web_search, web_fetch]
---

# 天气查询助手

你是天气查询专家，帮助用户获取天气信息。

## 工作流程
1. 使用 web_search 搜索城市天气
2. 使用 web_fetch 获取详细页面
3. 整理天气信息返回给用户
```

**特点**：
- ✅ 无需写代码，纯配置驱动
- ✅ 支持内置工具、MCP 工具、自定义工具
- ✅ 复制到 `skills/` 目录即可使用

### 方式二：Python Skill（自定义逻辑）

```python
from langchain_core.tools import tool
from ollamapilot import Skill

@tool
def my_tool(query: str) -> str:
    """我的工具"""
    return f"处理结果: {query}"

class MySkill(Skill):
    name = "my_skill"
    description = "我的自定义 Skill"
    triggers = ["关键词1", "关键词2"]
    
    def get_tools(self):
        return [my_tool]
    
    def get_system_prompt(self):
        return "你是专家，帮助用户..."
```

### 方式三：MCP 工具（接入外部服务）

```python
from ollamapilot import create_agent, create_mcp_middleware

mcp_mw = create_mcp_middleware(
    server_url="https://mcp.example.com/mcp",
    server_name="my-server",
    allowed_tools=["search", "query"]
)

agent = create_agent(model, middleware=[mcp_mw])
```

---

## 🏗️ 架构

### 核心设计

```
┌─────────────────────────────────────────────────────────────┐
│                    OllamaPilot Agent                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   Skill 1    │    │   Skill 2    │    │   Skill N    │  │
│  │  (天气查询)   │    │  (代码执行)   │    │  (文件操作)   │  │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘  │
│         │                   │                   │          │
│         └───────────────────┼───────────────────┘          │
│                             ▼                              │
│                    ┌─────────────────┐                     │
│  ┌──────────────┐  │  Skill Selector │  ┌──────────────┐  │
│  │  Tool Retry  │← │   Middleware    │ →│  Tool Limit  │  │
│  └──────────────┘  └────────┬────────┘  └──────────────┘  │
│                             ▼                              │
│                    ┌─────────────────┐                     │
│                    │  LangChain      │                     │
│                    │  create_agent   │                     │
│                    └─────────────────┘                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 技术栈

- **LangChain 1.0+**：`create_agent()` + `AgentMiddleware`
- **LangGraph**：`MemorySaver` 实现对话记忆
- **Ollama**：本地模型支持（qwen3.5:4b/9b/35b, glm-4.7-flash 等）
- **GraphRAG**：知识图谱 + 向量检索混合架构

---

## 🛠️ 内置工具

### 核心工具

| 工具 | 功能 | 示例 |
|------|------|------|
| `web_search` | 网络搜索（自动部署） | `web_search.invoke({"query": "天气"})` |
| `web_fetch` | 获取网页内容 | `web_fetch.invoke({"url": "https://..."})` |
| `python_exec` | 执行 Python 代码 | `python_exec.invoke({"code": "1+1"})` |
| `read_file` | 读取文件 | `read_file.invoke({"path": "file.txt"})` |
| `write_file` | 写入文件 | `write_file.invoke({"path": "file.txt", "content": "..."})` |
| `list_directory` | 列出目录 | `list_directory.invoke({"path": "."})` |
| `search_files` | 搜索文件 | `search_files.invoke({"pattern": "*.py"})` |
| `shell_exec` | 执行 Shell 命令 | `shell_exec.invoke({"command": "ls"})` |
| `shell_script` | 执行 Shell 脚本 | `shell_script.invoke({"script": "echo hello"})` |

### 🔍 Web 搜索配置

`web_search` 工具依赖 [SearXNG](https://github.com/searxng/searxng) 搜索引擎，支持**自动部署**：

```python
# 首次使用时会自动检测并启动 SearXNG
agent.invoke("搜索 Python 3.13 新特性")

# 手动管理服务
from ollamapilot.tools.builtin import web_search_setup

web_search_setup("status")  # 检查状态
web_search_setup("start")   # 启动服务
web_search_setup("stop")    # 停止服务
web_search_setup("logs")    # 查看日志
```

**部署方式**：

1. **全自动部署**（推荐）：首次使用时会自动完成：
   - 检查 Docker 是否安装
   - 自动拉取 `searxng/searxng` 镜像（如本地不存在）
   - 创建并启动容器
   - 等待服务就绪

2. **手动部署**：`docker run -d --name searxng -p 8080:8080 searxng/searxng`

3. **远程服务**：设置环境变量 `export SEARXNG_URL='http://your-searxng-url'`

### 📚 GraphRAG 配置

```bash
# 1. 创建知识库目录
mkdir knowledge_base

# 2. 放入文档（PDF、TXT、Markdown 等）
cp your_document.pdf knowledge_base/

# 3. 启动程序，自动索引
python main.py

# 4. 选择 Embedding 模型（可选）
#    - qwen3-embedding:0.6b (快速，质量一般)
#    - qwen3-embedding:4b (平衡)
#    - qwen3-embedding:8b (高质量，较慢)
```

**特性**：
- ✅ 后台异步索引，启动即可对话
- ✅ 多 Embedding 模型隔离存储，切换模型不丢失数据
- ✅ 自动实体抽取和关系构建
- ✅ 支持大文档智能分块

---

## ✅ 测试

### 运行测试

```bash
python tests/test_agent.py
```

### 测试结果

```
============================================================
OllamaPilot 测试套件
============================================================

✅ 发现模型: qwen3.5:4b
======================== test session starts ========================
platform win32 -- Python 3.11.5, pytest-8.4.2
collected 8 items

tests/test_agent.py::TestBasicFunctionality::test_list_models PASSED
tests/test_agent.py::TestBasicFunctionality::test_init_model PASSED
tests/test_agent.py::TestAgent::test_agent_creation PASSED
tests/test_agent.py::TestAgent::test_simple_chat PASSED
tests/test_agent.py::TestAgent::test_python_tool PASSED
tests/test_agent.py::TestAgent::test_file_tool PASSED
tests/test_agent.py::TestQuestions::test_question_1_weekday PASSED
tests/test_agent.py::TestQuestions::test_question_2_weather PASSED

============= 8 passed in 52.31s =============
```

### 测试覆盖

- ✅ 模型列表和初始化
- ✅ Agent 创建和配置
- ✅ 简单对话
- ✅ Python 工具执行
- ✅ 文件工具执行
- ✅ **测试题目 1**：明天星期几 ✓
- ✅ **测试题目 2**：明天苏州天气怎么样 ✓
- ✅ **记忆功能**：多轮对话记忆测试 ✓

---

## 🎯 实战演示

### 示例 1：天气查询（Web 搜索）

```
用户: 明天苏州天气怎么样
🎯 激活 Skill: weather
🔧 执行工具: web_search({'query': '苏州明天天气'})
   ✅ 结果: 🔍 搜索 '苏州明天天气' 的结果（5 条）:
           1. 苏州天气预报-中央气象台
              URL: https://www.nmc.cn/publish/forecast/AJS/suzhou.html
🔧 执行工具: web_fetch({'url': 'https://www.nmc.cn/publish/forecast/AJS/suzhou.html'})
   ✅ 结果: 📄 标题: 苏州天气预报
           🔗 URL: https://www.nmc.cn/publish/forecast/AJS/suzhou.html

AI: 根据查询结果，苏州明天的天气情况如下：

## 🌤️ 苏州明天天气预报

| 项目 | 详情 |
|------|------|
| **天气状况** | ☀️晴 |
| **温度范围** | 6 ~ 14℃ |
| **空气质量** | ✅优 |
```

### 示例 2：本地文档问答（GraphRAG）

```
用户: 伤寒论中桂枝汤的主要成分有哪些？
🎯 激活 Skill: graphrag
🔍 检索知识库...
   ✅ 找到 5 个相关片段
   ✅ 识别实体: 桂枝汤、桂枝、芍药、甘草、生姜、大枣
🔧 执行工具: search_knowledge({'query': '桂枝汤 成分'})
   ✅ 结果: 📚 从本地知识库检索到相关信息

AI: 根据《伤寒论》记载，桂枝汤的主要成分包括：

## 🍵 桂枝汤组成

| 药材 | 用量 | 功效 |
|------|------|------|
| **桂枝** | 三两 | 解肌发表，温通经脉 |
| **芍药** | 三两 | 养血敛阴，柔肝止痛 |
| **甘草** | 二两 | 调和诸药，补中益气 |
| **生姜** | 三两 | 温中止呕，解表散寒 |
| **大枣** | 十二枚 | 补脾益气，养血安神 |

**用法**：水煎服，温服取微汗。
```

### 示例 3：Python 计算

```
用户: 计算 1+2+3+4+5 等于多少
🔧 执行工具: python_exec({'code': 'sum([1,2,3,4,5])'})
   ✅ 结果: >>> Python 代码执行
           ==================================================
           代码:
           result = sum([1,2,3,4,5])
           print(result)
           ==================================================

           📤 输出:
           15

           ✅ 执行成功

AI: 1 + 2 + 3 + 4 + 5 = **15**
```

### 示例 4：记忆功能

```python
# 第一轮
agent.invoke("我叫张三", thread_id="session_1")
# AI: 你好张三！很高兴认识你。

# 第二轮（同一个 session）
agent.invoke("你还记得我叫什么名字吗？", thread_id="session_1")
# AI: 当然记得，你叫张三！
```

### 示例 5：文件操作

```
用户: 列出当前目录的文件
🔧 执行工具: list_directory({'path': '.'})
   ✅ 结果: 📁 目录: D:\xx\OllamaPilot
           ==================================================
           📂 .git/
           📂 ollamapilot/
           📂 skills/
           📂 tests/
           📄 README.md
           📄 main.py
           📄 pyproject.toml

AI: 当前目录下有以下内容和文件：

**📁 子目录：**
1. `.git/` - Git版本控制信息
2. `ollamapilot/` - OllamaPilot主包目录
3. `skills/` - 技能相关文件夹
4. `tests/` - 测试文件目录

**📄 文本文件：**
1. `README.md` (11.4KB) - 项目说明文档
2. `main.py` (3.7KB) - 主程序入口脚本
3. `pyproject.toml` (1.6KB) - Python项目配置文件
```

---

## 📁 项目结构

```
OllamaPilot/
├── ollamapilot/              # 核心包
│   ├── __init__.py
│   ├── agent.py              # Agent 实现（基于 create_agent）
│   ├── skill_middleware.py   # Skill 中间件
│   ├── models.py             # 模型初始化
│   └── tools/
│       ├── __init__.py
│       ├── builtin.py        # 9个内置工具
│       ├── mcp_tools.py      # MCP 工具支持
│       └── custom/           # 自定义工具目录
│           └── __init__.py
├── skills/                   # Skill 目录（用户自定义）
│   ├── graphrag/             # GraphRAG Skill - 知识图谱检索
│   ├── weather/              # 天气查询 Skill
│   └── web_search/           # Web 搜索 Skill
├── knowledge_base/           # 知识库目录（存放本地文档）
├── data/                     # 数据存储目录
│   └── graphrag/             # GraphRAG 索引数据
├── tests/
│   └── test_agent.py         # 测试套件
├── .gitignore
├── LICENSE
├── main.py                   # CLI 入口
├── pyproject.toml            # 项目配置
└── README.md                 # 本文档
```

---

## 🔧 高级配置

### 记忆功能

```python
# 启用记忆
agent = create_agent(model, enable_memory=True)

# 多轮对话
response1 = agent.invoke("我叫张三", thread_id="session_1")
response2 = agent.invoke("我叫什么名字？", thread_id="session_1")  # 记得！

# 清除记忆
agent.clear_history("session_1")
```

### 详细日志

```python
agent = create_agent(model, verbose=True)
# 输出：
# 📦 已加载 1 个 Skill
# 🎯 激活 Skill: weather
# 🔧 执行工具: web_search({'query': '苏州天气'})
#    ✅ 结果: ...
```

### 自定义中间件

```python
from langchain.agents.middleware import ToolRetryMiddleware

agent = create_agent(
    model=model,
    middleware=[
        ToolRetryMiddleware(max_retries=3),  # 工具重试
    ]
)
```

---

## 📊 性能优化

针对小模型的工程优化：

- **工具重试**：`ToolRetryMiddleware` 自动重试失败工具
- **调用限制**：`ToolCallLimitMiddleware` 防止无限循环
- **详细日志**：verbose 模式显示执行过程，便于调试
- **Skill 选择**：自动选择最合适的 Skill，减少模型困惑
- **后台索引**：GraphRAG 文档索引在后台进行，不阻塞对话

---

## 📜 版本历史

### v0.1.3 (当前) - 智能体"睁眼看世界"

**🎉 重磅更新：GraphRAG 知识图谱检索**

- ✅ **GraphRAG Skill**：基于知识图谱的本地文档问答
  - 自动索引 PDF、TXT、Markdown 等文档
  - 实体抽取和关系构建
  - 混合检索（实体关系 + 向量相似度）
  - 后台异步索引，启动即可对话
  - 多 Embedding 模型隔离存储
- ✅ **Web 搜索增强**：自动部署 SearXNG，连接互联网
- ✅ **双模式协同**：Web 搜索 + GraphRAG 联合回答
- ✅ **知识库管理**：`knowledge_base/` 目录自动扫描

### v0.1.1

- ✅ 修复 Skill 激活日志显示问题
- ✅ 优化工具调用日志显示

### v0.1.0

- ✅ 基于 LangChain 1.0+ `create_agent`
- ✅ Skill 系统（SKILL.md + Python）
- ✅ 内置 9 个工具
- ✅ MCP 工具支持
- ✅ 自定义工具支持
- ✅ 记忆功能
- ✅ 中间件系统
- ✅ 8 项测试全部通过

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 许可证

MIT License

---

<p align="center">
  Made with ❤️ for Local AI
</p>
