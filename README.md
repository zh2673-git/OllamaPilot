# OllamaPilot 🚀

<div align="center">

**让 4B 小模型也能稳定执行复杂任务的本地智能体框架**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0+-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-Supported-red.svg)](https://ollama.com/)

[English](#english) | [中文](#中文)

</div>

---

## ✨ 为什么选择 OllamaPilot？

### 🎯 三大核心优势

#### 1️⃣ **完全本地运行 - 零云端依赖**
- 🔒 **数据隐私**：所有模型运行在本地，数据不出你的电脑
- 💰 **零成本**：无需 API 费用，一次下载永久使用
- 🚀 **4B 模型可用**：专门优化，让小模型也能干大事

#### 2️⃣ **开箱即用 - 内置 9 大工具**
```
📁 文件系统工具  |  🔧 Shell 命令工具  |  💻 代码处理工具
   read_file         shell_exec          code_search
   write_file        shell_script        apply_patch
   list_directory                         code_stats
   search_files
```
**无需配置，即装即用！**

#### 3️⃣ **Web Skill - Docker 一键部署**
```bash
cd skills/web
docker-compose up -d
```
- ✅ **完全本地搜索**：使用 SearXNG，无需任何 API Key
- ✅ **一键启动**：一条命令搞定搜索引擎部署
- ✅ **隐私保护**：搜索数据完全在本地处理

---

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    用户交互层 (CLI)                      │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              OllamaPilot 核心引擎                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ 模型驱动路由  │  │ 中间件链     │  │ 流式决策展示  │ │
│  │ Auto Router  │  │ Middleware   │  │ Streaming    │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  内置工具层  │  │  Skill 层    │  │  Web Skill   │
│  (9个工具)   │  │  (可扩展)    │  │  (Docker)    │
└──────────────┘  └──────────────┘  └──────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              Ollama 本地模型 (4B-35B)                   │
│   qwen3.5:4b  |  glm-4.7-flash  |  其他模型...        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎬 4B 小模型实战演示

### 实例：查询 2026 年 3 月 6 日新闻

使用 **qwen3.5:4b**（仅 4B 参数）完成复杂任务的全过程：

```bash
$ python main.py

=========================================================
🤖 OllamaPilot - 本地智能体框架
=========================================================

📋 可用模型:
----------------------------------------
【glm-4.7-flash】  1. glm-4.7-flash:latest
【qwen3.5】       2. qwen3.5:35b
                  3. qwen3.5:4b    ← 选择这个！
                  4. qwen3.5:9b
----------------------------------------

请选择模型 (1-4): 3
✅ 已选择模型: qwen3.5:4b

=========================================================
🤖 智能助手已启动
=========================================================
模型: Ollama (qwen3.5:4b)
架构: 基座Agent + Skill (模型自主决策)
---------------------------------------------------------

🔄 正在初始化模型...
🔄 正在加载 Skill 元数据...
✓ 发现 Skill: web
✓ 发现 Skill: 自动笔记

=========================================================
💬 开始对话
=========================================================

你: 2026年3月6日新闻

==================================================
📝 用户: 2026年3月6日新闻
==================================================

🔍 分析意图...
📝 模型决策过程:
----------------------------------------
思考: 用户想查询特定日期的新闻，需要网络搜索。
web Skill 提供 web_search 工具，适合此任务。
回复: web
----------------------------------------
✅ 决策 Skill: web
📚 加载 Skill: web
   策略: full | 原始: 553字 | 估计: ~287 tokens
🔧 加载工具 (2个):
   - web_search (local)
   - web_fetch (local)
✅ 已加载 9 个内置工具
✅ 成功加载 11 个工具

🔄 迭代 1/5
💭 思维链: 用户询问未来日期的新闻，使用 web_search 搜索
🤖 回答: {"name": "web_search", "arguments": {"query": "2026年3月6日 新闻"}}
🔧 调用工具: web_search

🔄 迭代 2/5
💭 思维链: 搜索结果已返回，整理关键信息
🤖 回答: 
根据搜索结果，以下是关于2026年3月6日的新闻信息：

## 📰 2026年3月6日主要新闻摘要

### 🏛️ 经济主题记者会
- **时间**：下午3时
- **地点**：梅地亚中心新闻发布厅
- **参与嘉宾**：国家发改委、财政部、商务部、中国人民银行、证监会相关负责人

### 💰 经济政策预告
1. **2026年经济增长目标**：4.5%-5%
2. **居民医保补助**：人均财政补助标准提高24元
3. **基础养老金**：月最低标准再提高20元

> ⚠️ 注：以上信息来源于网络搜索结果，涉及未来日期的预测性内容。

✅ 完成
```

### 🎯 实例亮点

| 特性 | 说明 |
|------|------|
| ✅ **智能路由** | 4B 小模型准确识别意图，自动选择 web Skill |
| ✅ **稳定输出** | 标准化 JSON 工具调用，格式规范 |
| ✅ **自动迭代** | 系统自动处理工具调用 → 获取结果 → 整理输出 |
| ✅ **结构化结果** | 自动生成清晰的 Markdown 格式 |

---

## 🚀 快速开始

### 1. 安装 Ollama

```bash
# 访问 https://ollama.com 下载并安装
# Windows / macOS / Linux 全平台支持
```

### 2. 下载模型（推荐小模型）

```bash
# 4B 模型 - 轻量级，适合普通电脑
ollama pull qwen3.5:4b

# 或 9B 模型 - 性能更强
ollama pull qwen3.5:9b

# 或 GLM 模型
ollama pull glm-4.7-flash:latest
```

### 3. 克隆项目并安装依赖

```bash
git clone https://github.com/yourusername/OllamaPilot.git
cd OllamaPilot

pip install -e ".[ollama]"
```

### 4. 运行

```bash
python main.py
```

就这么简单！🎉

---

## 🔧 Web Skill - Docker 一键部署

### 为什么需要 Web Skill？

OllamaPilot 内置了 9 大工具，但网络搜索需要搜索引擎支持。我们为你准备了 **完全本地化的解决方案**：

### 部署步骤

```bash
# 1. 进入 web skill 目录
cd skills/web

# 2. 一键启动搜索引擎（Docker）
docker-compose up -d

# 3. 首次运行需要配置（可选）
./setup-searxng.ps1  # Windows
# 或
./setup-searxng.sh     # Linux/macOS
```

### 特性

- ✅ **零 API 依赖**：使用 SearXNG 开源搜索引擎
- ✅ **完全本地**：搜索数据不经过第三方服务器
- ✅ **一键部署**：Docker Compose 自动化配置
- ✅ **多引擎聚合**：支持 Google、Bing、DuckDuckGo 等

---

## 📦 内置工具列表

OllamaPilot 开箱即用，内置 9 大工具：

### 📁 文件系统工具 (`tools/builtin/filesystem.py`)

| 工具 | 功能 | 示例 |
|------|------|------|
| `read_file` | 读取文件内容 | 读取配置文件、日志等 |
| `write_file` | 写入文件内容 | 保存结果、生成代码 |
| `list_directory` | 列出目录内容 | 浏览项目结构 |
| `search_files` | 搜索文件 | 查找特定文件 |

### 🔧 Shell 命令工具 (`tools/builtin/shell.py`)

| 工具 | 功能 | 示例 |
|------|------|------|
| `shell_exec` | 执行 Shell 命令 | 运行测试、部署脚本 |
| `shell_script` | 执行 Shell 脚本 | 批量处理任务 |

### 💻 代码处理工具 (`tools/builtin/code.py`)

| 工具 | 功能 | 示例 |
|------|------|------|
| `code_search` | 代码搜索 | 查找函数、类定义 |
| `apply_patch` | 应用代码补丁 | 修复 bug、添加功能 |
| `code_stats` | 代码统计 | 分析代码量、复杂度 |

---

## 🧩 Skill 扩展系统

### 当前可用 Skill

```
skills/
├── web/           # 网络搜索 (Docker 一键部署)
└── 自动笔记/       # Obsidian 笔记管理
```

### 创建自定义 Skill

只需 3 步：

#### 1. 创建目录

```bash
mkdir skills/my_skill
cd skills/my_skill
```

#### 2. 编写 SKILL.md

```markdown
---
name: my_skill
description: 我的自定义技能
tags: ["自定义", "示例"]
triggers:
  - "测试"
  - "demo"
version: "1.0.0"
author: "Your Name"
tools:
  - name: my_tool
    type: local
    module: skills.my_skill.skill
---

# My Skill

## 功能描述
这是一个示例 Skill...
```

#### 3. 实现工具（可选）

```python
# skills/my_skill/skill.py
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """我的工具描述"""
    return f"结果: {query}"

TOOLS = [my_tool]
```

就这么简单！系统会自动发现并加载你的 Skill。

---

## 🏛️ 技术栈

| 技术 | 版本 | 用途 |
|------|------|------|
| Python | >=3.9 | 编程语言 |
| LangChain | >=1.0.0 | LLM 应用框架 |
| LangChain-Core | >=1.0.0 | 核心组件 |
| Pydantic | >=2.7.0 | 数据验证 |
| tiktoken | >=0.5.0 | Token 精确计算 |
| langchain-ollama | >=1.0.0 | Ollama 本地模型支持 |

---

## 📂 项目结构

```
OllamaPilot/
├── base_agent/              # 核心引擎
│   ├── agent.py           # Agent 实现
│   ├── skill/             # Skill 路由与加载
│   ├── middleware/        # 中间件链
│   ├── model/            # 模型客户端
│   └── tool/             # 工具加载器
│
├── skills/                # Skill 目录
│   ├── web/              # 网络搜索 (Docker)
│   │   ├── SKILL.md
│   │   ├── skill.py
│   │   ├── docker-compose.yml
│   │   └── setup-searxng.ps1
│   └── 自动笔记/          # Obsidian 笔记
│
├── tools/                 # 内置工具
│   ├── builtin/          # 9 大内置工具
│   │   ├── filesystem.py  # 文件系统
│   │   ├── shell.py       # Shell 命令
│   │   └── code.py        # 代码处理
│   └── registry.py        # 工具注册表
│
├── examples/              # 示例代码
├── tests/                # 测试用例
├── main.py               # 主入口
├── pyproject.toml        # 项目配置
└── README.md             # 本文档
```

---

## 🎯 核心特性详解

### 1. 模型驱动的 Skill 路由

让模型根据 Skill 描述和触发词智能选择最合适的 Skill：

```python
# 模型自动决策过程
用户输入: "帮我搜索最新的 AI 新闻"
  ↓
模型分析: 包含"搜索"关键词 → 匹配 web Skill
  ↓
自动加载: web Skill + web_search 工具
  ↓
执行任务: 调用搜索工具
```

### 2. 中间件链保障

5 大中间件确保小模型稳定运行：

| 中间件 | 作用 |
|--------|------|
| Skill Loader | 自动加载匹配的工具和上下文 |
| Context Inject | 动态注入 Skill 描述和工具说明 |
| Tool Retry | 工具调用失败自动重试 |
| Memory Manager | 维护对话上下文，支持多轮交互 |
| Tool Format Parser | 多格式解析，兼容小模型不规范输出 |

### 3. Token 精确计算

使用 tiktoken 精确计算，防止上下文溢出：

```python
# 智能加载策略
if estimated_tokens < 4000:
    strategy = "full"      # 完全加载
elif estimated_tokens < 8000:
    strategy = "layered"   # 分层加载
else:
    strategy = "retrieval" # 检索加载
```

---

## 📖 使用示例

### 示例 1：文件操作

```bash
你: 列出当前目录的所有文件
🤖: [调用 list_directory 工具]
✅ 已找到以下文件：main.py, README.md, ...

你: 读取 README.md 的前 50 行
🤖: [调用 read_file 工具]
✅ 已读取文件内容...
```

### 示例 2：代码搜索

```bash
你: 搜索项目中所有的 Agent 类
🤖: [调用 code_search 工具]
✅ 找到 3 个匹配结果：
   1. base_agent/agent.py - Agent 类定义
   2. examples/model_driven_demo.py - 使用示例
   ...
```

### 示例 3：网络搜索

```bash
你: 搜索最新的 Python 3.12 新特性
🤖: [自动选择 web Skill]
🤖: [调用 web_search 工具]
✅ 搜索结果：
   Python 3.12 新特性包括：
   1. 更好的错误消息
   2. 性能优化
   3. 新的语法特性
   ...
```

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 🙏 致谢

- [LangChain](https://python.langchain.com/) - 强大的 LLM 应用框架
- [Ollama](https://ollama.com/) - 本地模型运行平台
- [SearXNG](https://docs.searxng.org/) - 开源搜索引擎

---

## 📮 联系方式

- **GitHub Issues**: [提交问题](https://github.com/yourusername/OllamaPilot/issues)
- **Discussions**: [参与讨论](https://github.com/yourusername/OllamaPilot/discussions)

---

<div align="center">

**如果这个项目对你有帮助，请给个 ⭐ Star！**

Made with ❤️ by OllamaPilot Team

</div>

---

## English

# OllamaPilot 🚀

<div align="center">

**A Local Agent Framework That Makes 4B Models Capable of Complex Tasks**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-1.0+-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-Supported-red.svg)](https://ollama.com/)

</div>

---

## ✨ Why OllamaPilot?

### 🎯 Three Core Advantages

#### 1️⃣ **Fully Local - Zero Cloud Dependency**
- 🔒 **Data Privacy**: All models run locally, data never leaves your machine
- 💰 **Zero Cost**: No API fees, download once and use forever
- 🚀 **4B Model Ready**: Optimized for small models to do big tasks

#### 2️⃣ **Ready to Use - 9 Built-in Tools**
```
📁 File System  |  🔧 Shell Commands  |  💻 Code Processing
   read_file         shell_exec          code_search
   write_file        shell_script        apply_patch
   list_directory                         code_stats
   search_files
```
**No configuration needed, just install and use!**

#### 3️⃣ **Web Skill - One-Click Docker Deployment**
```bash
cd skills/web
docker-compose up -d
```
- ✅ **Fully Local Search**: Uses SearXNG, no API Key required
- ✅ **One-Click Setup**: One command to deploy search engine
- ✅ **Privacy Protected**: Search data processed entirely locally

---

## 🚀 Quick Start

### 1. Install Ollama

```bash
# Visit https://ollama.com to download and install
# Supports Windows / macOS / Linux
```

### 2. Download Models

```bash
# 4B model - Lightweight, suitable for average computers
ollama pull qwen3.5:4b

# Or 9B model - Better performance
ollama pull qwen3.5:9b

# Or GLM model
ollama pull glm-4.7-flash:latest
```

### 3. Clone and Install

```bash
git clone https://github.com/yourusername/OllamaPilot.git
cd OllamaPilot

pip install -e ".[ollama]"
```

### 4. Run

```bash
python main.py
```

That's it! 🎉

---

## 📦 Built-in Tools

OllamaPilot comes with 9 built-in tools:

### 📁 File System Tools

| Tool | Function | Example |
|------|----------|---------|
| `read_file` | Read file content | Read config files, logs |
| `write_file` | Write file content | Save results, generate code |
| `list_directory` | List directory contents | Browse project structure |
| `search_files` | Search files | Find specific files |

### 🔧 Shell Command Tools

| Tool | Function | Example |
|------|----------|---------|
| `shell_exec` | Execute shell commands | Run tests, deploy scripts |
| `shell_script` | Execute shell scripts | Batch processing |

### 💻 Code Processing Tools

| Tool | Function | Example |
|------|----------|---------|
| `code_search` | Search code | Find functions, class definitions |
| `apply_patch` | Apply code patches | Fix bugs, add features |
| `code_stats` | Code statistics | Analyze code size, complexity |

---

## 🔧 Web Skill - Docker Deployment

### Deployment Steps

```bash
# 1. Enter web skill directory
cd skills/web

# 2. One-click start search engine (Docker)
docker-compose up -d

# 3. First-time setup (optional)
./setup-searxng.ps1  # Windows
# or
./setup-searxng.sh     # Linux/macOS
```

### Features

- ✅ **Zero API Dependency**: Uses open-source SearXNG search engine
- ✅ **Fully Local**: Search data doesn't go through third-party servers
- ✅ **One-Click Deployment**: Docker Compose automated setup
- ✅ **Multi-Engine Aggregation**: Supports Google, Bing, DuckDuckGo, etc.

---

## 🧩 Skill Extension System

### Create Custom Skill in 3 Steps

#### 1. Create Directory

```bash
mkdir skills/my_skill
cd skills/my_skill
```

#### 2. Write SKILL.md

```markdown
---
name: my_skill
description: My custom skill
tags: ["custom", "example"]
triggers:
  - "test"
  - "demo"
version: "1.0.0"
author: "Your Name"
tools:
  - name: my_tool
    type: local
    module: skills.my_skill.skill
---

# My Skill

## Description
This is an example skill...
```

#### 3. Implement Tools (Optional)

```python
# skills/my_skill/skill.py
from langchain_core.tools import tool

@tool
def my_tool(query: str) -> str:
    """My tool description"""
    return f"Result: {query}"

TOOLS = [my_tool]
```

That's it! The system will automatically discover and load your Skill.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

---

<div align="center">

**If this project helps you, please give it a ⭐ Star!**

Made with ❤️ by OllamaPilot Team

</div>
