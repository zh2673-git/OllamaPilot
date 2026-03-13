# OllamaPilot

<p align="center">
  <strong>🚀 本地 AI 智能助手框架</strong><br>
  <em>LangChain 1.0+ × Ollama 小模型 × GraphRAG 知识图谱 × 零 API 网络搜索 × OpenCode 式终端交互</em>
</p>

<p align="center">
  <a href="#-五大核心亮点">亮点</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-核心功能">核心功能</a> •
  <a href="#-skill-系统">Skill 系统</a> •
  <a href="#-架构">架构</a>
</p>

---

## 🎯 五大核心亮点

### 1️⃣ LangChain 1.0+ 原生架构
- ✅ 基于 `create_agent()` 工厂函数，代码简洁优雅
- ✅ Skill 即 Middleware，符合 LangChain 设计哲学
- ✅ 自动获得内置中间件（重试、限流、日志）
- ✅ 跟随官方升级，维护简单

### 2️⃣ Ollama 小模型优化
- 🚀 **4B 模型也能干大事**：专门优化让小模型稳定执行复杂任务
- 💰 **零成本**：无需 API 费用，一次下载永久使用
- 🔒 **数据隐私**：所有模型本地运行，数据不出你的电脑
- 📦 **模型丰富**：支持 qwen、glm、llama 等主流模型
- ⚙️ **.env 配置管理**：通过配置文件灵活设置模型参数，便于前端集成

### 3️⃣ GraphRAG 知识图谱（Morifish 风格）
- 🗂️ **多格式文档解析**：PDF、TXT、Markdown 自动处理
- 🔗 **实体关系抽取**：自动构建知识网络，比纯向量检索更精准
- 🧩 **智能文档分块**：大文档自动切分，保持语义连贯
- ⚡ **后台异步索引**：启动即可对话，索引不阻塞
- 💾 **多模型隔离存储**：切换 Embedding 模型不丢失数据
- 🎯 **WordAligner 精确对齐**：实体精确映射到原文位置，支持溯源验证
- 📝 **手动文档管理**：通过指令控制文档索引，静默后台处理

### 4️⃣ 多引擎智能搜索（免费API优先）
- 🌐 **10+ 搜索引擎**：Tavily、SearXNG、Serper、Bing、Brave、DuckDuckGo、PubMed、GitHub 等
- 💰 **免费额度充足**：Tavily(1000次/月)、Serper(2500次/月)、Bing(1000次/月)、Brave(2000次/月)
- 🔄 **智能降级机制**：配额用完或失败时自动切换到备用引擎
- 🇨🇳 **国内直接访问**：所有推荐引擎均可在国内网络直接使用，无需翻墙
- 🔍 **三层搜索架构**：基础搜索 → 增强搜索 → 深度研究，满足不同场景需求
- 📊 **配额管理**：自动跟踪API使用情况，避免超额

### 5️⃣ OpenCode 式终端交互
- ⌨️ **斜杠命令体系**：`/model`、`/new`、`/sessions` 等命令快速操作
- 🔄 **运行时模型切换**：无需重启程序即可更换模型
- 💬 **多会话管理**：每个会话独立历史，随时切换
- 📊 **实时状态显示**：当前模型、会话、消息数一目了然
- ⚡ **真正并行处理**：索引和对话同时进行，互不阻塞
- 📝 **断点续传**：索引失败后可恢复任务，不重复工作

---

## 📱 远程控制 - 多渠道接入

**🎉 新功能：通过 QQ、飞书、钉钉远程控制 Agent**

Channels 模块让你可以通过多种即时通讯工具远程调用 Agent，特别适合：
- 🚀 在手机上通过 QQ 发送指令
- 🏠 远程控制本地运行的 AI 助手
- 📱 无需公网 IP，使用平台推送消息

```bash
# 安装依赖
pip install aiohttp tenacity pyyaml

# 复制配置模板
cp channels/config.example.yaml channels/config.yaml

# 编辑配置，填入你的 API Key
vim channels/config.yaml

# 启动渠道服务
python -m channels.runner
```

### 支持的渠道

| 渠道 | 状态 | 说明 |
|------|------|------|
| QQ | ✅ | 使用官方 Bot API，个人号可用 |
| 飞书 | ✅ | 企业自建应用 |
| 钉钉 | ✅ | 企业内部机器人 |

详见 [channels/DESIGN.md](channels/DESIGN.md)

---

## 🎯 本项目特色（实战案例）

### 📊 案例：伤寒论知识图谱构建

**测试环境**:
- 文档：伤寒论(上)+(下) 讲解版（倪海厦），5MB+
- 模型：qwen3.5:4b（对话）+ qwen3-embedding:0.6b（向量）
- 硬件：RTX 4060 8GB + 64GB内存

**索引结果**:
```
📚 文档列表
✅ 4.2伤寒论(上) - 135分块, 10,840实体
✅ 4.3伤寒论(下) - 175分块, 6,637实体
⏱️ 索引时间：约40分钟（批量优化后）
📊 总实体数：17,477
```

### 🚀 高性能批量索引

**速度提升5倍**:
- **批量LLM调用**：每批20个分块一次性处理
- **智能超时管理**：5分钟超时保护，大文档稳定处理
- **实时进度显示**：批次进度、实体数、预估时间一目了然

### 📚 全局预设词典系统

**7大领域764词条**:
| 词典 | 领域 | 词条数 |
|------|------|--------|
| global.json | 通用 | 109 |
| tcm.json | 中医 | 139 |
| medical.json | 现代医学 | 143 |
| legal.json | 法律 | 119 |
| finance.json | 金融 | 127 |
| technology.json | 科技 | 123 |
| custom.json | 自定义 | 4 |

**词典继承机制**：全局预设 + 文档私有，预置术语大幅提升速度

### 💡 智能问答演示

**本地知识搜索**:
```
你: 搜索本地知识库，麻黄汤的构成

助手: 🔧 执行工具: search_knowledge
   ✅ 结果: 找到5条相关记录
   📚 来源: 4.3伤寒论(下) | 相关度: 0.80

🤖 AI: ## 📚 麻黄汤的构成（基于知识库）

| 药物 | 作用 |
|------|------|
| ✅ 麻黄 | 主药，解表发汗、入肺经 |
| ✅ 桂枝 | 辅助推药至体表，配合麻黄发汗 |
| ✅ 甘草 | 调和诸药，增强疗效 |

### 📖 《伤寒论》中的配伍原理

根据知识库中《4.2伤寒论(上)》的记载：

> "麻黄汤里面为什么要有桂枝、甘草？就是利用桂枝、甘草，
> 把麻黄先推到皮肤表面上。后面桂枝没力了，就靠麻黄了，
> 用麻黄来解表发汗。"
```

**智能回退机制**：
- 本地知识不足时，自动联网搜索补充
- 确保回答完整准确
- 搜索结果可自动添加到本地知识库

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

### 运行交互式对话（推荐）

```bash
python main.py
```

**交互命令示例**：
```
你: /help              # 显示所有命令
你: /model             # 切换对话模型
你: /model qwen2.5:7b  # 直接指定模型
你: /new               # 创建新会话
你: /sessions          # 查看所有会话
你: /switch session_1  # 切换会话
你: /clear             # 清空当前对话
你: /index             # 索引 knowledge_base/ 文件夹
你: /index ./docs      # 索引指定文件夹
你: /docs              # 查看文档索引状态
你: /resume            # 恢复失败的索引任务
你: /reload            # 重新加载 .env 配置
```

---

## 🧠 核心功能

### 🌐 三层搜索架构 - 从基础到深度研究

OllamaPilot 提供三层搜索能力，满足不同场景需求：

```
┌─────────────────────────────────────────────────────────────┐
│  第三层：deep_research (深度研究)                            │
│  触发词："研究 xxx"                                          │
│  输出：Markdown 研究报告                                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第二层：enhanced_search (增强搜索)                          │
│  触发词："增强搜索 xxx"                                       │
│  类型：学术/代码/百科/综合                                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第一层：web_search (基础搜索)                               │
│  触发词："搜索 xxx"                                          │
│  特点：简单快速，基础通用                                    │
└─────────────────────────────────────────────────────────────┘
```

#### 第一层：基础搜索

```python
# 简单快速的基础搜索
agent.invoke("搜索 Python 3.13 新特性")
agent.invoke("查一下北京天气")
```

**引擎优先级**：Tavily → SearXNG → Serper → Bing → Brave → DuckDuckGo

#### 第二层：增强搜索

专业领域搜索，支持学术/代码/百科/综合四种类型：

```python
# 学术搜索 - PubMed 医学文献
agent.invoke("增强搜索 量子计算论文")

# 代码搜索 - GitHub 开源项目
agent.invoke("增强搜索 Python 爬虫框架")

# 百科搜索 - 百度百科
agent.invoke("增强搜索 什么是区块链")

# 综合搜索 - 多引擎聚合
agent.invoke("增强搜索 深度学习应用")
```

**可用引擎**：
| 引擎 | 类型 | 免费额度 | 需配置 |
|------|------|---------|--------|
| ✅ Tavily | 通用 | 1000次/月 | TAVILY_API_KEY |
| ✅ SearXNG | 聚合 | 无限 | 本地部署 |
| ✅ Serper | Google | 2500次/月 | SERPER_API_KEY |
| ✅ Bing | 必应 | 1000次/月 | BING_API_KEY |
| ✅ Brave | Brave | 2000次/月 | BRAVE_API_KEY |
| ✅ DuckDuckGo | 通用 | 无限 | 无需配置 |
| ✅ PubMed | 学术 | 无限 | 无需配置 |
| ✅ 百度百科 | 百科 | 无限 | 无需配置 |
| ✅ GitHub | 代码 | 500次/小时 | GITHUB_TOKEN |
| ✅ Gitee | 代码 | 无限 | 无需配置 |

#### 第三层：深度研究

多轮迭代研究，生成专业 Markdown 报告：

```python
# 深度研究自动生成报告
agent.invoke("研究 人工智能在医疗诊断中的应用")
agent.invoke("调研 区块链技术发展趋势")
```

**特点**：
- 多轮搜索和信息整合
- 自动生成结构化报告
- 报告保存到 `./reports/` 目录
- 优先使用 Tavily（专为AI研究设计）

---

### 📚 GraphRAG - 知识图谱增强检索

**GraphRAG（Graph Retrieval-Augmented Generation）** 让本地模型深度理解你的文档：

agent.invoke("搜索 Python 3.13 新特性")
```

### 📚 GraphRAG - 知识图谱增强检索

**GraphRAG（Graph Retrieval-Augmented Generation）** 让本地模型深度理解你的文档：

| 特性 | 说明 |
|------|------|
| 🗂️ **多格式支持** | PDF、TXT、Markdown 等文档自动解析 |
| 🔗 **知识图谱** | 自动提取实体和关系，构建知识网络 |
| 🧩 **智能分块** | 大文档自动切分，保持语义连贯 |
| ⚡ **后台索引** | 文档索引在后台进行，不阻塞对话 |
| 🎯 **精准检索** | 基于实体关系+向量相似度的混合检索 |
| 📍 **WordAligner** | 实体精确对齐到原文，支持溯源验证 |
| 📝 **手动管理** | 通过指令控制文档索引，静默处理 |

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
- **SearXNG**：零 API 网络搜索

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

---

## 📋 版本历史

### v0.3.0 (当前) - 多渠道远程控制

**🎉 全新功能：通过 QQ、飞书、钉钉远程控制 Agent**

- ✅ **Channels 模块**：独立的多渠道远程控制模块
- ✅ **QQ 开放平台**：使用官方 Bot API，不再依赖 go-cqhttp
- ✅ **飞书/钉钉**：支持企业微信、钉钉机器人
- ✅ **零侵入设计**：不修改原项目代码，独立运行
- ✅ **自动发现**：渠道注册表机制，新增渠道只需添加文件
- ✅ **错误重试**：使用 tenacity 实现指数退避重试
- ✅ **运行统计**：记录消息数、成功率、响应时间
- ✅ **配置模板**：提供 config.example.yaml，用户可安全 Fork

### v0.2.8 - 流式输出优化与异步中间件

**🎉 体验升级：更流畅的流式输出**

- ✅ **流式事件输出**：使用 LangChain 原生 astream_events，实时显示工具执行过程
- ✅ **异步中间件支持**：ToolLoggingMiddleware 支持异步 awrap_tool_call
- ✅ **工具执行可见**：清晰展示工具开始/结束状态
- ✅ **Event Loop 优化**：修复嵌套事件循环导致的崩溃问题
- ✅ **去除重复执行**：修复流式失败后回退导致的工具重复执行问题
- ✅ **回答防重复**：修复流式输出失败后重复打印回答的问题
- ✅ **on_chat_model_end 支持**：增加模型完成事件处理，确保回复完整输出
- ✅ **递归限制可配置**：新增 RECURSION_LIMIT 环境变量，默认 50 次

### v0.2.7 - 多引擎智能搜索与API配额管理

**🎉 重磅更新：免费API优先策略 + 智能降级机制**

- ✅ **10+ 搜索引擎**：Tavily、SearXNG、Serper、Bing、Brave、DuckDuckGo、PubMed、GitHub、Gitee、百度百科等
- ✅ **免费额度充足**：Tavily(1000次/月)、Serper(2500次/月)、Bing(1000次/月)、Brave(2000次/月)
- ✅ **智能降级机制**：配额用完或失败时自动切换到备用引擎
- ✅ **API配额管理**：自动跟踪使用情况，避免超额
- ✅ **国内直接访问**：所有推荐引擎均可在国内网络直接使用
- ✅ **零配置可用**：未配置API时自动使用免费引擎（DuckDuckGo、PubMed等）

### v0.2.6 - 多级分类知识库支持

**🎉 新增功能：分类知识库搜索**

- ✅ **多级文件夹结构**：支持一级/二级/多级分类
- ✅ **精准搜索**：在指定分类中搜索，避免知识污染
- ✅ **新增工具**：`search_knowledge_base`、`list_knowledge_categories`
- ✅ **向后兼容**：原有 `search_knowledge` 仍然可用

### v0.2.5 - 三层搜索架构与增强研究能力

**🎉 重磅更新：三层搜索架构 + 深度研究**

- ✅ **三层搜索架构**：基础搜索 → 增强搜索 → 深度研究
  - 第一层 `web_search`：基础通用搜索，简单快速
  - 第二层 `enhanced_search`：多引擎专业搜索（学术/代码/百科）
  - 第三层 `deep_research`：多轮迭代研究，生成专业报告
- ✅ **8个搜索引擎**：SearXNG、PubMed、百度百科、GitHub、arXiv、Wikipedia等
- ✅ **完全免费**：所有引擎均无需 API Key
- ✅ **国内可用**：优先选择国内可访问的引擎
- ✅ **Markdown Skill 统一格式**：所有搜索 Skill 使用 SKILL.md 配置

### v0.2.4 - 批量索引优化与搜索体验提升

**🎉 重磅更新：高性能批量索引与即时搜索**

- ✅ **批量LLM调用优化**：每批20个分块，速度提升5倍
  - 伤寒论 5MB+ PDF（310分块）约40分钟完成索引
  - 提取 17,477 个实体，知识图谱构建完整
- ✅ **智能超时管理**：5分钟超时保护，大文档稳定处理
- ✅ **全局预设词典系统**：7大领域764词条，预置术语加速索引
- ✅ **索引完成即时搜索**：自动清除缓存，无需重启即可搜索
- ✅ **对话超时保护**：索引期间对话不阻塞，Ctrl+C可中断
- ✅ **实体数实时显示**：进度更新时同步显示已提取实体数

<details>
<summary>查看历史版本</summary>

### v0.2.3 - 全局预设词典与多领域支持

- ✅ **全局预设词典系统**：config/dictionaries/ 目录管理
- ✅ **词典继承机制**：全局词典 + 文档私有词典
- ✅ **词典管理器**：DictionaryManager 统一管理

### v0.2.2 - 搜索性能优化与多文档管理

- ✅ **GraphRAGService 缓存机制**
- ✅ **修复知识图谱为空问题**
- ✅ **批量索引优化**

### v0.2.1 - 混合模式实体抽取与通用词典

- ✅ **混合模式实体抽取**：词典匹配 + LLM抽取 + 动态学习
- ✅ **通用实体词典**：80个跨领域通用实体

### v0.2.0 - 移除Ollama锁，真正并行处理

- ✅ **移除 Ollama 锁**：索引和对话真正并行
- ✅ **断点续传**：`/resume` 命令

### v0.1.8 - WordAligner 与手动文档管理

- ✅ **WordAligner 精确对齐**
- ✅ **手动文档管理**

### v0.1.7 - 模型切换与会话管理

- ✅ **斜杠命令体系**
- ✅ **运行时模型切换**

</details>

### v0.1.3 - GraphRAG 知识图谱

- ✅ **GraphRAG Skill**：基于知识图谱的本地文档问答
- ✅ **Web 搜索增强**：自动部署 SearXNG
- ✅ **双模式协同**：Web 搜索 + GraphRAG 联合回答

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

## 📄 许可证

MIT License

---

<p align="center">
  Made with ❤️ for Local AI
</p>