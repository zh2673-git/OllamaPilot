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

### 3️⃣ GraphRAG 知识图谱（本项目特色）

**📊 实测效果（伤寒论 5MB+ PDF）**
```
📚 文档列表
✅ 4.2伤寒论(上) - 135分块, 10840实体
✅ 4.3伤寒论(下) - 175分块, 6637实体
⏱️ 索引时间：约40分钟（qwen3.5:4b + qwen3-embedding:0.6b）
```

**🚀 高性能批量索引**
- **批量LLM调用**：每批20个分块一次性处理，速度提升5倍
- **智能超时管理**：5分钟超时保护，大文档稳定处理
- **实时进度显示**：批次进度、实体数、预估时间一目了然

**📚 全局预设词典系统**
- **7大领域764词条**：中医、现代医学、法律、金融、科技等
- **词典继承机制**：全局预设 + 文档私有，预置术语大幅提升速度
- **动态学习积累**：LLM发现新实体自动加入文档词典

**🎯 精准实体对齐**
- **WordAligner算法**：实体精确映射到原文位置，支持溯源验证
- **混合抽取模式**：词典匹配 + 规则匹配 + LLM抽取，准确率更高
- **多模型隔离存储**：切换 Embedding 模型不丢失数据

**⚡ 智能搜索体验**
- **即时搜索**：索引完成后立即可搜索，无需重启
- **缓存自动刷新**：索引完成自动清除缓存，新数据实时生效
- **多文档联合搜索**：跨文档知识关联，全局视角回答
- **智能回退**：本地知识不足时自动联网搜索，确保回答完整

### 4️⃣ 零 API 网络搜索
- 🌐 **自动部署 SearXNG**：无需配置，首次使用自动拉取 Docker 镜像
- 🔍 **实时信息获取**：天气、股价、新闻，联网即查
- 🛡️ **隐私保护**：搜索通过自建 SearXNG，不依赖第三方 API
- 📡 **远程服务支持**：可配置外部 SearXNG 实例

### 5️⃣ OpenCode 式终端交互
- ⌨️ **斜杠命令体系**：`/model`、`/new`、`/sessions` 等命令快速操作
- 🔄 **运行时模型切换**：无需重启程序即可更换模型
- 💬 **多会话管理**：每个会话独立历史，随时切换
- 📊 **实时状态显示**：当前模型、会话、消息数一目了然
- ⚡ **真正并行处理**：索引和对话同时进行，互不阻塞
- 📝 **断点续传**：索引失败后可恢复任务，不重复工作

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

### 🌐 Web 搜索 - 零 API 连接互联网

通过内置的 `web_search` 和 `web_fetch` 工具，智能体可以：
- 🔍 实时搜索网络信息
- 📰 获取最新新闻和动态
- 🌤️ 查询实时天气、股价等信息

```python
# 自动激活 Web 搜索 Skill
agent.invoke("今天比特币价格是多少？")
# AI 会自动搜索网络并返回实时价格
```

**自动部署 SearXNG**：
```python
# 首次使用 web_search 时自动完成：
# 1. 检查 Docker 是否安装
# 2. 自动拉取 searxng/searxng 镜像
# 3. 创建并启动容器
# 4. 等待服务就绪

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

### v0.2.4 (当前) - 批量索引优化与搜索体验提升

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