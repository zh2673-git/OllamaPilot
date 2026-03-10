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

### v0.2.2 (当前) - 搜索性能优化与多文档管理

**🎉 重磅更新：知识图谱搜索性能优化**

- ✅ **GraphRAGService 缓存机制**：避免每次搜索重复加载索引
  - 按文档ID和模型名称缓存服务实例
  - 首次搜索创建，后续直接复用
  - 搜索性能提升：从 O(n×m) 到 O(n)
- ✅ **修复知识图谱为空问题**：支持搜索所有已索引的文档
  - DocumentManager 管理多文档索引
  - 自动遍历所有文档子目录进行搜索
  - 全局统计信息汇总
- ✅ **优化实体抽取**：恢复规则匹配机制
  - 词典匹配 + 规则匹配 + LLM抽取
  - 多层过滤减少误匹配
  - 通用词典覆盖常见实体类型

### v0.2.1 - 混合模式实体抽取与通用词典

**🎉 智能实体抽取与知识图谱优化**

- ✅ **混合模式实体抽取**：词典匹配 + LLM抽取 + 动态学习
- ✅ **通用实体词典**：80个跨领域通用实体
- ✅ **人工干预接口**：支持导出/导入词典
- ✅ **修复实体抽取失效**：添加基础词典
- ✅ **清理冗余文件**：删除 `ontology_generator.py`

### v0.2.0 - 移除Ollama锁，真正并行处理

**🎉 架构简化与性能优化**

- ✅ **移除 Ollama 锁**：索引和对话真正并行，互不阻塞
- ✅ **修复索引性能问题**：移除 GraphRAGService 重复的模型测试（每次60秒）
- ✅ **简化架构**：删除 `ollama_lock.py` 模块，减少复杂度
- ✅ **main.py 重构**：模块化结构
- ✅ **断点续传**：`/resume` 命令
- ✅ **命令自动补全**：Tab 键支持
- ✅ **默认知识库**：`/index` 默认路径
- ✅ **异步后台索引**：后台运行
- ✅ **技能触发优化**：显式触发

### v0.1.9 - 代码重构与稳定性增强

**🎉 架构优化与并发保护**

- ✅ **main.py 重构**：模块化结构
- ✅ **Ollama 并发保护**：全局锁机制（已移除）
- ✅ **断点续传**：`/resume` 命令
- ✅ **命令自动补全**：Tab 键支持
- ✅ **默认知识库**：`/index` 默认路径
- ✅ **异步后台索引**：后台运行
- ✅ **技能触发优化**：显式触发

### v0.1.8 - WordAligner 与手动文档管理

**🎉 知识图谱增强与配置管理**

- ✅ **WordAligner 精确对齐**：移植 LangExtract 算法，实体精确映射到原文位置
- ✅ **手动文档管理**：通过指令控制文档索引，静默后台处理
- ✅ **模型+文档名隔离存储**：如 `伤寒论_qwen3_embedding_0_6b/`，方便管理
- ✅ **.env 配置管理**：支持配置文件设置模型参数，便于前端集成
- ✅ **配置热重载**：运行时重新加载配置，无需重启

### v0.1.7 - 模型切换与会话管理

**🎉 重磅更新：OpenCode 式终端交互**

- ✅ **斜杠命令体系**：`/model`、`/new`、`/sessions` 等命令
- ✅ **运行时模型切换**：无需重启即可切换对话模型
- ✅ **多会话管理**：每个会话独立 thread_id，历史隔离
- ✅ **Embedding 动态切换**：可随时切换或禁用 Embedding 模型

### v0.1.6 - 修复 Skill 冲突

- ✅ 修复 GraphRAG 拦截其他 Skill 的问题
- ✅ 添加 active_skill 状态跟踪

### v0.1.5 - 流式输出优化

- ✅ 修复流式输出重复显示问题
- ✅ 添加聊天问题检测，智能跳过知识库检索

### v0.1.4 - 数据目录隔离

- ✅ 添加 .gitignore 排除 knowledge_base/ 和 data/ 目录
- ✅ 修复流式输出偶发无响应问题

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