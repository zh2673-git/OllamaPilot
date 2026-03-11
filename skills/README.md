# OllamaPilot 三层搜索架构

> 基于 "agent 基座 + skill" 架构的多层搜索系统

## 架构概览

```
┌─────────────────────────────────────────────────────────────┐
│  第三层：DeepResearchSkill (研究流程层)                      │
│  ─────────────────────────────────────                      │
│  多轮迭代研究，生成专业报告                                   │
│  参考: Open Deep Research 的 Supervisor-Agent 模式          │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第二层：EnhancedSearchSkill (增强搜索层)                    │
│  ─────────────────────────────────────                      │
│  多引擎聚合，专业领域搜索                                     │
│  参考: Local Deep Research 的搜索引擎工厂                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  第一层：web_search (基础工具层)                             │
│  ─────────────────────────────                              │
│  单一引擎，简单快速，基础通用搜索                             │
│  现有实现，保持简单稳定                                      │
└─────────────────────────────────────────────────────────────┘
```

## 核心特点

- ✅ **完全免费**: 不使用任何付费 API
- ✅ **国内可用**: 所有引擎在国内网络环境下可访问
- ✅ **分层清晰**: 三层架构，职责分离，避免功能重复
- ✅ **易于扩展**: 新引擎接入成本低

## 已实现的搜索引擎

| 引擎 | 类别 | 免费 | 国内可用 | 特点 |
|------|------|------|---------|------|
| **SearXNG** | general | ✅ | ✅ | 本地部署，聚合多源 |
| **arXiv** | academic | ✅ | ✅ | 物理/数学/CS论文 |
| **PubMed** | academic | ✅ | ✅ | 生物医学文献 |
| **Wikipedia** | encyclopedia | ✅ | ✅ | 多语言百科 |
| **百度百科** | encyclopedia | ✅ | ✅ | 中文百科 |
| **GitHub** | code | ✅ | ✅ | 国际开源代码 |
| **Gitee** | code | ✅ | ✅ | 国内代码托管 |
| **Wikipedia(EN)** | encyclopedia | ✅ | ✅ | 英文百科 |

## 快速开始

### 1. 运行测试

```bash
# 运行基础测试
python tests/test_enhanced_search.py

# 运行演示
python examples/enhanced_search_demo.py
```

### 2. 在 Agent 中使用

```python
from ollamapilot import init_ollama_model, OllamaPilotAgent
from skills.enhanced_search import EnhancedSearchSkill
from skills.deep_research import DeepResearchSkill

# 初始化模型
model = init_ollama_model("qwen3.5:4b")

# 创建 Agent
agent = OllamaPilotAgent(model, skills_dir="skills")

# Skill 会自动从 skills/ 目录加载
# 或者手动注册
enhanced_search = EnhancedSearchSkill()
deep_research = DeepResearchSkill()

agent.skill_registry.register(enhanced_search)
agent.skill_registry.register(deep_research)

# 使用示例
response = agent.chat("学术搜索量子计算最新进展")
response = agent.chat("深度研究 AI 在医疗诊断中的应用")
```

### 3. 直接使用 Skill

```python
from skills.enhanced_search import EnhancedSearchSkill

# 创建 Skill
skill = EnhancedSearchSkill()

# 学术搜索
result = skill._academic_search("machine learning", num_results=5)
print(result)

# 代码搜索
result = skill._code_search("python web framework", num_results=5)
print(result)

# 百科搜索
result = skill._encyclopedia_search("人工智能", num_results=5)
print(result)

# 多引擎搜索
result = skill._multi_engine_search("深度学习", num_results=10)
print(result)
```

## 文件结构

```
skills/
├── enhanced_search/              # 第二层: 增强搜索
│   ├── __init__.py
│   ├── skill.py                  # EnhancedSearchSkill 主类
│   ├── aggregator.py             # 结果聚合器
│   └── engines/                  # 搜索引擎实现
│       ├── __init__.py
│       ├── base.py               # 搜索引擎基类 + 工厂
│       ├── searxng.py            # SearXNG (复用内置)
│       ├── arxiv.py              # arXiv 论文
│       ├── wikipedia.py          # Wikipedia (中英文)
│       ├── baidu_baike.py        # 百度百科
│       ├── pubmed.py             # PubMed 医学
│       └── github.py             # GitHub + Gitee
│
└── deep_research/                # 第三层: 深度研究
    ├── __init__.py
    ├── skill.py                  # DeepResearchSkill 主类
    └── state.py                  # 研究状态定义
```

## 触发词

### EnhancedSearchSkill

- **学术搜索**: 学术搜索、论文搜索、文献查找、arxiv、pubmed
- **代码搜索**: 代码搜索、github搜索、gitee搜索、开源项目
- **百科搜索**: 百科搜索、wikipedia、维基百科、百度百科
- **多引擎**: 多引擎搜索、聚合搜索、全面搜索

### DeepResearchSkill

- 深度研究、研究报告、调研
- 详细分析、全面调查

## 依赖安装

```bash
# 基础依赖（已包含在项目 requirements 中）
pip install langchain-core

# 可选：深度研究功能需要
pip install langgraph
```

## 配置说明

在 `.env` 文件中添加：

```env
# SearXNG 配置
SEARXNG_URL=http://localhost:8080

# 深度研究配置
DEEP_RESEARCH_MAX_ITERATIONS=6
DEEP_RESEARCH_SAVE_REPORTS=true
DEEP_RESEARCH_REPORTS_DIR=./reports
```

## 参考项目

| 项目 | 用途 | 链接 |
|------|------|------|
| Open Deep Research | Supervisor-Agent 研究流程 | https://github.com/langchain-ai/open_deep_research |
| Local Deep Research | 搜索引擎工厂 + 结果聚合 | https://github.com/LearningCircuit/local-deep-research |
| LangGraph | 状态图流程编排 | https://langchain-ai.github.io/langgraph/ |

## 许可证

MIT License
