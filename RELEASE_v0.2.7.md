# OllamaPilot v0.2.7 发布说明

## 🎉 重磅更新：多引擎智能搜索与API配额管理

### 1. 免费API优先搜索策略

支持 **10+ 搜索引擎**，优先使用国内可免费使用的API：

| 引擎 | 类型 | 免费额度 | 配置项 | 国内可用 |
|------|------|---------|--------|---------|
| Tavily | 通用 | 1000次/月 | TAVILY_API_KEY | ✅ |
| SearXNG | 聚合 | 无限 | 本地部署 | ✅ |
| Serper | Google | 2500次/月 | SERPER_API_KEY | ✅ |
| Bing | 必应 | 1000次/月 | BING_API_KEY | ✅ |
| Brave | Brave | 2000次/月 | BRAVE_API_KEY | ✅ |
| DuckDuckGo | 通用 | 无限 | 无需配置 | ✅ |
| PubMed | 学术 | 无限 | 无需配置 | ✅ |
| 百度百科 | 百科 | 无限 | 无需配置 | ✅ |
| GitHub | 代码 | 500次/小时 | GITHUB_TOKEN | ✅ |
| Gitee | 代码 | 无限 | 无需配置 | ✅ |

### 2. 智能降级机制

当主要引擎不可用时，自动降级到备用引擎：

```
通用搜索优先级：
Tavily → SearXNG → Serper → Bing → Brave → DuckDuckGo

学术搜索优先级：
PubMed → arXiv

代码搜索优先级：
GitHub → Gitee
```

### 3. API配额管理

- 自动跟踪各API使用情况
- 配额用完时自动切换到备用引擎
- 支持配额警告（默认80%阈值）
- 配额信息持久化存储

### 4. 零配置可用

即使不配置任何API，系统也能正常工作：
- 通用搜索：DuckDuckGo（免费）
- 学术搜索：PubMed + arXiv（免费）
- 代码搜索：Gitee（免费）
- 百科搜索：百度百科（免费）

---

## 🔧 新增功能

### 新增搜索引擎

| 引擎 | 文件 | 说明 |
|------|------|------|
| Tavily | `engines/tavily.py` | 专为AI应用设计，质量最佳 |
| Serper | `engines/serper.py` | Google搜索结果 |
| Bing | `engines/bing.py` | 微软必应搜索 |
| Brave | `engines/brave.py` | Brave独立搜索引擎 |
| DuckDuckGo | `engines/duckduckgo.py` | 隐私保护搜索 |

### 新增核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| 配额管理器 | `quota_manager.py` | 跟踪API使用情况 |
| 引擎路由器 | `engine_router.py` | 智能选择和降级 |

### 新增工具

| 工具 | 功能 |
|------|------|
| `get_search_quota_report()` | 查看API配额使用情况 |
| `check_search_engine_availability()` | 检查各引擎可用性 |

---

## 📖 配置指南

### 快速配置（推荐）

在 `.env` 文件中配置API Key：

```bash
# 第1优先：Tavily（质量最佳）
TAVILY_API_KEY=your_tavily_key_here

# 第2优先：SearXNG（本地部署）
SEARXNG_URL=http://localhost:8080

# 其他可选
SERPER_API_KEY=your_serper_key_here
BING_API_KEY=your_bing_key_here
BRAVE_API_KEY=your_brave_key_here
GITHUB_TOKEN=your_github_token_here
```

### 申请地址

- **Tavily**: https://tavily.com (1000次/月)
- **Serper**: https://serper.dev (2500次/月)
- **Bing**: https://www.microsoft.com/en-us/bing/apis/bing-web-search-api (1000次/月)
- **Brave**: https://brave.com/search/api/ (2000次/月)
- **GitHub**: https://github.com/settings/tokens (500次/小时)

---

## 📝 使用示例

### 基础搜索

```
你: 搜索苏州明天天气
助手: 🔍 使用 Tavily 搜索...
     ✅ Tavily 返回 5 条结果
```

### 查看配额

```
你: 查看搜索配额
助手: 📊 API配额使用报告
     ✅ Tavily: 限额 1000 | 已用 45 | 剩余 955 | 使用率 4.5%
     ✅ SearXNG: 无限额度
     ...
```

### 检查可用性

```
你: 检查搜索引擎
助手: 🔍 搜索引擎可用性检查
     ✅ 通用搜索: tavily, searxng, serper, bing, brave, duckduckgo
     ✅ 学术搜索: pubmed, arxiv
     ✅ 代码搜索: github, gitee
     ✅ 百科搜索: baidu_baike, wikipedia
```

---

## 🔧 技术细节

### 引擎注册机制

所有引擎通过 `@register_engine` 装饰器自动注册：

```python
@register_engine
class TavilySearchEngine(SearchEngineBase):
    name = "tavily"
    ...
```

### 实时环境变量读取

引擎在 `is_available()` 中实时读取环境变量，确保 `.env` 配置生效：

```python
def is_available(self) -> bool:
    api_key = os.environ.get("TAVILY_API_KEY")
    return bool(api_key)
```

### 配额持久化

配额信息保存到 `data/api_quotas.json`，程序重启后仍然有效。

---

## 📋 版本信息

- **版本号**: v0.2.7
- **发布日期**: 2026-03-12
- **兼容性**: 完全兼容 v0.2.x

## 🔄 更新日志

```
v0.2.7 (2026-03-12)
├── 新增
│   ├── Tavily 搜索引擎
│   ├── Serper 搜索引擎
│   ├── Bing 搜索引擎
│   ├── Brave 搜索引擎
│   ├── DuckDuckGo 搜索引擎
│   ├── API配额管理器
│   ├── 搜索引擎路由器
│   ├── get_search_quota_report 工具
│   └── check_search_engine_availability 工具
├── 改进
│   ├── 通用搜索支持多引擎优先级
│   ├── 智能降级机制
│   ├── .env 配置自动设置到环境变量
│   └── 深度研究使用实际搜索
└── 文档
    ├── 更新 README.md 搜索说明
    ├── 新增 API配置指南.md
    └── 新增 免费API搜索策略设计.md
```

---

**完整文档**: [README.md](https://github.com/zh2673-git/OllamaPilot/blob/main/README.md)  
**配置指南**: [API配置指南.md](https://github.com/zh2673-git/OllamaPilot/blob/main/笔记/OllamaPilot/API配置指南.md)  
**问题反馈**: [GitHub Issues](https://github.com/zh2673-git/OllamaPilot/issues)
