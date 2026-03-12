---
name: web_search
description: 智能网络搜索 - 支持多引擎优先级和自动降级
triggers:
  # 基础搜索 - 简单直接的查询
  - 搜索
  - 查
  - 找
  - 看看
  - 了解一下
tools:
  - web_search
  - web_search_setup
  - web_fetch
---

# Web 搜索 Skill

## 基本信息

- **name**: web_search
- **description**: 智能网络搜索 - 支持多引擎优先级和自动降级
- **version**: 2.0.0
- **author**: OllamaPilot

## 核心特性

### 多引擎优先级策略

`web_search` 工具会按以下优先级尝试多个搜索引擎：

```
搜索优先级队列:
1. SearXNG (本地部署)    → 质量最佳，无限额度
2. DuckDuckGo           → 免费备用，无需配置
3. 其他配置API          → Serper/Bing/Brave (如果配置了)
```

### 自动降级机制

- **SearXNG 可用** → 使用 SearXNG（聚合多个搜索引擎）
- **SearXNG 不可用** → 自动降级到 DuckDuckGo
- **配置了其他 API** → 按优先级尝试其他引擎
- **所有引擎失败** → 返回友好提示

## 系统提示词

你是一个专业的网络搜索助手。当用户需要搜索网络信息时，你会使用 web_search 工具进行搜索。

### 搜索策略

1. **智能引擎选择**：web_search 工具会自动选择最佳可用引擎
2. **关键词优化**：将用户的自然语言查询转换为高效的搜索关键词
3. **结果筛选**：从搜索结果中提取最相关的信息，去除广告和低质量内容
4. **信息整合**：将多个来源的信息整合成连贯的回答

### 使用流程

1. 直接使用 `web_search(query="搜索内容")` 进行搜索
2. 工具会自动选择最佳引擎（SearXNG → DuckDuckGo → 其他）
3. 如果搜索失败，使用 `web_search_setup(action="status")` 检查服务状态
4. 如需手动管理 SearXNG，使用 `web_search_setup(action="start/stop/logs")`

### 注意事项

- **无需配置也能用**：如果没有配置任何 API，会自动使用 DuckDuckGo（免费）
- **推荐部署 SearXNG**：本地部署可获得最佳搜索质量
- **支持通过环境变量配置**：使用远程 SearXNG 实例或其他搜索 API

## 配置

### 环境变量

```bash
# SearXNG 配置（推荐本地部署，质量最佳）
SEARXNG_URL=http://localhost:8080

# 可选：其他搜索 API（提升搜索质量）
SERPER_API_KEY=your_key_here      # Google 搜索，2500次/月
BING_API_KEY=your_key_here        # 必应搜索，1000次/月
BRAVE_API_KEY=your_key_here       # Brave 搜索，2000次/月
```

### 配置优先级

| 配置级别 | 需要配置 | 搜索质量 | 说明 |
|---------|---------|---------|------|
| **零配置** | 无需配置 | ⭐⭐⭐ | 使用 DuckDuckGo（免费） |
| **标准配置** | SearXNG | ⭐⭐⭐⭐⭐ | 本地部署，聚合多引擎 |
| **增强配置** | SearXNG + API Keys | ⭐⭐⭐⭐⭐ | 最佳搜索体验 |

### Docker 部署

**全自动部署（推荐）**：
```python
# 首次使用时会自动完成：
# 1. 检查 Docker 是否安装
# 2. 自动拉取 searxng/searxng 镜像（如本地不存在）
# 3. 创建并启动容器
# 4. 等待服务就绪
web_search_setup(action="start")

# 或者直接搜索，会自动触发部署
web_search(query="Python 教程")
```

**手动部署**：
```bash
docker run -d --name searxng -p 8080:8080 searxng/searxng
```

## 使用示例

**用户**: 搜索一下最新的 Python 3.13 特性

**助手**: 使用 web_search 工具搜索

```python
web_search(query="Python 3.13 new features", count=5)
```

**用户**: 为什么搜索不了？

**助手**: 检查搜索服务状态

```python
web_search_setup(action="status")
```

## 引擎选择逻辑

当用户调用 `web_search` 时，系统会：

1. **首先尝试智能路由器**（如果增强搜索模块可用）
   - 按优先级尝试 SearXNG → DuckDuckGo → 其他 API
   - 自动选择第一个可用的引擎

2. **如果智能路由器失败**
   - 回退到原始 SearXNG 逻辑
   - 尝试启动 SearXNG（如果配置了 auto_start）

3. **如果 SearXNG 也失败**
   - 返回错误提示和解决方案

## 相关链接

- SearXNG 项目: https://github.com/searxng/searxng
- SearXNG 文档: https://docs.searxng.org/
- DuckDuckGo: https://duckduckgo.com/
