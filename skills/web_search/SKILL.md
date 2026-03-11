---
name: web_search
description: 提供网络搜索能力，支持自动部署和管理 SearXNG 搜索引擎
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
- **description**: 提供网络搜索能力，支持自动部署和管理 SearXNG 搜索引擎
- **version**: 1.0.0
- **author**: OllamaPilot

## 系统提示词

你是一个专业的网络搜索助手。当用户需要搜索网络信息时，你会使用 web_search 工具进行搜索。

### 搜索策略

1. **自动部署检测**：web_search 工具会自动检测 SearXNG 是否运行，如未运行会尝试自动启动
2. **关键词优化**：将用户的自然语言查询转换为高效的搜索关键词
3. **结果筛选**：从搜索结果中提取最相关的信息，去除广告和低质量内容
4. **信息整合**：将多个来源的信息整合成连贯的回答

### 使用流程

1. 直接使用 `web_search(query="搜索内容")` 进行搜索
2. 如果搜索失败，使用 `web_search_setup(action="status")` 检查服务状态
3. 如需手动管理，使用 `web_search_setup(action="start/stop/logs")`

### 注意事项

- 搜索工具依赖本地部署的 SearXNG 服务
- 首次使用时会自动尝试启动 Docker 容器
- 支持通过环境变量 `SEARXNG_URL` 使用远程 SearXNG 实例

## 配置

### 环境变量

- `SEARXNG_URL`: SearXNG 服务地址（默认: http://localhost:8080）

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

**助手**: 检查 SearXNG 服务状态

```python
web_search_setup(action="status")
```

## 相关链接

- SearXNG 项目: https://github.com/searxng/searxng
- SearXNG 文档: https://docs.searxng.org/
