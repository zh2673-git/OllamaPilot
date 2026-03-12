---
name: enhanced_search
description: 增强搜索 - 多引擎专业搜索（学术/代码/百科）
triggers:
  # 增强搜索 - 专业领域搜索
  - 增强
  - 专业搜索
  - 高级搜索
tools:
  - academic_search
  - code_search
  - encyclopedia_search
  - multi_engine_search
  - get_search_quota_report
  - check_search_engine_availability
---

# 增强搜索 Skill

## 基本信息

- **name**: enhanced_search
- **description**: 增强搜索 - 多引擎专业搜索（学术/代码/百科）
- **version**: 1.1.0
- **author**: OllamaPilot

## 核心特性

### 免费API优先策略

本Skill优先使用**国内可免费使用的API**，无需翻墙：

| 搜索类型 | 优先引擎 | 备用引擎 | 额度 |
|---------|---------|---------|------|
| 学术搜索 | PubMed | arXiv | 无限 |
| 代码搜索 | GitHub | Gitee | GitHub: 500次/小时 |
| 百科搜索 | 百度百科 | Wikipedia | 无限 |
| 通用搜索 | SearXNG | DuckDuckGo | 无限 |

### 智能降级机制

当主要引擎不可用时，自动降级到备用引擎：
- GitHub配额用完 → 自动切换到 Gitee
- SearXNG不可用 → 自动切换到 DuckDuckGo
- 所有引擎失败 → 返回友好提示

## 系统提示词

你是增强搜索助手，被用户显式调用（用户输入包含"增强"关键词）。

### 工具选择指南

**1. academic_search(query)** - 学术文献
- 用户想查：论文、学术资料、研究成果、医学文献
- 使用引擎：PubMed（优先）→ arXiv（备用）
- 示例："增强搜索 量子计算论文" → 用 academic_search

**2. code_search(query)** - 代码仓库  
- 用户想查：开源项目、GitHub、代码示例、编程库
- 使用引擎：GitHub（优先）→ Gitee（备用）
- 示例："增强搜索 Python爬虫" → 用 code_search

**3. encyclopedia_search(query)** - 百科知识
- 用户想查：概念解释、定义、百科知识
- 使用引擎：百度百科（优先）→ Wikipedia（备用）
- 示例："增强搜索 什么是区块链" → 用 encyclopedia_search

**4. multi_engine_search(query)** - 综合搜索
- 用户想查：全面信息、多来源验证
- 使用引擎：SearXNG（优先）→ DuckDuckGo（备用）
- 示例："增强搜索 深度学习应用" → 用 multi_engine_search

**5. get_search_quota_report()** - 配额报告
- 用户想查看API使用情况
- 示例："查看搜索配额" → 用 get_search_quota_report

**6. check_search_engine_availability()** - 可用性检查
- 用户想检查哪些引擎可用
- 示例："检查搜索引擎" → 用 check_search_engine_availability

### 简单规则

- 提到"论文"/"学术"/"文献" → academic_search
- 提到"代码"/"GitHub"/"项目" → code_search  
- 提到"什么是"/"概念"/"定义" → encyclopedia_search
- 提到"配额"/"额度" → get_search_quota_report
- 提到"可用"/"状态" → check_search_engine_availability
- 其他情况 → multi_engine_search

## 可用工具

### academic_search(query: str, num_results: int = 10) -> str
搜索学术论文和文献

**引擎策略：**
1. PubMed（医学文献，无限额度）
2. arXiv（学术论文，无限额度，不稳定时跳过）

**参数：**
- query: 搜索查询
- num_results: 返回结果数量（默认10）

**示例：**
```python
academic_search(query="machine learning", num_results=5)
```

### code_search(query: str, num_results: int = 10, language: str = None) -> str
搜索开源代码仓库

**引擎策略：**
1. GitHub（500次/小时，需配置 GITHUB_TOKEN）
2. Gitee（无限额度，国内友好）

**参数：**
- query: 搜索查询
- num_results: 返回结果数量（默认10）
- language: 编程语言过滤（可选）

**示例：**
```python
code_search(query="python web framework", num_results=5)
code_search(query="machine learning", language="python")
```

**配置说明：**
在 `.env` 文件中配置 GitHub Token：
```bash
GITHUB_TOKEN=your_github_token_here
```

### encyclopedia_search(query: str, num_results: int = 10) -> str
搜索百科知识

**引擎策略：**
1. 百度百科（中文优化，无限额度）
2. Wikipedia（无限额度，不稳定时跳过）

**参数：**
- query: 搜索查询
- num_results: 返回结果数量（默认10）

**示例：**
```python
encyclopedia_search(query="人工智能", num_results=5)
```

### multi_engine_search(query: str, num_results: int = 10, engines: list = None) -> str
多引擎聚合搜索

**引擎策略：**
1. SearXNG（本地聚合，无限额度，需Docker部署）
2. DuckDuckGo（免费，无需API Key）

**参数：**
- query: 搜索查询
- num_results: 返回结果数量（默认10）
- engines: 指定搜索引擎列表（可选）

**示例：**
```python
multi_engine_search(query="深度学习", num_results=10)
```

### get_search_quota_report() -> str
获取API配额使用报告

**返回：**
各引擎的配额使用情况，包括：
- 限额和已使用次数
- 剩余配额
- 使用率百分比

**示例：**
```python
get_search_quota_report()
```

### check_search_engine_availability() -> str
检查各搜索引擎可用性

**返回：**
各搜索类别的可用引擎列表

**示例：**
```python
check_search_engine_availability()
```

## 使用示例

**用户**: 增强搜索 量子计算论文

**助手**: 使用 academic_search 工具搜索学术论文

```python
academic_search(query="quantum computing", num_results=5)
```

**用户**: 增强搜索 Python爬虫

**助手**: 使用 code_search 工具搜索代码仓库

```python
code_search(query="python crawler", num_results=5)
```

**用户**: 增强搜索 什么是区块链

**助手**: 使用 encyclopedia_search 工具搜索百科

```python
encyclopedia_search(query="区块链", num_results=5)
```

**用户**: 增强搜索 深度学习应用

**助手**: 使用 multi_engine_search 工具进行综合搜索

```python
multi_engine_search(query="深度学习应用", num_results=10)
```

**用户**: 查看搜索配额

**助手**: 使用 get_search_quota_report 工具查看配额

```python
get_search_quota_report()
```

## 配置说明

### 可选配置（在 .env 文件中）

```bash
# SearXNG 配置（推荐本地部署）
SEARXNG_URL=http://localhost:8080

# GitHub API Token（提高代码搜索稳定性）
GITHUB_TOKEN=your_github_token_here

# 可选：其他搜索引擎API（免费额度）
SERPER_API_KEY=your_serper_key_here      # 2500次/月
BING_API_KEY=your_bing_key_here          # 1000次/月
BRAVE_API_KEY=your_brave_key_here        # 2000次/月
```

### 推荐配置组合

| 配置级别 | 需要配置 | 搜索质量 |
|---------|---------|---------|
| **极简** | 无需配置 | ⭐⭐⭐ |
| **标准** | GITHUB_TOKEN | ⭐⭐⭐⭐ |
| **增强** | GITHUB_TOKEN + SearXNG | ⭐⭐⭐⭐⭐ |

## 注意事项

1. **免费优先**：所有主要引擎均为免费API，国内可直接访问
2. **自动降级**：配额用完或引擎失败时自动切换
3. **配额管理**：GitHub等有限额引擎会自动跟踪使用情况
4. **无需翻墙**：所有推荐引擎均可在国内网络直接使用
5. **SearXNG部署**：如需最佳体验，建议本地Docker部署 SearXNG

## 故障排除

**问题**：搜索结果为空
- 检查网络连接
- 运行 `check_search_engine_availability()` 查看可用引擎
- 检查是否配置了必要的API Token

**问题**：GitHub搜索失败
- 检查 `.env` 中是否配置了 GITHUB_TOKEN
- 运行 `get_search_quota_report()` 查看是否超出配额
- 系统会自动降级到 Gitee

**问题**：SearXNG不可用
- 检查 Docker 是否运行
- 检查 SEARXNG_URL 配置是否正确
- 系统会自动降级到 DuckDuckGo
