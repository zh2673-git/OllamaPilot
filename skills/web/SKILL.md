---
name: web
description: 网络工具，包括网页搜索、内容获取、HTTP请求等功能
tags: ["网络", "搜索", "HTTP", "网页", "API"]
triggers:
  - "搜索"
  - "查找"
  - "获取网页"
  - "网络搜索"
  - "google"
  - "baidu"
  - "http"
  - "下载"
  - "网页内容"
version: "1.0.0"
author: "BaseAgent Team"
# 工具配置 - 只配置额外工具，内置工具自动加载
tools:
  # Web Skill 特有工具（内置工具如 read_file/write_file 会自动加载）
  - name: web_search
    type: local
    module: skills.web.skill
  - name: web_fetch
    type: local
    module: skills.web.skill
---

# Web Skill - 网络工具

提供网络相关的实用工具，包括网页搜索、内容获取、HTTP请求等功能。

## 🎯 核心能力

- **网页搜索**: 使用搜索引擎查找信息
- **内容获取**: 抓取网页内容并提取文本
- **HTTP请求**: 发送 GET/POST 等 HTTP 请求
- **数据下载**: 下载文件或资源

## 🔧 可用工具

> **说明**：所有 Skill 自动拥有 9 个内置工具（文件操作、Shell 命令、代码处理等）

### Web Skill 特有工具
- `web_search` - 网页搜索
- `web_fetch` - 获取网页内容

## 💡 使用示例

### 示例 1: 搜索并保存
```
用户: 搜索 Python 最新版本的信息并保存到文件

执行:
1. web_search(query="Python 3.12 新特性")
2. write_file(file_path="python_news.txt", content=搜索结果)
```

### 示例 2: 获取网页内容
```
用户: 获取某个网页的内容

执行:
web_fetch(url="https://example.com/article")
```
