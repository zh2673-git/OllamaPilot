# Changelog

所有项目的显著变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
并且本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.2.0] - 2026-03-06

### Added
- **新增 Python Solver Skill** - 强大的 Python 代码执行能力
  - `python_exec` - 执行 Python 代码片段，支持快速计算和数据处理
  - `python_script` - 执行完整的 Python 脚本文件
  - `python_notebook` - 交互式代码执行（类似 Jupyter Notebook）
  - `python_install` - 安装 Python 包
  - 内置安全机制，防止危险操作
  - 预导入常用库（pandas, numpy, matplotlib 等）
  - 触发关键词：python, 脚本, 计算, 数据处理, 自动化等

### Example
使用示例：
```
用户: 计算 9797890 乘以 44322
Agent: 调用 python_exec 工具执行计算
结果: 434,262,080,580
```

## [0.1.0] - 2025

### Added
- 初始版本发布
- 基座 Agent + Skill 架构
- 内置 9 个基础工具（文件系统、Shell、代码处理）
- Web Skill - 本地搜索引擎支持（SearXNG）
- 模型驱动路由决策
- 小模型优化中间件链
- 流式决策展示

---

## 版本说明

- **MAJOR** 版本：不兼容的 API 修改
- **MINOR** 版本：向下兼容的功能新增
- **PATCH** 版本：向下兼容的问题修复
