---
name: python-solver
description: Use when 需要用 Python 解决各类问题，包括数据分析、文件处理、文本处理、计算任务、自动化脚本等。当用户提到"用 Python 解决"、"写个脚本"、"处理数据"、"计算"、"分析"等关键词时触发。
tags: ["python", "脚本", "数据处理", "计算", "自动化"]
triggers:
  - "python"
  - "脚本"
  - "用 Python"
  - "写个脚本"
  - "处理数据"
  - "数据分析"
  - "计算"
  - "自动化"
  - "批量处理"
version: "1.0.0"
author: "BaseAgent Team"
tools:
  - name: python_exec
    type: local
    module: skills.python_solver.skill
  - name: python_script
    type: local
    module: skills.python_solver.skill
  - name: python_notebook
    type: local
    module: skills.python_solver.skill
---

# Python Solver - Python 问题解决工具

提供强大的 Python 代码执行能力，帮助解决各类编程和数据处理问题。

## 🎯 核心能力

- **代码执行**: 直接执行 Python 代码片段
- **脚本运行**: 运行完整的 Python 脚本文件
- **数据分析**: 处理 CSV、JSON、Excel 等数据文件
- **文本处理**: 字符串操作、正则表达式、文本转换
- **文件处理**: 批量重命名、格式转换、文件整理
- **计算任务**: 数学计算、统计分析、算法实现
- **可视化**: 生成图表、数据可视化

## 🔧 可用工具

> **说明**: 所有 Skill 自动拥有 9 个内置工具（文件操作、Shell 命令、代码处理等）

### Python Solver 特有工具

| 工具名 | 功能 | 适用场景 |
|--------|------|----------|
| `python_exec` | 执行单行/多行 Python 代码 | 快速计算、简单处理 |
| `python_script` | 执行 Python 脚本文件 | 复杂任务、多步骤处理 |
| `python_notebook` | 交互式代码执行（类似 Jupyter） | 数据分析、逐步探索 |

## 💡 使用示例

### 示例 1: 快速计算
```
用户: 计算 1 到 100 的和

执行:
python_exec(code="sum(range(1, 101))")
```

### 示例 2: 数据处理
```
用户: 读取 CSV 文件并统计行数

执行:
python_script(script="""
import pandas as pd
df = pd.read_csv('data.csv')
print(f"总行数: {len(df)}")
print(f"列名: {list(df.columns)}")
print(df.head())
""")
```

### 示例 3: 文件批量处理
```
用户: 批量重命名当前目录下的所有 .txt 文件

执行:
python_script(script="""
import os
for i, filename in enumerate(os.listdir('.')):
    if filename.endswith('.txt'):
        new_name = f'doc_{i:03d}.txt'
        os.rename(filename, new_name)
        print(f'{filename} -> {new_name}')
""")
```

### 示例 4: 文本处理
```
用户: 提取这段文本中的所有邮箱地址

执行:
python_exec(code="""
import re
text = '''联系邮箱: john@example.com, support@company.org
备用邮箱: admin@test.com'''
emails = re.findall(r'[\w.-]+@[\w.-]+\.\w+', text)
print(emails)
""")
```

### 示例 5: 数据可视化
```
用户: 生成一个简单的折线图

执行:
python_script(script="""
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.title('正弦函数')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.savefig('sine_wave.png')
print("图表已保存为 sine_wave.png")
""")
```

## 📋 支持的库

Python Solver 预装以下常用库：

- **数据处理**: `pandas`, `numpy`
- **文本处理**: `re` (内置), `json`, `csv`
- **文件操作**: `pathlib`, `shutil`, `os`
- **可视化**: `matplotlib`, `seaborn`
- **网络**: `requests`, `urllib`
- **日期时间**: `datetime`, `time`
- **数学**: `math`, `random`, `statistics`

## ⚠️ 安全限制

- 禁止访问系统敏感目录
- 禁止执行系统命令（使用 shell 工具代替）
- 网络访问受限（使用 web skill 代替）
- 文件操作限制在工作目录

## 🔗 相关 Skills

- `web` - 需要网络搜索或 HTTP 请求时
- `filesystem` - 复杂的文件系统操作
- `shell` - 需要执行系统命令时
