"""
Markdown 格式记忆存储系统

兼顾可读性和性能：
- 人类可读的 Markdown 格式存储
- 使用 frontmatter 存储元数据（YAML 格式）
- 自动缓存到 JSON 以提升读取性能
- 支持手动编辑和注入记忆

存储结构：
- memories.md       # 主记忆文件（人类可读）
- memories.md.json  # 自动缓存（性能优化）
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ollamapilot.memory.types import MemoryType, MemoryEntry


class MarkdownMemoryStorage:
    """
    Markdown 格式记忆存储
    
    特性：
    - Markdown 格式便于人类阅读和编辑
    - YAML frontmatter 存储结构化元数据
    - 自动 JSON 缓存提升读取性能
    - 文件修改时自动重新加载
    
    使用示例：
        >>> storage = MarkdownMemoryStorage("./data/memories")
        >>> storage.add_entry(MemoryEntry(...))
        >>> entries = storage.load_entries()
    """
    
    def __init__(self, storage_dir: str = "./data/memories"):
        """
        初始化存储
        
        Args:
            storage_dir: 存储目录
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # 主文件路径
        self.md_file = self.storage_dir / "memories.md"
        self.cache_file = self.storage_dir / "memories.md.json"
        
        # 初始化文件
        self._init_files()
    
    def _init_files(self):
        """初始化存储文件"""
        if not self.md_file.exists():
            # 创建带说明的模板文件
            template = self._generate_template()
            self.md_file.write_text(template, encoding='utf-8')
        
        # 如果缓存不存在，从 Markdown 生成
        if not self.cache_file.exists() and self.md_file.exists():
            self._rebuild_cache()
    
    def _generate_template(self) -> str:
        """生成模板文件"""
        return """# System Memory - 系统记忆

> 此文件由 OllamaPilot 自动生成和维护
> 您可以手动编辑此文件来添加、修改或删除记忆
> 格式：YAML frontmatter + Markdown 内容

---

## 记忆条目格式说明

每个记忆条目格式如下：

```markdown
---
id: "唯一标识"
type: "semantic"  # semantic(语义) | procedural(程序) | episodic(情景)
category: "general"  # 分类
timestamp: "2024-01-15T10:30:00"
importance: 0.9  # 0-1 重要性评分
---

记忆内容写在这里，支持 Markdown 格式。
可以是多行文本。
```

---

## 语义记忆 (Semantic)

用户偏好、重要事实

---

## 程序记忆 (Procedural)

Skill 使用模式、操作习惯

---

## 情景记忆 (Episodic)

重要对话摘要、历史事件

---

"""
    
    def _parse_frontmatter(self, text: str) -> tuple[Dict[str, Any], str]:
        """
        解析 YAML frontmatter
        
        Args:
            text: Markdown 文本
            
        Returns:
            (frontmatter_dict, content)
        """
        pattern = r'^---\s*\n(.*?)\n---\s*\n?(.*)$'
        match = re.match(pattern, text.strip(), re.DOTALL)
        
        if not match:
            return {}, text.strip()
        
        yaml_text = match.group(1)
        content = match.group(2).strip()
        
        # 简单 YAML 解析
        frontmatter = {}
        for line in yaml_text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"\'')
                
                # 类型转换
                if value.lower() in ('true', 'false'):
                    frontmatter[key] = value.lower() == 'true'
                elif value.replace('.', '').isdigit():
                    frontmatter[key] = float(value) if '.' in value else int(value)
                else:
                    frontmatter[key] = value
        
        return frontmatter, content
    
    def _generate_frontmatter(self, data: Dict[str, Any]) -> str:
        """生成 YAML frontmatter"""
        lines = ["---"]
        for key, value in data.items():
            if isinstance(value, str):
                # 如果字符串包含特殊字符，使用引号
                if any(c in value for c in [':', '#', '\n', '"', "'"]):
                    value = f'"{value.replace("\\", "\\\\").replace('"', '\\"')}"'
                lines.append(f"{key}: {value}")
            elif isinstance(value, (int, float)):
                lines.append(f"{key}: {value}")
            elif isinstance(value, bool):
                lines.append(f"{key}: {str(value).lower()}")
        lines.append("---")
        return "\n".join(lines)
    
    def _entry_to_markdown(self, entry: MemoryEntry) -> str:
        """将记忆条目转换为 Markdown"""
        frontmatter_data = {
            "id": entry.id,
            "type": entry.type.value,
            "timestamp": entry.timestamp.isoformat(),
            "importance": entry.importance,
        }
        
        # 添加 metadata 中的其他字段
        for key, value in entry.metadata.items():
            if key not in frontmatter_data:
                frontmatter_data[key] = value
        
        frontmatter = self._generate_frontmatter(frontmatter_data)
        return f"{frontmatter}\n\n{entry.content}\n\n---\n\n"
    
    def _markdown_to_entry(self, markdown_text: str) -> Optional[MemoryEntry]:
        """将 Markdown 转换为记忆条目"""
        try:
            frontmatter, content = self._parse_frontmatter(markdown_text)
            
            if not frontmatter.get('id') or not frontmatter.get('type'):
                return None
            
            # 提取 metadata（排除标准字段）
            standard_fields = {'id', 'type', 'timestamp', 'importance'}
            metadata = {k: v for k, v in frontmatter.items() if k not in standard_fields}
            
            return MemoryEntry(
                id=frontmatter['id'],
                type=MemoryType(frontmatter['type']),
                content=content,
                metadata=metadata,
                timestamp=datetime.fromisoformat(frontmatter.get('timestamp', datetime.now().isoformat())),
                importance=frontmatter.get('importance', 1.0),
            )
        except Exception:
            return None
    
    def _rebuild_cache(self):
        """从 Markdown 重建 JSON 缓存"""
        entries = self._parse_markdown_file()
        cache_data = {
            "version": "1.0",
            "last_modified": datetime.now().isoformat(),
            "entries": [e.to_dict() for e in entries]
        }
        
        with open(self.cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    
    def _parse_markdown_file(self) -> List[MemoryEntry]:
        """解析 Markdown 文件中的所有条目"""
        if not self.md_file.exists():
            return []
        
        content = self.md_file.read_text(encoding='utf-8')
        
        # 按 --- 分隔符分割（跳过文件开头的说明）
        # 找到第一个实际的条目
        entries = []
        
        # 使用正则匹配所有 frontmatter 块
        pattern = r'---\s*\n(.*?)\n---\s*\n?(.*?)(?=\n---|\Z)'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            yaml_text = match.group(1)
            entry_content = match.group(2).strip()
            
            # 跳过说明部分（没有 id 的块）
            if 'id:' not in yaml_text:
                continue
            
            full_text = f"---\n{yaml_text}\n---\n\n{entry_content}"
            entry = self._markdown_to_entry(full_text)
            if entry:
                entries.append(entry)
        
        return entries
    
    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效（文件未修改）"""
        if not self.cache_file.exists():
            return False
        
        if not self.md_file.exists():
            return True  # 只有缓存，没有 Markdown
        
        # 比较修改时间
        md_mtime = self.md_file.stat().st_mtime
        cache_mtime = self.cache_file.stat().st_mtime
        
        return cache_mtime >= md_mtime
    
    def load_entries(self) -> List[MemoryEntry]:
        """
        加载所有记忆条目
        
        优先使用缓存以提升性能，缓存失效时自动重建
        """
        # 如果缓存有效，使用缓存
        if self._is_cache_valid():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    return [MemoryEntry.from_dict(e) for e in cache_data.get('entries', [])]
            except Exception:
                pass  # 缓存损坏，重新解析
        
        # 从 Markdown 解析
        entries = self._parse_markdown_file()
        
        # 重建缓存
        self._rebuild_cache()
        
        return entries
    
    def save_entries(self, entries: List[MemoryEntry]):
        """
        保存所有记忆条目
        
        同时更新 Markdown 文件和 JSON 缓存
        """
        # 按类型分组
        grouped = {
            MemoryType.SEMANTIC: [],
            MemoryType.PROCEDURAL: [],
            MemoryType.EPISODIC: [],
        }
        
        for entry in entries:
            grouped[entry.type].append(entry)
        
        # 生成 Markdown 内容
        sections = [
            "# System Memory - 系统记忆",
            "",
            "> 此文件由 OllamaPilot 自动生成和维护",
            "> 您可以手动编辑此文件来添加、修改或删除记忆",
            "> 格式：YAML frontmatter + Markdown 内容",
            "",
            "---",
            "",
        ]
        
        # 语义记忆
        sections.extend([
            "## 语义记忆 (Semantic)",
            "",
            "用户偏好、重要事实",
            "",
        ])
        for entry in grouped[MemoryType.SEMANTIC]:
            sections.append(self._entry_to_markdown(entry))
        
        # 程序记忆
        sections.extend([
            "## 程序记忆 (Procedural)",
            "",
            "Skill 使用模式、操作习惯",
            "",
        ])
        for entry in grouped[MemoryType.PROCEDURAL]:
            sections.append(self._entry_to_markdown(entry))
        
        # 情景记忆
        sections.extend([
            "## 情景记忆 (Episodic)",
            "",
            "重要对话摘要、历史事件",
            "",
        ])
        for entry in grouped[MemoryType.EPISODIC]:
            sections.append(self._entry_to_markdown(entry))
        
        # 写入文件
        content = "\n".join(sections)
        self.md_file.write_text(content, encoding='utf-8')
        
        # 重建缓存
        self._rebuild_cache()
    
    def add_entry(self, entry: MemoryEntry):
        """添加单个条目"""
        entries = self.load_entries()
        
        # 检查是否已存在
        existing_idx = None
        for i, e in enumerate(entries):
            if e.id == entry.id:
                existing_idx = i
                break
        
        if existing_idx is not None:
            # 更新现有条目
            entries[existing_idx] = entry
        else:
            # 添加新条目
            entries.append(entry)
        
        self.save_entries(entries)
    
    def delete_entry(self, entry_id: str) -> bool:
        """删除条目"""
        entries = self.load_entries()
        original_len = len(entries)
        entries = [e for e in entries if e.id != entry_id]
        
        if len(entries) < original_len:
            self.save_entries(entries)
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        entries = self.load_entries()
        
        return {
            "total_entries": len(entries),
            "by_type": {
                "semantic": len([e for e in entries if e.type == MemoryType.SEMANTIC]),
                "procedural": len([e for e in entries if e.type == MemoryType.PROCEDURAL]),
                "episodic": len([e for e in entries if e.type == MemoryType.EPISODIC]),
            },
            "md_file_size": self.md_file.stat().st_size if self.md_file.exists() else 0,
            "cache_file_size": self.cache_file.stat().st_size if self.cache_file.exists() else 0,
            "cache_valid": self._is_cache_valid(),
        }


# 向后兼容：MarkdownSystemMemory 继承自 SystemMemory
class MarkdownSystemMemoryMixin:
    """
    为 SystemMemory 添加 Markdown 存储支持的 Mixin
    
    使用方式：
        class SystemMemoryWithMarkdown(SystemMemory, MarkdownSystemMemoryMixin):
            pass
    """
    
    def _init_storage(self):
        """初始化 Markdown 存储"""
        self.md_storage = MarkdownMemoryStorage(str(self.storage_dir))
    
    def _load_entries(self, mem_type: MemoryType) -> List[MemoryEntry]:
        """从 Markdown 加载条目"""
        if hasattr(self, 'md_storage'):
            all_entries = self.md_storage.load_entries()
            return [e for e in all_entries if e.type == mem_type]
        return super()._load_entries(mem_type) if hasattr(super(), '_load_entries') else []
    
    def _save_entry(self, entry: MemoryEntry):
        """保存到 Markdown"""
        if hasattr(self, 'md_storage'):
            self.md_storage.add_entry(entry)
        elif hasattr(super(), '_save_entry'):
            super()._save_entry(entry)
