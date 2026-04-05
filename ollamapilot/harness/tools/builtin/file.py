"""
文件操作工具 - 使用新 Tool 基类

借鉴 Claude Code 的文件工具设计
"""

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ollamapilot.harness.tools.base import (
    Tool, ToolContext, ToolResult, ValidationResult, PermissionResult
)


class FileReadTool(Tool):
    """文件读取工具"""
    
    name = "read_file"
    description = "读取文件内容，支持文本文件和 PDF"
    is_read_only = True
    is_destructive = False
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文件路径（相对或绝对）"
                },
                "offset": {
                    "type": "integer",
                    "description": "起始行号",
                    "default": 0
                },
                "limit": {
                    "type": "integer",
                    "description": "最大读取行数",
                    "default": 1000
                }
            },
            "required": ["file_path"]
        }
    
    async def validate(self, input_data: Dict[str, Any]) -> ValidationResult:
        base_result = await super().validate(input_data)
        if not base_result.valid:
            return base_result
        
        file_path = input_data.get('file_path', '')
        path = Path(file_path)
        
        if not path.exists():
            return ValidationResult.failure(f"文件不存在: {file_path}")
        
        if not path.is_file():
            return ValidationResult.failure(f"不是文件: {file_path}")
        
        # 检查是否是设备文件
        if path.is_block_device() or path.is_char_device():
            return ValidationResult.failure("不能读取设备文件")
        
        return ValidationResult.success()
    
    async def execute(
        self, 
        input_data: Dict[str, Any], 
        context: ToolContext,
        on_progress: Optional[Callable[[str, float], None]] = None
    ) -> ToolResult:
        file_path = input_data.get('file_path', '')
        offset = input_data.get('offset', 0)
        limit = input_data.get('limit', 1000)
        
        path = Path(file_path)
        
        try:
            # 检查文件类型
            suffix = path.suffix.lower()
            
            if suffix == '.pdf':
                return await self._read_pdf(path, limit)
            
            # 读取文本文件
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i < offset:
                        continue
                    if len(lines) >= limit:
                        lines.append(f"\n... (已截断，共读取 {limit} 行)")
                        break
                    lines.append(line.rstrip())
                
                content = '\n'.join(lines)
                return ToolResult.success_result(
                    f"📄 文件: {file_path}\n{'='*50}\n{content}",
                    {"lines_read": len(lines), "total_offset": offset}
                )
                
        except Exception as e:
            return ToolResult.error_result(str(e))
    
    async def _read_pdf(self, path: Path, limit: int) -> ToolResult:
        """读取 PDF 文件"""
        try:
            import PyPDF2
            
            text = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                total_pages = len(reader.pages)
                
                for i, page in enumerate(reader.pages):
                    if i >= limit:
                        text.append(f"\n... (已截断，共读取 {limit} 页，总共 {total_pages} 页)")
                        break
                    page_text = page.extract_text()
                    if page_text:
                        text.append(f"\n--- 第 {i+1} 页 ---\n{page_text}")
            
            content = '\n'.join(text)
            return ToolResult.success_result(
                f"📄 PDF 文件: {path}\n{'='*50}\n{content}",
                {"pages_read": min(limit, total_pages), "total_pages": total_pages}
            )
            
        except ImportError:
            return ToolResult.error_result("未安装 PyPDF2，无法读取 PDF。请运行: pip install PyPDF2")
        except Exception as e:
            return ToolResult.error_result(f"读取 PDF 失败: {e}")


class FileWriteTool(Tool):
    """文件写入工具"""
    
    name = "write_file"
    description = "写入文件内容"
    is_read_only = False
    is_destructive = True
    
    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "文件路径"
                },
                "content": {
                    "type": "string",
                    "description": "文件内容"
                },
                "append": {
                    "type": "boolean",
                    "description": "是否追加模式",
                    "default": False
                }
            },
            "required": ["file_path", "content"]
        }
    
    async def check_permission(
        self, 
        input_data: Dict[str, Any], 
        context: ToolContext
    ) -> PermissionResult:
        """写入操作需要确认"""
        file_path = input_data.get('file_path', '')
        append = input_data.get('append', False)
        path = Path(file_path)
        
        # 如果文件已存在且不是追加模式，需要确认
        if path.exists() and not append:
            return PermissionResult.ask(
                f"文件 {file_path} 已存在，是否覆盖？"
            )
        
        return PermissionResult.allow()
    
    async def execute(
        self, 
        input_data: Dict[str, Any], 
        context: ToolContext,
        on_progress: Optional[Callable[[str, float], None]] = None
    ) -> ToolResult:
        file_path = input_data.get('file_path', '')
        content = input_data.get('content', '')
        append = input_data.get('append', False)
        
        try:
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            mode = 'a' if append else 'w'
            with open(path, mode, encoding='utf-8') as f:
                f.write(content)
            
            action = "追加" if append else "写入"
            return ToolResult.success_result(
                f"✅ {action}成功: {file_path}\n   大小: {len(content)} 字符",
                {"file_path": str(path.absolute()), "size": len(content)}
            )
            
        except Exception as e:
            return ToolResult.error_result(str(e))
