"""
文件处理工具模块

支持：
1. 文档内容提取（PDF, DOCX, TXT, MD等）
2. 图片内容识别（OCR + 视觉模型）
3. 文件下载和临时存储
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
import aiohttp
import asyncio


class FileProcessor:
    """文件处理器"""

    # 支持的文档类型
    SUPPORTED_DOCS = {
        'application/pdf': '.pdf',
        'application/msword': '.doc',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
        'text/plain': '.txt',
        'text/markdown': '.md',
        'text/x-markdown': '.md',
        'application/json': '.json',
        'text/csv': '.csv',
        'application/vnd.ms-excel': '.xls',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
    }

    # 支持的图片类型
    SUPPORTED_IMAGES = {
        'image/jpeg': '.jpg',
        'image/png': '.png',
        'image/gif': '.gif',
        'image/webp': '.webp',
        'image/bmp': '.bmp',
    }

    def __init__(self, temp_dir: str = "./data/temp_documents"):
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    async def download_file(self, url: str, filename: Optional[str] = None) -> Path:
        """异步下载文件"""
        if not filename:
            filename = url.split('/')[-1] or "downloaded_file"

        file_path = self.temp_dir / filename

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    with open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            f.write(chunk)
                    return file_path
                else:
                    raise Exception(f"下载失败: HTTP {response.status}")

    def extract_text_content(self, file_path: Path) -> str:
        """提取文档文本内容"""
        suffix = file_path.suffix.lower()

        try:
            if suffix == '.pdf':
                return self._extract_pdf(file_path)
            elif suffix in ['.doc', '.docx']:
                return self._extract_word(file_path)
            elif suffix in ['.txt', '.md', '.json', '.csv']:
                return self._extract_text(file_path)
            elif suffix in ['.xls', '.xlsx']:
                return self._extract_excel(file_path)
            else:
                return f"[不支持的文件格式: {suffix}]"
        except Exception as e:
            return f"[文件解析失败: {str(e)}]"

    def _extract_pdf(self, file_path: Path) -> str:
        """提取PDF内容"""
        try:
            import PyPDF2
            text = []
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text() or "")
            return "\n".join(text)
        except ImportError:
            return "[需要安装 PyPDF2: pip install PyPDF2]"

    def _extract_word(self, file_path: Path) -> str:
        """提取Word内容"""
        try:
            import docx
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        except ImportError:
            return "[需要安装 python-docx: pip install python-docx]"

    def _extract_text(self, file_path: Path) -> str:
        """提取文本文件"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        return "[无法解码文件内容]"

    def _extract_excel(self, file_path: Path) -> str:
        """提取Excel内容"""
        try:
            import pandas as pd
            df = pd.read_excel(file_path)
            return df.to_string(index=False)
        except ImportError:
            return "[需要安装 pandas: pip install pandas openpyxl]"

    async def analyze_image(self, image_path: Path, query: str = "描述这张图片的内容", model_name: str = "qwen3.5:4b") -> str:
        """分析图片内容（使用多模态模型）"""
        try:
            # 读取图片为base64
            import base64
            with open(image_path, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')

            # 使用 Ollama API 直接调用视觉模型
            import aiohttp

            # 构建 Ollama 多模态请求
            payload = {
                "model": model_name,  # 使用配置的视觉模型
                "messages": [
                    {
                        "role": "user",
                        "content": query,
                        "images": [image_base64]
                    }
                ],
                "stream": False
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:11434/api/chat",
                    json=payload
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("message", {}).get("content", "[图片分析返回空]")
                    else:
                        error_text = await response.text()
                        print(f"⚠️ 视觉模型调用失败: {error_text}")
                        # 失败时回退到OCR
                        return await self._ocr_image(image_path)

        except Exception as e:
            print(f"⚠️ 图片分析失败: {e}")
            # 失败时回退到OCR
            return await self._ocr_image(image_path)

    async def _ocr_image(self, image_path: Path) -> str:
        """OCR识别图片文字"""
        try:
            import pytesseract
            from PIL import Image

            image = Image.open(image_path)
            text = pytesseract.image_to_string(image, lang='chi_sim+eng')
            return f"[OCR识别结果]\n{text}" if text.strip() else "[图片中未识别到文字]"

        except ImportError:
            return "[需要安装OCR依赖: pip install pytesseract pillow]"
        except Exception as e:
            return f"[OCR识别失败: {str(e)}]"

    def cleanup(self, file_path: Path):
        """清理临时文件"""
        try:
            if file_path.exists():
                file_path.unlink()
        except Exception:
            pass

    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """获取文件信息"""
        stat = file_path.stat()
        return {
            'name': file_path.name,
            'size': stat.st_size,
            'suffix': file_path.suffix,
            'path': str(file_path)
        }


# 全局文件处理器实例
_file_processor: Optional[FileProcessor] = None


def get_file_processor() -> FileProcessor:
    """获取文件处理器实例"""
    global _file_processor
    if _file_processor is None:
        _file_processor = FileProcessor()
    return _file_processor
