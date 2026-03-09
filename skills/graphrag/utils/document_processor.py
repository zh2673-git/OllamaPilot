"""
文档处理器

提供文本提取、智能分块等功能。
"""

from typing import List, Optional
from pathlib import Path
import re


class DocumentProcessor:
    """
    文档处理器

    提供文档读取、文本提取、智能分块等功能。
    """

    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 200):
        """
        初始化文档处理器

        Args:
            chunk_size: 分块大小（字符数），默认 2000
            chunk_overlap: 块间重叠大小，默认 200
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def read_document(self, file_path: str) -> Optional[str]:
        """
        读取文档内容

        Args:
            file_path: 文件路径

        Returns:
            文档内容，失败返回 None
        """
        path = Path(file_path)

        if not path.exists():
            return None

        try:
            # 根据文件类型选择读取方式
            suffix = path.suffix.lower()

            if suffix == '.pdf':
                return self._read_pdf(path)
            elif suffix in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json']:
                return self._read_text(path)
            elif suffix in ['.docx', '.doc']:
                return self._read_docx(path)
            else:
                # 尝试作为文本读取
                return self._read_text(path)
        except Exception as e:
            print(f"⚠️ 读取文档失败: {e}")
            return None

    def _read_text(self, path: Path) -> str:
        """读取文本文件"""
        encodings = ['utf-8', 'gbk', 'gb2312', 'utf-16']

        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding, errors='ignore') as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # 如果都失败，使用二进制读取
        with open(path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')

    def _read_pdf(self, path: Path) -> Optional[str]:
        """读取 PDF 文件"""
        try:
            # 尝试使用 PyPDF2
            import PyPDF2

            text = []
            with open(path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text.append(page.extract_text())

            return '\n'.join(text)
        except ImportError:
            print("⚠️ 未安装 PyPDF2，无法读取 PDF。请运行: pip install PyPDF2")
            return None
        except Exception as e:
            print(f"⚠️ 读取 PDF 失败: {e}")
            return None

    def _read_docx(self, path: Path) -> Optional[str]:
        """读取 Word 文档"""
        try:
            import docx

            doc = docx.Document(path)
            text = []

            for para in doc.paragraphs:
                text.append(para.text)

            return '\n'.join(text)
        except ImportError:
            print("⚠️ 未安装 python-docx，无法读取 Word。请运行: pip install python-docx")
            return None
        except Exception as e:
            print(f"⚠️ 读取 Word 失败: {e}")
            return None

    def chunk_text(self, text: str) -> List[str]:
        """
        智能分块

        优先在段落边界分块，保持语义完整性。

        Args:
            text: 原始文本

        Returns:
            文本块列表
        """
        if not text:
            return []

        # 清理文本
        text = self._clean_text(text)

        # 如果文本长度小于块大小，直接返回
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []

        # 按段落分割
        paragraphs = re.split(r'\n\s*\n', text)

        current_chunk = []
        current_length = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_length = len(para)

            # 如果当前段落太长，需要进一步分割
            if para_length > self.chunk_size:
                # 先保存当前块
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # 分割长段落
                sub_chunks = self._split_long_paragraph(para)
                chunks.extend(sub_chunks)
                continue

            # 检查添加当前段落后是否超过块大小
            if current_length + para_length + 2 > self.chunk_size:
                # 保存当前块
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))

                # 开始新块，保留重叠部分
                if current_chunk and self.chunk_overlap > 0:
                    # 计算需要保留的重叠文本
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = [overlap_text, para] if overlap_text else [para]
                    current_length = len('\n\n'.join(current_chunk))
                else:
                    current_chunk = [para]
                    current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length + 2

        # 保存最后一个块
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))

        return chunks

    def _split_long_paragraph(self, paragraph: str) -> List[str]:
        """分割长段落"""
        chunks = []

        # 尝试按句子分割
        sentences = re.split(r'(?<=[。！？.!?])\s+', paragraph)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size:
                if current_chunk:
                    chunks.append(''.join(current_chunk))

                # 如果单个句子超过块大小，强制分割
                if sentence_length > self.chunk_size:
                    for i in range(0, sentence_length, self.chunk_size - self.chunk_overlap):
                        chunk = sentence[i:i + self.chunk_size]
                        if chunk:
                            chunks.append(chunk)
                    current_chunk = []
                    current_length = 0
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        if current_chunk:
            chunks.append(''.join(current_chunk))

        return chunks

    def _get_overlap_text(self, chunks: List[str]) -> str:
        """获取重叠文本"""
        if not chunks:
            return ""

        # 从最后几个段落中提取重叠文本
        overlap_chunks = []
        overlap_length = 0

        for chunk in reversed(chunks):
            if overlap_length + len(chunk) <= self.chunk_overlap:
                overlap_chunks.insert(0, chunk)
                overlap_length += len(chunk)
            else:
                break

        return '\n\n'.join(overlap_chunks)

    def _clean_text(self, text: str) -> str:
        """清理文本"""
        # 移除多余的空白字符
        text = re.sub(r'[ \t]+', ' ', text)
        # 统一换行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        # 移除多余的空行
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    def process_document(self, file_path: str) -> List[str]:
        """
        处理文档：读取并分块

        Args:
            file_path: 文件路径

        Returns:
            文本块列表
        """
        text = self.read_document(file_path)
        if text is None:
            return []

        return self.chunk_text(text)
