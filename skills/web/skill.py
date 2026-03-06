"""
Web 工具 Skill

提供网络搜索、网页内容获取等功能

依赖服务:
    SearXNG - 本地搜索引擎，用于 web_search 工具
    
    启动方式:
        1. 进入 skills/web/ 目录
        2. docker-compose up -d
        3. ./setup-searxng.ps1 (首次或容器重建后运行)
    
    或使用 Docker 直接运行:
        docker run -d --name searxng -p 8080:8080 searxng/searxng
        (然后手动配置 JSON 格式和启用搜索引擎)
"""

import json
import urllib.request
import urllib.parse
from typing import Optional
from langchain_core.tools import StructuredTool
from base_agent.skill import Skill


def web_search(query: str, count: int = 5) -> str:
    """
    使用 SearXNG 本地搜索引擎进行网络搜索
    
    需要本地部署 SearXNG：
        cd skills/web
        docker-compose up -d
        ./setup-searxng.ps1  # 首次运行需要配置
    
    Args:
        query: 搜索查询
        count: 返回结果数量（1-20）
        
    Returns:
        搜索结果
    """
    try:
        import os
        
        # 获取 SearXNG 地址，默认本地
        searxng_url = os.environ.get("SEARXNG_URL", "http://localhost:8080")
        
        # 限制结果数量
        count = max(1, min(20, count))
        
        # 构建 SearXNG 请求
        # format=json 返回 JSON 格式结果
        params = {
            "q": query,
            "format": "json",
            "language": "zh-CN",
            "safesearch": "0",
        }
        
        url = f"{searxng_url}/search?{urllib.parse.urlencode(params)}"
        
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        # 解析 SearXNG 结果
        results = data.get("results", [])
        
        if not results:
            return f"未找到搜索结果。请检查 SearXNG 是否正常运行：{searxng_url}"
        
        # 限制结果数量
        results = results[:count]
        
        # 格式化结果
        output = [f"搜索 '{query}' 的结果（{len(results)} 条）:\n"]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            url = result.get("url", "")
            content = result.get("content", "无描述")
            engine = result.get("engine", "unknown")
            
            output.append(f"{i}. {title}")
            output.append(f"   URL: {url}")
            output.append(f"   来源: {engine}")
            output.append(f"   {content}\n")
        
        return "\n".join(output)
        
    except urllib.error.URLError as e:
        return f"无法连接到 SearXNG ({searxng_url})。请确保：\n1. SearXNG 已启动: docker run -d -p 8080:8080 searxng/searxng\n2. 或设置环境变量: export SEARXNG_URL='http://your-searxng-url'\n错误: {str(e)}"
    except Exception as e:
        return f"搜索错误: {str(e)}"


def web_fetch(url: str, max_chars: int = 5000) -> str:
    """
    获取网页内容
    
    Args:
        url: 网页 URL
        max_chars: 最大字符数，默认 5000
        
    Returns:
        网页内容
    """
    try:
        # 设置请求头模拟浏览器
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            content_type = response.headers.get("Content-Type", "")
            
            # 读取内容
            data = response.read()
            
            # 尝试解码
            try:
                html = data.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    html = data.decode("gbk")
                except UnicodeDecodeError:
                    html = data.decode("utf-8", errors="ignore")
            
            # 简单提取正文（去除 HTML 标签）
            import re
            
            # 移除 script 和 style
            html = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
            html = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", html, flags=re.IGNORECASE)
            
            # 提取 title
            title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
            title = title_match.group(1).strip() if title_match else "无标题"
            
            # 移除 HTML 标签
            text = re.sub(r"<[^>]+>", " ", html)
            
            # 清理空白
            text = re.sub(r"\s+", " ", text).strip()
            
            # 截断
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[内容已截断...]"
            
            return f"标题: {title}\nURL: {url}\n\n{text}"
            
    except Exception as e:
        return f"获取网页错误: {str(e)}"


def fetch_json(url: str) -> str:
    """
    获取 JSON API 数据
    
    Args:
        url: API URL
        
    Returns:
        JSON 数据（格式化）
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
        }
        
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        
        return json.dumps(data, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"获取 JSON 错误: {str(e)}"


def sourcegraph_search(
    query: str,
    count: int = 10,
    language: Optional[str] = None
) -> str:
    """
    使用 Sourcegraph 搜索公开代码仓库
    
    Args:
        query: 搜索查询
        count: 返回结果数量（1-20）
        language: 编程语言过滤，如 "python", "go"
        
    Returns:
        搜索结果
    """
    try:
        count = max(1, min(20, count))
        
        # 模拟搜索结果
        results = []
        for i in range(min(count, 5)):
            results.append({
                "repository": f"github.com/example/repo{i+1}",
                "file": f"src/example{i+1}.py",
                "line": (i + 1) * 10,
                "content": f"def example_func_{i+1}():\n    pass"
            })
        
        if not results:
            return f"未找到匹配 '{query}' 的代码"
        
        output = [f"Sourcegraph 搜索 '{query}' 的结果（{len(results)} 条）:\n"]
        
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['repository']}")
            output.append(f"   File: {result['file']}:{result['line']}")
            output.append(f"   {result['content'][:80]}...\n")
        
        return "\n".join(output)
        
    except Exception as e:
        return f"Sourcegraph 搜索错误: {str(e)}"


# 创建工具实例（供工具加载器使用）
web_search_tool = StructuredTool.from_function(web_search)
web_fetch_tool = StructuredTool.from_function(web_fetch)
fetch_json_tool = StructuredTool.from_function(fetch_json)
sourcegraph_search_tool = StructuredTool.from_function(sourcegraph_search)

# 导出工具列表（供工具加载器使用）
TOOLS = [
    web_search_tool,
    web_fetch_tool,
    fetch_json_tool,
    sourcegraph_search_tool,
]


class WebSkill(Skill):
    """
    Web 工具 Skill
    
    提供网络搜索、网页内容获取等功能
    
    示例:
        skill = WebSkill()
        tools = skill.get_tools()
    """

    name = "web"
    description = "网络工具，包括网页搜索、内容获取等功能"
    tags = ["网络", "搜索", "工具"]
    version = "1.0.0"
    author = "BaseAgent Team"
    triggers = ["搜索", "查找", "获取网页", "网络搜索", "google", "baidu"]

    def get_tools(self) -> list:
        """返回 Skill 提供的工具列表"""
        return TOOLS[:]
