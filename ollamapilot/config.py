"""
配置管理模块

支持从 .env 文件读取配置，便于前端集成和动态配置。
"""

import os
from typing import Optional
from pathlib import Path


class Config:
    """配置类"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        初始化配置
        
        Args:
            env_file: .env 文件路径，默认使用项目根目录的 .env
        """
        if env_file is None:
            # 默认使用项目根目录的 .env
            project_root = Path(__file__).parent.parent
            env_file = project_root / ".env"
        
        self.env_file = Path(env_file)
        self._config = {}
        
        # 加载配置
        self._load_env_file()
        
    def _load_env_file(self):
        """从 .env 文件加载配置"""
        if not self.env_file.exists():
            print(f"⚠️ 配置文件不存在: {self.env_file}")
            print(f"   将使用默认配置")
            self._load_defaults()
            return

        try:
            with open(self.env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    # 跳过空行和注释
                    if not line or line.startswith('#'):
                        continue

                    # 解析键值对
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # 移除引号
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        self._config[key] = value
                        # 同时设置到环境变量，供其他模块使用
                        if key not in os.environ:
                            os.environ[key] = value

            print(f"✅ 已加载配置: {self.env_file}")

        except Exception as e:
            print(f"⚠️ 加载配置文件失败: {e}")
            self._load_defaults()
    
    def _load_defaults(self):
        """加载默认配置"""
        self._config = {
            'OLLAMA_BASE_URL': 'http://localhost:11434',
            'CHAT_MODEL': 'qwen3.5:4b',
            'CHAT_TEMPERATURE': '0.7',
            'CHAT_NUM_CTX': '8192',
            'CHAT_NUM_PREDICT': '2048',
            'EMBEDDING_MODEL': 'qwen3-embedding:0.6b',
            'GRAPH_RAG_ENABLE_WORD_ALIGNER': 'true',
            'GRAPH_RAG_FUZZY_THRESHOLD': '0.75',
            'GRAPH_RAG_KNOWLEDGE_BASE_DIR': './knowledge_base',
            'GRAPH_RAG_PERSIST_DIR': './data/graphrag',
            'VERBOSE': 'false',
            'DEFAULT_THREAD_ID': 'default',
            'CLI_TIMEOUT': '120',  # CLI 响应超时时间（秒）
        }
    
    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """获取配置值"""
        # 首先检查环境变量
        env_value = os.environ.get(key)
        if env_value is not None:
            return env_value
        
        # 然后检查配置文件
        return self._config.get(key, default)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔值配置"""
        value = self.get(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """获取整数配置"""
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """获取浮点数配置"""
        value = self.get(key)
        if value is None:
            return default
        try:
            return float(value)
        except ValueError:
            return default
    
    # 便捷属性
    @property
    def ollama_base_url(self) -> str:
        return self.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    
    @property
    def chat_model(self) -> str:
        return self.get('CHAT_MODEL', 'qwen3.5:4b')
    
    @property
    def chat_temperature(self) -> float:
        return self.get_float('CHAT_TEMPERATURE', 0.7)
    
    @property
    def chat_num_ctx(self) -> int:
        return self.get_int('CHAT_NUM_CTX', 8192)
    
    @property
    def chat_num_predict(self) -> int:
        return self.get_int('CHAT_NUM_PREDICT', 2048)
    
    @property
    def embedding_model(self) -> str:
        return self.get('EMBEDDING_MODEL', 'qwen3-embedding:0.6b')
    
    @property
    def graph_rag_enable_word_aligner(self) -> bool:
        return self.get_bool('GRAPH_RAG_ENABLE_WORD_ALIGNER', True)
    
    @property
    def graph_rag_fuzzy_threshold(self) -> float:
        return self.get_float('GRAPH_RAG_FUZZY_THRESHOLD', 0.75)
    
    @property
    def graph_rag_knowledge_base_dir(self) -> str:
        return self.get('GRAPH_RAG_KNOWLEDGE_BASE_DIR', './knowledge_base')
    
    @property
    def graph_rag_persist_dir(self) -> str:
        return self.get('GRAPH_RAG_PERSIST_DIR', './data/graphrag')
    
    @property
    def verbose(self) -> bool:
        return self.get_bool('VERBOSE', False)
    
    @property
    def default_thread_id(self) -> str:
        return self.get('DEFAULT_THREAD_ID', 'default')
    
    @property
    def cli_timeout(self) -> int:
        """CLI 响应超时时间（秒），默认 120 秒"""
        return self.get_int('CLI_TIMEOUT', 120)
    
    def reload(self):
        """重新加载配置"""
        self._config.clear()
        self._load_env_file()
        print("✅ 配置已重新加载")
    
    def to_dict(self) -> dict:
        """导出为字典"""
        return {
            'ollama_base_url': self.ollama_base_url,
            'chat_model': self.chat_model,
            'chat_temperature': self.chat_temperature,
            'chat_num_ctx': self.chat_num_ctx,
            'chat_num_predict': self.chat_num_predict,
            'embedding_model': self.embedding_model,
            'graph_rag_enable_word_aligner': self.graph_rag_enable_word_aligner,
            'graph_rag_fuzzy_threshold': self.graph_rag_fuzzy_threshold,
            'graph_rag_knowledge_base_dir': self.graph_rag_knowledge_base_dir,
            'graph_rag_persist_dir': self.graph_rag_persist_dir,
            'verbose': self.verbose,
            'default_thread_id': self.default_thread_id,
            'cli_timeout': self.cli_timeout,
        }
    
    def print_config(self):
        """打印当前配置"""
        print("\n📋 当前配置:")
        print("-" * 50)
        config_dict = self.to_dict()
        for key, value in config_dict.items():
            print(f"  {key}: {value}")
        print("-" * 50)


# 全局配置实例
_config_instance: Optional[Config] = None


def get_config(env_file: Optional[str] = None) -> Config:
    """
    获取全局配置实例
    
    Args:
        env_file: .env 文件路径
        
    Returns:
        Config 实例
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(env_file)
    return _config_instance


def reload_config():
    """重新加载全局配置"""
    global _config_instance
    if _config_instance is not None:
        _config_instance.reload()


# 便捷函数
def get_chat_model() -> str:
    """获取对话模型"""
    return get_config().chat_model


def get_embedding_model() -> str:
    """获取 Embedding 模型"""
    return get_config().embedding_model


def get_ollama_base_url() -> str:
    """获取 Ollama 服务地址"""
    return get_config().ollama_base_url
