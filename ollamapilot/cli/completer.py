"""
命令自动补全模块
"""

# 尝试导入 readline 用于命令自动补全
try:
    import readline
    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False


class CommandCompleter:
    """命令自动补全器"""
    
    # 所有可用命令
    COMMANDS = [
        '/help', '/model', '/embedding', '/new', '/sessions',
        '/switch', '/clear', '/info', '/messages', '/docs', '/index',
        '/reload', '/resume', 'quit', 'exit', 'q', 'bye'
    ]
    
    def __init__(self):
        self.current_candidates = []
    
    def complete(self, text, state):
        """补全函数"""
        if state == 0:
            # 第一次调用，生成候选列表
            if text.startswith('/'):
                self.current_candidates = [cmd for cmd in self.COMMANDS if cmd.startswith(text)]
            else:
                self.current_candidates = []
        
        # 返回当前状态的候选
        if state < len(self.current_candidates):
            return self.current_candidates[state]
        return None
    
    def setup(self):
        """设置自动补全"""
        if HAS_READLINE:
            readline.set_completer(self.complete)
            readline.parse_and_bind('tab: complete')
            # 设置补全分隔符，让 / 也被视为单词的一部分
            readline.set_completer_delims(' \t\n;')
            return True
        return False
