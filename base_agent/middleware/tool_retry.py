"""
Tool Retry 中间件 - 增强版本

工具调用失败时自动重试，并支持自动修复常见错误
"""

import json
import time
from typing import Any, Optional, Callable
from .base import AgentMiddleware, AgentState


class ToolRetryMiddleware(AgentMiddleware):
    """
    工具重试中间件（增强版）
    
    针对本地小模型的特点，提供：
    1. 工具调用失败自动重试
    2. 常见格式错误自动修复
    3. 参数类型自动转换
    4. 详细的错误提示
    
    示例:
        middleware = ToolRetryMiddleware(max_retries=3, delay=1.0)
        
        # 工具失败时会自动重试3次，每次间隔1秒
        # 如果参数格式错误，会自动尝试修复
    """
    
    def __init__(
        self, 
        max_retries: int = 3,
        delay: float = 0.5,
        retry_exceptions: Optional[tuple] = None,
        auto_fix: bool = True
    ):
        """
        初始化中间件
        
        Args:
            max_retries: 最大重试次数
            delay: 重试间隔（秒）
            retry_exceptions: 需要重试的异常类型，默认所有 Exception
            auto_fix: 是否启用自动修复功能
        """
        self.max_retries = max_retries
        self.delay = delay
        self.retry_exceptions = retry_exceptions or (Exception,)
        self.auto_fix = auto_fix
        self._fix_stats = {"total": 0, "fixed": 0}  # 修复统计
    
    def _validate_tool_call(self, tool_args: dict) -> tuple[bool, Optional[str]]:
        """
        验证工具调用参数格式
        
        Args:
            tool_args: 工具参数
            
        Returns:
            (是否有效, 错误信息)
        """
        if not isinstance(tool_args, dict):
            return False, f"参数必须是字典，但得到 {type(tool_args).__name__}"
        
        # 检查是否有空值或非法类型
        for key, value in tool_args.items():
            if value is None:
                return False, f"参数 '{key}' 不能为 None"
        
        return True, None
    
    def _fix_tool_args(self, tool_args: Any, error_msg: str) -> Optional[dict]:
        """
        尝试修复工具参数错误
        
        针对小模型常见的参数格式错误进行修复：
        1. 字符串转字典（如：'{"key": "value"}' -> dict）
        2. 基本类型包装（如："text" -> {"input": "text"}）
        3. 列表转字符串（如：["a", "b"] -> "a, b"）
        
        Args:
            tool_args: 原始参数
            error_msg: 错误信息
            
        Returns:
            修复后的参数，无法修复则返回 None
        """
        self._fix_stats["total"] += 1
        
        # 修复1: 字符串转字典
        if isinstance(tool_args, str):
            try:
                # 尝试解析 JSON
                fixed = json.loads(tool_args)
                if isinstance(fixed, dict):
                    self._fix_stats["fixed"] += 1
                    return fixed
            except json.JSONDecodeError:
                # 不是 JSON，包装为 input 参数
                self._fix_stats["fixed"] += 1
                return {"input": tool_args}
        
        # 修复2: 列表转字符串（某些工具需要字符串参数）
        if isinstance(tool_args, list):
            self._fix_stats["fixed"] += 1
            return {"input": ", ".join(str(x) for x in tool_args)}
        
        # 修复3: 数字/布尔值包装
        if isinstance(tool_args, (int, float, bool)):
            self._fix_stats["fixed"] += 1
            return {"value": tool_args}
        
        # 修复4: 如果参数为空，尝试从错误信息中提取
        if tool_args is None or tool_args == {}:
            # 尝试从错误信息中提取参数
            if "path" in error_msg.lower():
                return {"path": "."}
            elif "url" in error_msg.lower() or "http" in error_msg.lower():
                return {"url": "http://example.com"}
        
        return None
    
    def _convert_param_types(self, tool_args: dict, tool_func: Callable) -> dict:
        """
        根据工具函数的参数类型，自动转换参数类型
        
        Args:
            tool_args: 原始参数
            tool_func: 工具函数
            
        Returns:
            类型转换后的参数
        """
        import inspect
        
        try:
            sig = inspect.signature(tool_func)
            converted = tool_args.copy()
            
            for param_name, param in sig.parameters.items():
                if param_name not in converted:
                    continue
                
                value = converted[param_name]
                param_type = param.annotation
                
                # 尝试类型转换
                if param_type == str and not isinstance(value, str):
                    converted[param_name] = str(value)
                elif param_type == int and isinstance(value, str):
                    try:
                        converted[param_name] = int(value)
                    except ValueError:
                        pass
                elif param_type == float and isinstance(value, str):
                    try:
                        converted[param_name] = float(value)
                    except ValueError:
                        pass
                elif param_type == bool and isinstance(value, str):
                    converted[param_name] = value.lower() in ('true', '1', 'yes', 'on')
            
            return converted
        except Exception:
            return tool_args
    
    def wrap_tool_call(
        self,
        tool_name: str,
        tool_args: dict,
        state: AgentState,
        config: Optional[dict] = None
    ) -> Optional[Callable]:
        """
        包装工具调用，实现重试和自动修复逻辑
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            state: 当前状态
            config: 运行配置
            
        Returns:
            包装后的工具调用函数
        """
        def wrapper(tool_func: Callable) -> Any:
            """包装函数"""
            last_exception = None
            current_args = tool_args
            
            for attempt in range(self.max_retries + 1):
                try:
                    # 验证参数格式
                    is_valid, error_msg = self._validate_tool_call(current_args)
                    
                    if not is_valid and self.auto_fix and attempt > 0:
                        # 尝试修复参数
                        fixed_args = self._fix_tool_args(current_args, error_msg)
                        if fixed_args is not None:
                            current_args = fixed_args
                            is_valid, error_msg = self._validate_tool_call(current_args)
                    
                    if not is_valid:
                        raise ValueError(f"参数验证失败: {error_msg}")
                    
                    # 尝试类型转换
                    current_args = self._convert_param_types(current_args, tool_func)
                    
                    # 执行工具
                    result = tool_func(**current_args)
                    return result
                    
                except self.retry_exceptions as e:
                    last_exception = e
                    error_str = str(e).lower()
                    
                    if attempt < self.max_retries:
                        # 检查是否可以自动修复
                        if self.auto_fix and attempt >= 1:
                            fixed_args = self._fix_tool_args(current_args, str(e))
                            if fixed_args is not None:
                                current_args = fixed_args
                                if config and config.get("verbose"):
                                    print(f"  🔧 自动修复参数，重试 {attempt + 1}/{self.max_retries}")
                        
                        # 重试前等待
                        if self.delay > 0:
                            time.sleep(self.delay)
                    else:
                        # 所有重试都失败，返回友好的错误信息
                        error_detail = self._format_error(tool_name, current_args, str(e))
                        raise Exception(error_detail)
            
            return None
        
        return wrapper
    
    def _format_error(self, tool_name: str, tool_args: dict, error: str) -> str:
        """
        格式化错误信息，提供友好的提示
        
        Args:
            tool_name: 工具名称
            tool_args: 工具参数
            error: 原始错误
            
        Returns:
            格式化的错误信息
        """
        lines = [
            f"❌ 工具 '{tool_name}' 调用失败",
            f"参数: {json.dumps(tool_args, ensure_ascii=False, indent=2)}",
            f"错误: {error}",
        ]
        
        if self._fix_stats["total"] > 0:
            lines.append(f"修复尝试: {self._fix_stats['fixed']}/{self._fix_stats['total']} 成功")
        
        lines.append("\n建议:")
        lines.append("1. 检查参数格式是否正确")
        lines.append("2. 确保所有必需参数都已提供")
        lines.append("3. 检查参数类型是否匹配")
        
        return "\n".join(lines)
    
    def after_tool_call(
        self,
        tool_name: str,
        tool_result: Any,
        state: AgentState,
        config: Optional[dict] = None
    ) -> Optional[dict[str, Any]]:
        """
        工具调用后记录结果
        
        Args:
            tool_name: 工具名称
            tool_result: 工具执行结果
            state: 当前状态
            config: 运行配置
            
        Returns:
            状态更新或 None
        """
        # 记录工具调用统计
        if "tool_stats" not in state:
            state["tool_stats"] = {"total": 0, "success": 0, "failed": 0}
        
        state["tool_stats"]["total"] += 1
        
        if isinstance(tool_result, Exception):
            state["tool_stats"]["failed"] += 1
        else:
            state["tool_stats"]["success"] += 1
        
        return None
    
    def get_stats(self) -> dict:
        """获取修复统计信息"""
        return self._fix_stats.copy()
