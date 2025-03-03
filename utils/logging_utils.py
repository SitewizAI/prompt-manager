"""Logging utilities for Prompt Manager"""

from datetime import datetime
from typing import Optional, ClassVar, Dict, List
from dataclasses import dataclass, field
import time
from functools import wraps



def measure_time(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time
        print(f"⏱️ {func.__name__} took {duration:.2f} seconds")
        return result
    return wrapper

@dataclass
class ToolMessageTracker:
    """Track messages and responses from tool calls."""
    messages: List[Dict[str, str]] = field(default_factory=list)
    _instance: ClassVar[Optional['ToolMessageTracker']] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.messages = []
        return cls._instance

    def add_message(self, tool_name: str, input_msg: str, response: str, error: Optional[str] = None):
        """Add a tool message with its response."""
        self.messages.append({
            "tool": tool_name,
            "input": input_msg,
            "response": response,
            "error": error,
            "timestamp": datetime.now().isoformat()
        })

    def get_context(self) -> str:
        """Get tool interaction context as string."""
        if not self.messages:
            return ""
        
        context = "\nTool Interaction History:\n"
        for msg in self.messages:
            context += f"\nTool: {msg['tool']}\n"
            context += f"Input: {msg['input']}\n"
            context += f"Response: {msg['response']}\n"
            if msg['error']:
                context += f"Error: {msg['error']}\n"
            context += f"Time: {msg['timestamp']}\n"
            context += "-" * 40 + "\n"
        return context

    def clear(self):
        """Clear message history."""
        self.messages = []

def log_debug(message: str):
    """Debug logging with tracking."""
    print(f"DEBUG: {message}")
    ToolMessageTracker().add_message(
        tool_name="debug",
        input_msg="",
        response=message
    )

def log_error(message: str, error: Exception = None):
    """Error logging with tracking."""
    error_msg = f"ERROR: {message}"
    if error:
        error_msg += f" - {str(error)}"
    print(error_msg)
    ToolMessageTracker().add_message(
        tool_name="error",
        input_msg=message,
        response="",
        error=str(error) if error else None
    )
