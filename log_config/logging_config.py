import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import os

# Create logs directory if it doesn't exist
LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

class ConversationLogger:
    """Logs conversations and token usage to JSON files."""
    
    def __init__(self, log_dir: Path = LOGS_DIR, dev: bool = False):
        self.log_dir = log_dir
        self.dev = dev
        self.log_dir.mkdir(exist_ok=True)
        
        # Create separate log files with dev prefix if in dev mode
        prefix = "dev_" if dev else ""
        self.conversation_log_path = self.log_dir / f"{prefix}conversations.jsonl"
        self.token_usage_log_path = self.log_dir / f"{prefix}token_usage.jsonl"
        self.error_log_path = self.log_dir / f"{prefix}errors.jsonl"
    
    def log_conversation(
        self,
        user_input: str,
        assistant_response: str,
        wake_word: str,
        tone: str,
        timestamp: Optional[datetime] = None,
        conversation_id: Optional[str] = None,
    ) -> None:
        """Log a conversation exchange to JSONL file."""
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "conversation_id": conversation_id,
            "wake_word": wake_word,
            "tone": tone,
            "user_input": user_input,
            "assistant_response": assistant_response,
        }
        
        self._write_jsonl(self.conversation_log_path, log_entry)
    
    def log_token_usage(
        self,
        agent_name: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        model: str,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log token usage for LLM calls."""
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "agent_name": agent_name,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }
        
        self._write_jsonl(self.token_usage_log_path, log_entry)
    
    def log_error(
        self,
        error_type: str,
        error_message: str,
        context: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Log errors to JSONL file."""
        if timestamp is None:
            timestamp = datetime.now()
        
        log_entry = {
            "timestamp": timestamp.isoformat(),
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {},
        }
        
        self._write_jsonl(self.error_log_path, log_entry)
    
    @staticmethod
    def _write_jsonl(file_path: Path, entry: Dict[str, Any]) -> None:
        """Append a JSON object to a JSONL file."""
        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Error writing to log file {file_path}: {e}")

# Global logger instance
_conversation_logger: Optional[ConversationLogger] = None

def get_conversation_logger(dev: bool = False) -> ConversationLogger:
    """Get or create the global conversation logger."""
    global _conversation_logger
    if _conversation_logger is None:
        _conversation_logger = ConversationLogger(dev=dev)
    return _conversation_logger

# Python logging configuration
def setup_python_logging(log_level=logging.INFO):
    """Setup standard Python logging to both console and file."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("home_assistant")
    logger.setLevel(log_level)
    
    # File handler
    fh = logging.FileHandler(logs_dir / "application.log")
    fh.setLevel(log_level)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# Get the Python logger
logger = setup_python_logging()
