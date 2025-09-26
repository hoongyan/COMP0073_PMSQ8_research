from pathlib import Path
import logging
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

class LogSettings(BaseModel):
    """Logging configuration settings."""
    directory: str = Field(default="logs", description="Base directory for log files")
    subdirectories: dict = Field(
        default={
            "preprocessing": "preprocessing",
            "agent": "agent",
            "rag": "rag",
            "database": "database",
            "setup": "setup",
            "tests": "tests",
            "evaluation": "evaluation"
        },
        description="Subdirectories for component-specific logs"
    )

def setup_logger(name: str, subdirectory: str = "agent") -> logging.Logger:
    """Configure a logger for a specific component."""
    settings = LogSettings(directory=os.getenv("LOG_DIRECTORY", "logs"))
    log_dir = Path(settings.directory) / subdirectory
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{name.lower()}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='a')
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    logger.info(f"Logging initialized to {log_file}")
    return logger