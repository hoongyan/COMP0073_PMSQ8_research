from pathlib import Path
from filelock import FileLock
from typing import List, Optional
from config.settings import get_settings
from config.logging_config import setup_logger


class RecordCounter:
    """Manages cycling through records for victim profiles specific to victim chatbot."""
    
    def __init__(self, record_index_file: str = "record_index.txt"):
        """Initialize with record index file path."""
        self.settings = get_settings()
        self.record_index_file = record_index_file
        self.lock_file = f"{record_index_file}.lock"
        self.logger = setup_logger("RecordCounter", self.settings.log.subdirectories["agent"])
    
    def reset(self):
        """Reset the record index to 0."""
        try:
            with FileLock(self.lock_file):
                with open(self.record_index_file, "w", encoding="utf-8") as f:
                    f.write("-1")
                self.logger.info(f"Reset record index to -1 in {self.record_index_file}")
        except Exception as e:
            self.logger.error(f"Error resetting record index: {str(e)}", exc_info=True)
            raise
    
    def get_next_index(self, total_records: int) -> int:
        """Get the next record index, cycling back to 0 if at the end."""
        try:
            with FileLock(self.lock_file):
                current_index = -1
                if Path(self.record_index_file).exists():
                    with open(self.record_index_file, "r", encoding="utf-8") as f:
                        try:
                            current_index = int(f.read().strip())
                        except ValueError:
                            self.logger.warning(f"Invalid index in {self.record_index_file}, resetting to -1")
                next_index = (current_index + 1) % total_records if total_records > 0 else 0
                with open(self.record_index_file, "w", encoding="utf-8") as f:
                    f.write(str(next_index))
                self.logger.debug(f"Current index: {current_index}, Next index: {next_index}, Total records: {total_records}")
                return next_index
        except Exception as e:
            self.logger.error(f"Error getting next record index: {str(e)}", exc_info=True)
            return 0
            
            
#Additional utility functions for agents
def build_query_with_history(current_query: str, history: List[str], max_history: Optional[int] = None, max_tokens: int = 2000) -> str:
    """Build RAG query with history, limited by items (N) and tokens. This is to ensure to provide flexibility in case of context overload.
    
    Args:
        current_query: Current user input.
        history: List of past queries.
        max_history: Max past items (None = all).
        max_tokens: Token limit estimate (truncate if exceeded).
    
    Returns:
        Concatenated query string.
    """
    if max_history is not None:
        history = history[-max_history:]  # Last N
    
    full_text = " ".join(history) + " " + current_query
    
    # Token fallback 
    word_count = len(full_text.split())
    est_tokens = int(word_count * 1.3)
    if est_tokens > max_tokens:
        # Truncate words to fit (reverse to keep recent inputs)
        words = full_text.split()
        trunc_words = words[-(int(max_tokens / 1.3)):]
        full_text = " ".join(trunc_words)
    
    return full_text

