import os
import csv
from pathlib import Path
from filelock import FileLock
from config.settings import get_settings
from config.logging_config import setup_logger

class IDManager:
    """Manages incremental integer conversation IDs."""
    
    def __init__(self, csv_file: str, id_file: str = "last_conversation_id.txt"):
        """Initialize with CSV and ID file paths."""
        self.csv_file = csv_file
        self.id_file = id_file
        self.lock_file = f"{id_file}.lock"
        Path(self.id_file).parent.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("IDManager", get_settings().log.subdirectories["agent"])
        self.last_id = self._load_last_id()
    
    def _load_last_id(self) -> int:
        """Load the last used conversation ID from CSV or ID file."""
        max_id = 0
        if os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    ids = [int(row["conversation_id"]) for row in reader if row["conversation_id"].isdigit()]
                    max_id = max(ids) if ids else 0

            except Exception as e:
                self.logger.error(f"Error reading conversation_id from CSV: {str(e)}")
        
        if os.path.exists(self.id_file):
            try:
                with open(self.id_file, "r") as f:
                    file_id = int(f.read().strip())
                    max_id = max(max_id, file_id)
          
            except Exception as e:
                self.logger.error(f"Error reading last_id from file: {str(e)}")
        
        return max_id
    
    def get_next_id(self) -> int:
        """Generate the next conversation ID with file locking."""
        with FileLock(self.lock_file):
            self.last_id += 1
            try:
                with open(self.id_file, "w") as f:
                    f.write(str(self.last_id))
        
                return self.last_id
            except Exception as e:
                self.logger.error(f"Error writing last_id to file: {str(e)}")
                raise