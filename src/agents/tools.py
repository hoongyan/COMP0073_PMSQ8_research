import sys
import os
import json
import csv
import os
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
from filelock import FileLock

from langchain_core.tools import tool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.database.vector_operations import VectorStore
from src.database.database_operations import DatabaseManager, CRUDOperations
from src.models.data_model import Strategy
from config.settings import get_settings
from config.logging_config import setup_logger

class PoliceTools:
    """Tools for the police chatbot."""

    def __init__(self, rag_csv_path: str = "rag_invocations.csv"):
        """Initialize with VectorStore and settings."""
        self.settings = get_settings()
        self.logger = setup_logger("Tools", self.settings.log.subdirectories["agent"])
        self.csv_file = rag_csv_path  # Use the passed path
        # Ensure the directory for csv_file exists
        Path(self.csv_file).parent.mkdir(parents=True, exist_ok=True)
        self.index_counter = self._load_last_index()
        
        db_manager = DatabaseManager()
        self.vector_store = VectorStore(session_factory=db_manager.session_factory)
        self.strategy_crud = CRUDOperations(Strategy, db_manager.session_factory)
        
    def _load_last_index(self) -> int:
        """Load the last used index from rag_invocations.csv."""
        csv_file = "rag_invocations.csv"
        max_index = 0
        if os.path.exists(csv_file):
            try:
                with FileLock(f"{csv_file}.lock"):
                    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        if "index" in reader.fieldnames:
                            indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
                            max_index = max(indices) if indices else 0
                        else:
                            self.logger.warning(f"CSV {csv_file} does not contain 'index' field. Starting with index 0.")
                self.logger.debug(f"Loaded max index {max_index} from {csv_file}")
            except Exception as e:
                self.logger.error(f"Error reading index from CSV: {str(e)}")
        return max_index

    def _log_rag_invocation(self, conversation_id: int, query: str, top_k: int, scam_results: list, strategy_results: list, scam_distances: list, strategy_distances: list, llm_model: str = None):
        """Log RAG invocation details to rag_invocations.csv with separate columns. This will be used for simple evaluation on RAG retrieval accuracy."""
        file_exists = os.path.isfile(self.csv_file)
        effective_conversation_id = conversation_id if conversation_id is not None else -1
        try:
            with FileLock(f"{self.csv_file}.lock", timeout=10):
                with open(self.csv_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "index", "conversation_id", "timestamp", "query", "top_k",
                            "scam_results", "scam_distances",  "strategy_results", "llm_model"
                        ])
                    self.index_counter += 1
                    writer.writerow([
                        str(self.index_counter),
                        str(effective_conversation_id),
                        datetime.now().isoformat(),
                        query,
                        top_k,
                        json.dumps(scam_results, ensure_ascii=False),
                        json.dumps(strategy_results, ensure_ascii=False),
                        json.dumps(scam_distances, ensure_ascii=False),
                        json.dumps(strategy_distances, ensure_ascii=False),
                        llm_model or ""
                    ])
                    self.logger.info(f"Logged RAG invocation: index={self.index_counter}, conversation_id={effective_conversation_id}")
        except Exception as e:
            self.logger.error(f"Failed to log RAG invocation: {str(e)}")

    def get_tools(self):
        """Return tools for RAG retrieval agent to retrieve scam reports only."""
        @tool
        def retrieve_scam_reports(query: str, top_k: int = 5, conversation_id: int = None, llm_model: str = None, metadata_filter: Optional[Dict] = None) -> str:
            """
            Retrieve scam reports from the database only.
            
            Args:
                query (str): The query to search for similar scam reports.
                top_k (int, optional): Number of top results to return. Defaults to 5.
                conversation_id (int, optional): ID of the conversation for logging.
                llm_model (str, optional): LLM model used by the police agent.
            
            Returns:
                str: JSON string of scam reports.
            """
            try:
                self.logger.debug(f"Executing retrieve_scam_reports: query='{query}', top_k={top_k}, conversation_id={conversation_id}, llm_model={llm_model}, metadata_filter = {metadata_filter}")
                
                #Retrieve scam results
                scam_results, scam_distances = self.vector_store.retrieve_scam_reports(query, top_k, metadata_filter)
                self._log_rag_invocation(
                    conversation_id, query, top_k, scam_results, scam_distances, [], llm_model
                )
                return json.dumps(scam_results, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"Error in retrieve_scam_reports: {str(e)}")
                return json.dumps({"error": f"Error in retrieve_scam_reports: {str(e)}"})
        
        return [retrieve_scam_reports]

    def get_augmented_tools(self):
        """Return tools for self-augmenting RAG (scam reports and strategies)."""
        @tool
        def augmented_police_tool(query: str, user_profile: str, top_k: int = 5, conversation_id: int = None, llm_model: str = None) -> str:
            """
            Retrieve scam reports and questioning strategies from database.
            
            Args:
                query (str): The query to search for similar scam reports and strategies.
                top_k (int, optional): Number of top results to return for each table. Defaults to 5.
                conversation_id (int, optional): ID of the conversation for logging.
                llm_model (str, optional): LLM model used by the police agent.
            
            Returns:
                str: JSON string containing scam reports and strategy types.
            """
            try:
                self.logger.debug(f"Executing augmented_police_tool: query='{query}', top_k={top_k}, conversation_id={conversation_id}, llm_model={llm_model}")
                                
                # Process user_profile
                user_profile_dict = json.loads(user_profile) if isinstance(user_profile, str) else user_profile
                
                search_profile = {}
                for dim in ['tech_literacy', 'language_proficiency', 'emotional_state']:
                    if dim in user_profile_dict and isinstance(user_profile_dict[dim], dict) and 'level' in user_profile_dict[dim]:
                        search_profile[dim] = {'level': user_profile_dict[dim]['level']}  # Only level (process: skip score/confidence if present)
                    else:
                        self.logger.warning(f"Missing/invalid '{dim}' in user_profile - skipping for search")

                self.logger.debug(f"Processed search profile: {search_profile}")

                # Retrieve strategies
                strategy_results = self.strategy_crud.retrieve_strategies(search_profile, top_k)
                self.logger.debug(f"Retrieved {len(strategy_results)} strategies (structured search)")
                
                # Retrieve scam reports
                scam_results, scam_distances = self.vector_store.retrieve_scam_reports(query, top_k)
  
                # Combine results
                combined_results = {
                    "scam_reports": scam_results,
                    "strategies": strategy_results
                }
                
                self._log_rag_invocation(
                    conversation_id, query, top_k, scam_results, scam_distances, strategy_results, llm_model
                )
                return json.dumps(combined_results, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"Error in augmented_police_tool: {str(e)}")
                return json.dumps({"error": f"Error in augmented_police_tool: {str(e)}"})
        
        return [augmented_police_tool]

