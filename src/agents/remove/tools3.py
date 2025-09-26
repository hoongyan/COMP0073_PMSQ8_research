# tools.py
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from src.database.vector_operations import VectorStore
from src.database.database_operations import DatabaseManager
from src.models.data_model import ScamReport
from config.settings import get_settings
import json
import csv
import os
from datetime import datetime
from pathlib import Path
import logging

class RetrieveScamReportsArgs(BaseModel):
    query: str = Field(..., description="The query to search for similar scam reports")
    top_k: int = Field(default=5, description="Number of top results to return")
    conversation_id: str = Field(default="0", description="ID of the conversation")
    llm_model: str = Field(default="llama3.1:8b", description="LLM model used")

class PoliceTools:
    def __init__(self):
        self.settings = get_settings()
        self._setup_logging()
        self.index_counter = self._load_last_index()
        db_manager = DatabaseManager()
        self.vector_store = VectorStore(session_factory=db_manager.session_factory)

    def _setup_logging(self):
        log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "tools.log"
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='a'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized to {log_file}")

    def _load_last_index(self) -> int:
        csv_file = "rag_invocations.csv"
        max_index = 0
        if os.path.exists(csv_file):
            try:
                with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
                    max_index = max(indices) if indices else 0
                self.logger.debug(f"Loaded max index {max_index} from {csv_file}")
            except Exception as e:
                self.logger.error(f"Error reading index from CSV: {str(e)}")
        return max_index

    def _log_rag_invocation(self, conversation_id: str, query: str, top_k: int, results: list, distances: list, llm_model: str):
        csv_file = "rag_invocations.csv"
        file_exists = os.path.isfile(csv_file)
        try:
            with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["index", "conversation_id", "timestamp", "query", "top_k", "results", "distances", "llm_model"])
                self.index_counter += 1
                writer.writerow([
                    str(self.index_counter),
                    str(conversation_id),
                    datetime.now().isoformat(),
                    query,
                    top_k,
                    json.dumps(results, ensure_ascii=False),
                    json.dumps(distances, ensure_ascii=False),
                    llm_model
                ])
                self.logger.info(f"Wrote RAG invocation to {csv_file}: index={self.index_counter}")
        except Exception as e:
            self.logger.error(f"Failed to write RAG invocation: {str(e)}")

    def get_tools(self):
        def retrieve_scam_reports(query: str, top_k: int, conversation_id: str, llm_model: str) -> str:
            try:
                self.logger.debug(f"Executing retrieve_scam_reports: query='{query}', top_k={top_k}")
                df = self.vector_store.similarity_search(query_text=query, model=ScamReport, limit=top_k)
                formatted_results = []
                distances = []
                if not df.empty:
                    for record in df.to_dict(orient="records"):
                        result = {
                            "report_id": record.get("report_id"),
                            "scam_incident_date": record.get("scam_incident_date").isoformat() if record.get("scam_incident_date") else None,
                            "scam_type": record.get("scam_type"),
                            "scam_approach_platform": record.get("scam_approach_platform"),
                            "scam_communication_platform": record.get("scam_communication_platform"),
                            "scam_transaction_type": record.get("scam_transaction_type"),
                            "scam_beneficiary_platform": record.get("scam_beneficiary_platform"),
                            "scam_beneficiary_identifier": record.get("scam_beneficiary_identifier"),
                            "scam_contact_no": record.get("scam_contact_no"),
                            "scam_moniker": record.get("scam_moniker"),
                            "scam_amount_lost": record.get("scam_amount_lost"),
                            "scam_incident_description": record.get("scam_incident_description"),
                            "scam_specific_details": record.get("scam_specific_slots", {})
                        }
                        formatted_results.append(result)
                    distances = df["distance"].tolist() if "distance" in df else []
                self._log_rag_invocation(conversation_id, query, top_k, formatted_results, distances, llm_model)
                return json.dumps(formatted_results, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"Error retrieving scam reports: {str(e)}")
                return json.dumps({"error": f"Error retrieving scam reports: {str(e)}"})

        return [StructuredTool.from_function(
            func=retrieve_scam_reports,
            name="retrieve_scam_reports",
            description="Retrieve relevant scam reports from the database.",
            args_schema=RetrieveScamReportsArgs
        )]
        
class VictimTools:
    def __init__(self):
        self.settings = get_settings()
        self._setup_logging()

    def _setup_logging(self):
        log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "tools.log"
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='a'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized to {log_file}")

    def get_tools(self):
        return []