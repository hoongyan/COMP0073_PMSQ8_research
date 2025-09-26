# from langchain_core.tools import StructuredTool
# from pydantic import BaseModel, Field
# from src.database.vector_operations import VectorStore
# from src.database.database_operations import DatabaseManager
# from src.models.data_model import ScamReport
# from config.settings import get_settings
# from config.id_manager import IDManager
# import logging
# from pathlib import Path
# import json
# import csv
# import os
# from datetime import datetime
# from filelock import FileLock

# class RetrieveScamReportsArgs(BaseModel):
#     """Arguments for the retrieve_scam_reports tool."""
#     query: str = Field(..., description="The query to search for similar scam reports")
#     top_k: int = Field(default=5, description="Number of top results to return")
#     conversation_id: int = Field(default=None, description="ID of the conversation for logging")
#     llm_model: str = Field(default=None, description="LLM model used by the police agent")

# class PoliceTools:
#     """Tools for the police chatbot, including scam report retrieval."""

#     def __init__(self):
#         """Initialize with VectorStore and settings."""
#         self.settings = get_settings()
#         self._setup_logging()
#         self.index_counter = self._load_last_index()
#         self.id_manager = IDManager(csv_file="rag_invocations.csv")
#         db_manager = DatabaseManager()
#         self.vector_store = VectorStore(session_factory=db_manager.session_factory)

#     def _setup_logging(self):
#         """Configure logging to write to tool-specific log file."""
#         log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
#         log_dir.mkdir(parents=True, exist_ok=True)
#         log_file = log_dir / "tools.log"

#         logging.basicConfig(
#             level=logging.DEBUG,
#             format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#             handlers=[
#                 logging.FileHandler(log_file, mode='a'),
#                 logging.StreamHandler()
#             ]
#         )
#         self.logger = logging.getLogger(__name__)
#         self.logger.info(f"Logging initialized to {log_file}")

#     def _load_last_index(self) -> int:
#         """Load the last used index from rag_invocations.csv."""
#         csv_file = "rag_invocations.csv"
#         max_index = 0
#         if os.path.exists(csv_file):
#             try:
#                 with FileLock(f"{csv_file}.lock"):
#                     with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                         reader = csv.DictReader(f)
#                         if "index" in reader.fieldnames:
#                             indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
#                             max_index = max(indices) if indices else 0
#                         else:
#                             self.logger.warning(f"CSV {csv_file} does not contain 'index' field. Starting with index 0.")
#                 self.logger.debug(f"Loaded max index {max_index} from {csv_file}")
#             except Exception as e:
#                 self.logger.error(f"Error reading index from CSV: {str(e)}", exc_info=True)
#         return max_index

#     def _log_rag_invocation(self, conversation_id: int, query: str, top_k: int, results: list, distances: list, llm_model: str = None):
#         """Log RAG invocation details to rag_invocations.csv."""
#         csv_file = "rag_invocations.csv"
#         file_exists = os.path.isfile(csv_file)

#         effective_conversation_id = conversation_id if conversation_id is not None else -1
#         self.logger.debug(f"Logging RAG invocation: conversation_id={effective_conversation_id}, query={query}, top_k={top_k}, results_length={len(results)}")
#         try:
#             with FileLock(f"{csv_file}.lock", timeout=10):
#                 with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
#                     writer = csv.writer(f)
#                     if not file_exists:
#                         writer.writerow([
#                             "index", "conversation_id", "timestamp", "query", "top_k", "results", "distances", "llm_model"
#                         ])
#                     self.index_counter += 1
#                     row = [
#                         str(self.index_counter),
#                         str(effective_conversation_id),
#                         datetime.now().isoformat(),
#                         query,
#                         top_k,
#                         json.dumps(results, ensure_ascii=False),
#                         json.dumps(distances, ensure_ascii=False),
#                         llm_model or ""
#                     ]
#                     writer.writerow(row)
#                     self.logger.info(f"Successfully wrote RAG invocation to {csv_file}: index={self.index_counter}, conversation_id={effective_conversation_id}")
#         except Exception as e:
#             self.logger.error(f"Failed to write RAG invocation to {csv_file}: {str(e)}", exc_info=True)
#             raise

#     def get_tools(self):
#         """Return the list of tools for the police agent."""
#         def retrieve_scam_reports(query: str, top_k: int = 5, conversation_id: int = None, llm_model: str = None) -> str:
#             try:
#                 self.logger.debug(f"Executing retrieve_scam_reports: query='{query}', top_k={top_k}, conversation_id={conversation_id}, llm_model={llm_model}")
#                 df = self.vector_store.similarity_search(
#                     query_text=query,
#                     model=ScamReport,
#                     limit=top_k
#                 )
#                 formatted_results = []
#                 distances = []
#                 if not df.empty:
#                     for record in df.to_dict(orient="records"):
#                         result = {
#                             "report_id": record.get("report_id"),
#                             "scam_incident_date": record.get("scam_incident_date").isoformat() if record.get("scam_incident_date") else None,
#                             "scam_report_date": record.get("scam_report_date").isoformat() if record.get("scam_report_date") else None,
#                             "scam_type": record.get("scam_type"),
#                             "scam_approach_platform": record.get("scam_approach_platform"),
#                             "scam_communication_platform": record.get("scam_communication_platform"),
#                             "scam_transaction_type": record.get("scam_transaction_type"),
#                             "scam_beneficiary_platform": record.get("scam_beneficiary_platform"),
#                             "scam_beneficiary_identifier": record.get("scam_beneficiary_identifier"),
#                             "scam_contact_no": record.get("scam_contact_no"),
#                             "scam_email": record.get("scam_email"),
#                             "scam_moniker": record.get("scam_moniker"),
#                             "scam_url_link": record.get("scam_url_link"),
#                             "scam_amount_lost": record.get("scam_amount_lost"),
#                             "scam_incident_description": record.get("scam_incident_description"),
#                             "scam_specific_details": record.get("scam_specific_slots", {})
#                         }
#                         formatted_results.append(result)
#                     distances = df["distance"].tolist() if "distance" in df else []
#                 self._log_rag_invocation(conversation_id, query, top_k, formatted_results, distances, llm_model)
#                 return json.dumps(formatted_results, ensure_ascii=False)
#             except Exception as e:
#                 self.logger.error(f"Error retrieving scam reports: {str(e)}", exc_info=True)
#                 return json.dumps({"error": f"Error retrieving scam reports: {str(e)}"})
        
#         return [StructuredTool.from_function(
#             func=retrieve_scam_reports,
#             name="retrieve_scam_reports",
#             description="Retrieve relevant scam reports from the database based on the query.",
#             args_schema=RetrieveScamReportsArgs,
#             return_direct=True,
#             strict=True
#         )]
    
# class VictimTools:
#     """Tools for the victim chatbot."""

#     def __init__(self):
#         """Initialize with settings."""
#         self.settings = get_settings()
#         self._setup_logging()

#     def _setup_logging(self):
#         """Configure logging to write to tool-specific log file."""
#         log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
#         log_dir.mkdir(parents=True, exist_ok=True)
#         log_file = log_dir / "tools.log"

#         logging.basicConfig(
#             level=logging.DEBUG,
#             format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#             handlers=[
#                 logging.FileHandler(log_file, mode='a'),
#                 logging.StreamHandler()
#             ]
#         )
#         self.logger = logging.getLogger(__name__)
#         self.logger.info(f"Logging initialized to {log_file}")

#     def get_tools(self):
#         """Return the list of tools for the victim agent."""
#         return []


from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from src.database.vector_operations import VectorStore
from src.database.database_operations import DatabaseManager
from src.models.data_model import ScamReport
from config.settings import get_settings
from config.id_manager import IDManager
import logging
from pathlib import Path
import json
import csv
import os
from datetime import datetime
from filelock import FileLock

class RetrieveScamReportsArgs(BaseModel):
    """Arguments for the retrieve_scam_reports tool."""
    query: str = Field(..., description="The query to search for similar scam reports")
    top_k: int = Field(..., description="Number of top results to return")
    conversation_id: int = Field(..., description="ID of the conversation for logging")
    llm_model: str = Field(..., description="LLM model used by the police agent")

class PoliceTools:
    def __init__(self):
        self.settings = get_settings()
        self._setup_logging()
        self.index_counter = self._load_last_index()
        self.id_manager = IDManager(csv_file="rag_invocations.csv")
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
                self.logger.error(f"Error reading index from CSV: {str(e)}", exc_info=True)
        return max_index

    def _log_rag_invocation(self, conversation_id: int, query: str, top_k: int, results: list, distances: list, llm_model: str):
        csv_file = "rag_invocations.csv"
        file_exists = os.path.isfile(csv_file)
        self.logger.debug(f"Logging RAG invocation: conversation_id={conversation_id}, query={query}, top_k={top_k}, results_length={len(results)}")
        try:
            with FileLock(f"{csv_file}.lock", timeout=10):
                with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "index", "conversation_id", "timestamp", "query", "top_k", "results", "distances", "llm_model"
                        ])
                    self.index_counter += 1
                    row = [
                        str(self.index_counter),
                        str(conversation_id),
                        datetime.now().isoformat(),
                        query,
                        top_k,
                        json.dumps(results, ensure_ascii=False),
                        json.dumps(distances, ensure_ascii=False),
                        llm_model
                    ]
                    writer.writerow(row)
                    self.logger.info(f"Successfully wrote RAG invocation to {csv_file}: index={self.index_counter}, conversation_id={conversation_id}")
        except Exception as e:
            self.logger.error(f"Failed to write RAG invocation to {csv_file}: {str(e)}", exc_info=True)
            raise

    def get_tools(self):
        def retrieve_scam_reports(query: str, top_k: int, conversation_id: int, llm_model: str) -> str:
            try:
                self.logger.debug(f"Executing retrieve_scam_reports: query='{query}', top_k={top_k}, conversation_id={conversation_id}, llm_model={llm_model}")
                df = self.vector_store.similarity_search(
                    query_text=query,
                    model=ScamReport,
                    limit=top_k
                )
                formatted_results = []
                distances = []
                if not df.empty:
                    for record in df.to_dict(orient="records"):
                        result = {
                            "report_id": record.get("report_id"),
                            "scam_incident_date": record.get("scam_incident_date").isoformat() if record.get("scam_incident_date") else None,
                            "scam_report_date": record.get("scam_report_date").isoformat() if record.get("scam_report_date") else None,
                            "scam_type": record.get("scam_type"),
                            "scam_approach_platform": record.get("scam_approach_platform"),
                            "scam_communication_platform": record.get("scam_communication_platform"),
                            "scam_transaction_type": record.get("scam_transaction_type"),
                            "scam_beneficiary_platform": record.get("scam_beneficiary_platform"),
                            "scam_beneficiary_identifier": record.get("scam_beneficiary_identifier"),
                            "scam_contact_no": record.get("scam_contact_no"),
                            "scam_email": record.get("scam_email"),
                            "scam_moniker": record.get("scam_moniker"),
                            "scam_url_link": record.get("scam_url_link"),
                            "scam_amount_lost": record.get("scam_amount_lost"),
                            "scam_incident_description": record.get("scam_incident_description"),
                            "scam_specific_details": record.get("scam_specific_slots", {})
                        }
                        formatted_results.append(result)
                    distances = df["distance"].tolist() if "distance" in df else []
                self._log_rag_invocation(conversation_id, query, top_k, formatted_results, distances, llm_model)
                return json.dumps(formatted_results, ensure_ascii=False)
            except Exception as e:
                self.logger.error(f"Error retrieving scam reports: {str(e)}", exc_info=True)
                return json.dumps({"error": f"Error retrieving scam reports: {str(e)}"})
        
        return [StructuredTool.from_function(
            func=retrieve_scam_reports,
            name="retrieve_scam_reports",
            description="Retrieve relevant scam reports from the database based on the query.",
            args_schema=RetrieveScamReportsArgs,
            return_direct=True,
            strict=True
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