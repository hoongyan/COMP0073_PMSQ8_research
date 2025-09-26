import unittest
import os
import sys
import pandas as pd
import json
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.database.database_operations import DatabaseManager
from src.database.vector_operations import VectorStore
from src.models.data_model import ScamReport
from config.settings import get_settings
from config.id_manager import IDManager
import logging
import csv

# Configure logging for tests
def setup_logging():
    settings = get_settings()
    log_dir = Path(settings.log.directory) / settings.log.subdirectories["tests"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "test_retrieval.log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class TestRetrieval(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.settings = get_settings()
        self.db_manager = DatabaseManager()
        self.vector_store = VectorStore(self.db_manager.session_factory)
        self.id_manager = IDManager()
        self.csv_file = "rag_invocations.csv"
        self.conversation_id = self.id_manager.get_next_id()
        self.id_file = "last_conversation_id.txt"
        self.models = ["gemma2:9b", "llama3.2", "qwen2.5:7b"]
        
        # Clean up files before tests
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)
        if os.path.exists(self.id_file):
            os.remove(self.id_file)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)
        if os.path.exists(self.id_file):
            os.remove(self.id_file)
    
    def test_database_connection(self):
        """Test database connection."""
        logger.debug("Testing database connection")
        self.assertTrue(self.db_manager.test_connection(), "Database connection failed")
        logger.info("Database connection test passed")
    
    def test_pgvector_extension(self):
        """Test enabling pgvector extension."""
        logger.debug("Testing pgvector extension")
        self.assertTrue(self.db_manager.enable_pgvector(), "Failed to enable pgvector extension")
        logger.info("pgvector extension test passed")
    
    def test_hnsw_index(self):
        """Test creation of HNSW index."""
        logger.debug("Testing HNSW index creation")
        self.assertTrue(self.db_manager.create_hnsw_index("scam_reports", "embedding"), "Failed to create HNSW index")
        logger.info("HNSW index creation test passed")
    
    def test_similarity_search_no_filter(self):
        """Test similarity search without metadata filter."""
        logger.debug("Testing similarity search without filter")
        sample_query = "I received a suspicious SMS from DBS asking me to click a link"
        results = self.vector_store.similarity_search(
            query_text=sample_query,
            model=ScamReport,
            limit=5,
            metadata_filter=None
        )
        
        if not results.empty:
            self.assertIn("report_id", results.columns, "Results missing report_id")
            self.assertIn("distance", results.columns, "Results missing distance")
            self.assertTrue(all(results["distance"] >= 0), "Invalid distance values")
            logger.info(f"Similarity search (no filter) returned {len(results)} records")
        else:
            logger.warning("No results returned from similarity search (no filter)")
    
    def test_similarity_search_with_filter(self):
        """Test similarity search with metadata filter."""
        logger.debug("Testing similarity search with filter")
        sample_query = "I received a suspicious SMS from DBS asking me to click a link"
        results = self.vector_store.similarity_search(
            query_text=sample_query,
            model=ScamReport,
            limit=5,
            metadata_filter={"scam_type": "PHISHING"}
        )
        
        if not results.empty:
            self.assertIn("report_id", results.columns, "Results missing report_id")
            self.assertIn("distance", results.columns, "Results missing distance")
            self.assertTrue(all(results["scam_type"] == "PHISHING"), "Filter not applied correctly")
            self.assertTrue(all(results["distance"] >= 0), "Invalid distance values")
            logger.info(f"Similarity search (with filter) returned {len(results)} records")
        else:
            logger.warning("No results returned from similarity search (with filter)")
    
    def test_rag_invocation_logging(self):
        """Test RAG invocation logging for multiple models."""
        for model in self.models:
            with self.subTest(model=model):
                logger.debug(f"Testing RAG invocation logging with model {model}")
                from src.agents.tools import PoliceTools
                
                police_tools = PoliceTools()
                tools = police_tools.get_tools()
                retrieve_tool = tools[0]  # retrieve_scam_reports
                
                sample_query = "I received a suspicious SMS from DBS asking me to click a link"
                result = retrieve_tool.invoke({"query": sample_query, "top_k": 5, "conversation_id": self.conversation_id, "police_model": model})
                
                self.assertTrue(os.path.exists(self.csv_file), "RAG invocation CSV not created")
                with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    self.assertEqual(len(rows), 1, "RAG invocation not logged")
                    self.assertEqual(int(rows[0]["conversation_id"]), self.conversation_id)
                    self.assertEqual(rows[0]["query"], sample_query)
                    self.assertEqual(rows[0]["top_k"], "5")
                    self.assertTrue(rows[0]["results"], "Results not logged")
                    self.assertTrue(rows[0]["distances"], "Distances not logged")
                    self.assertEqual(rows[0]["police_model"], model)
                
                logger.info(f"RAG invocation logging test passed for model {model}")
                self.conversation_id = self.id_manager.get_next_id()  # Increment for next subtest

if __name__ == "__main__":
    unittest.main()