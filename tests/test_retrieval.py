import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.database_operations import DatabaseManager
from src.database.vector_operations import VectorStore
from src.models.data_model import ScamReport
from config.settings import get_settings
from config.logging_config import setup_logger
import pandas as pd
import traceback
import logging

def test_similarity_search():
    logger = setup_logger("TestRetrieval", get_settings().log.subdirectories["tests"])
    logger.debug("Starting test_similarity_search")
    try:
        print("Starting test_similarity_search...")
        settings = get_settings()
        print("Settings loaded successfully")
        logger.debug("Settings loaded: %s", settings)

        db_manager = DatabaseManager()
        print("DatabaseManager initialized")
        logger.debug("DatabaseManager initialized")

        if not db_manager.test_connection():
            print("Database connection failed")
            logger.error("Database connection failed")
            return
        print("Database connection successful")
        logger.debug("Database connection successful")

        vector_store = VectorStore(db_manager.session_factory)
        print("VectorStore initialized")
        logger.debug("VectorStore initialized")

        if not db_manager.enable_pgvector():
            print("Failed to enable pgvector extension")
            logger.error("Failed to enable pgvector extension")
            return
        print("pgvector extension enabled")
        logger.debug("pgvector extension enabled")

        if not db_manager.create_hnsw_index("scam_reports", "embedding"):
            print("Failed to create HNSW index")
            logger.error("Failed to create HNSW index")
            return
        print("HNSW index created")
        logger.debug("HNSW index created")

        sample_query = "I received a suspicious SMS from DBS asking me to click a link"
        print(f"Running similarity search with query: {sample_query}")
        logger.debug("Running similarity search with query: %s", sample_query)

        # Test without metadata filter
        results = vector_store.similarity_search(
            sample_query,
            ScamReport,
            limit=5,
            metadata_filter=None
        )
        print("Similarity search results (no filter):")
        logger.debug("Similarity search (no filter) completed, results: %s", "empty" if results.empty else f"{len(results)} records")
        if not results.empty:
            print(results[['report_id', 'scam_type', 'scam_incident_description', 'scam_url_link', 'scam_specific_slots', 'distance']])
        else:
            print("No results returned from similarity search (no filter)")

        # Test with metadata filter
        results = vector_store.similarity_search(
            sample_query,
            ScamReport,
            limit=5,
            metadata_filter={"scam_type": "PHISHING"}
        )
        print("Similarity search results (with scam_type filter):")
        logger.debug("Similarity search (with scam_type filter) completed, results: %s", "empty" if results.empty else f"{len(results)} records")
        if not results.empty:
            print(results[['report_id', 'scam_type', 'scam_incident_description', 'scam_url_link', 'scam_specific_slots', 'distance']])
        else:
            print("No results returned from similarity search (with scam_type filter)")

    except Exception as e:
        print(f"Error in test_similarity_search: {str(e)}")
        logger.error("Error in test_similarity_search: %s", str(e), exc_info=True)
        traceback.print_exc()

if __name__ == "__main__":
    test_similarity_search()