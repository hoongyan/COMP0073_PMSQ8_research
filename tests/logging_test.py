import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import get_settings
from config.id_manager import IDManager
from config.logging_config import setup_logger
from src.agents.utils import RecordCounter, save_conversation_history
from src.database.database_operations import DatabaseManager, CRUDOperations
from src.database.vector_operations import VectorStore
from src.preprocessing.preprocess import Preprocessor
from scripts.setup.load_data import DataLoader
from scripts.setup.init_db import DatabaseInitializer
from datetime import datetime
import json
import csv
import os

def test_logging():
    logger = setup_logger("TestLogging", get_settings().log.subdirectories["tests"])
    
    # Test settings
    try:
        settings = get_settings()
        logger.info("Settings initialized successfully")
    except Exception as e:
        logger.error(f"Settings initialization failed: {str(e)}", exc_info=True)
        return
    
    # Test IDManager
    id_manager = IDManager()
    conversation_id = id_manager.get_next_id()
    logger.info(f"Generated conversation_id: {conversation_id}")
    
    # Test RecordCounter
    record_counter = RecordCounter()
    total_records = 5
    record_index = record_counter.get_next_index(total_records)
    logger.info(f"Got record index: {record_index}")
    
    # Test conversation history saving
    history = [
        {
            "role": "victim",
            "content": "Test victim message",
            "timestamp": datetime.now().isoformat(),
            "autonomous": False
        },
        {
            "role": "police",
            "content": "Test police response",
            "timestamp": datetime.now().isoformat(),
            "structured_data": {
                "scam_incident_date": "",
                "scam_type": "TEST",
                "scam_approach_platform": "",
                "scam_communication_platform": "",
                "scam_transaction_type": "",
                "scam_beneficiary_platform": "",
                "scam_beneficiary_identifier": "",
                "scam_contact_no": "",
                "scam_email": "",
                "scam_moniker": "",
                "scam_url_link": "",
                "scam_amount_lost": 0.0,
                "scam_incident_description": "Test incident",
                "scam_specific_details": {"scam_subcategory": "test"},
                "rag_invoked": True
            },
            "rag_invoked": True,
            "autonomous": False
        }
    ]
    save_conversation_history(conversation_id, history, model_name="test_model", logger=logger)
    logger.info("Conversation history saved successfully")
    
    # Test DatabaseManager
    db_manager = DatabaseManager()
    if db_manager.test_connection():
        logger.info("Database connection successful")
    else:
        logger.error("Database connection failed")
        return
    
    # Test CRUDOperations
    crud = CRUDOperations(None, db_manager.session_factory)
    logger.info("CRUDOperations initialized")
    
    # Test VectorStore
    vector_store = VectorStore(db_manager.session_factory)
    logger.info("VectorStore initialized")
    
    # Test Preprocessor
    preprocessor = Preprocessor()
    logger.info("Preprocessor initialized")
    
    # Test DataLoader
    data_loader = DataLoader()
    logger.info("DataLoader initialized")
    
    # Test DatabaseInitializer
    db_initializer = DatabaseInitializer()
    logger.info("DatabaseInitializer initialized")
    
    # Verify log files
    log_files = [
        "logs/setup/settings.log",
        "logs/agent/idmanager.log",
        "logs/agent/recordcounter.log",
        "logs/tests/testlogging.log",
        "logs/agent/conversationsaver.log",
        "logs/database/databasemanager.log",
        "logs/database/crudoperations.log",
        "logs/database/vectorstore.log",
        "logs/preprocessing/preprocessor.log",
        "logs/database/dataloader.log",
        "logs/database/databaseinitializer.log"
    ]
    for log_file in log_files:
        if os.path.exists(log_file):
            logger.info(f"Log file exists: {log_file}")
        else:
            logger.error(f"Log file missing: {log_file}")
    
    # Verify CSV content
    with open("conversation_history.csv", mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["conversation_id"] == str(conversation_id):
                logger.info(f"Found conversation_id {conversation_id} in CSV: {row}")

if __name__ == "__main__":
    test_logging()