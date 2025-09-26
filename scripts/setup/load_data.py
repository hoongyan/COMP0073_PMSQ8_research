import argparse
import sys
import os
from pathlib import Path
import json
import pandas as pd 
from sqlalchemy import text
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..')))
from src.database.database_operations import DatabaseManager, CRUDOperations
from src.database.vector_operations import VectorStore
from src.preprocessing.preprocess import ScamReportPreprocessor
from src.models.data_model import ScamReport, Strategy
from config.settings import get_settings
from config.logging_config import setup_logger
from scripts.setup.init_db import DatabaseInitializer


class DataLoader:
    """Manages data loading and storage into the database."""
    
    def __init__(self):
        """Initialize components and setup logging."""
        self.settings = get_settings()
        self.logger = setup_logger("DataLoader", self.settings.log.subdirectories["database"])
        self.db_manager = DatabaseManager()
        self.strategy_crud = CRUDOperations(Strategy, self.db_manager.session_factory)
        self.crud = CRUDOperations(ScamReport, self.db_manager.session_factory)
        self.vector_store = VectorStore(self.db_manager.session_factory)
        self.preprocessor = ScamReportPreprocessor(self.vector_store)
    
    def load_and_store_data(self, initialize_schema: bool = False):
        """Clear existing data and load new scam reports into the database."""
        try:
            # Initialize database schema if requested (through flag)
            if initialize_schema:
                initializer = DatabaseInitializer()
                initializer.initialize_database()
            
            # Clear existing data
            deleted_count = self.crud.delete_all()
            self.logger.info(f"Deleted {deleted_count} existing records")
            print(f"Deleted {deleted_count} existing records")
            
            # Preprocess data
            processed_df = self.preprocessor.preprocess()
            
            # Validate DataFrame columns
            expected_columns = [
                "report_id", "scam_incident_date", "scam_report_date", "scam_type",
                "scam_approach_platform", "scam_communication_platform", "scam_transaction_type",
                "scam_beneficiary_platform", "scam_beneficiary_identifier", "scam_contact_no",
                "scam_email", "scam_moniker", "scam_url_link", "scam_amount_lost",
                "scam_incident_description", "embedding"
            ]
           
            
            if not all(col in processed_df.columns for col in expected_columns):
                missing = [col for col in expected_columns if col not in processed_df.columns]
                self.logger.error(f"Missing columns in DataFrame: {missing}")
                raise ValueError(f"Missing columns in DataFrame: {missing}")
            
            # Insert data
            inserted_count = self.crud.create_bulk(processed_df)
            self.logger.info(f"Inserted {inserted_count} new records")
            print(f"Inserted {inserted_count} new records")
            
            # Verify sample record
            if not processed_df.empty:
                sample_id = processed_df.iloc[0]["report_id"]
                record = self.crud.read(sample_id)
                if record:
                    self.logger.info(f"Verified sample record: {sample_id}")
                    print(f"Verified sample record: {sample_id}")
                else:
                    self.logger.warning(f"Sample record {sample_id} not found")
                    print(f"Sample record {sample_id} not found")
                    
            # Load strategies from seed file
            self.logger.info("Loading strategies from seed file...")
            
            # Clear existing strategies
            deleted_strategy_count = self.strategy_crud.delete_all()
            self.logger.info(f"Deleted {deleted_strategy_count} existing strategies")
            print(f"Deleted {deleted_strategy_count} existing strategies")
            
            # Reset the ID sequence to start from 1 
            try:
                with self.db_manager.session_factory() as db:
                    db.execute(text("ALTER SEQUENCE strategy_id_seq RESTART WITH 1;"))
                    db.commit()
                self.logger.info("Reset strategy ID sequence to start from 1")
                print("Reset strategy ID sequence to start from 1")
            except Exception as e:
                self.logger.error(f"Failed to reset strategy ID sequence: {str(e)}")
                print(f"Failed to reset strategy ID sequence: {str(e)}")
                raise  # Stop if it fails, to avoid partial setup

            # Load seed JSON
            seed_path = Path(self.settings.data.strategy_seed_json)
            if not seed_path.exists():
                self.logger.error(f"Seed file not found: {seed_path}")
                raise FileNotFoundError(f"Seed file not found: {seed_path}")
            
            with open(seed_path, 'r') as f:
                strategies = json.load(f)
            
            # Convert to DataFrame for bulk insert
            strategies_df = pd.DataFrame(strategies)
            
            # Validate DataFrame columns for strategies
            expected_strategy_columns = [
                "strategy_type", "response", "success_score", "user_profile",
            ]
            
            if not all(col in strategies_df.columns for col in expected_strategy_columns):
                missing = [col for col in expected_strategy_columns if col not in strategies_df.columns]
                self.logger.error(f"Missing columns in strategies DataFrame: {missing}")
                raise ValueError(f"Missing columns in strategies DataFrame: {missing}")
            
            # Insert strategies
            inserted_strategy_count = self.strategy_crud.create_bulk(strategies_df)
            self.logger.info(f"Inserted {inserted_strategy_count} new strategies")
            print(f"Inserted {inserted_strategy_count} new strategies")
            
            # Verify sample strategy
            if not strategies_df.empty:
                sample_strategy = strategies_df.iloc[0]
                
                # Since Strategy has autoincrement id, read the first one 
                read_strategies = self.strategy_crud.read_all(limit=1)
                if read_strategies:
                    self.logger.info(f"Verified sample strategy with type: {read_strategies[0].strategy_type}")
                    print(f"Verified sample strategy with type: {read_strategies[0].strategy_type}")
                else:
                    self.logger.warning("No strategies found after insertion")
                    print("No strategies found after insertion")
                
        except Exception as e:
            self.logger.error(f"Error in data loading: {str(e)}")
            print(f"Error: {str(e)}")
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load scam report data into the database.")
    parser.add_argument("--initialize-schema", action="store_true", help="Initialize database schema before loading data")
    args = parser.parse_args()
    loader = DataLoader()
    loader.load_and_store_data(initialize_schema=args.initialize_schema)