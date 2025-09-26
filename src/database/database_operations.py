import sys
import os
from typing import Any, List, Optional, Type, Dict, Union, Tuple
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, text, update, func
from sqlalchemy.ext.declarative import DeclarativeMeta
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.settings import get_settings
from config.logging_config import setup_logger
from src.models.data_model import Strategy


class DatabaseManager:
    """Manages database connection and setup."""
    
    def __init__(self):
        """Initialize database connection and session factory."""
        self.settings = get_settings()
        self.logger = setup_logger("DatabaseManager", self.settings.log.subdirectories["database"])
        try:
            self.engine = create_engine(
                self.settings.database.url,
                echo=self.settings.database.echo
            )
            self.session_factory = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            self.logger.info("Database engine created successfully")
        except Exception as e:
            self.logger.error(f"Failed to create database engine: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            with self.session_factory() as session:
                result = session.execute(text("SELECT NOW();"))
                self.logger.info(f"Database connection test successful: {result.fetchone()[0]}")
                return True
        except Exception as e:
            self.logger.error(f"Database connection test failed: {str(e)}")
            return False
    
    def enable_pgvector(self) -> bool:
        """Enable pgvector extension if not already enabled."""
        try:
            with self.session_factory() as session:
                result = session.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';"))
                if result.fetchone():
                    self.logger.info("pgvector extension is enabled")
                    return True
                self.logger.info("Enabling pgvector extension")
                session.execute(text("CREATE EXTENSION vector;"))
                session.commit()
                self.logger.info("pgvector extension enabled successfully")
                return True
        except Exception as e:
            self.logger.error(f"Failed to enable pgvector extension: {str(e)}")
            return False
        
    def create_hnsw_index(self, table_name: str, column_name: str = "embedding") -> bool:
        """Create HNSW index on specified table and column."""
        index_name = f"{table_name}_{column_name}_idx"  # Make index name dynamic
        try:
            with self.session_factory() as session:
                session.execute(text(f"CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} USING hnsw ({column_name} vector_cosine_ops);"))
                session.commit()
                self.logger.info(f"HNSW index created on {table_name}.{column_name}")
                return True
        except Exception as e:
            self.logger.error(f"Failed to create HNSW index: {str(e)}")
            return False


class CRUDOperations:
    """Generic CRUD operations for SQLAlchemy models."""
    
    def __init__(self, model: Type[DeclarativeMeta], session_factory: sessionmaker):
        """Initialize with SQLAlchemy model and session factory."""
        self.model = model
        self.session_factory = session_factory
        self.logger = setup_logger("CRUDOperations", get_settings().log.subdirectories["database"])
        self.pk_column = [col for col in self.model.__table__.primary_key.columns][0].name  # Cache PK name
    
    def create(self, data: Dict[str, Any]) -> Optional[Any]:
        """Create a single record."""
        with self.session_factory() as db:
            try:
                record = self.model(**data)
                db.add(record)
                db.commit()
                db.refresh(record)
                self.logger.info(f"Created record with {self.pk_column}: {getattr(record, self.pk_column, 'unknown')}")
                return record
            except Exception as e:
                db.rollback()
                self.logger.error(f"Error creating record: {str(e)}")
                return None
    
    
    def create_bulk(self, df: pd.DataFrame) -> int:
        """Create multiple records from a DataFrame."""
        with self.session_factory() as db:
            try:
                count = 0
                for _, row in df.iterrows():
                    try:
                        record = self.model(**row.to_dict())
                        db.add(record)
                        count += 1
                    except Exception as e:
                        self.logger.error(f"Error creating record: {str(e)}")
                        continue
                db.commit()
                self.logger.info(f"Created {count} records")
                return count
            except Exception as e:
                db.rollback()
                self.logger.error(f"Error creating bulk records: {str(e)}")
                return 0
            
    
    def read(self, record_id: Union[str, int]) -> Optional[Any]:
        """Read a single record by ID."""
        with self.session_factory() as db:
            try:
                filter_expr = getattr(self.model, self.pk_column) == record_id
                record = db.query(self.model).filter(filter_expr).first()
                if record:
                    self.logger.info(f"Read record with {self.pk_column}: {record_id}")
                else:
                    self.logger.warning(f"No record found with {self.pk_column}: {record_id}")
                return record
            except Exception as e:
                self.logger.error(f"Error reading record: {str(e)}")
                return None
    
    def read_all(self, limit: int = 100, offset: int = 0) -> List[Any]:
        """Read all records."""
        with self.session_factory() as db:
            try:
                records = db.query(self.model).offset(offset).limit(limit).all()
                self.logger.info(f"Read {len(records)} records")
                return records
            except Exception as e:
                self.logger.error(f"Error reading records: {str(e)}")
                return []
    
    def update(self, record_id: Union[str, int], data: Dict[str, Any]) -> Optional[Any]:
        """Update a record by ID."""
        with self.session_factory() as db:
            try:
                filter_expr = getattr(self.model, self.pk_column) == record_id
                record = db.query(self.model).filter(filter_expr).first()
                if record:
                    for key, value in data.items():
                        setattr(record, key, value)
                    db.commit()
                    db.refresh(record)
                    self.logger.info(f"Updated record with {self.pk_column}: {record_id}")
                    return record
                self.logger.warning(f"No record found with {self.pk_column}: {record_id}")
                return None
            except Exception as e:
                db.rollback()
                self.logger.error(f"Error updating record: {str(e)}")
                return None
            
    def update_embedding(self, record_id: Union[str, int], embedding: List[float], column_name: str = "embedding") -> bool:
        """Update a single record's specified embedding column."""
        with self.session_factory() as db:
            try:
                filter_expr = getattr(self.model, self.pk_column) == record_id
                emb_col = getattr(self.model, column_name)
                updated = db.query(self.model).filter(filter_expr).update({emb_col: embedding})
                if updated == 0:
                    self.logger.warning(f"No record found with {self.pk_column}: {record_id}")
                    return False
                db.commit()
                self.logger.info(f"Updated {column_name} for record {self.pk_column}: {record_id}")
                return True
            except Exception as e:
                db.rollback()
                self.logger.error(f"Error updating {column_name}: {str(e)}")
                return False
    
    def update_embedding_bulk(self, embeddings: Dict[Union[str, int], List[float]], column_name: str = "embedding") -> int:
        """Bulk update specified embedding column for multiple records."""
        with self.session_factory() as db:
            try:
                emb_col = getattr(self.model, column_name)
                count = 0
                for record_id, embedding in embeddings.items():
                    filter_expr = getattr(self.model, self.pk_column) == record_id
                    updated = db.query(self.model).filter(filter_expr).update({emb_col: embedding})
                    count += updated
                db.commit()
                self.logger.info(f"Updated {column_name} for {count} records")
                return count
            except Exception as e:
                db.rollback()
                self.logger.error(f"Error bulk updating {column_name}: {str(e)}")
                return 0


    def delete(self, record_id: Union[str, int]) -> bool:
        """Delete a record by its primary key."""
        
        with self.session_factory() as db:
            try:
                filter_expr = getattr(self.model, self.pk_column) == record_id
                record = db.query(self.model).filter(filter_expr).first()
                if record:
                    db.delete(record)
                    db.commit()
                    self.logger.info(f"Deleted record with {self.pk_column}: {record_id}")
                    return True
                self.logger.warning(f"No record found with {self.pk_column}: {record_id}")
                return False
            except Exception as e:
                db.rollback()
                self.logger.error(f"Error deleting record: {str(e)}")
                return False
    
    def delete_all(self) -> int:
        """Delete all records in the table."""
        with self.session_factory() as db:
            try:
                count = db.query(self.model).delete()
                db.commit()
                self.logger.info(f"Deleted {count} records from {self.model.__tablename__}")
                return count
            except Exception as e:
                db.rollback()
                self.logger.error(f"Error deleting all records: {str(e)}")
                return 0
            
    def count_records(self) -> int:
        """Get the count of records in the table."""
        with self.session_factory() as db:
            try:
                count = db.query(self.model).count()
                self.logger.debug(f"Record count for {self.model.__tablename__}: {count}")
                return count
            except Exception as e:
                self.logger.error(f"Error counting records: {str(e)}")
                return 0
            
    def strategy_search(
            self,
            user_profile: Dict,
            limit: int = 5,
            metadata_filter: Optional[dict] = None
        ) -> pd.DataFrame:
        """Specialized structured search for Strategy: exact profile level matches, then rank by success_score."""
        
        with self.session_factory() as db:
            try:
                kb_size = db.query(Strategy).count()
                if kb_size == 0:
                    self.logger.warning("No strategies in KB")
                    return pd.DataFrame()
                    
                # Start query
                query = db.query(Strategy)
                    
                # Hard filter to find exact match on each profile level 
                for dim in ['tech_literacy', 'language_proficiency', 'emotional_state']:
                    if dim in user_profile and 'level' in user_profile[dim]:
                        level = user_profile[dim]['level']
                            
                        # Safer filter: Use SQL function for nested JSON (avoids index error)
                        query = query.filter(
                            func.jsonb_extract_path_text(Strategy.user_profile, dim, 'level') == level
                        )
                    
                # Apply metadata_filter if any
                if metadata_filter:
                    for key, value in metadata_filter.items():
                        query = query.filter(getattr(Strategy, key) == value)
                    
                # Rank by success_score DESC, limit
                query = query.order_by(Strategy.success_score.desc()).limit(limit)
                results = query.all()
                    
                if not results:
                    self.logger.warning("No matching strategies found")
                    return pd.DataFrame()
                    
                # Increment retrieval_count for selected
                for strategy in results:
                    db.execute(update(Strategy).where(Strategy.id == strategy.id).values(retrieval_count=Strategy.retrieval_count + 1))
                    
                records = []
                for strat in results:
                    record_data = {c.name: getattr(strat, c.name) for c in Strategy.__table__.columns}
                    records.append(record_data)
                    
                db.commit()
                df = pd.DataFrame(records)
                self.logger.info(f"Structured search returned {len(df)} strategies, ranked by success_score DESC")
                return df
            except Exception as e:
                self.logger.error(f"Error in strategy_search: {str(e)}")
                return pd.DataFrame()
                
    def retrieve_strategies(self, user_profile: Dict, top_k: int = 5, metadata_filter: Optional[Dict] = None) -> Tuple[List[Dict], List[float]]:
        """Retrieve strategy types using structured search."""
        try:
            if isinstance(user_profile, str):
                user_profile_dict = json.loads(user_profile)  # Parse if str
            elif isinstance(user_profile, dict):
                user_profile_dict = user_profile  # Use directly if dict
            else:
                raise ValueError("user_profile must be str (JSON) or dict")
                
            df = self.strategy_search(user_profile=user_profile, limit=top_k, metadata_filter=metadata_filter)
            results = []
            if not df.empty:
                for record in df.to_dict(orient="records"):
                    result = {
                        "strategy_type": record.get("strategy_type")
                        }
                    results.append(result)
            return results  # No distances since no embeddings
        except Exception as e:
            self.logger.error(f"Error retrieving strategies: {str(e)}")
            return []
                
    def prune_strategies(self, min_kb_size: int = 50) -> int:
        """Prune low-score, low-retrieval strategies only if knowledgebase is large enough."""
        with self.session_factory() as db:
            try:
                kb_size = db.query(Strategy).count() #ensure kb size is large enough
                if kb_size < min_kb_size:
                    self.logger.info(f"Skipping prune: KB size {kb_size} < {min_kb_size}")
                    return 0
                
                deleted = db.query(Strategy).filter(
                    Strategy.success_score < 0.6,
                    Strategy.retrieval_count < 5
                ).delete()
                db.commit()
                self.logger.info(f"Pruned {deleted} strategies")
                return deleted
            except Exception as e:
                db.rollback()
                self.logger.error(f"Error pruning strategies: {str(e)}")
                return 0
            
            
if __name__ == "__main__":
    from datetime import date
    from src.models.data_model import ScamReport, Base  # Import the model

    # Initialize DatabaseManager and create tables if not exist
    db_manager = DatabaseManager()
    Base.metadata.create_all(db_manager.engine)  # Ensure table exists

    # Initialize CRUD for ScamReport
    crud = CRUDOperations(ScamReport, db_manager.session_factory)

    # Sample data for single create
    sample_data = {
        "report_id": "test_report_1",
        "scam_incident_date": date(2023, 1, 1),
        "scam_report_date": date(2023, 1, 2),
        "scam_type": "phishing",
        "scam_approach_platform": "email",
        "scam_communication_platform": "phone",
        "scam_transaction_type": "wire",
        "scam_beneficiary_platform": "bank",
        "scam_beneficiary_identifier": "123456",
        "scam_contact_no": "+123456789",
        "scam_email": "scam@example.com",
        "scam_moniker": "Scammer",
        "scam_url_link": "http://scam.com",
        "scam_amount_lost": 100.0,
        "scam_incident_description": "Test description",
        "embedding": [0.1] * 384  # Dummy embedding vector
    }

    # Test single create
    created_record = crud.create(sample_data)
    if created_record:
        print(f"Created record: {created_record.report_id}")
    else:
        print("Failed to create record")

    # Test read
    read_record = crud.read("test_report_1")
    if read_record:
        print(f"Read record: {read_record.scam_type}")

    # Test update
    update_data = {"scam_amount_lost": 200.0, "scam_type": "updated_phishing"}
    updated_record = crud.update("test_report_1", update_data)
    if updated_record:
        print(f"Updated record: {updated_record.scam_amount_lost}")

    # Test read after update
    read_updated = crud.read("test_report_1")
    if read_updated:
        print(f"Read updated: {read_updated.scam_type}")

    # Test update embedding
    new_embedding = [0.2] * 384
    embedding_updated = crud.update_embedding("test_report_1", new_embedding)
    print(f"Embedding updated: {embedding_updated}")

    # Sample DataFrame for bulk create
    bulk_data = pd.DataFrame([
        {
            "report_id": "test_report_2",
            "scam_incident_date": date(2023, 2, 1),
            "scam_report_date": date(2023, 2, 2),
            "scam_type": "ecommerce",
            "scam_approach_platform": "web",
            "scam_communication_platform": "chat",
            "scam_transaction_type": "card",
            "scam_beneficiary_platform": "online",
            "scam_beneficiary_identifier": "789012",
            "scam_contact_no": "+987654321",
            "scam_email": "bulkscam@example.com",
            "scam_moniker": "BulkScammer",
            "scam_url_link": "http://bulkscam.com",
            "scam_amount_lost": 50.0,
            "scam_incident_description": "Bulk test description",
            "embedding": [0.3] * 384
        },
        {
            "report_id": "test_report_3",
            "scam_incident_date": date(2023, 3, 1),
            "scam_report_date": date(2023, 3, 2),
            "scam_type": "government",
            "scam_approach_platform": "social",
            "scam_communication_platform": "app",
            "scam_transaction_type": "transfer",
            "scam_beneficiary_platform": "wallet",
            "scam_beneficiary_identifier": "345678",
            "scam_contact_no": "+112233445",
            "scam_email": "bulkscam2@example.com",
            "scam_moniker": "BulkScammer2",
            "scam_url_link": "http://bulkscam2.com",
            "scam_amount_lost": 75.0,
            "scam_incident_description": "Bulk test description 2",
            "embedding": [0.4] * 384
        }
    ])

    # Test bulk create
    bulk_count = crud.create_bulk(bulk_data)
    print(f"Bulk created {bulk_count} records")

    # Test read one from bulk
    read_bulk = crud.read("test_report_2")
    if read_bulk:
        print(f"Read bulk record: {read_bulk.scam_type}")

    # Test bulk update embedding
    bulk_embeddings = {
        "test_report_2": [0.5] * 384,
        "test_report_3": [0.6] * 384
    }
    bulk_embedding_updated = crud.update_embedding_bulk(bulk_embeddings)
    print(f"Bulk embedding updated for {bulk_embedding_updated} records")

    # Delete the test records
    for test_id in ["test_report_1", "test_report_2", "test_report_3"]:
        deleted = crud.delete(test_id)
        print(f"Deleted {test_id}: {deleted}")

    # Verify deletion
    for test_id in ["test_report_1", "test_report_2", "test_report_3"]:
        read_deleted = crud.read(test_id)
        if not read_deleted:
            print(f"Confirmed deletion of {test_id}")
            
    #Test delete all (if needed)
    # deleted_count = crud.delete_all()
    # print(f"Deleted all {deleted_count} records")
    # # Verify no records left (optional)
    # all_records = crud.read_all()
    # print(f"Records after delete_all: {len(all_records)}")