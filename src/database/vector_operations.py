import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from typing import List, Optional, Dict, Tuple, Type
from sqlalchemy import text, update, func
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from config.settings import get_settings
from config.logging_config import setup_logger
from sqlalchemy.ext.declarative import DeclarativeMeta
from src.models.data_model import ScamReport, Strategy
import json  

class VectorStore:
    """Manages vector operations for pgvector-based similarity searches."""
    
    def __init__(self, session_factory: callable):
        """Initialize with session factory and embedding model."""
        self.settings = get_settings()
        self.logger = setup_logger("VectorStore", self.settings.log.subdirectories["database"])
        self.model = SentenceTransformer(self.settings.vector.embedding_model)
        self.session_factory = session_factory
        self.embedding_dimensions = self.settings.vector.embedding_dimensions
        self.logger.info("VectorStore initialized successfully")
    
    def get_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text."""
        try:
            text = text.replace("\n", " ") if isinstance(text, str) else json.dumps(text)  
            embedding = self.model.encode(text).tolist()
            self.logger.debug(f"Generated embedding for text: {text[:50]}...")
            return embedding
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            raise
        
    
    #Applies to tables with default 'embedding' column
    def similarity_search(
        self,
        query: str,
        model: Type[DeclarativeMeta],
        limit: int = 5,
        metadata_filter: Optional[dict] = None,
        embedding_column: str = 'embedding'
    ) -> pd.DataFrame:
        """Generic flat similarity search applicable to any model with an embedding column."""
        with self.session_factory() as db:
            try:
                record_count = db.query(model).count()
                if record_count == 0:
                    self.logger.warning(f"No records in {model.__tablename__}")
                    return pd.DataFrame()
                
                query_embedding = np.array(self.get_embedding(query), dtype=np.float32)
                emb_col = getattr(model, embedding_column)
                db_query = db.query(
                    model,
                    emb_col.cosine_distance(query_embedding).label('distance')
                ).filter(emb_col.isnot(None))
                
                if metadata_filter:
                    for key, value in metadata_filter.items():
                        db_query = db_query.filter(getattr(model, key) == value)
                
                db_query = db_query.order_by('distance').limit(limit)
                results = db_query.all()
                
                records = []
                for r in results:
                    record, distance = r
                    record_data = {c.name: getattr(record, c.name) for c in model.__table__.columns}
                    record_data['distance'] = distance
                    records.append(record_data)
                
                df = pd.DataFrame(records)
                self.logger.info(f"Generic similarity search returned {len(df)} results")
                return df
            except Exception as e:
                self.logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
                return pd.DataFrame()
    
 
    def retrieve_scam_reports(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict] = None) -> Tuple[List[Dict], List[float]]:
        """Retrieve scam reports using generic flat similarity search."""
        try:
            df = self.similarity_search(query=query, model=ScamReport, limit=top_k, metadata_filter=metadata_filter)
            results = []
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
                    }
                    results.append(result)
                distances = df["distance"].tolist() if "distance" in df else []
            return results, distances
        except Exception as e:
            self.logger.error(f"Error retrieving scam reports: {str(e)}")
            return [], []
    

if __name__ == "__main__":

    from src.database.database_operations import DatabaseManager, CRUDOperations
    db_manager = DatabaseManager()
    vector_store = VectorStore(db_manager.session_factory)
    strategy_crud = CRUDOperations(Strategy, db_manager.session_factory)
    sample_query = "I received a call from a person claiming to be the police."
    
    
    # Test retrieve_scam_reports
    scam_results, scam_dist = vector_store.retrieve_scam_reports(query=sample_query, top_k=3)
    print(f"Retrieved scam reports: {len(scam_results)}")
    
