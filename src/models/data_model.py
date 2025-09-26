from sqlalchemy import Column, String, Date, Float, Text, DateTime, CheckConstraint, Integer
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB
from datetime import datetime
from sqlalchemy import func

Base = declarative_base()

class ScamReport(Base):
    """
    Data model for scam_reports table in PostgreSQL with pgvector.
    Stores scam generic details. scam_incident_description stored as embeddings for similarity searches. 
    """
    __tablename__ = 'scam_reports'

    report_id = Column(String, primary_key=True, nullable=False)
    scam_incident_date = Column(Date, nullable=False)
    scam_report_date = Column(Date, nullable=False)
    scam_type = Column(String)
    scam_approach_platform = Column(String)
    scam_communication_platform = Column(String)
    scam_transaction_type = Column(String)
    scam_beneficiary_platform = Column(String)
    scam_beneficiary_identifier = Column(String)
    scam_contact_no = Column(String)
    scam_email = Column(String)
    scam_moniker = Column(String)
    scam_url_link = Column(String)
    scam_amount_lost = Column(Float)
    scam_incident_description = Column(String)
    embedding = Column(Vector(384))

class Strategy(Base):
    """
    Data model for strategy table in PostgreSQL with pgvector.
    Stores strategies used to guide profile_rag_ie_kb agent. 
    Knowledgebase Agent in pipeline uses this table for knowledgebase augmentation of interaction/communication strategies.
    """
    __tablename__ = 'strategy'

    id = Column(Integer, primary_key=True, autoincrement=True, nullable=False)
    strategy_type = Column(String,CheckConstraint('LENGTH(strategy_type) >= 3'), nullable=False)
    response = Column(Text, CheckConstraint('LENGTH(response) >= 5'), nullable=False,)
    success_score = Column(Float, CheckConstraint('success_score >= 0.7 AND success_score <= 1.0'), nullable=False,)
    user_profile = Column(JSONB, nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False, server_default=func.now())  # Use server default for timestamp
    retrieval_count = Column(Integer, nullable=False, default=0)  # For pruning low-utility strategies