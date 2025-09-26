from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os
import logging
from functools import lru_cache
from pathlib import Path
from config.logging_config import setup_logger, LogSettings

load_dotenv()

class DatabaseSettings(BaseModel):
    """Database configuration settings."""
    url: str = Field(..., description="PostgreSQL database connection URL")
    echo: bool = Field(default=False, description="Enable SQLAlchemy query logging")
    user: str = Field(..., description="PostgreSQL user")
    password: str = Field(..., description="PostgreSQL password")
    db_name: str = Field(..., description="PostgreSQL database name")
    port: int = Field(..., description="PostgreSQL port")

class VectorSettings(BaseModel):
    """Vector store configuration settings."""
    embedding_model: str = Field(default="all-MiniLM-L6-v2", description="Embedding model for vector generation")
    embedding_dimensions: int = Field(default=384, description="Embedding vector dimensions")
    scam_reports_table: str = Field(default="scam_reports", description="Table name for ScamReport table")
    strategy_table: str = Field(default="strategy", description="Table name for Strategy table")

class DataSettings(BaseModel):
    """Data file configuration settings."""

    # Scam report data paths
    scam_report_csv: str = Field(default="data/scam_report/scam_reports.csv", description="Path to scam report CSV file for training")
    scam_report_csv_processed: str = Field(default="data/scam_report/scam_report_processed.csv", description="Path to scam report embeddings CSV file for training")

    # Victim profile data paths
    full_scam_scenario_csv: str = Field(default="data/victim_profile/full_scam_scenario.csv", description="Path to combined scam report CSV file for training")
    victim_details_json: str = Field(default="data/victim_profile/victim_details.json", description="Path to victim details JSON file for simulations")
    victim_details_human_eval_json: str = Field(default="data/victim_profile/victim_details_human_eval.json", description="Path to victim details JSON file for small-scale simulations and human evaluation")
    
    # Person details data path
    person_details_csv: str = Field(default="data/person_details/person_details.csv", description="Path to victim details CSV file for database ingestion")

    # Strategy seed data path
    strategy_seed_json: str = Field(default="data/strategy/strategy_seed_augmented.json", description="Path to strategy seed JSON file")
    
    # Configuration file
    scam_templates_json: str = Field(default="config/scam_templates.json", description="Path to scam templates JSON configuration file")
    
    # Phase 2 Simulations
    
    #RAG invocation log 
    rag_invocation_rag_ie_autonomous: str = Field(default="simulations/phase_2/rag_ie/autonomous_rag_invocation.csv", description="Path to vanilla RAG autonomous RAG invocation log CSV file")
    rag_invocation_rag_ie_nonautonomous: str = Field(default="simulations/phase_2/rag_ie/nonautonomous_rag_invocation.csv", description="Path to vanilla RAG nonautonomous RAG invocation log CSV file")
    rag_invocation_profile_rag_ie_autonomous: str = Field(default="simulations/phase_2/profile_rag_ie/autonomous_rag_invocation.csv", description="Path to self-augmenting RAG autonomous RAG invocation log CSV file")
    rag_invocation_profile_rag_ie_nonautonomous: str = Field(default="simulations/phase_2/profile_rag_ie/non_autonomous_rag_invocation.csv", description="Path to self-augmenting RAG nonautonomous RAG invocation log CSV file")
    rag_invocation_profile_rag_ie_kb_autonomous: str = Field(default="simulations/phase_2/profile_rag_ie_kb/autonomous_rag_invocation.csv", description="Path to self-augmenting RAG autonomous RAG invocation log CSV file")
    rag_invocation_profile_rag_ie_kb_nonautonomous: str = Field(default="simulations/phase_2/profile_rag_ie_kb/non_autonomous_rag_invocation.csv", description="Path to self-augmenting RAG nonautonomous RAG invocation log CSV file")

    #Conversation history
    conversation_history_ie_autonomous: str = Field(default="simulations/phase_2/ie/autonomous_conversation_history.csv", description="Path to baseline IE autonomous conversation history CSV file")
    conversation_history_ie_nonautonomous: str = Field(default="simulations/phase_2/ie/nonautonomous_conversation_history.csv", description="Path to baseline IE nonautonomous conversation history CSV file")
    conversation_history_rag_ie_autonomous: str = Field(default="simulations/phase_2/rag_ie/autonomous_conversation_history.csv", description="Path to vanilla RAG autonomous conversation history CSV file")
    conversation_history_rag_ie_nonautonomous: str = Field(default="simulations/phase_2/rag_ie/nonautonomous_conversation_history.csv", description="Path to vanilla RAG nonautonomous conversation history CSV file")
    conversation_history_profile_rag_ie_autonomous: str = Field(default="simulations/phase_2/profile_rag_ie/autonomous_conversation_history.csv", description="Path to baseline profile_rag_ie autonomous conversation history CSV file")
    conversation_history_profile_rag_ie_nonautonomous: str = Field(default="simulations/phase_2/profile_rag_ie/nonautonomous_conversation_history.csv", description="Path to baseline profile_rag_ie nonautonomous conversation history CSV file")
    conversation_history_profile_rag_ie_kb_autonomous: str = Field(default="simulations/phase_2/profile_rag_ie_kb/autonomous_conversation_history.csv", description="Path to self-augmenting RAG autonomous conversation history CSV file")
    conversation_history_profile_rag_ie_kb_nonautonomous: str = Field(default="simulations/phase_2/profile_rag_ie_kb/nonautonomous_conversation_history.csv", description="Path to self-augmenting RAG nonautonomous conversation history CSV file")
    
    #ID files for conversation tracking
    id_file_ie_autonomous: str = Field(default="simulations/phase_2/locks/ie_autonomous_last_id.txt", description="Path to last conversation ID file for baseline IE autonomous simulations")
    id_file_ie_nonautonomous: str = Field(default="simulations/phase_2/locks/ie_nonautonomous_last_id.txt", description="Path to last conversation ID file for baseline IE non-autonomous simulations")
    id_file_rag_ie_autonomous: str = Field(default="simulations/phase_2/locks/rag_ie_autonomous_last_id.txt", description="Path to last conversation ID file for vanilla RAG autonomous simulations")
    id_file_rag_ie_nonautonomous: str = Field(default="simulations/phase_2/locks/rag_ie_nonautonomous_last_id.txt", description="Path to last conversation ID file for vanilla RAG non-autonomous simulations")
    id_file_profile_rag_ie_autonomous: str = Field(default="simulations/phase_2/locks/profile_rag_ie_autonomous_last_id.txt", description="Path to last conversation ID file for profile_rag_ie autonomous simulations")
    id_file_profile_rag_ie_nonautonomous: str = Field(default="simulations/phase_2/locks/profile_rag_ie_nonautonomous_last_id.txt", description="Path to last conversation ID file for profile_rag_ie non-autonomous simulations")
    id_file_profile_rag_ie_kb_autonomous: str = Field(default="simulations/phase_2/locks/profile_rag_ie_kb_autonomous_last_id.txt", description="Path to last conversation ID file for self-augmenting RAG autonomous simulations")
    id_file_profile_rag_ie_kb_nonautonomous: str = Field(default="simulations/phase_2/locks/profile_rag_ie_kb_nonautonomous_last_id.txt", description="Path to last conversation ID file for self-augmenting RAG non-autonomous simulations")

    evaluation_results_dir: str = Field(default="evaluation/results/", description="Base directory for evaluation results and outputs (e.g., plots, metrics CSVs).")
    
class AgentSettings(BaseModel):
    """Agent configuration settings."""
    ollama_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")

class Settings(BaseModel):
    """Main application configuration settings."""
    log: LogSettings
    database: DatabaseSettings
    vector: VectorSettings
    data: DataSettings
    agents: AgentSettings

@lru_cache()
def get_settings() -> Settings:
    """
    Load and cache application settings from environment variables.
    """
    logger = setup_logger("Settings", "setup")
    database_url = os.getenv("DATABASE_URL")
    postgres_user = os.getenv("POSTGRES_USER")
    postgres_password = os.getenv("POSTGRES_PASSWORD")
    postgres_db = os.getenv("POSTGRES_DB")
    postgres_port = os.getenv("POSTGRES_PORT")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    if not all([database_url, postgres_user, postgres_password, postgres_db, postgres_port]):
        missing = [k for k, v in {
            "DATABASE_URL": database_url,
            "POSTGRES_USER": postgres_user,
            "POSTGRES_PASSWORD": postgres_password,
            "POSTGRES_DB": postgres_db,
            "POSTGRES_PORT": postgres_port
        }.items() if not v]
        raise ValueError(f"Missing environment variables: {missing}")
    
    settings = Settings(
        log=LogSettings(directory=os.getenv("LOG_DIRECTORY", "logs")), #defaults to log if LOG_DIRECTORY is not set in .env
        database=DatabaseSettings(
            url=database_url,
            user=postgres_user,
            password=postgres_password,
            db_name=postgres_db,
            port=int(postgres_port),
            echo=False
        ),
        vector=VectorSettings(),
        data=DataSettings(),  
        agents=AgentSettings(ollama_base_url=ollama_base_url)
    )
    
    logging.info("Settings initialized successfully")
    return settings