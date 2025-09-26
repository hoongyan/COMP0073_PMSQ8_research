import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agents.conversation_manager import ConversationManager, ConversationMode, ConversationType
from config.settings import get_settings
from config.logging_config import setup_logger

# Setup logging
settings = get_settings()
logger = setup_logger("BatchSimulation", settings.log.subdirectories["agent"])

# Constants
JSON_FILE = settings.data.victim_details_json
INITIAL_QUERIES = ["Hi, I am a police AI assistant. How can I help you?"] 
POLICE_MODELS = [
    ("qwen2.5:7b", "Ollama"),
    ("gpt-4o-mini", "OpenAI"),
    ("granite3.2:8b", "Ollama"),
    ("mistral:7b", "Ollama")
]
VICTIM_MODELS = [("gpt-4o-mini", "OpenAI")]  # Always this
MAX_TURNS = 15
N_SIMULATIONS = 48  # 24 profiles * 2 replicates (cycling ensures each profile twice)

# List of all conversation types to simulate
CONVERSATION_TYPES = [
    ConversationType.IE,
    ConversationType.RAG_IE,
    ConversationType.PROFILE_RAG_IE,
    ConversationType.PROFILE_RAG_IE_KB
]

# Load profiles once (shared across all types)
with open(JSON_FILE, "r") as f:
    test_profiles = json.loads(f.read())[:]

# Run simulations for each conversation type
for conv_type in CONVERSATION_TYPES:
    print(f"\nStarting batch simulation for conversation type: {conv_type.value}")
    
    manager = ConversationManager(
        mode=ConversationMode.AUTONOMOUS,
        conversation_type=conv_type,
        max_turns=MAX_TURNS,
        json_file=JSON_FILE
    )
    
    try:
        batch_results = manager.batch_run_autonomous(
            n=N_SIMULATIONS,
            initial_queries=INITIAL_QUERIES,
            police_models=POLICE_MODELS,
            victim_models=VICTIM_MODELS, 
            profiles=test_profiles  
        )
        print(f"Batch simulation for {conv_type.value} completed successfully.")
        print(f"Batch Test Results for {conv_type.value}:", json.dumps(batch_results, indent=2))
    except Exception as e:
        manager.logger.error(f"Batch test error for {conv_type.value}: {str(e)}", exc_info=True)
        raise