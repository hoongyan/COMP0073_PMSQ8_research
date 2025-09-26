import sys
import os
import json
from pathlib import Path
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agents.conversation_manager import ConversationManager, ConversationMode, ConversationType
from config.settings import get_settings
from config.logging_config import setup_logger

# Setup logging
settings = get_settings()
logger = setup_logger("BatchSimulation", settings.log.subdirectories["agent"])

# Constants
JSON_FILE = settings.data.victim_details_human_eval_json
INITIAL_QUERIES = ["Hi, I am a police AI assistant. How can I help you?"] 
POLICE_MODELS = [
    ("qwen2.5:7b", "Ollama"),
    ("gpt-4o-mini", "OpenAI"),
    ("granite3.2:8b", "Ollama"),
    ("mistral:7b", "Ollama")
]
VICTIM_MODELS = [("gpt-4o-mini", "OpenAI")]  # Always this
MAX_TURNS = 10
N_SIMULATIONS = 12  # 6 profiles * 2 replicates (cycling ensures each profile twice)


#Vanilla RAG
manager = ConversationManager(
    mode=ConversationMode.AUTONOMOUS,
    conversation_type=ConversationType.VANILLA_RAG,
    max_turns=15,
    json_file=JSON_FILE
    )

try:
     # Load profiles from JSON for test 
    with open(manager.json_file, "r") as f:
            test_profiles = json.loads(f.read())[:]  
        
    # police_models = [
    #         ("qwen2.5:7b", "Ollama"),
    #         ("gpt-4o-mini", "OpenAI"),
    #         ("granite3.2:8b", "Ollama"),
    #         ("mistral:7b", "Ollama")
    #         ] 

    # victim_models = [("gpt-4o-mini", "OpenAI")]
    batch_results = manager.batch_run_autonomous(
        n=N_SIMULATIONS,
        initial_queries=INITIAL_QUERIES,
        police_models=POLICE_MODELS,
        victim_models=VICTIM_MODELS, 
        profiles=test_profiles  
        )
    print("Batch simulation completed successfuly.")
    print("Batch Test Results:", json.dumps(batch_results, indent=2))
except Exception as e:
    manager.logger.error(f"Batch test error: {str(e)}", exc_info=True)
    raise
    
# #Self-Augmenting RAG
# manager = ConversationManager(
#     mode=ConversationMode.AUTONOMOUS,
#     conversation_type=ConversationType.SELF_AUGMENTING,  # Ensure this is set to use the right paths/agent
#     max_turns=15,
#     json_file=JSON_FILE
#     )

# try:
#     # Load test profiles (adjust slice for more variety)
#     with open(manager.json_file, "r") as f:
#         test_profiles = json.loads(f.read())[:]  
    
#     # police_models = [
#     #         ("qwen2.5:7b", "Ollama"),
#     #         ("gpt-4o-mini", "OpenAI"),
#     #         ("granite3.2:8b", "Ollama"),
#     #         ("mistral:7b", "Ollama")
#     #     ] 

#     # victim_models = [("gpt-4o-mini", "OpenAI")]
    
#     # Run batch simulation with Cartesian product
#     batch_results = manager.batch_run_autonomous(
#         n=N_SIMULATIONS, 
#         initial_queries=INITIAL_QUERIES,
#         police_models=POLICE_MODELS,
#         victim_models=VICTIM_MODELS,
#         profiles=test_profiles
#     )
#     print("Batch simulation completed successfuly.")
#     print("Batch Test Results:", json.dumps(batch_results, indent=2))
# except Exception as e:
#     manager.logger.error(f"Batch test error: {str(e)}", exc_info=True)
#     raise