import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from config.settings import get_settings
from config.logging_config import setup_logger
from src.models.response_model import UserProfile, RetrievalOutput
from src.agents.prompt import Prompt
from src.agents.baseline.profile_rag_ie_kb_agent import GraphState  # TypedDict definition

# Initialize settings and logger
settings = get_settings()
logger = setup_logger("UserProfileAgentTest", settings.log.subdirectories["agent"])

# Initialize LLM with UserProfile schema
def initialize_llm(schema):
    return ChatOllama(
        model="qwen2.5:7b",
        base_url=settings.agents.ollama_base_url,
        format="json",
        response_format={"type": "json_schema", "json_schema": schema}
    )

# UserProfileAgent logic (isolated node)
def user_profile_node(state: GraphState) -> GraphState:
    logger.debug(f"UserProfileAgent processing query: {state['query']}")
    llm = initialize_llm(schema=UserProfile.model_json_schema())
    prompt = ChatPromptTemplate.from_template(Prompt.template["user_profile"])
    try:
        response = llm.invoke(prompt.format_prompt(query=state['query'], query_history=state['query_history']).to_messages())
        logger.debug(f"Raw LLM response: {response.content}")
        state['user_profile'] = UserProfile(**json.loads(response.content))
        logger.debug(f"UserProfileAgent output: {state['user_profile']}")
    except Exception as e:
        logger.error(f"UserProfileAgent error: {str(e)}")
        state['user_profile'] = UserProfile()  # Default profile on error
    return state

# Test setup
query = "I received a call from someone claiming to be a government official."
query_history = []
state = {
    "query": query,
    "query_history": query_history,
    "user_profile": UserProfile().model_dump(),  # Start with default
    "templates": RetrievalOutput().model_dump(),  # Use default instance as dict
    "ie_output": None,
    "unfilled_slots": {"scam_incident_date": True},
    "prev_unfilled_slots": {},
    "conversation_id": 1,
    "rag_invoked": True,
    "target_slots": [],
    "questions": []
}

logger.info("Starting UserProfileAgent test")
result_state = user_profile_node(state)
print(json.dumps({
    "query": query,
    "user_profile": result_state["user_profile"].model_dump(),
    "error": None if result_state["user_profile"] != UserProfile() else "Failed to infer profile"
}, indent=2))