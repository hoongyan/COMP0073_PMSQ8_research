import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from config.settings import get_settings
from config.logging_config import setup_logger
from src.models.response_model import UserProfile, RetrievalOutput, IEOutput, PipelineState, PoliceResponseSlots, ProcessedOutput
from src.agents.prompt import Prompt
from src.agents.tools import PoliceTools
from src.database.database_operations import DatabaseManager
from src.database.vector_operations import VectorStore

# Initialize settings and logger
settings = get_settings()
logger = setup_logger("RetrievalIEAgentTest", settings.log.subdirectories["agent"])

# Initialize LLM
def initialize_llm(schema):
    return ChatOllama(
        model="qwen2.5:7b",
        base_url=settings.agents.ollama_base_url,
        format="json",
        response_format={"type": "json_schema", "json_schema": schema}
    )

# RetrievalAgent logic
def retrieval_node(state: PipelineState) -> PipelineState:
    logger.debug(f"RetrievalAgent processing query: {state.query}")
    police_tools = PoliceTools()
    try:
        tool = police_tools.get_augmented_tools()[0]  # augmented_police_tool
        results = tool.invoke({
            "query": state.query,
            "top_k": 1,
            "conversation_id": 1,
            "llm_model": "qwen2.5:7b"
        })
        result_dict = json.loads(results) if isinstance(results, str) else results
        state.templates = RetrievalOutput(
            scam_report=[json.dumps(report) for report in result_dict.get("scam_reports", [])],
            strategy_type=[{"strategy_type": s["strategy_type"], "question": s["question"]} for s in result_dict.get("strategies", [])]
        )
        logger.debug(f"RetrievalAgent output: {state.templates}")
    except Exception as e:
        logger.error(f"RetrievalAgent error: {str(e)}")
        state.templates = RetrievalOutput(
            scam_report=[],
            strategy_type=[
                {"strategy_type": "ask with simple terms", "question": "Can you describe what happened?"},
                {"strategy_type": "empathetic", "question": "I'm sorry to hear that. How did the scammer contact you?"}
            ]
        )
    return state

# IEAgent logic
def ie_node(state: PipelineState) -> PipelineState:
    logger.debug(f"IEAgent processing query: {state.query}")
    llm = initialize_llm(schema=IEOutput.model_json_schema())
    prompt = ChatPromptTemplate.from_template(Prompt.template["ie"])
    try:
        response = llm.invoke(prompt.format_prompt(
            query=state.query,
            query_history=state.query_history[-2:],
            scam_reports=state.templates.scam_report
        ).to_messages())
        print("IEAgent Raw LLM response:", response.content)
        logger.debug(f"IEAgent Raw LLM response: {response.content}")
        output = json.loads(response.content)
        state.ie_output = IEOutput(**output)

    except Exception as e:
        logger.error(f"IEAgent error: {str(e)}")
        state.ie_output = IEOutput(
            scam_type="GOVERNMENT OFFICIALS IMPERSONATION" if "government official" in state.query.lower() else "",
            scam_approach_platform="CALL" if "call" in state.query.lower() else "",
            scam_specific_details={}
        )
    return state

# Test setup
query = "I received a call from someone claiming to be a government official."
query_history = []
state = PipelineState(
    query=query,
    query_history=query_history,
    user_profile=UserProfile(
        age_group="20-29",
        tech_literacy="moderate",
        language_proficiency="high",
        emotional_state="anxious"
    ),
    templates=RetrievalOutput(),
    ie_output=IEOutput(
        conversational_response="Can you describe what happened?",
        scam_specific_details={}
    ),
    output_json=ProcessedOutput(
        question="",
        user_profile=UserProfile()
    )
)

# Run test
logger.info("Starting RetrievalAgent and IEAgent test")
state = retrieval_node(state)
state = ie_node(state)
print(json.dumps({
    "query": query,
    "retrieval_output": state.templates.model_dump(),
    "ie_output": state.ie_output.model_dump(),
    "error": None if state.templates.scam_report or state.ie_output.conversational_response else "Failed to process"
}, indent=2))