# from typing import Dict, List, Optional
# from langgraph.graph import StateGraph, END
# from langgraph.prebuilt import create_react_agent
# from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
# from langchain_community.chat_message_histories import ChatMessageHistory
# from src.agents.llm_providers2 import LLMProvider
# from src.agents.tools2 import PoliceTools, VictimTools
# from src.models.response_model import PoliceResponse
# from src.config.prompt import Prompt
# from config.settings import get_settings
# from config.id_manager import IDManager
# import logging
# import json
# import csv
# import os
# from datetime import datetime
# from pathlib import Path
# import requests
# from filelock import FileLock
# from pydantic import BaseModel, ValidationError, Field
# import re

# class AgentState(BaseModel):
#     """State for the LangGraph RAG pipeline."""
#     query: str = Field(..., description="The input query for the agent")
#     conversation_id: int = Field(..., description="Unique conversation ID")
#     conversation_history: List[Dict[str, str | Dict]] = Field(default_factory=list, description="Conversation history with structured data")
#     rag_results: Optional[str] = Field(default=None, description="Results from RAG tool invocation")
#     llm_response: Optional[str] = Field(default=None, description="Raw LLM response")
#     structured_data: Optional[Dict] = Field(default=None, description="Structured output data")
#     rag_invoked: bool = Field(default=False, description="Whether RAG was invoked")

# class Agent:
#     """Base class for chatbot agents."""
    
#     def __init__(self, agent_id: str, llm_provider: str, model: str, max_turns: int):
#         """Initialize agent with configuration."""
#         self.settings = get_settings()
#         self.agent_id = agent_id
#         self.llm_provider = llm_provider
#         self.model = model
#         self.max_turns = max_turns
#         self.agent = None
#         self.llm_provider_instance = LLMProvider()
#         self._setup_logging()
#         self._validate_inputs()
    
#     def _setup_logging(self):
#         """Configure logging to write to agent-specific log file."""
#         log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
#         log_dir.mkdir(parents=True, exist_ok=True)
#         log_file = log_dir / "agent.log"
        
#         logging.basicConfig(
#             level=logging.DEBUG,
#             format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#             handlers=[
#                 logging.FileHandler(log_file, mode='a'),
#                 logging.StreamHandler()
#             ]
#         )
#         self.logger = logging.getLogger(__name__)
#         self.logger.info(f"Logging initialized to {log_file}")
    
#     def _validate_inputs(self):
#         """Validate LLM provider and model."""
#         supported_models = self.llm_provider_instance.get_supported_models()
#         if self.llm_provider not in supported_models:
#             raise ValueError(f"Invalid llm_provider: {self.llm_provider}. Must be one of {list(supported_models.keys())}")
#         if self.model not in supported_models[self.llm_provider]:
#             raise ValueError(f"Invalid model: {self.model}. Must be one of {supported_models[self.llm_provider]}")
        
#         if self.llm_provider == "Ollama":
#             try:
#                 response = requests.get(f"{self.settings.agents.ollama_base_url}/api/tags")
#                 if response.status_code != 200:
#                     raise ConnectionError(f"Ollama server not responding: {response.status_code}")
#                 models = response.json().get("models", [])
#                 if not any(m["name"] == self.model for m in models):
#                     raise ValueError(f"Model {self.model} not loaded in Ollama server")
#                 self.logger.debug(f"Ollama server is accessible and model {self.model} is loaded")
#             except Exception as e:
#                 self.logger.error(f"Failed to connect to Ollama server: {str(e)}")
#                 raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")
#         else:
#             self.logger.debug(f"Skipping Ollama server check for provider: {self.llm_provider}")

# class PoliceAgent(Agent):
#     def __init__(self, llm_provider: str = "OpenAI", model: str = "gpt-4o-mini", max_turns: int = 10):
#         super().__init__("police", llm_provider, model, max_turns)
#         self._initialize_agent()
    
#     def _initialize_agent(self):
#         """Initialize the police agent with LLM and tools."""
#         try:
#             llm = self.llm_provider_instance.get_structured_llm(
#                 provider=self.llm_provider, model=self.model, structured_model=PoliceResponse
#             )
#             self.logger.debug(f"Successfully initialized LLM: {self.llm_provider}/{self.model}")
#             tools = PoliceTools().get_tools()
#             self.logger.debug(f"Tools initialized: {[t.name for t in tools]}")
#             self.agent = create_react_agent(model=llm, tools=tools)
#             self.logger.debug("Successfully created ReAct agent")
#             self.logger.info(f"Police chatbot created with agent_id: {self.agent_id}, model: {self.model}")
#         except Exception as e:
#             self.logger.error(f"Failed to create police chatbot: {str(e)}", exc_info=True)
#             raise

# class VictimAgent(Agent):
#     """Victim chatbot agent with plain text output and no tools."""
    
#     def __init__(self, llm_provider: str = "OpenAI", model: str = "gpt-4o-mini", max_turns: int = 10):
#         super().__init__("victim", llm_provider, model, max_turns)
#         self._initialize_agent()
    
#     def _initialize_agent(self):
#         """Initialize the victim agent with LLM and no tools."""
#         try:
#             llm = self.llm_provider_instance.get_llm(provider=self.llm_provider, model=self.model)
#             self.logger.debug(f"Successfully initialized LLM: {self.llm_provider}/{self.model}")
#             tools = VictimTools().get_tools()
#             self.logger.debug(f"Tools initialized: {[t.name for t in tools] if tools else 'No tools'}")
#             self.agent = create_react_agent(model=llm, tools=tools)
#             self.logger.debug("Successfully created ReAct agent")
#             self.logger.info(f"Victim chatbot created with agent_id: {self.agent_id}, model: {self.model}")
#         except Exception as e:
#             self.logger.error(f"Failed to create victim chatbot: {str(e)}", exc_info=True)
#             raise

# class RAGPipeline:
#     """LangGraph pipeline to enforce mandatory RAG invocation for police agent responses."""
    
#     def __init__(self, police_agent: PoliceAgent, police_tools: PoliceTools):
#         self.police_agent = police_agent
#         self.police_tools = police_tools
#         self.logger = logging.getLogger(__name__)
#         self._build_graph()
    
#     def _build_graph(self):
#         """Build the LangGraph workflow for mandatory RAG invocation."""
#         workflow = StateGraph(AgentState)
        
#         workflow.add_node("invoke_rag", self._invoke_rag)
#         workflow.add_node("invoke_llm", self._invoke_llm)
#         workflow.add_node("parse_response", self._parse_response)
        
#         workflow.add_edge("invoke_rag", "invoke_llm")
#         workflow.add_edge("invoke_llm", "parse_response")
#         workflow.add_edge("parse_response", END)
        
#         workflow.set_entry_point("invoke_rag")
#         self.graph = workflow.compile()
    
#     def _invoke_rag(self, state: AgentState) -> AgentState:
#         """Invoke the RAG tool to retrieve scam reports."""
#         try:
#             tools = self.police_tools.get_tools()
#             retrieve_scam_reports = tools[0]
#             query = state.query
#             rag_result = retrieve_scam_reports.invoke({
#                 "query": query,
#                 "top_k": 5,
#                 "conversation_id": state.conversation_id,
#                 "llm_model": self.police_agent.model
#             })
#             self.logger.debug(f"RAG invocation result: {rag_result}")
#             state.rag_results = rag_result
#             state.rag_invoked = True
#             return state
#         except Exception as e:
#             self.logger.error(f"RAG invocation failed: {str(e)}")
#             state.rag_results = json.dumps({"error": f"RAG invocation failed: {str(e)}"})
#             state.rag_invoked = False
#             return state
    
#     def _invoke_llm(self, state: AgentState) -> AgentState:
#         """Invoke the LLM with RAG results included in the prompt."""
#         try:
#             history = ChatMessageHistory()
#             for msg in state.conversation_history:
#                 if msg["role"] == "victim":
#                     history.add_user_message(msg["content"])
#                 elif msg["role"] == "police":
#                     history.add_ai_message(msg["content"])
            
#             prompt = Prompt.template["baseline_police"]
#             if state.rag_results and state.rag_invoked:
#                 rag_data = json.loads(state.rag_results)
#                 if isinstance(rag_data, list) and rag_data:
#                     rag_summary = "\n\nBased on our records, we found similar scams involving:\n" + "\n".join(
#                         f"- {report.get('scam_type', 'unknown')} on {report.get('scam_approach_platform', 'unknown')} (e.g., {report.get('scam_incident_description', 'no details available')[:100]}...)"
#                         for report in rag_data[:3]  # Limit to 3 for brevity
#                     )
#                     prompt += rag_summary
#                 else:
#                     prompt += "\n\nNo relevant scam reports found in the database."
            
#             state_dict = {
#                 "messages": [
#                     SystemMessage(content=prompt),
#                     *history.messages,
#                     HumanMessage(content=state.query)
#                 ]
#             }
            
#             response = self.police_agent.agent.invoke(state_dict)
#             # Handle string response for Ollama models
#             if isinstance(response, str):
#                 try:
#                     response = json.loads(response)
#                 except json.JSONDecodeError:
#                     self.logger.warning(f"Failed to parse string response as JSON: {response}")
#                     state.llm_response = response
#                     return state
#             state.llm_response = response.get("messages", [])[-1].content if response.get("messages") else "I'm sorry, I need more details about the incident to assist you."
#             return state
#         except Exception as e:
#             self.logger.error(f"LLM invocation failed: {str(e)}")
#             state.llm_response = "I'm sorry, I need more details about the incident to assist you."
#             return state
        
    

#     def _parse_response(self, state: AgentState) -> AgentState:
#         """Parse and validate the LLM response to ensure structured output."""
#         try:
#             # Initialize all variables
#             scam_incident_date = ""
#             scam_type = "UNKNOWN"
#             scam_approach_platform = "UNKNOWN"
#             scam_communication_platform = "UNKNOWN"
#             scam_transaction_type = "UNKNOWN"
#             scam_beneficiary_platform = "UNKNOWN"
#             scam_beneficiary_identifier = "UNKNOWN"
#             scam_contact_no = "UNKNOWN"
#             scam_email = ""
#             scam_moniker = "UNKNOWN"
#             scam_url_link = ""
#             scam_amount_lost = 0.0
#             scam_incident_description = state.query
#             scam_specific_details = {}
            
#             # Extract details from victim input
#             query = state.query.lower()
#             if "email" in query:
#                 scam_type = "PHISHING"
#                 scam_approach_platform = "EMAIL"
#                 scam_communication_platform = "EMAIL"
#                 scam_specific_details = {
#                     "scam_subcategory": "EMAIL BANK PHISHING",
#                     "scam_impersonation_type": "LEGITIMATE ENTITY",
#                     "scam_first_impersonated_entity": "UNKNOWN",
#                     "scam_first_impersonated_entity_name": "UNKNOWN",
#                     "scam_phished_details": "",
#                     "scam_use_of_phished_details": "",
#                     "scam_pretext_for_phishing": ""
#                 }
#             elif "facebook" in query:
#                 scam_type = "ECOMMERCE" if "ticket" in query else "PHISHING"
#                 scam_approach_platform = "FACEBOOK"
#                 scam_communication_platform = "FACEBOOK"
#                 if "ticket" in query:
#                     scam_specific_details = {
#                         "scam_subcategory": "FAILURE TO DELIVER GOODS AND SERVICES",
#                         "scam_item_involved": "TAYLOR SWIFT CONCERT TICKET",
#                         "scam_item_type": "TICKETS"
#                     }
#                 else:
#                     scam_specific_details = {
#                         "scam_subcategory": "SOCIAL MEDIA PHISHING",
#                         "scam_impersonation_type": "",
#                         "scam_first_impersonated_entity": "",
#                         "scam_first_impersonated_entity_name": "",
#                         "scam_phished_details": "",
#                         "scam_use_of_phished_details": "",
#                         "scam_pretext_for_phishing": ""
#                     }
            
#             # Extract from conversation history
#             for msg in state.conversation_history:
#                 if msg["role"] == "victim":
#                     content = msg["content"].lower()
#                     if "facebook" in content:
#                         scam_approach_platform = "FACEBOOK"
#                         scam_communication_platform = "FACEBOOK"
#                     if "ticket" in content:
#                         scam_type = "ECOMMERCE"
#                         scam_specific_details = {
#                             "scam_subcategory": "FAILURE TO DELIVER GOODS AND SERVICES",
#                             "scam_item_involved": "TAYLOR SWIFT CONCERT TICKET",
#                             "scam_item_type": "TICKETS"
#                         }
#                     if "bank transfer" in content:
#                         scam_transaction_type = "BANK TRANSFER"
#                     match = re.search(r'\$(\d+\.?\d*)', content)
#                     if match:
#                         scam_amount_lost = float(match.group(1))
#                     match = re.search(r'(\d{4}-\d{2}-\d{2})', content)
#                     if match:
#                         scam_incident_date = match.group(1)
#                     if "wilkinsonthomas" in content:
#                         scam_moniker = "wilkinsonthomas"
#                     if "cimb" in content:
#                         scam_beneficiary_platform = "CIMB"
#                     match = re.search(r'(\d{8})', content)
#                     if match:
#                         scam_beneficiary_identifier = match.group(1)
            
#             try:
#                 parsed_response = json.loads(state.llm_response)
#                 if isinstance(parsed_response, dict) and "name" in parsed_response and parsed_response["name"] == "retrieve_scam_reports":
#                     self.logger.debug("Detected tool call in LLM response, using victim input with RAG defaults")
#                     rag_data = json.loads(state.rag_results) if state.rag_results else []
#                     if isinstance(rag_data, list) and rag_data:
#                         first_result = rag_data[0]
#                         # Use RAG to fill missing fields
#                         scam_type = scam_type if scam_type != "UNKNOWN" else first_result.get("scam_type", "UNKNOWN")
#                         scam_approach_platform = scam_approach_platform if scam_approach_platform != "UNKNOWN" else first_result.get("scam_approach_platform", "UNKNOWN")
#                         scam_communication_platform = scam_communication_platform if scam_communication_platform != "UNKNOWN" else first_result.get("scam_communication_platform", "UNKNOWN")
#                         scam_transaction_type = scam_transaction_type if scam_transaction_type != "UNKNOWN" else first_result.get("scam_transaction_type", "UNKNOWN")
#                         scam_beneficiary_platform = scam_beneficiary_platform if scam_beneficiary_platform != "UNKNOWN" else first_result.get("scam_beneficiary_platform", "UNKNOWN")
#                         scam_beneficiary_identifier = scam_beneficiary_identifier if scam_beneficiary_identifier != "UNKNOWN" else first_result.get("scam_beneficiary_identifier", "UNKNOWN")
#                         scam_contact_no = scam_contact_no if scam_contact_no != "UNKNOWN" else first_result.get("scam_contact_no", "UNKNOWN")
#                         scam_email = scam_email if scam_email else first_result.get("scam_email", "")
#                         scam_moniker = scam_moniker if scam_moniker != "UNKNOWN" else first_result.get("scam_moniker", "UNKNOWN")
#                         scam_url_link = scam_url_link if scam_url_link else first_result.get("scam_url_link", "")
#                         scam_amount_lost = scam_amount_lost if scam_amount_lost != 0.0 else first_result.get("scam_amount_lost", 0.0)
#                         scam_incident_description = scam_incident_description if scam_incident_description != state.query else first_result.get("scam_incident_description", state.query)
#                         if not scam_specific_details:
#                             specific_details = first_result.get("scam_specific_details", {})
#                             scam_specific_details = {k: str(v) for k, v in specific_details.items()}
#                     state.structured_data = {
#                         "rag_invoked": state.rag_invoked,
#                         "scam_incident_date": scam_incident_date,
#                         "scam_type": scam_type,
#                         "scam_approach_platform": scam_approach_platform,
#                         "scam_communication_platform": scam_communication_platform,
#                         "scam_transaction_type": scam_transaction_type,
#                         "scam_beneficiary_platform": scam_beneficiary_platform,
#                         "scam_beneficiary_identifier": scam_beneficiary_identifier,
#                         "scam_contact_no": scam_contact_no,
#                         "scam_email": scam_email,
#                         "scam_moniker": scam_moniker,
#                         "scam_url_link": scam_url_link,
#                         "scam_amount_lost": scam_amount_lost,
#                         "scam_incident_description": scam_incident_description,
#                         "scam_specific_details": scam_specific_details
#                     }
#                     state.llm_response = (
#                         f"I'm sorry to hear about your experience. Based on your input, it sounds like a {scam_type.lower()} scam "
#                         f"on {scam_approach_platform.lower()}. Please provide more details about your incident, such as any links "
#                         f"you clicked, messages you received, or payments you made, so we can assist with the investigation."
#                     )
#                 else:
#                     # Update with LLM response if available
#                     specific_details = parsed_response.get("scam_specific_details", {})
#                     if isinstance(specific_details, str):
#                         specific_details = {}
#                     scam_specific_details = specific_details if specific_details else scam_specific_details
#                     parsed_response["scam_specific_details"] = scam_specific_details
#                     parsed_response["scam_incident_date"] = scam_incident_date if scam_incident_date else parsed_response.get("scam_incident_date", "")
#                     parsed_response["scam_type"] = scam_type if scam_type != "UNKNOWN" else parsed_response.get("scam_type", "UNKNOWN")
#                     parsed_response["scam_approach_platform"] = scam_approach_platform if scam_approach_platform != "UNKNOWN" else parsed_response.get("scam_approach_platform", "UNKNOWN")
#                     parsed_response["scam_communication_platform"] = scam_communication_platform if scam_communication_platform != "UNKNOWN" else parsed_response.get("scam_communication_platform", "UNKNOWN")
#                     parsed_response["scam_transaction_type"] = scam_transaction_type if scam_transaction_type != "UNKNOWN" else parsed_response.get("scam_transaction_type", "UNKNOWN")
#                     parsed_response["scam_beneficiary_platform"] = scam_beneficiary_platform if scam_beneficiary_platform != "UNKNOWN" else parsed_response.get("scam_beneficiary_platform", "UNKNOWN")
#                     parsed_response["scam_beneficiary_identifier"] = scam_beneficiary_identifier if scam_beneficiary_identifier != "UNKNOWN" else parsed_response.get("scam_beneficiary_identifier", "UNKNOWN")
#                     parsed_response["scam_contact_no"] = scam_contact_no if scam_contact_no != "UNKNOWN" else parsed_response.get("scam_contact_no", "UNKNOWN")
#                     parsed_response["scam_email"] = scam_email if scam_email else parsed_response.get("scam_email", "")
#                     parsed_response["scam_moniker"] = scam_moniker if scam_moniker != "UNKNOWN" else parsed_response.get("scam_moniker", "UNKNOWN")
#                     parsed_response["scam_url_link"] = scam_url_link if scam_url_link else parsed_response.get("scam_url_link", "")
#                     parsed_response["scam_amount_lost"] = scam_amount_lost if scam_amount_lost != 0.0 else parsed_response.get("scam_amount_lost", 0.0)
#                     parsed_response["scam_incident_description"] = scam_incident_description if scam_incident_description != state.query else parsed_response.get("scam_incident_description", state.query)
#                     police_response = PoliceResponse(**parsed_response)
#                     state.structured_data = {"rag_invoked": state.rag_invoked}
#                     state.structured_data.update(police_response.dict(exclude={"conversational_response"}))
#                     state.llm_response = police_response.conversational_response.strip()
#             except json.JSONDecodeError:
#                 # Fallback for non-JSON responses
#                 self.logger.warning(f"Failed to parse LLM response as JSON, falling back to raw text: {state.llm_response}")
#                 state.structured_data = {
#                     "rag_invoked": state.rag_invoked,
#                     "scam_incident_date": scam_incident_date,
#                     "scam_type": scam_type,
#                     "scam_approach_platform": scam_approach_platform,
#                     "scam_communication_platform": scam_communication_platform,
#                     "scam_transaction_type": scam_transaction_type,
#                     "scam_beneficiary_platform": scam_beneficiary_platform,
#                     "scam_beneficiary_identifier": scam_beneficiary_identifier,
#                     "scam_contact_no": scam_contact_no,
#                     "scam_email": scam_email,
#                     "scam_moniker": scam_moniker,
#                     "scam_url_link": scam_url_link,
#                     "scam_amount_lost": scam_amount_lost,
#                     "scam_incident_description": scam_incident_description,
#                     "scam_specific_details": scam_specific_details
#                 }
#                 state.llm_response = (
#                     f"I'm sorry to hear about your experience. Based on your input, it sounds like a {scam_type.lower()} scam "
#                     f"on {scam_approach_platform.lower()}. Please provide more details about your incident, such as any links "
#                     f"you clicked, messages you received, or payments you made, so we can assist with the investigation."
#                 )
#         except ValidationError as e:
#             self.logger.warning(f"Failed to validate LLM response: {str(e)}. Raw response: {state.llm_response}")
#             state.structured_data = {
#                 "rag_invoked": state.rag_invoked,
#                 "scam_incident_date": scam_incident_date,
#                 "scam_type": scam_type,
#                 "scam_approach_platform": scam_approach_platform,
#                 "scam_communication_platform": scam_communication_platform,
#                 "scam_transaction_type": scam_transaction_type,
#                 "scam_beneficiary_platform": scam_beneficiary_platform,
#                 "scam_beneficiary_identifier": scam_beneficiary_identifier,
#                 "scam_contact_no": scam_contact_no,
#                 "scam_email": scam_email,
#                 "scam_moniker": scam_moniker,
#                 "scam_url_link": scam_url_link,
#                 "scam_amount_lost": scam_amount_lost,
#                 "scam_incident_description": scam_incident_description,
#                 "scam_specific_details": scam_specific_details
#             }
#             state.llm_response = (
#                 f"I'm sorry to hear about your experience. Based on your input, it sounds like a {scam_type.lower()} scam "
#                 f"on {scam_approach_platform.lower()}. Please provide more details about your incident, such as any links "
#                 f"you clicked, messages you received, or payments you made, so we can assist with the investigation."
#             )
#         return state
    
#     def run(self, query: str, conversation_id: int, conversation_history: List[Dict[str, str | Dict]]) -> Dict:
#         """Execute the RAG pipeline."""
#         try:
#             initial_state = AgentState(
#                 query=query,
#                 conversation_id=conversation_id,
#                 conversation_history=conversation_history
#             )
#             result = self.graph.invoke(initial_state.dict())
#             return {
#                 "response": result["llm_response"],
#                 "structured_data": result["structured_data"],
#                 "rag_invoked": result["rag_invoked"]
#             }
#         except Exception as e:
#             self.logger.error(f"Error in RAG pipeline execution: {str(e)}")
#             raise

# class ConversationManager:
#     """Manages conversation history and CSV storage."""
    
#     def __init__(self):
#         """Initialize conversation manager with settings and logging."""
#         self.settings = get_settings()
#         self.conversation_registry = {}
#         self.id_manager = IDManager()
#         self._setup_logging()
#         self.index_counter = self._load_last_index()
#         self.police_tools = PoliceTools()
    
#     def _setup_logging(self):
#         """Configure logging to write to conversation-specific log file."""
#         log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
#         log_dir.mkdir(parents=True, exist_ok=True)
#         log_file = log_dir / "conversation.log"
        
#         logging.basicConfig(
#             level=logging.DEBUG,
#             format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#             handlers=[
#                 logging.FileHandler(log_file, mode='a'),
#                 logging.StreamHandler()
#             ]
#         )
#         self.logger = logging.getLogger(__name__)
#         self.logger.info(f"Logging initialized to {log_file}")
    
#     def _load_last_index(self) -> int:
#         """Load the last used index from conversation_history.csv."""
#         csv_file = "conversation_history.csv"
#         max_index = 0
#         if os.path.exists(csv_file):
#             try:
#                 with FileLock(f"{csv_file}.lock"):
#                     with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                         reader = csv.DictReader(f)
#                         if "index" in reader.fieldnames:
#                             indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
#                             max_index = max(indices) if indices else 0
#                         else:
#                             self.logger.warning(f"CSV {csv_file} does not contain 'index' field. Starting with index 0.")
#                 self.logger.debug(f"Loaded max index {max_index} from {csv_file}")
#             except Exception as e:
#                 self.logger.error(f"Error reading index from CSV: {str(e)}")
#         return max_index
    
#     def _generate_conversation_id(self) -> int:
#         """Generate a unique integer conversation ID."""
#         return self.id_manager.get_next_id()
    
#     def _save_to_csv(self, conversation_id: int, conversation_type: str, messages: List[Dict], structured_data: Dict = None, llm_model: str = None):
#         """Save conversation messages to CSV, avoiding duplicates and consolidating structured data."""
#         csv_file = "conversation_history.csv"
#         file_exists = os.path.isfile(csv_file)
        
#         existing_entries = set()
#         if file_exists:
#             try:
#                 with FileLock(f"{csv_file}.lock"):
#                     with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                         reader = csv.DictReader(f)
#                         for row in reader:
#                             key = (row["conversation_id"], row["sender_type"], row["content"], row["timestamp"])
#                             existing_entries.add(key)
#             except Exception as e:
#                 self.logger.error(f"Error reading CSV for deduplication: {str(e)}")
#                 raise
        
#         with FileLock(f"{csv_file}.lock"):
#             try:
#                 with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
#                     writer = csv.writer(f)
#                     if not file_exists:
#                         writer.writerow([
#                             "index", "conversation_id", "conversation_type", "sender_type", "content", "timestamp",
#                             "llm_model", "scam_generic_details"
#                         ])
                    
#                     for msg in messages[-2:]:
#                         key = (str(conversation_id), msg["role"], msg["content"], msg["timestamp"])
#                         if key not in existing_entries:
#                             self.index_counter += 1
#                             row = [
#                                 str(self.index_counter),
#                                 str(conversation_id),
#                                 conversation_type,
#                                 msg["role"],
#                                 msg["content"],
#                                 msg["timestamp"],
#                                 llm_model or "",
#                                 json.dumps(structured_data, ensure_ascii=False) if msg["role"] == "police" and structured_data else ""
#                             ]
#                             writer.writerow(row)
#                             existing_entries.add(key)
#                             self.logger.debug(f"Wrote new message to CSV: conversation_id={conversation_id}, sender={msg['role']}")
#                         else:
#                             self.logger.debug(f"Skipped duplicate message: conversation_id={conversation_id}, sender={msg['role']}")
#             except Exception as e:
#                 self.logger.error(f"Error writing to CSV: {str(e)}")
#                 raise
    
#     def get_nonautonomous_response(
#         self, agent: Agent, query: str, prompt: str = None, conversation_id: int = None,
#         conversation_history: Optional[List[Dict[str, str | Dict]]] = None
#     ) -> Dict:
#         """Get a response from an agent for a non-autonomous conversation."""
#         if not query.strip():
#             self.logger.error("Query cannot be empty")
#             return {"error": "Query cannot be empty"}
        
#         if not agent.agent:
#             self.logger.error(f"Agent not initialized for agent_id: {agent.agent_id}")
#             return {"error": f"Agent not initialized for agent_id: {agent.agent_id}"}
        
#         effective_prompt = prompt.strip() if prompt else Prompt.template[f"baseline_{agent.agent_id}"]
#         if not effective_prompt:
#             self.logger.error("Prompt cannot be empty")
#             return {"error": "Prompt cannot be empty"}
        
#         conversation_id = conversation_id or self._generate_conversation_id()
#         conversation_history = conversation_history or []
        
#         valid_roles = {"victim", "police"}
#         for msg in conversation_history:
#             if msg.get("role") not in valid_roles:
#                 self.logger.error(f"Invalid role in conversation history: {msg.get('role')}")
#                 return {"error": f"Invalid role in conversation history: {msg.get('role')}"}
        
#         try:
#             if agent.agent_id == "police":
#                 pipeline = RAGPipeline(police_agent=agent, police_tools=self.police_tools)
#                 pipeline_result = pipeline.run(query, conversation_id, conversation_history)
#                 conversational_response = pipeline_result["response"]
#                 structured_data = pipeline_result["structured_data"]
#                 rag_invoked = pipeline_result["rag_invoked"]
#             else:
#                 history = ChatMessageHistory()
#                 for msg in conversation_history:
#                     if msg["role"] == "victim":
#                         history.add_user_message(msg["content"])
#                     elif msg["role"] == "police":
#                         history.add_ai_message(msg["content"])
                
#                 state = {
#                     "messages": [
#                         SystemMessage(content=effective_prompt),
#                         *history.messages,
#                         HumanMessage(content=query)
#                     ]
#                 }
                
#                 self.logger.debug(f"Invoking agent for query: {query}, conversation_id: {conversation_id}, agent_id: {agent.agent_id}")
#                 response = agent.agent.invoke(state)
#                 self.logger.debug(f"Raw agent response: {response}")
                
#                 messages = response.get("messages", [])
#                 ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
#                 conversational_response = ai_messages[-1] if ai_messages else "I'm sorry, I need more information to assist you."
#                 structured_data = {}
#                 rag_invoked = False
            
#             if not conversational_response.strip():
#                 self.logger.warning(f"Empty response from {agent.agent_id}. Using fallback response.")
#                 conversational_response = "I'm sorry, I need more details about the incident to assist you."
            
#             updated_history = conversation_history + [
#                 {"role": "victim", "content": query, "timestamp": datetime.now().isoformat()},
#                 {"role": agent.agent_id, "content": conversational_response, "timestamp": datetime.now().isoformat(), "structured_data": structured_data if agent.agent_id == "police" else {}}
#             ]
#             self.conversation_registry[conversation_id] = updated_history
#             self._save_to_csv(conversation_id, "non_autonomous", updated_history, structured_data, llm_model=agent.model)
            
#             self.logger.debug(f"Response generated and saved for agent_id: {agent.agent_id}, conversation_id: {conversation_id}")
#             return {
#                 "response": conversational_response,
#                 "structured_data": structured_data,
#                 "conversation_id": conversation_id,
#                 "conversation_history": updated_history,
#                 "conversation_type": "non_autonomous",
#                 "llm_model": agent.model
#             }
        
#         except Exception as e:
#             self.logger.error(f"Error in get_nonautonomous_response for agent_id {agent.agent_id}, conversation_id {conversation_id}: {str(e)}")
#             return {"error": f"Failed to get response: {str(e)}"}
    
#     def get_autonomous_response(
#         self,
#         police_agent: PoliceAgent,
#         victim_agent: VictimAgent,
#         police_prompt: str = None,
#         victim_prompt: str = None,
#         initial_query: str = "Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?",
#         conversation_id: int = None,
#         max_turns: int = 10
#     ) -> Dict:
#         """Simulate an autonomous conversation between police and victim agents."""
#         if not police_agent.agent or not victim_agent.agent:
#             self.logger.error(f"Agent not initialized: police={police_agent.agent_id}, victim={victim_agent.agent_id}")
#             return {"error": f"Agent not initialized: police={police_agent.agent_id}, victim={victim_agent.agent_id}"}
        
#         police_prompt = police_prompt.strip() if police_prompt and isinstance(police_prompt, str) else Prompt.template["baseline_police"]
#         victim_prompt = victim_prompt.strip() if victim_prompt and isinstance(victim_prompt, str) else Prompt.template["baseline_victim"]
#         if not police_prompt or not victim_prompt:
#             self.logger.error("Prompts cannot be empty")
#             return {"error": "Prompts cannot be empty"}
        
#         conversation_id = conversation_id or self._generate_conversation_id()
#         conversation_history = self.conversation_registry.get(conversation_id, [])
#         current_query = initial_query
#         last_messages = []
#         structured_data_history = []
        
#         try:
#             max_memory_turns = min(police_agent.max_turns, victim_agent.max_turns)
#             pipeline = RAGPipeline(police_agent=police_agent, police_tools=self.police_tools)
            
#             for turn in range(max_turns):
#                 # Victim response
#                 victim_history = ChatMessageHistory()
#                 for msg in conversation_history:
#                     if msg["role"] == "police":
#                         victim_history.add_user_message(msg["content"])
#                     elif msg["role"] == "victim":
#                         victim_history.add_ai_message(msg["content"])
                
#                 victim_state = {
#                     "messages": [
#                         SystemMessage(content=victim_prompt),
#                         *victim_history.messages,
#                         HumanMessage(content=current_query)
#                     ]
#                 }
                
#                 self.logger.debug(f"Invoking victim agent for query: {current_query}, conversation_id: {conversation_id}")
#                 try:
#                     victim_response = victim_agent.agent.invoke(victim_state)
#                     self.logger.debug(f"Raw victim response: {victim_response}")
#                 except Exception as e:
#                     self.logger.error(f"LLM invocation failed for victim agent: {str(e)}")
#                     victim_message = "I'm sorry, I'm a bit upset. Can you ask that again?"
#                     conversation_history.append({
#                         "role": "victim",
#                         "content": victim_message,
#                         "timestamp": datetime.now().isoformat()
#                     })
#                     self._save_to_csv(conversation_id, "autonomous", conversation_history, llm_model=victim_agent.model)
#                     break
                
#                 victim_messages = victim_response.get("messages", [])
#                 victim_ai_messages = [msg.content for msg in victim_messages if isinstance(msg, AIMessage)]
#                 victim_message = victim_ai_messages[-1] if victim_ai_messages else "I'm sorry, I'm a bit upset. Can you ask that again?"
                
#                 if not victim_message.strip():
#                     self.logger.warning(f"Empty victim response in turn {turn+1}. Using fallback response.")
#                     victim_message = "I'm sorry, I'm a bit upset. Can you ask that again?"
                
#                 conversation_history.append({
#                     "role": "victim",
#                     "content": victim_message,
#                     "timestamp": datetime.now().isoformat()
#                 })
#                 self._save_to_csv(conversation_id, "autonomous", conversation_history, llm_model=victim_agent.model)
                
#                 if "[END_CONVERSATION]" in victim_message or "thank you for your cooperation" in victim_message.lower():
#                     self.logger.info(f"Conversation ended: {victim_message}")
#                     break
                
#                 last_messages.append(victim_message)
#                 if len(last_messages) > 6:
#                     last_messages.pop(0)
#                     if len(set(last_messages)) <= 4:
#                         self.logger.warning(f"Detected potential repetition: {last_messages}")
#                         break
                
#                 # Police response using RAG pipeline
#                 pipeline_result = pipeline.run(victim_message, conversation_id, conversation_history)
#                 conversational_response = pipeline_result["response"]
#                 structured_data = pipeline_result["structured_data"]
                
#                 conversation_history.append({
#                     "role": "police",
#                     "content": conversational_response,
#                     "timestamp": datetime.now().isoformat(),
#                     "structured_data": structured_data
#                 })
#                 if structured_data:
#                     structured_data_history.append(structured_data)
                
#                 self._save_to_csv(conversation_id, "autonomous", conversation_history, structured_data, llm_model=police_agent.model)
                
#                 last_messages.append(conversational_response)
#                 if len(last_messages) > 6:
#                     last_messages.pop(0)
#                     if len(set(last_messages)) <= 4:
#                         self.logger.warning(f"Detected potential repetition: {last_messages}")
#                         break
                
#                 current_query = conversational_response
                
#                 if "thank you for your cooperation" in conversational_response.lower():
#                     self.logger.info("Police ended conversation")
#                     break
            
#             self.conversation_registry[conversation_id] = conversation_history
#             final_structured_data = structured_data_history[-1] if structured_data_history else {}
            
#             self.logger.debug(f"Autonomous conversation completed for conversation_id: {conversation_id}")
#             return {
#                 "status": "Conversation completed",
#                 "conversation_id": conversation_id,
#                 "conversation_history": conversation_history,
#                 "conversation_type": "autonomous",
#                 "structured_data": final_structured_data,
#                 "police_model": police_agent.model,
#                 "victim_model": victim_agent.model
#             }
        
#         except Exception as e:
#             self.logger.error(f"Error in get_autonomous_response for conversation_id {conversation_id}: {str(e)}")
#             return {"error": f"Failed to simulate conversation: {str(e)}"}


from typing import Dict, List, Optional
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from src.agents.remove.llm_providers2 import LLMProvider
from src.agents.remove.tools2 import PoliceTools, VictimTools
from src.models.response_model import PoliceResponse
from src.config.prompt import Prompt
from config.settings import get_settings
from config.id_manager import IDManager
import logging
import json
import csv
import os
from datetime import datetime
from pathlib import Path
import requests
from filelock import FileLock
from pydantic import BaseModel, ValidationError, Field
import re

class AgentState(BaseModel):
    query: str = Field(..., description="The input query for the agent")
    conversation_id: int = Field(..., description="Unique conversation ID")
    conversation_history: List[Dict[str, str | Dict]] = Field(default_factory=list, description="Conversation history with structured data")
    rag_results: Optional[str] = Field(default=None, description="Results from RAG tool invocation")
    llm_response: Optional[str] = Field(default=None, description="Raw LLM response")
    structured_data: Optional[Dict] = Field(default=None, description="Structured output data")
    rag_invoked: bool = Field(default=False, description="Whether RAG was invoked")

class Agent:
    def __init__(self, agent_id: str, llm_provider: str, model: str, max_turns: int):
        self.settings = get_settings()
        self.agent_id = agent_id
        self.llm_provider = llm_provider
        self.model = model
        self.max_turns = max_turns
        self.agent = None
        self.llm_provider_instance = LLMProvider()
        self._setup_logging()
        self._validate_inputs()
    
    def _setup_logging(self):
        log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "agent.log"
        
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='a'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized to {log_file}")
    
    def _validate_inputs(self):
        supported_models = self.llm_provider_instance.get_supported_models()
        if self.llm_provider not in supported_models:
            raise ValueError(f"Invalid llm_provider: {self.llm_provider}. Must be one of {list(supported_models.keys())}")
        if self.model not in supported_models[self.llm_provider]:
            raise ValueError(f"Invalid model: {self.model}. Must be one of {supported_models[self.llm_provider]}")
        
        if self.llm_provider == "Ollama":
            try:
                response = requests.get(f"{self.settings.agents.ollama_base_url}/api/tags")
                if response.status_code != 200:
                    raise ConnectionError(f"Ollama server not responding: {response.status_code}")
                models = response.json().get("models", [])
                if not any(m["name"] == self.model for m in models):
                    raise ValueError(f"Model {self.model} not loaded in Ollama server")
                self.logger.debug(f"Ollama server is accessible and model {self.model} is loaded")
            except Exception as e:
                self.logger.error(f"Failed to connect to Ollama server: {str(e)}")
                raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")

class PoliceAgent(Agent):
    def __init__(self, llm_provider: str = "Ollama", model: str = "llama3.2", max_turns: int = 10):
        super().__init__("police", llm_provider, model, max_turns)
        self._initialize_agent()
    
    def _initialize_agent(self):
        try:
            llm = self.llm_provider_instance.get_structured_llm(
                provider=self.llm_provider, model=self.model, structured_model=PoliceResponse
            )
            self.logger.debug(f"Successfully initialized LLM: {self.llm_provider}/{self.model}")
            tools = PoliceTools().get_tools()
            self.logger.debug(f"Tools initialized: {[t.name for t in tools]}")
            self.agent = create_react_agent(model=llm, tools=tools)
            self.logger.debug("Successfully created ReAct agent")
            self.logger.info(f"Police chatbot created with agent_id: {self.agent_id}, model: {self.model}")
        except Exception as e:
            self.logger.error(f"Failed to create police chatbot: {str(e)}", exc_info=True)
            raise

class VictimAgent(Agent):
    def __init__(self, llm_provider: str = "Ollama", model: str = "llama3.2", max_turns: int = 10):
        super().__init__("victim", llm_provider, model, max_turns)
        self._initialize_agent()
    
    def _initialize_agent(self):
        try:
            llm = self.llm_provider_instance.get_llm(provider=self.llm_provider, model=self.model)
            self.logger.debug(f"Successfully initialized LLM: {self.llm_provider}/{self.model}")
            tools = VictimTools().get_tools()
            self.logger.debug(f"Tools initialized: {[t.name for t in tools] if tools else 'No tools'}")
            self.agent = create_react_agent(model=llm, tools=tools)
            self.logger.debug("Successfully created ReAct agent")
            self.logger.info(f"Victim chatbot created with agent_id: {self.agent_id}, model: {self.model}")
        except Exception as e:
            self.logger.error(f"Failed to create victim chatbot: {str(e)}", exc_info=True)
            raise

class RAGPipeline:
    def __init__(self, police_agent: PoliceAgent, police_tools: PoliceTools):
        self.police_agent = police_agent
        self.police_tools = police_tools
        self.logger = logging.getLogger(__name__)
        self._build_graph()
    
    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("invoke_rag", self._invoke_rag)
        workflow.add_node("invoke_llm", self._invoke_llm)
        workflow.add_node("parse_response", self._parse_response)
        workflow.add_edge("invoke_rag", "invoke_llm")
        workflow.add_edge("invoke_llm", "parse_response")
        workflow.add_edge("parse_response", END)
        workflow.set_entry_point("invoke_rag")
        self.graph = workflow.compile()
    
    def _invoke_rag(self, state: AgentState) -> AgentState:
        try:
            tools = self.police_tools.get_tools()
            retrieve_scam_reports = tools[0]
            query = state.query
            self.logger.debug(f"Invoking RAG tool with query: {query}")
            rag_result = retrieve_scam_reports.invoke({
                "query": query,
                "top_k": 5,
                "conversation_id": state.conversation_id,
                "llm_model": self.police_agent.model
            })
            self.logger.info(f"RAG tool invoked successfully: {rag_result}")
            state.rag_results = rag_result
            state.rag_invoked = True
            return state
        except Exception as e:
            self.logger.error(f"RAG invocation failed: {str(e)}")
            state.rag_results = json.dumps({"error": f"RAG invocation failed: {str(e)}"})
            state.rag_invoked = False
            return state
    
    def _invoke_llm(self, state: AgentState) -> AgentState:
        try:
            history = ChatMessageHistory()
            for msg in state.conversation_history:
                if msg["role"] == "victim":
                    history.add_user_message(msg["content"])
                elif msg["role"] == "police":
                    history.add_ai_message(msg["content"])
            
            prompt = Prompt.template["baseline_police"]
            if state.rag_results and state.rag_invoked:
                try:
                    rag_data = json.loads(state.rag_results)
                    if isinstance(rag_data, list) and rag_data:
                        rag_summary = "\n\nBased on our records, we found similar scams involving:\n" + "\n".join(
                            f"- {report.get('scam_type', 'unknown')} on {report.get('scam_approach_platform', 'unknown')} (e.g., {report.get('scam_incident_description', 'no details available')[:100]}...)"
                            for report in rag_data[:3]
                        )
                        prompt += rag_summary
                    else:
                        prompt += "\n\nNo relevant scam reports found in the database."
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse RAG results")
                    prompt += "\n\nNo relevant scam reports found in the database."
            
            state_dict = {
                "messages": [
                    SystemMessage(content=prompt),
                    *history.messages,
                    HumanMessage(content=state.query)
                ]
            }
            
            response = self.police_agent.agent.invoke(state_dict)
            if isinstance(response, str):
                try:
                    response = json.loads(response)
                except json.JSONDecodeError:
                    self.logger.warning(f"Failed to parse string response as JSON: {response}")
                    state.llm_response = response
                    return state
            state.llm_response = response.get("messages", [])[-1].content if response.get("messages") else "I'm sorry, I need more details about the incident to assist you."
            return state
        except Exception as e:
            self.logger.error(f"LLM invocation failed: {str(e)}")
            state.llm_response = "I'm sorry, I need more details about the incident to assist you."
            return state
    
    def _parse_response(self, state: AgentState) -> AgentState:
        try:
            # Initialize structured data with defaults
            structured_data = {
                "conversational_response": "I'm sorry, I need more details about the incident to assist you.",
                "scam_incident_date": "",
                "scam_type": "UNKNOWN",
                "scam_approach_platform": "UNKNOWN",
                "scam_communication_platform": "UNKNOWN",
                "scam_transaction_type": "UNKNOWN",
                "scam_beneficiary_platform": "UNKNOWN",
                "scam_beneficiary_identifier": "UNKNOWN",
                "scam_contact_no": "UNKNOWN",
                "scam_email": "",
                "scam_moniker": "UNKNOWN",
                "scam_url_link": "",
                "scam_amount_lost": 0.0,
                "scam_incident_description": state.query,
                "scam_specific_details": {},
                "rag_invoked": state.rag_invoked
            }
            
            # Extract details from victim input and conversation history
            sources = [state.query] + [msg["content"] for msg in state.conversation_history if msg["role"] == "victim"]
            for content in sources:
                content_lower = content.lower()
                
                # Extract scam type
                if any(kw in content_lower for kw in ["phishing", "email", "sms", "login", "credential"]):
                    structured_data["scam_type"] = "PHISHING"
                elif any(kw in content_lower for kw in ["ecommerce", "ticket", "product", "non-delivery", "purchase"]):
                    structured_data["scam_type"] = "ECOMMERCE"
                elif any(kw in content_lower for kw in ["investment", "crypto", "bitcoin"]):
                    structured_data["scam_type"] = "INVESTMENT"
                elif any(kw in content_lower for kw in ["government", "irs", "police", "authority"]):
                    structured_data["scam_type"] = "GOVERNMENT IMPERSONATION"
                
                # Extract platforms
                platforms = ["facebook", "whatsapp", "instagram", "email", "sms", "call", "telegram"]
                for platform in platforms:
                    if platform in content_lower:
                        structured_data["scam_approach_platform"] = platform.upper()
                        structured_data["scam_communication_platform"] = platform.upper()
                
                # Extract transaction type
                if "bank transfer" in content_lower:
                    structured_data["scam_transaction_type"] = "BANK TRANSFER"
                elif any(kw in content_lower for kw in ["cryptocurrency", "bitcoin", "crypto"]):
                    structured_data["scam_transaction_type"] = "CRYPTOCURRENCY"
                elif "credit card" in content_lower:
                    structured_data["scam_transaction_type"] = "CREDIT CARD"
                
                # Extract amount lost
                amount_match = re.search(r'\$\d+\.\d{2}', content_lower)
                if amount_match:
                    structured_data["scam_amount_lost"] = float(amount_match.group(0).replace('$', ''))
                
                # Extract date
                date_match = re.search(r'\d{4}-\d{2}-\d{2}', content_lower)
                if date_match:
                    structured_data["scam_incident_date"] = date_match.group(0)
                elif "january" in content_lower:
                    structured_data["scam_incident_date"] = "2025-01-01"  # Example default
                elif "february" in content_lower:
                    structured_data["scam_incident_date"] = "2025-02-01"
                # Add other month mappings as needed
                
                # Extract moniker
                moniker_match = re.search(r"'[A-Za-z0-9_]{4,}'", content_lower)
                if moniker_match:
                    structured_data["scam_moniker"] = moniker_match.group(0).strip("'")
                
                # Extract beneficiary platform
                banks = ["cimb", "dbs", "scb", "trust", "hsbc", "gxs", "citibank"]
                for bank in banks:
                    if bank in content_lower:
                        structured_data["scam_beneficiary_platform"] = bank.upper()
                
                # Extract beneficiary identifier
                account_match = re.search(r'\b\d{6,12}\b', content_lower)
                if account_match:
                    structured_data["scam_beneficiary_identifier"] = account_match.group(0)
                
                # Extract email
                email_match = re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', content_lower)
                if email_match:
                    structured_data["scam_email"] = email_match.group(0)
                
                # Extract URL
                url_match = re.search(r'https?://[^\s]+', content_lower)
                if url_match:
                    structured_data["scam_url_link"] = url_match.group(0)
                
                # Set scam-specific details based on scam type
                if structured_data["scam_type"] == "PHISHING":
                    structured_data["scam_specific_details"] = {
                        "scam_subcategory": "PHISHING",
                        "scam_impersonation_type": "LEGITIMATE ENTITY" if any(kw in content_lower for kw in ["bank", "government"]) else "",
                        "scam_first_impersonated_entity": "",
                        "scam_first_impersonated_entity_name": "",
                        "scam_phished_details": "",
                        "scam_use_of_phished_details": "",
                        "scam_pretext_for_phishing": ""
                    }
                elif structured_data["scam_type"] == "ECOMMERCE":
                    structured_data["scam_specific_details"] = {
                        "scam_subcategory": "FAILURE TO DELIVER GOODS AND SERVICES" if any(kw in content_lower for kw in ["ticket", "non-delivery"]) else "ECOMMERCE",
                        "scam_item_involved": "TICKET" if "ticket" in content_lower else "",
                        "scam_item_type": "TICKETS" if "ticket" in content_lower else ""
                    }
                elif structured_data["scam_type"] == "INVESTMENT":
                    structured_data["scam_specific_details"] = {
                        "scam_subcategory": "INVESTMENT SCAM",
                        "scam_investment_type": "CRYPTOCURRENCY" if any(kw in content_lower for kw in ["crypto", "bitcoin"]) else ""
                    }
                elif structured_data["scam_type"] == "GOVERNMENT IMPERSONATION":
                    structured_data["scam_specific_details"] = {
                        "scam_subcategory": "",
                        "scam_impersonation_type": "",
                        "scam_first_impersonated_entity": "",
                        "scam_first_impersonated_entity_name": "",
                        "scam_second_impersonated_entity": "",
                        "scam_second_impersonated_entity_name": ""
                    }
            
            # Use RAG results only for missing fields
            if state.rag_results and state.rag_invoked:
                try:
                    rag_data = json.loads(state.rag_results)
                    if isinstance(rag_data, list) and rag_data:
                        first_result = rag_data[0]
                        for key in ["scam_type", "scam_approach_platform", "scam_communication_platform",
                                    "scam_transaction_type", "scam_beneficiary_platform",
                                    "scam_beneficiary_identifier", "scam_contact_no", "scam_email",
                                    "scam_moniker", "scam_url_link", "scam_amount_lost", "scam_incident_date"]:
                            if structured_data[key] in ["", "UNKNOWN", 0.0]:
                                structured_data[key] = first_result.get(key, structured_data[key])
                        if not structured_data["scam_specific_details"]:
                            structured_data["scam_specific_details"] = first_result.get("scam_specific_details", {})
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse RAG results")
            
            # Try to parse LLM response as JSON, prioritizing victim input
            try:
                parsed_response = json.loads(state.llm_response)
                if isinstance(parsed_response, dict) and "name" not in parsed_response:  # Exclude tool calls
                    for key, value in parsed_response.items():
                        if key in structured_data and structured_data[key] in ["", "UNKNOWN", 0.0]:
                            structured_data[key] = value
                    if "scam_specific_details" in parsed_response and not structured_data["scam_specific_details"]:
                        structured_data["scam_specific_details"] = parsed_response["scam_specific_details"]
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse LLM response as JSON: {state.llm_response}")
            
            # Update conversational response with extracted details
            scam_type = structured_data["scam_type"].lower() if structured_data["scam_type"] != "UNKNOWN" else "possible scam"
            platform = structured_data["scam_approach_platform"].lower() if structured_data["scam_approach_platform"] != "UNKNOWN" else "unknown platform"
            structured_data["conversational_response"] = (
                f"I'm sorry to hear about your experience. It sounds like a {scam_type} on {platform}. "
                f"Please provide more details about your incident, such as any links you clicked, "
                f"messages you received, or payments you made, to assist with the investigation."
            )
            
            # Validate structured data
            try:
                police_response = PoliceResponse(**structured_data)
                state.structured_data = {"rag_invoked": state.rag_invoked}
                state.structured_data.update(police_response.dict(exclude={"conversational_response"}))
                state.llm_response = police_response.conversational_response
            except ValidationError as e:
                self.logger.error(f"Failed to validate structured data: {str(e)}")
                state.structured_data = structured_data
            
            return state
        except Exception as e:
            self.logger.error(f"Error in parsing response: {str(e)}")
            state.structured_data = structured_data
            return state
    
    def run(self, query: str, conversation_id: int, conversation_history: List[Dict[str, str | Dict]]) -> Dict:
        try:
            initial_state = AgentState(
                query=query,
                conversation_id=conversation_id,
                conversation_history=conversation_history
            )
            result = self.graph.invoke(initial_state.dict())
            return {
                "response": result["llm_response"],
                "structured_data": result["structured_data"],
                "rag_invoked": result["rag_invoked"]
            }
        except Exception as e:
            self.logger.error(f"Error in RAG pipeline execution: {str(e)}")
            raise

class ConversationManager:
    def __init__(self):
        self.settings = get_settings()
        self.conversation_registry = {}
        self.id_manager = IDManager()
        self._setup_logging()
        self.index_counter = self._load_last_index()
        self.police_tools = PoliceTools()
    
    def _setup_logging(self):
        log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "conversation.log"
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='a'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized to {log_file}")
    
    def _load_last_index(self) -> int:
        csv_file = "conversation_history.csv"
        max_index = 0
        if os.path.exists(csv_file):
            try:
                with FileLock(f"{csv_file}.lock"):
                    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        if "index" in reader.fieldnames:
                            indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
                            max_index = max(indices) if indices else 0
                        else:
                            self.logger.warning(f"CSV {csv_file} does not contain 'index' field. Starting with index 0.")
                self.logger.debug(f"Loaded max index {max_index} from {csv_file}")
            except Exception as e:
                self.logger.error(f"Error reading index from CSV: {str(e)}")
        return max_index
    
    def _generate_conversation_id(self) -> int:
        return self.id_manager.get_next_id()
    
    def _save_to_csv(self, conversation_id: int, conversation_type: str, messages: List[Dict], structured_data: Dict = None, llm_model: str = None):
        csv_file = "conversation_history.csv"
        file_exists = os.path.isfile(csv_file)
        existing_entries = set()
        if file_exists:
            try:
                with FileLock(f"{csv_file}.lock"):
                    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            key = (row["conversation_id"], row["sender_type"], row["content"], row["timestamp"])
                            existing_entries.add(key)
            except Exception as e:
                self.logger.error(f"Error reading CSV for deduplication: {str(e)}")
                raise
        
        with FileLock(f"{csv_file}.lock"):
            try:
                with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "index", "conversation_id", "conversation_type", "sender_type", "content", "timestamp",
                            "llm_model", "scam_generic_details"
                        ])
                    new_entries = []
                    for msg in messages[-2:]:
                        key = (str(conversation_id), msg["role"], msg["content"], msg["timestamp"])
                        if key not in existing_entries:
                            self.index_counter += 1
                            structured_data_str = json.dumps(structured_data, ensure_ascii=False) if msg["role"] == "police" and structured_data else ""
                            row = [
                                str(self.index_counter),
                                str(conversation_id),
                                conversation_type,
                                msg["role"],
                                msg["content"],
                                msg["timestamp"],
                                llm_model or "",
                                structured_data_str
                            ]
                            new_entries.append((key, row))
                    for key, row in new_entries:
                        writer.writerow(row)
                        existing_entries.add(key)
                        self.logger.debug(f"Wrote new message to CSV: conversation_id={conversation_id}, sender={row[3]}")
            except Exception as e:
                self.logger.error(f"Error writing to CSV: {str(e)}")
                raise
    
    def get_nonautonomous_response(
        self, agent: Agent, query: str, prompt: str = None, conversation_id: int = None,
        conversation_history: Optional[List[Dict[str, str | Dict]]] = None
    ) -> Dict:
        if not query.strip():
            self.logger.error("Query cannot be empty")
            return {"error": "Query cannot be empty"}
        
        if not agent.agent:
            self.logger.error(f"Agent not initialized for agent_id: {agent.agent_id}")
            return {"error": f"Agent not initialized for agent_id: {agent.agent_id}"}
        
        effective_prompt = prompt.strip() if prompt else Prompt.template[f"baseline_{agent.agent_id}"]
        if not effective_prompt:
            self.logger.error("Prompt cannot be empty")
            return {"error": "Prompt cannot be empty"}
        
        conversation_id = conversation_id or self._generate_conversation_id()
        conversation_history = conversation_history or []
        
        valid_roles = {"victim", "police"}
        for msg in conversation_history:
            if msg.get("role") not in valid_roles:
                self.logger.error(f"Invalid role in conversation history: {msg.get('role')}")
                return {"error": f"Invalid role in conversation history: {msg.get('role')}"}
        
        try:
            if agent.agent_id == "police":
                pipeline = RAGPipeline(police_agent=agent, police_tools=self.police_tools)
                pipeline_result = pipeline.run(query, conversation_id, conversation_history)
                conversational_response = pipeline_result["response"]
                structured_data = pipeline_result["structured_data"]
                rag_invoked = pipeline_result["rag_invoked"]
            else:
                history = ChatMessageHistory()
                for msg in conversation_history:
                    if msg["role"] == "victim":
                        history.add_user_message(msg["content"])
                    elif msg["role"] == "police":
                        history.add_ai_message(msg["content"])
                
                state = {
                    "messages": [
                        SystemMessage(content=effective_prompt),
                        *history.messages,
                        HumanMessage(content=query)
                    ]
                }
                
                self.logger.debug(f"Invoking agent for query: {query}, conversation_id: {conversation_id}, agent_id: {agent.agent_id}")
                response = agent.agent.invoke(state)
                self.logger.debug(f"Raw agent response: {response}")
                
                messages = response.get("messages", [])
                ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
                conversational_response = ai_messages[-1] if ai_messages else "I'm sorry, I'm a bit upset. Can you ask that again?"
                structured_data = {}
                rag_invoked = False
            
            if not conversational_response.strip():
                self.logger.warning(f"Empty response from {agent.agent_id}. Using fallback response.")
                conversational_response = "I'm sorry, I need more details about the incident to assist you."
            
            updated_history = conversation_history[:]
            new_victim_msg = {"role": "victim", "content": query, "timestamp": datetime.now().isoformat()}
            # Check for duplicates based on role and content only
            if not any(msg["role"] == new_victim_msg["role"] and msg["content"] == new_victim_msg["content"] for msg in updated_history):
                updated_history.append(new_victim_msg)
            
            new_agent_msg = {
                "role": agent.agent_id,
                "content": conversational_response,
                "timestamp": datetime.now().isoformat(),
                "structured_data": structured_data if agent.agent_id == "police" else {}
            }
            if not any(msg["role"] == new_agent_msg["role"] and msg["content"] == new_agent_msg["content"] for msg in updated_history):
                updated_history.append(new_agent_msg)
            
            self.conversation_registry[conversation_id] = updated_history
            self._save_to_csv(conversation_id, "non_autonomous", updated_history, structured_data, llm_model=agent.model)
            
            self.logger.debug(f"Response generated and saved for agent_id: {agent.agent_id}, conversation_id: {conversation_id}")
            return {
                "response": conversational_response,
                "structured_data": structured_data,
                "conversation_id": conversation_id,
                "conversation_history": updated_history,
                "conversation_type": "non_autonomous",
                "llm_model": agent.model
            }
        except Exception as e:
            self.logger.error(f"Error in get_nonautonomous_response for agent_id {agent.agent_id}, conversation_id {conversation_id}: {str(e)}")
            return {"error": f"Failed to get response: {str(e)}"}
    
    def get_autonomous_response(
        self,
        police_agent: PoliceAgent,
        victim_agent: VictimAgent,
        police_prompt: str = None,
        victim_prompt: str = None,
        initial_query: str = "Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?",
        conversation_id: int = None,
        max_turns: int = 10
    ) -> Dict:
        if not police_agent.agent or not victim_agent.agent:
            self.logger.error(f"Agent not initialized: police={police_agent.agent_id}, victim={victim_agent.agent_id}")
            return {"error": f"Agent not initialized: police={police_agent.agent_id}, victim={victim_agent.agent_id}"}
        
        police_prompt = police_prompt.strip() if police_prompt and isinstance(police_prompt, str) else Prompt.template["baseline_police"]
        victim_prompt = victim_prompt.strip() if victim_prompt and isinstance(victim_prompt, str) else Prompt.template["baseline_victim"]
        if not police_prompt or not victim_prompt:
            self.logger.error("Prompts cannot be empty")
            return {"error": "Prompts cannot be empty"}
        
        conversation_id = conversation_id or self._generate_conversation_id()
        conversation_history = self.conversation_registry.get(conversation_id, [])
        current_query = initial_query
        last_messages = []
        structured_data_history = []
        
        try:
            max_memory_turns = min(police_agent.max_turns, victim_agent.max_turns)
            pipeline = RAGPipeline(police_agent=police_agent, police_tools=self.police_tools)
            
            for turn in range(max_turns):
                victim_history = ChatMessageHistory()
                for msg in conversation_history:
                    if msg["role"] == "police":
                        victim_history.add_user_message(msg["content"])
                    elif msg["role"] == "victim":
                        victim_history.add_ai_message(msg["content"])
                
                victim_state = {
                    "messages": [
                        SystemMessage(content=victim_prompt),
                        *victim_history.messages,
                        HumanMessage(content=current_query)
                    ]
                }
                
                self.logger.debug(f"Invoking victim agent for query: {current_query}, conversation_id: {conversation_id}")
                try:
                    victim_response = victim_agent.agent.invoke(victim_state)
                    self.logger.debug(f"Raw victim response: {victim_response}")
                except Exception as e:
                    self.logger.error(f"LLM invocation failed for victim agent: {str(e)}")
                    victim_message = "I'm sorry, I'm a bit upset. Can you ask that again?"
                    conversation_history.append({
                        "role": "victim",
                        "content": victim_message,
                        "timestamp": datetime.now().isoformat()
                    })
                    self._save_to_csv(conversation_id, "autonomous", conversation_history, llm_model=victim_agent.model)
                    break
                
                victim_messages = victim_response.get("messages", [])
                victim_ai_messages = [msg.content for msg in victim_messages if isinstance(msg, AIMessage)]
                victim_message = victim_ai_messages[-1] if victim_ai_messages else "I'm sorry, I'm a bit upset. Can you ask that again?"
                
                if not victim_message.strip():
                    self.logger.warning(f"Empty victim response in turn {turn+1}. Using fallback response.")
                    victim_message = "I'm sorry, I'm a bit upset. Can you ask that again?"
                
                if not any(msg["role"] == "victim" and msg["content"] == victim_message and msg["timestamp"] == datetime.now().isoformat() for msg in conversation_history):
                    conversation_history.append({
                        "role": "victim",
                        "content": victim_message,
                        "timestamp": datetime.now().isoformat()
                    })
                self._save_to_csv(conversation_id, "autonomous", conversation_history, llm_model=victim_agent.model)
                
                if "[END_CONVERSATION]" in victim_message or "thank you for your cooperation" in victim_message.lower():
                    self.logger.info(f"Conversation ended: {victim_message}")
                    break
                
                last_messages.append(victim_message)
                if len(last_messages) > 6:
                    last_messages.pop(0)
                    if len(set(last_messages)) <= 4:
                        self.logger.warning(f"Detected potential repetition: {last_messages}")
                        break
                
                pipeline_result = pipeline.run(victim_message, conversation_id, conversation_history)
                conversational_response = pipeline_result["response"]
                structured_data = pipeline_result["structured_data"]
                
                if not any(msg["role"] == "police" and msg["content"] == conversational_response and msg["timestamp"] == datetime.now().isoformat() for msg in conversation_history):
                    conversation_history.append({
                        "role": "police",
                        "content": conversational_response,
                        "timestamp": datetime.now().isoformat(),
                        "structured_data": structured_data
                    })
                if structured_data:
                    structured_data_history.append(structured_data)
                
                self._save_to_csv(conversation_id, "autonomous", conversation_history, structured_data, llm_model=police_agent.model)
                
                last_messages.append(conversational_response)
                if len(last_messages) > 6:
                    last_messages.pop(0)
                    if len(set(last_messages)) <= 4:
                        self.logger.warning(f"Detected potential repetition: {last_messages}")
                        break
                
                current_query = conversational_response
                
                if "thank you for your cooperation" in conversational_response.lower():
                    self.logger.info("Police ended conversation")
                    break
            
            self.conversation_registry[conversation_id] = conversation_history
            final_structured_data = structured_data_history[-1] if structured_data_history else {}
            
            self.logger.debug(f"Autonomous conversation completed for conversation_id: {conversation_id}")
            return {
                "status": "Conversation completed",
                "conversation_id": conversation_id,
                "conversation_history": conversation_history,
                "conversation_type": "autonomous",
                "structured_data": final_structured_data,
                "police_model": police_agent.model,
                "victim_model": victim_agent.model
            }
        except Exception as e:
            self.logger.error(f"Error in get_autonomous_response for conversation_id {conversation_id}: {str(e)}")
            return {"error": f"Failed to simulate conversation: {str(e)}"}