# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_ollama import ChatOllama
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import MessagesState
# from typing import TypedDict, Annotated, Sequence
# from pydantic import BaseModel
# import json
# import logging
# import csv
# import os
# from datetime import datetime
# from pathlib import Path
# from filelock import FileLock
# from config.settings import get_settings
# from src.database.vector_operations import VectorStore
# from src.database.database_operations import DatabaseManager
# from src.models.data_model import ScamReport
# from src.models.response_model import PoliceResponse
# from src.agents.tools import PoliceTools
# from config.id_manager import IDManager

# # Configure logging
# def setup_logging():
#     settings = get_settings()
#     log_dir = Path(settings.log.directory) / settings.log.subdirectories["agent"]
#     log_dir.mkdir(parents=True, exist_ok=True)
#     log_file = log_dir / "chatbot.log"
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler(log_file, mode='a'),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)

# logger = setup_logging()

# # Define state
# class ChatbotState(TypedDict):
#     messages: Annotated[Sequence[HumanMessage | AIMessage], "Chat history"]
#     conversation_id: int
#     police_response: dict
#     rag_results: list

# # Updated prompt template with escaped curly braces
# prompt_template = ChatPromptTemplate.from_messages([
#     ("system", """
# You are a professional police AI assistant helping victims report scams. Extract scam related details from the victim's query and prompt them for additional details. Use the provided scam reports to inform your response. Extract details from the query and incrementally fill fields based on {rag_results}. Respond strictly in JSON format conforming to the PoliceResponse model. Do not include additional text or duplicate JSON objects. Use empty strings or 0.0 for missing fields. Prompt the victim for additional details as needed. Set `rag_invoked` to true if {rag_results} is used.
# The `conversational_response` must be a non-empty natural language prompt requesting specific missing details (e.g., date, amount lost, scammer's contact).

# **PoliceResponse Schema**:
# {{
#   "conversational_response": "str",  // Natural language response prompting victim for more details. This is a non-empty field and must be filled.
#   "scam_incident_date": "str",      // YYYY-MM-DD format, e.g., "2025-02-22"
#   "scam_type": "str",               // e.g., "ECOMMERCE", "PHISHING", "GOVERNMENT"
#   "scam_approach_platform": "str",  // e.g., "FACEBOOK", "SMS", "WHATSAPP"
#   "scam_communication_platform": "str",  // e.g., "EMAIL", "WHATSAPP"
#   "scam_transaction_type": "str",   // e.g., "BANK TRANSFER", "CRYPTOCURRENCY"
#   "scam_beneficiary_platform": "str",  // e.g., "CIMB", "HSBC"
#   "scam_beneficiary_identifier": "str",  // e.g., bank account number
#   "scam_contact_no": "str",         // Scammer's phone number
#   "scam_email": "str",             // Scammer's email
#   "scam_moniker": "str",           // Scammer's alias
#   "scam_url_link": "str",          // URLs used in scam
#   "scam_amount_lost": "float",     // Amount lost, e.g., 450.0
#   "scam_incident_description": "str",  // Detailed description of scam in first person
#   "scam_specific_details": "dict",  // Specific details, e.g., {{"scam_subcategory": "FAILURE TO DELIVER"}}
#   "rag_invoked": "bool"            // True if RAG results are used
# }}

# **Instructions**:
# 1. Extract details from the victim's query, e.g., "Facebook" maps to "scam_approach_platform": "FACEBOOK".
# 2. Use {rag_results} to standardize terminology and populate `scam_specific_details` (e.g., {{"scam_subcategory": "FAILURE TO DELIVER"}}).
# 3. Fill fields incrementally, preserving previously extracted information from conversation history.
# 4. For missing fields, use empty strings ("") or 0.0 and include a prompt in `conversational_response` to request details.
# 5. Output only the JSON object matching the PoliceResponse schema.

# **Example**:
# {{
#   "conversational_response": "I'm sorry to hear you were scammed on Facebook. Can you provide the date, amount paid, and the seller's details?",
#   "scam_incident_date": "",
#   "scam_type": "ECOMMERCE",
#   "scam_approach_platform": "FACEBOOK",
#   "scam_communication_platform": "",
#   "scam_transaction_type": "",
#   "scam_beneficiary_platform": "",
#   "scam_beneficiary_identifier": "",
#   "scam_contact_no": "",
#   "scam_email": "",
#   "scam_moniker": "",
#   "scam_url_link": "",
#   "scam_amount_lost": 0.0,
#   "scam_incident_description": "I was scammed on Facebook buying a concert ticket.",
#   "scam_specific_details": {{}},
#   "rag_invoked": true
# }}
# """),
#     MessagesPlaceholder(variable_name="messages"),
#     ("human", "{user_input}"),
# ])

# class PoliceChatbot:
#     def __init__(self, model_name: str = "llama3.1:8b"):
#         self.settings = get_settings()
#         self.id_manager = IDManager()
#         self.model_name = model_name
#         self.conversation_id = self.id_manager.get_next_id()
#         self.conversation_history = []
#         self.police_tools = PoliceTools()
#         self.workflow = self._build_workflow()
#         logger.info(f"Police chatbot initialized with model: {model_name}, conversation_id: {self.conversation_id}")

#     def _initialize_llm_and_tools(self):
#         llm = ChatOllama(
#             model=self.model_name,
#             base_url=self.settings.agents.ollama_base_url,
#             format="json",
#             response_format={
#                 "type": "json_schema",
#                 "json_schema": PoliceResponse.model_json_schema()
#             }
#         )
#         tools = self.police_tools.get_tools()
#         return llm.bind_tools(tools), tools

#     def _build_workflow(self):
#         llm, tools = self._initialize_llm_and_tools()
#         workflow = StateGraph(ChatbotState)
#         workflow.add_node("invoke_rag", lambda state: self._invoke_rag(state, tools))
#         workflow.add_node("process_llm", lambda state: self._process_llm(state, llm))
#         workflow.add_edge("invoke_rag", "process_llm")
#         workflow.add_edge("process_llm", END)
#         workflow.set_entry_point("invoke_rag")
#         return workflow.compile()

#     def _invoke_rag(self, state: ChatbotState, tools):
#         logger.debug(f"Invoking RAG for query: {state['messages'][-1].content}")
#         user_input = state["messages"][-1].content
#         rag_tool = tools[0]  # retrieve_scam_reports
#         try:
#             rag_results = rag_tool.invoke({
#                 "query": user_input,
#                 "top_k": 5,
#                 "conversation_id": state["conversation_id"],
#                 "llm_model": self.model_name
#             })
#             rag_results = json.loads(rag_results)
#         except Exception as e:
#             logger.error(f"RAG invocation failed: {str(e)}")
#             rag_results = []
#         return {"rag_results": rag_results}

#     def _process_llm(self, state: ChatbotState, llm):
#         logger.debug("Processing LLM response")
#         messages = state["messages"]
#         rag_results = state.get("rag_results", [])
#         logger.debug(f"Formatting prompt with messages: {messages}, user_input: {messages[-1].content}, rag_results: {rag_results}")
#         prompt = prompt_template.format(
#             messages=messages,
#             user_input=messages[-1].content,
#             rag_results=json.dumps(rag_results)
#         )
#         try:
#             response = llm.invoke(prompt)
#             logger.debug(f"Raw LLM response: {response.content}")
#             police_response = json.loads(response.content)
            
#             # # Map incorrect field names to PoliceResponse fields
#             # field_mappings = {
#             #     "platform": "scam_approach_platform",
#             #     "transaction_type": "scam_transaction_type"
#             # }
#             # for old_key, new_key in field_mappings.items():
#             #     if old_key in police_response:
#             #         police_response[new_key] = police_response.pop(old_key)
            
#             # Ensure all PoliceResponse fields are present
#             default_response = {
#                 "conversational_response": "I'm sorry, I need more information to assist you.",
#                 "scam_incident_date": "",
#                 "scam_type": "",
#                 "scam_approach_platform": "",
#                 "scam_communication_platform": "",
#                 "scam_transaction_type": "",
#                 "scam_beneficiary_platform": "",
#                 "scam_beneficiary_identifier": "",
#                 "scam_contact_no": "",
#                 "scam_email": "",
#                 "scam_moniker": "",
#                 "scam_url_link": "",
#                 "scam_amount_lost": 0.0,
#                 "scam_incident_description": "",
#                 "scam_specific_details": {},
#                 "rag_invoked": bool(rag_results)
#             }
#             default_response.update(police_response)
#             police_response = default_response
#             # Validate against PoliceResponse model
#             PoliceResponse(**police_response)
#             return {
#                 "messages": [AIMessage(content=json.dumps(police_response))],
#                 "police_response": police_response
#             }
#         except Exception as e:
#             logger.error(f"LLM processing failed: {str(e)}")
#             fallback_response = {
#                 "conversational_response": "Sorry, I encountered an error. Can you provide more details about the scam?",
#                 "scam_incident_date": "",
#                 "scam_type": "",
#                 "scam_approach_platform": "",
#                 "scam_communication_platform": "",
#                 "scam_transaction_type": "",
#                 "scam_beneficiary_platform": "",
#                 "scam_beneficiary_identifier": "",
#                 "scam_contact_no": "",
#                 "scam_email": "",
#                 "scam_moniker": "",
#                 "scam_url_link": "",
#                 "scam_amount_lost": 0.0,
#                 "scam_incident_description": "",
#                 "scam_specific_details": {},
#                 "rag_invoked": bool(rag_results)
#             }
#             return {
#                 "messages": [AIMessage(content=json.dumps(fallback_response))],
#                 "police_response": fallback_response
#             }

#     def _save_conversation(self):
#         csv_file = "conversation_history.csv"
#         file_exists = os.path.isfile(csv_file)
#         existing_entries = set()
#         index_counter = 0

#         if file_exists:
#             try:
#                 with FileLock(f"{csv_file}.lock"):
#                     with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                         reader = csv.DictReader(f)
#                         if "index" in reader.fieldnames:
#                             indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
#                             index_counter = max(indices) if indices else 0
#                         for row in reader:
#                             key = (row["conversation_id"], row["sender_type"], row["content"], row["timestamp"])
#                             existing_entries.add(key)
#             except Exception as e:
#                 logger.error(f"Error reading CSV for deduplication: {str(e)}")
#                 return

#         with FileLock(f"{csv_file}.lock"):
#             try:
#                 with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
#                     writer = csv.writer(f)
#                     if not file_exists:
#                         writer.writerow([
#                             "index", "conversation_id", "conversation_type", "sender_type", "content",
#                             "timestamp", "llm_model", "scam_generic_details", "scam_specific_details"
#                         ])
#                     for msg in self.conversation_history:
#                         key = (str(self.conversation_id), msg["role"], msg["content"], msg["timestamp"])
#                         if key not in existing_entries:
#                             index_counter += 1
#                             structured_data = msg.get("structured_data", {})
#                             row = [
#                                 str(index_counter),
#                                 str(self.conversation_id),
#                                 "non_autonomous",
#                                 msg["role"],
#                                 msg["content"],
#                                 msg["timestamp"],
#                                 self.model_name if msg["role"] == "police" else "",
#                                 json.dumps(structured_data, ensure_ascii=False) if msg["role"] == "police" and structured_data else "",
#                                 ""
#                             ]
#                             writer.writerow(row)
#                             existing_entries.add(key)
#                             logger.debug(f"Wrote message to CSV: conversation_id={self.conversation_id}, sender={msg['role']}")
#             except Exception as e:
#                 logger.error(f"Error writing to CSV: {str(e)}")

#     def process_query(self, query: str):
#         if not query.strip():
#             logger.error("Query cannot be empty")
#             return {"error": "Query cannot be empty"}

#         state = {
#             "messages": [HumanMessage(content=query)],
#             "conversation_id": self.conversation_id,
#             "police_response": {},
#             "rag_results": []
#         }
#         result = self.workflow.invoke(state)
#         police_response = result["police_response"]
#         conversational_response = police_response.get("conversational_response", "I'm sorry, I need more information to assist you.")

#         self.conversation_history.append({
#             "role": "victim",
#             "content": query,
#             "timestamp": datetime.now().isoformat()
#         })
#         self.conversation_history.append({
#             "role": "police",
#             "content": conversational_response,
#             "timestamp": datetime.now().isoformat(),
#             "structured_data": {k: v for k, v in police_response.items() if k != "conversational_response"}
#         })

#         return {
#             "response": conversational_response,
#             "structured_data": police_response,
#             "conversation_id": self.conversation_id,
#             "conversation_history": self.conversation_history
#         }

#     def end_conversation(self):
#         self._save_conversation()
#         logger.info(f"Conversation {self.conversation_id} saved to CSV")
#         return {"status": "Conversation ended", "conversation_id": self.conversation_id}

# if __name__ == "__main__":
#     chatbot = PoliceChatbot(model_name="llama3.1:8b")
#     query = "I received an SMS claiming to be from a bank."
#     response = chatbot.process_query(query)
#     print(json.dumps(response, indent=2))
#     chatbot.end_conversation()

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_ollama import ChatOllama
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import MessagesState
# from typing import TypedDict, Annotated, Sequence
# from pydantic import BaseModel
# import json
# import logging
# import csv
# import os
# from datetime import datetime
# from pathlib import Path
# from filelock import FileLock
# from config.settings import get_settings
# from config.logging_config import setup_logger
# from src.database.vector_operations import VectorStore
# from src.database.database_operations import DatabaseManager
# from src.models.data_model import ScamReport
# from src.models.response_model import PoliceResponse
# from src.agents.tools import PoliceTools
# from config.id_manager import IDManager
# from langchain_core.exceptions import LangChainException
# from src.agents.prompt import Prompt


# # Define state
# class ChatbotState(TypedDict):
#     messages: Annotated[Sequence[HumanMessage | AIMessage], "Chat history"]
#     conversation_id: int
#     police_response: dict
#     rag_results: list
#     rag_invoked: bool


# class PoliceChatbot:
#     def __init__(self, model_name: str = "qwen2.5:7b"):
#         self.settings = get_settings()
#         self.logger = setup_logger("VanillaRag_PoliceAgent", self.settings.log.subdirectories["agent"])
#         self.id_manager = IDManager()
#         self.model_name = model_name
#         self.conversation_id = self.id_manager.get_next_id()
#         self.conversation_history = []
        
#         #Initialize tools and chatbot prompt
#         self.police_tools = PoliceTools()
#         self.prompt_template = self._generate_prompt_template()
        
#         #Build LangGraph workflow
#         self.workflow = self._build_workflow()
        
#         self.logger.info(f"Police chatbot initialized with model: {model_name}, conversation_id: {self.conversation_id}")

#     def _generate_prompt_template(self) -> ChatPromptTemplate:
#         """
#         Generate the ChatPromptTemplate using the baseline_police prompt from Prompt class.
#         """
#         if "baseline_police" not in Prompt.template:
#             self.logger.error("Prompt type 'baseline_police' not found in Prompt.template")
#             raise ValueError("Prompt type 'baseline_police' not found")
        
#         prompt_template = ChatPromptTemplate.from_messages([
#             ("system", Prompt.template["baseline_police"]),
#             MessagesPlaceholder(variable_name="messages"),
#             ("human", "{user_input}"),
#         ])
#         self.logger.debug("Generated baseline_police prompt template")
#         return prompt_template
    
#     def _initialize_llm_and_tools(self):
#         llm = ChatOllama(
#             model=self.model_name,
#             base_url=self.settings.agents.ollama_base_url,
#             format="json",
#             response_format={
#                 "type": "json_schema",
#                 "json_schema": PoliceResponse.model_json_schema()
#             }
#         )
#         tools = self.police_tools.get_tools()
#         return llm.bind_tools(tools), tools

#     def _build_workflow(self):
#         llm, tools = self._initialize_llm_and_tools()
#         workflow = StateGraph(ChatbotState)
#         workflow.add_node("invoke_rag", lambda state: self._invoke_rag(state, tools))
#         workflow.add_node("process_llm", lambda state: self._process_llm(state, llm))
#         workflow.add_edge("invoke_rag", "process_llm")
#         workflow.add_edge("process_llm", END)
#         workflow.set_entry_point("invoke_rag")
#         return workflow.compile()

#     def _invoke_rag(self, state: ChatbotState, tools):
#         self.logger.debug(f"Invoking RAG for query: {state['messages'][-1].content}")
#         user_input = state["messages"][-1].content
#         rag_tool = tools[0]  # retrieve_scam_reports
#         try:
#             rag_results = rag_tool.invoke({
#                 "query": user_input,
#                 "top_k": 1,
#                 "conversation_id": state["conversation_id"],
#                 "llm_model": self.model_name
#             })
#             rag_results = json.loads(rag_results)
#             self.logger.debug(f"RAG results: {rag_results}")
#         except Exception as e:
#             self.logger.error(f"RAG invocation failed: {str(e)}", exc_info=True)
#             rag_results = []
#         self.logger.info("RAG invoked successfully")
#         return {
#             "rag_results": rag_results,
#             "rag_invoked": True
#         }

#     def _process_llm(self, state: ChatbotState, llm):
#         self.logger.debug("Processing LLM response")
#         messages = state["messages"]
#         rag_results = state.get("rag_results", [])
#         rag_invoked = state.get("rag_invoked", False)
        
#         prompt = self.prompt_template.format(
#             messages=messages,
#             user_input=messages[-1].content,
#             rag_results=json.dumps(rag_results)
#         )
#         try:
#             response = llm.invoke(prompt)
#             # logger.info(f"Raw LLM response for model {self.model_name}: '{response.content}'")
#             if not response.content.strip():
#                 self.logger.error("LLM returned empty response")
#                 raise ValueError("Empty response from LLM")
#             police_response = json.loads(response.content)
#             default_response = {
#                 "conversational_response": "I'm sorry, I need more information to assist you.",
#                 "scam_incident_date": "",
#                 "scam_type": "",
#                 "scam_approach_platform": "",
#                 "scam_communication_platform": "",
#                 "scam_transaction_type": "",
#                 "scam_beneficiary_platform": "",
#                 "scam_beneficiary_identifier": "",
#                 "scam_contact_no": "",
#                 "scam_email": "",
#                 "scam_moniker": "",
#                 "scam_url_link": "",
#                 "scam_amount_lost": 0.0,
#                 "scam_incident_description": "",
#                 "scam_specific_details": {}
#             }
#             default_response.update(police_response)
#             police_response = default_response
#             PoliceResponse(**police_response)
#             self.logger.debug(f"Validated PoliceResponse: {police_response}")
#             return {
#                 "messages": [AIMessage(content=json.dumps(police_response))],
#                 "police_response": police_response,
#                 "rag_invoked": rag_invoked
#             }
#         except Exception as e:
#             self.logger.error(f"LLM processing failed for model {self.model_name}: {str(e)}", exc_info=True)
#             fallback_response = {
#                 "conversational_response": "Sorry, I encountered an error. Can you provide more details about the scam, such as the date, what the caller said, and any contact details provided?",
#                 "scam_incident_date": "",
#                 "scam_type": "GOVERNMENT" if "government" in messages[-1].content.lower() else "",
#                 "scam_approach_platform": "PHONE_CALL" if "call" in messages[-1].content.lower() else "",
#                 "scam_communication_platform": "PHONE_CALL" if "call" in messages[-1].content.lower() else "",
#                 "scam_transaction_type": "",
#                 "scam_beneficiary_platform": "",
#                 "scam_beneficiary_identifier": "",
#                 "scam_contact_no": "",
#                 "scam_email": "",
#                 "scam_moniker": "",
#                 "scam_url_link": "",
#                 "scam_amount_lost": 0.0,
#                 "scam_incident_description": messages[-1].content,
#                 "scam_specific_details": {"scam_subcategory": "government impersonation"} if "government" in messages[-1].content.lower() else {},
#             }
#             return {
#                 "messages": [AIMessage(content=json.dumps(fallback_response))],
#                 "police_response": fallback_response,
#                 "rag_invoked": rag_invoked
#             }

#     def _save_conversation(self):
#         csv_file = "conversation_history.csv"
#         file_exists = os.path.isfile(csv_file)
#         existing_entries = set()
#         index_counter = 0

#         if file_exists:
#             try:
#                 with FileLock(f"{csv_file}.lock"):
#                     with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                         reader = csv.DictReader(f)
#                         if "index" in reader.fieldnames:
#                             indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
#                             index_counter = max(indices) if indices else 0
#                         f.seek(0)  # Reset file pointer for second pass
#                         reader = csv.DictReader(f)
#                         for row in reader:
#                             key = (row["conversation_id"], row["sender_type"], row["content"], row["timestamp"])
#                             existing_entries.add(key)
#             except Exception as e:
#                 self.logger.error(f"Error reading CSV for deduplication: {str(e)}", exc_info=True)
#                 return

#         with FileLock(f"{csv_file}.lock"):
#             try:
#                 with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
#                     writer = csv.writer(f)
#                     if not file_exists:
#                         writer.writerow([
#                             "index", "conversation_id", "conversation_type", "sender_type", "content",
#                             "timestamp", "llm_model", "scam_specific_details", "rag_invoked"
#                         ])
#                     for msg in self.conversation_history:
#                         key = (str(self.conversation_id), msg["role"], msg["content"], msg["timestamp"])
#                         if key not in existing_entries:
#                             index_counter += 1
#                             structured_data = msg.get("structured_data", {})
#                             row = [
#                                 str(index_counter),
#                                 str(self.conversation_id),
#                                 "non_autonomous",
#                                 msg["role"],
#                                 msg["content"],
#                                 msg["timestamp"],
#                                 self.model_name if msg["role"] == "police" else "",
#                                 json.dumps(structured_data, ensure_ascii=False) if msg["role"] == "police" and structured_data else "",
#                                 str(msg.get("rag_invoked", False)).lower() if msg["role"] == "police" else ""
#                             ]
#                             writer.writerow(row)
#                             existing_entries.add(key)
#                             self.logger.debug(f"Wrote message to CSV: conversation_id={self.conversation_id}, sender={msg['role']}")
#             except Exception as e:
#                 self.logger.error(f"Error writing to CSV: {str(e)}", exc_info=True)

#     def process_query(self, query: str):
#         if not query.strip():
#             self.logger.error("Query cannot be empty")
#             return {"error": "Query cannot be empty"}

#         state = {
#             "messages": [HumanMessage(content=query)],
#             "conversation_id": self.conversation_id,
#             "police_response": {},
#             "rag_results": [],
#             "rag_invoked": False
#         }
#         result = self.workflow.invoke(state)
#         police_response = result["police_response"]
#         rag_invoked = result.get("rag_invoked", False)
#         conversational_response = police_response.get("conversational_response", "I'm sorry, I need more information to assist you.")

#         self.conversation_history.append({
#             "role": "victim",
#             "content": query,
#             "timestamp": datetime.now().isoformat()
#         })
#         self.conversation_history.append({
#             "role": "police",
#             "content": conversational_response,
#             "timestamp": datetime.now().isoformat(),
#             "structured_data": {k: v for k, v in police_response.items() if k != "conversational_response"},
#             "rag_invoked": rag_invoked
#         })

#         self.logger.debug(f"Processed query response: {conversational_response}")
#         return {
#             "response": conversational_response,
#             "structured_data": police_response,
#             "conversation_id": self.conversation_id,
#             "conversation_history": self.conversation_history
#         }

#     def end_conversation(self):
#         self._save_conversation()
#         self.logger.info(f"Conversation {self.conversation_id} saved to CSV")
#         return {"status": "Conversation ended", "conversation_id": self.conversation_id}

# if __name__ == "__main__":
    
#     logger = setup_logger("VanillaRag_PoliceAgent", get_settings().log.subdirectories["agent"])
#     # List of models to test
#     # models = ["qwen3:8b" (coding), "granite3.3:8b" (coding),"llama3.2:latest"] 
#     models = ["qwen2.5:7b", "granite3.2:8b", "mistral:7b"] #supports structured output, chat and tool binding functionalities
#     query = "I received a call from someone claiming to be a government official."
    
#     # Dictionary to store results for each model
#     results = {}

#     logger.info("Starting model testing")
#     for model_name in models:
#         logger.info(f"--- Testing model: {model_name} ---")
#         try:
#             # Initialize chatbot with the current model
#             chatbot = PoliceChatbot(model_name=model_name)
            
#             # Process the query
#             response = chatbot.process_query(query)
            
#             # Store the response
#             results[model_name] = response
            
#             logger.info(f"Successfully processed query with model {model_name}: {json.dumps(response, indent=2)}")
            
#             # End conversation to save history
#             chatbot.end_conversation()
#         except LangChainException as e:
#             logger.error(f"LangChain error with model {model_name}: {str(e)}", exc_info=True)
#             results[model_name] = {"error": f"LangChain error: {str(e)}"}
#         except Exception as e:
#             logger.error(f"Unexpected error with model {model_name}: {str(e)}", exc_info=True)
#             results[model_name] = {"error": f"Unexpected error: {str(e)}"}

#     # Print results for all models
#     logger.info("Completed model testing")
#     print(json.dumps(results, indent=2))
    
    
#     # Define a sequence of victim queries for multi-turn testing (simulating a government impersonation scam conversation)
#     queries = [
#         "I received a call from someone claiming to be a government official.",
#         "It happened last week, on July 20, 2025. They said I owed taxes.",
#         "They impersonated the IRS, and asked for my bank details.",
#         "I transferred $1500 to their account via bank transfer.",
#         "The bank was HSBC, account number 1234567890, and the contact number was +1-123-456-7890."
#     ]
    
#     # Dictionary to store multi-turn results for each model (list of responses per model)
#     results = {}

#     logger.info("Starting multi-turn model testing")
#     for model_name in models:
#         logger.info(f"--- Testing model: {model_name} ---")
#         try:
#             # Initialize chatbot with the current model
#             chatbot = PoliceChatbot(model_name=model_name)
            
#             # Initialize list to collect responses for this model
#             model_responses = []
            
#             # Process each query in sequence
#             for i, query in enumerate(queries, 1):
#                 logger.info(f"Processing turn {i} with query: '{query}'")
#                 response = chatbot.process_query(query)
#                 model_responses.append({
#                     "turn": i,
#                     "query": query,
#                     "response": response["response"],
#                     "structured_data": response["structured_data"]
#                 })
#                 logger.info(f"Turn {i} response: {json.dumps(response, indent=2)}")
            
#             # Store the responses for this model
#             results[model_name] = model_responses
            
#             # End conversation to save history
#             chatbot.end_conversation()
#         except LangChainException as e:
#             logger.error(f"LangChain error with model {model_name}: {str(e)}", exc_info=True)
#             results[model_name] = {"error": f"LangChain error: {str(e)}"}
#         except Exception as e:
#             logger.error(f"Unexpected error with model {model_name}: {str(e)}", exc_info=True)
#             results[model_name] = {"error": f"Unexpected error: {str(e)}"}

#     # Print results for all models
#     logger.info("Completed multi-turn model testing")
#     print(json.dumps(results, indent=2))


# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_ollama import ChatOllama
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages
# from typing import TypedDict, Annotated, Sequence
# from pydantic import BaseModel, ValidationError
# import json
# import logging
# import csv
# import os
# from datetime import datetime
# from pathlib import Path
# from filelock import FileLock
# from config.settings import get_settings
# from config.logging_config import setup_logger
# from src.database.vector_operations import VectorStore
# from src.database.database_operations import DatabaseManager
# from src.models.data_model import ScamReport
# from src.models.response_model import PoliceResponse, QuestionSlotMap, ValidatedQuestionSlotMap
# from src.agents.tools import PoliceTools
# from config.id_manager import IDManager
# from langchain_core.exceptions import LangChainException
# from src.agents.prompt import Prompt
# from src.agents.utils import build_query_with_history
# import re


# # Define state
# class ChatbotState(TypedDict):
#     messages: Annotated[Sequence[AnyMessage], add_messages]
#     conversation_id: int
#     police_response: dict
#     rag_results: list
#     rag_invoked: bool


# class PoliceChatbot:
#     def __init__(self, model_name: str = "qwen2.5:7b"):
#         self.settings = get_settings()
#         self.logger = setup_logger("VanillaRag_PoliceAgent", self.settings.log.subdirectories["agent"])
#         self.id_manager = IDManager()
#         self.model_name = model_name
#         self.conversation_id = self.id_manager.get_next_id()
#         self.conversation_history = []
#         self.messages: Sequence[AnyMessage] = []
#         self.scam_type: str = ""
        
#         #Initialize tools and chatbot prompt
#         self.police_tools = PoliceTools()
#         self.tools = self.police_tools.get_tools()
        
#         self.main_prompt_template = self._generate_prompt_template()
#         self.llm = self._get_structured_llm(PoliceResponse)
#         self.target_slot_prompt = ChatPromptTemplate.from_template(Prompt.template["target_slot_agent"])
        
#         #Build LangGraph workflow
#         self.workflow = self._build_workflow()
        
#         self.logger.info(f"Police chatbot initialized with model: {model_name}, conversation_id: {self.conversation_id}")

#     def _generate_prompt_template(self) -> ChatPromptTemplate:
#         """
#         Generate the ChatPromptTemplate using the baseline_police prompt from Prompt class.
#         """
#         if "baseline_police" not in Prompt.template:
#             self.logger.error("Prompt type 'baseline_police' not found in Prompt.template")
#             raise ValueError("Prompt type 'baseline_police' not found")
        
#         prompt_template = ChatPromptTemplate.from_messages([
#             ("system", Prompt.template["baseline_police"]),
#             MessagesPlaceholder(variable_name="messages"),
#             ("human", "{user_input}"),
#         ])
#         self.logger.debug("Generated baseline_police prompt template")
#         return prompt_template
    
#     def _get_structured_llm(self, schema: BaseModel) -> ChatOllama:
#         return ChatOllama(
#             model=self.model_name,
#             base_url=self.settings.agents.ollama_base_url,
#             format="json",
#             response_format={
#                 "type": "json_schema",
#                 "json_schema": schema.model_json_schema()
#             }
#         )
    
#     def _build_workflow(self):
#         workflow = StateGraph(ChatbotState)
#         workflow.add_node("invoke_rag", lambda state: self._invoke_rag(state, self.tools))
#         workflow.add_node("process_llm", lambda state: self._process_llm(state))
#         workflow.add_node("map_target_slots", lambda state: self._map_target_slots(state))
#         workflow.add_edge("invoke_rag", "process_llm")
#         workflow.add_edge("process_llm", "map_target_slots")
#         workflow.add_edge("map_target_slots", END)
#         workflow.set_entry_point("invoke_rag")
#         return workflow.compile()

#     def _invoke_rag(self, state: ChatbotState, tools):
#         self.logger.debug(f"Invoking RAG for query: {state['messages'][-1].content}")
#         # current_query = state["messages"][-1].content
#         # N = None  # can set via config if needed
#         # history_msgs = [msg.content for msg in state["messages"][:-1] if isinstance(msg, HumanMessage)]
#         # if N is not None:
#         #     history_msgs = history_msgs[-N:]  # Limit context window if N is set
#         # history_query = " ".join(history_msgs + [current_query]) #adjust context length of past queries accordingly
#         user_query = build_query_with_history(state["messages"][-1].content, [m.content for m in state["messages"][:-1] if isinstance(m, HumanMessage)], max_history=5)
#         rag_tool = tools[0]  # retrieve_scam_reports
#         try:
#             rag_results = rag_tool.invoke({
#                 "query": user_query,
#                 "top_k": 2,
#                 "conversation_id": state["conversation_id"],
#                 "llm_model": self.model_name,
#                 # "metadata_filter": {"scam_type": self.scam_type.upper()} if self.scam_type else None
#             })
#             rag_results = json.loads(rag_results)
            
#             #post-filter
#             # if self.scam_type:  # Filter by scam_type if identified
#             #     rag_results = [r for r in rag_results if r.get("scam_type") == self.scam_type.upper()]
#             self.logger.debug(f"RAG results: {rag_results}")
#         except Exception as e:
#             self.logger.error(f"RAG invocation failed: {str(e)}", exc_info=True)
#             rag_results = []
#         self.logger.info("RAG invoked successfully")
#         return {
#             "rag_results": rag_results,
#             "rag_invoked": True
#         }

#     def _process_llm(self, state: ChatbotState):
#         self.logger.debug("Processing LLM response")
#         messages = state["messages"]
#         rag_results = state.get("rag_results", [])
#         rag_invoked = state.get("rag_invoked", False)
        
#         prompt = self.main_prompt_template.format(
#             messages=messages,
#             user_input=messages[-1].content,
#             rag_results=json.dumps(rag_results)
#         )

        
#         try:
#             response = self.llm.invoke(prompt)
#             # logger.info(f"Raw LLM response for model {self.model_name}: '{response.content}'")
#             if not response.content.strip():
#                 self.logger.error("LLM returned empty response")
#                 raise ValueError("Empty response from LLM")
#             police_response = json.loads(response.content)
#             default_response = {
#                 "conversational_response": "I'm sorry, I need more information to assist you.",
#                 "scam_incident_date": "",
#                 "scam_type": "",
#                 "scam_approach_platform": "",
#                 "scam_communication_platform": "",
#                 "scam_transaction_type": "",
#                 "scam_beneficiary_platform": "",
#                 "scam_beneficiary_identifier": "",
#                 "scam_contact_no": "",
#                 "scam_email": "",
#                 "scam_moniker": "",
#                 "scam_url_link": "",
#                 "scam_amount_lost": 0.0,
#                 "scam_incident_description": "",
#                 # "target_slots": []
#                 # "scam_specific_details": {}
#             }
#             default_response.update(police_response)
#             police_response = default_response
#             PoliceResponse(**police_response)
#             # Persist scam_type if newly identified and non-empty - to trigger metadata filtering for rag if necessary
#             if police_response["scam_type"] and not self.scam_type:
#                 self.scam_type = police_response["scam_type"]
#             self.logger.debug(f"Validated PoliceResponse: {police_response}")
#             return {
#                 "messages": [AIMessage(content=json.dumps(police_response))],
#                 "police_response": police_response,
#                 "rag_invoked": rag_invoked
#             }
#         except Exception as e:
#             self.logger.error(f"LLM processing failed for model {self.model_name}: {str(e)}", exc_info=True)
#             fallback_response = {
#                 "conversational_response": "Sorry, I encountered an error. Can you provide more details about the scam, such as the date, what the caller said, and any contact details provided?",
#                 "scam_incident_date": "",
#                 "scam_type": "",
#                 "scam_approach_platform": "",
#                 "scam_communication_platform":"",
#                 "scam_transaction_type": "",
#                 "scam_beneficiary_platform": "",
#                 "scam_beneficiary_identifier": "",
#                 "scam_contact_no": "",
#                 "scam_email": "",
#                 "scam_moniker": "",
#                 "scam_url_link": "",
#                 "scam_amount_lost": 0.0,
#                 "scam_incident_description": messages[-1].content,
#                 # "target_slots": []
#                 # "scam_specific_details": {},
#             }
#             return {
#                 "messages": [AIMessage(content=json.dumps(fallback_response))],
#                 "police_response": fallback_response,
#                 "rag_invoked": rag_invoked
#             }
            
#     def detect_issues(self, mapped_data: dict) -> list[str]:
#         issues = []
#         for slot, questions in mapped_data.items():
#             #Check for empty list
#             if not questions:
#                 issues.append(f"Empty list for {slot}")
#             for q in questions:
#                 #Check for questions with less than 3 words or begins with connectors such as and/or/if
#                 if len(q.strip().split()) < 3 or re.match(r'^(and|or|if)\b', q.strip().lower()):
#                     issues.append(f"Fragment detected in {slot}: {q}")
#         return issues
    

#     def rule_based_fallback(self,conversational: str) -> dict:
#         slots = {}
#         keywords = {
#             'scam_incident_date': ['date', 'when'],
#             'scam_amount_lost': ['amount', 'lost', 'paid', 'money', 'dollars', 'sum', 'total', 'cost', 'transferred'],
#             'scam_type': ['type', 'kind', 'scam', 'government', 'ecommerce', 'phishing'],
#             'scam_approach_platform': ['approached', 'contacted', 'reached'],
#             'scam_communication_platform': ['communicated', 'talked', 'messaged', 'called'],
#             'scam_transaction_type': ['transaction', 'payment', 'transfer', 'method', 'type', 'bank'],
#             'scam_beneficiary_platform': ['beneficiary', 'sent to', 'paid to', 'account', 'bank'],
#             'scam_beneficiary_identifier': ['identifier', 'account number'],
#             'scam_contact_no': ['contact', 'phone', 'number'],
#             'scam_email': ['email', 'gmail', 'yahoo', 'outlook'],
#             'scam_moniker': ['name', 'alias', 'moniker', 'called themselves', 'username', 'handle', 'nickname'],
#             'scam_url_link': ['url', 'link', 'website', 'site','clicked', 'visited', 'domain'],
#             'scam_incident_description': ['description', 'happened', 'details', 'story', 'explain', 'what occurred', 'incident', 'how']
#         }
#         for slot, kws in keywords.items():
#             if any(kw in conversational.lower() for kw in kws):
#                 slots[slot] = [conversational]  # Duplicate full as fallback
#         return slots

#     # def _map_target_slots(self, state: ChatbotState):
#     #     self.logger.debug("Mapping target slots")
#     #     police_response = state["police_response"]
#     #     conversational = police_response["conversational_response"]
        
#     #     # Get LLM
#     #     llm = self._get_structured_llm(QuestionSlotMap)
        
#     #     prompt = self.target_slot_prompt.format(conversational_response=conversational)
        
#     #     try:
#     #         response = llm.invoke(prompt)
#     #         if not response.content.strip():
#     #             raise ValueError("Target slot LLM returned empty response")
#     #         mapped = json.loads(response.content)
#     #         issues =self.detect_issues(mapped)
#     #         if issues:
#     #             self.logger.warning(f"Issues in mapping: {issues}. Retrying...")
#     #             corrective_prompt = self.target_slot_prompt.format(conversational_response=conversational) + \
#     #                                 f"\nYour previous output had issues: {', '.join(issues)}. Fix by duplicating full questions without fragments or empties."
#     #             response = llm.invoke(corrective_prompt)  # Retry
#     #             mapped = json.loads(response.content)  # Re-parse
            
#     #         validated_map = QuestionSlotMap(**mapped)
#     #         police_response["target_slot_map"] = validated_map.root  # Use .root for RootModel
#     #         police_response["target_slots"] = list(validated_map.root.keys())
#     #         self.logger.debug(f"Mapped target slots: {police_response['target_slots']}")
#     #     except Exception as e:
#     #         self.logger.error(f"Target slot mapping failed: {str(e)}", exc_info=True)
#     #         police_response["target_slot_map"] = {}
#     #         police_response["target_slots"] = []
        
#     #     return {
#     #         "police_response": police_response
#     #     }
    
#     def _map_target_slots(self, state: ChatbotState):
#         police_response = state["police_response"] 
#         conversational = police_response["conversational_response"]
#         prompt = self.target_slot_prompt.format(conversational_response=conversational)
#         llm = self._get_structured_llm(QuestionSlotMap)
        
#         max_retries = 2
#         for attempt in range(max_retries):
#             response = llm.invoke(prompt)
#             self.logger.debug(f"LLM raw response for mapping (attempt {attempt+1}): {response.content}")
            
#             if not response.content.strip():
#                 self.logger.error("Empty LLM response for mapping. Retrying..." if attempt < max_retries-1 else "Max retries reached.")
#                 continue
            
#             try:
#                 mapped = json.loads(response.content)
#                 issues = self.detect_issues(mapped)
#                 self.logger.debug(f"Mapping issues: {issues}")
                
#                 if issues:
#                     if attempt < max_retries - 1:
#                         corrective_prompt = prompt + f"\nPrevious output had issues: {', '.join(issues)}. Fix by extracting full questions and mapping to relevant slots without empties."
#                         prompt = corrective_prompt  # Update for next invoke
#                         continue
#                     else:
#                         self.logger.warning("Max retries; using fallback.")
#                         mapped = self.rule_based_fallback(conversational)
#                 break  # Success
#             except json.JSONDecodeError as e:
#                 self.logger.error(f"JSON decode error: {e}. Retrying...")
#                 continue
#             except Exception as e:
#                 self.logger.error(f"Mapping error: {e}")
#                 mapped = self.rule_based_fallback(conversational)
#                 break

#         # Final validation
#         try:
#             validated = ValidatedQuestionSlotMap(root=mapped)
#             police_response["target_slot_map"] = validated.root
#             police_response["target_slots"] = list(validated.root.keys())
#         except ValidationError as e:
#             self.logger.error(f"Validation failed: {e}. Using fallback.")
#             police_response["target_slot_map"] = self.rule_based_fallback(conversational)
#             police_response["target_slots"] = list(police_response["target_slot_map"].keys())

#         return {"police_response": police_response}
        
#     def _save_conversation(self):
#         csv_file = "conversation_history.csv"
#         file_exists = os.path.isfile(csv_file)
#         existing_entries = set()
#         index_counter = 0

#         if file_exists:
#             try:
#                 with FileLock(f"{csv_file}.lock"):
#                     with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                         reader = csv.DictReader(f)
#                         if "index" in reader.fieldnames:
#                             indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
#                             index_counter = max(indices) if indices else 0
#                         f.seek(0)  # Reset file pointer for second pass
#                         reader = csv.DictReader(f)
#                         for row in reader:
#                             key = (row["conversation_id"], row["sender_type"], row["content"], row["timestamp"])
#                             existing_entries.add(key)
#             except Exception as e:
#                 self.logger.error(f"Error reading CSV for deduplication: {str(e)}", exc_info=True)
#                 return

#         with FileLock(f"{csv_file}.lock"):
#             try:
#                 with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
#                     writer = csv.writer(f)
#                     if not file_exists:
#                         writer.writerow([
#                             "index", "conversation_id", "conversation_type", "sender_type", "content",
#                             "timestamp", "llm_model", "scam_details", "rag_invoked","target_slot_map"
#                         ])
#                     for msg in self.conversation_history:
#                         key = (str(self.conversation_id), msg["role"], msg["content"], msg["timestamp"])
#                         if key not in existing_entries:
#                             index_counter += 1
#                             structured_data = msg.get("structured_data", {})
#                             row = [
#                                 str(index_counter),
#                                 str(self.conversation_id),
#                                 "non_autonomous",
#                                 msg["role"],
#                                 msg["content"],
#                                 msg["timestamp"],
#                                 self.model_name if msg["role"] == "police" else "",
#                                 json.dumps(structured_data, ensure_ascii=False) if msg["role"] == "police" and structured_data else "",
#                                 str(msg.get("rag_invoked", False)).lower() if msg["role"] == "police" else "",
#                                 json.dumps(structured_data.get("target_slot_map", {}), ensure_ascii=False) if msg["role"] == "police" else "",  # New column
#                             ]
#                             writer.writerow(row)
#                             existing_entries.add(key)
#                             self.logger.debug(f"Wrote message to CSV: conversation_id={self.conversation_id}, sender={msg['role']}")
#             except Exception as e:
#                 self.logger.error(f"Error writing to CSV: {str(e)}", exc_info=True)

#     def process_query(self, query: str):
#         if not query.strip():
#             self.logger.error("Query cannot be empty")
#             return {"error": "Query cannot be empty"}

#         state = {
#             "messages": self.messages + [HumanMessage(content=query)],
#             "conversation_id": self.conversation_id,
#             "police_response": {},
#             "rag_results": [],
#             "rag_invoked": False
#         }
#         result = self.workflow.invoke(state)
#         self.messages = result["messages"]
#         police_response = result["police_response"]
#         rag_invoked = result.get("rag_invoked", False)
#         conversational_response = police_response.get("conversational_response", "I'm sorry, I need more information to assist you.")

#         self.conversation_history.append({
#             "role": "victim",
#             "content": query,
#             "timestamp": datetime.now().isoformat()
#         })
#         self.conversation_history.append({
#             "role": "police",
#             "content": conversational_response,
#             "timestamp": datetime.now().isoformat(),
#             "structured_data": {k: v for k, v in police_response.items() if k != "conversational_response"},
#             "rag_invoked": rag_invoked
#         })

#         self.logger.debug(f"Processed query response: {conversational_response}")
#         return {
#             "response": conversational_response,
#             "structured_data": police_response,
#             "conversation_id": self.conversation_id,
#             "conversation_history": self.conversation_history
#         }

#     def end_conversation(self):
#         self._save_conversation()
#         self.logger.info(f"Conversation {self.conversation_id} saved to CSV")
#         return {"status": "Conversation ended", "conversation_id": self.conversation_id}

# if __name__ == "__main__":
    
#     logger = setup_logger("VanillaRag_PoliceAgent", get_settings().log.subdirectories["agent"])
#     # List of models to test
#     # models = ["qwen3:8b" (coding), "granite3.3:8b" (coding),"llama3.2:latest"] 
#     models = ["qwen2.5:7b", "granite3.2:8b", "mistral:7b"]
#     # models = ["granite3.2:8b"] #supports structured output, chat and tool binding functionalities
#     # query = "I received a call from someone claiming to be a government official."
    
#     # # Dictionary to store results for each model
#     # results = {}

#     # logger.info("Starting singel-turn model testing")
#     # for model_name in models:
#     #     logger.info(f"--- Testing model: {model_name} ---")
#     #     try:
#     #         # Initialize chatbot with the current model
#     #         chatbot = PoliceChatbot(model_name=model_name)
            
#     #         # Process the query
#     #         response = chatbot.process_query(query)
            
#     #         # Store the response
#     #         results[model_name] = response
            
#     #         logger.info(f"Successfully processed query with model {model_name}: {json.dumps(response, indent=2)}")
            
#     #         # End conversation to save history
#     #         chatbot.end_conversation()
#     #     except LangChainException as e:
#     #         logger.error(f"LangChain error with model {model_name}: {str(e)}", exc_info=True)
#     #         results[model_name] = {"error": f"LangChain error: {str(e)}"}
#     #     except Exception as e:
#     #         logger.error(f"Unexpected error with model {model_name}: {str(e)}", exc_info=True)
#     #         results[model_name] = {"error": f"Unexpected error: {str(e)}"}

#     # # Print results for all models
#     # logger.info("Completed model testing")
#     # print(json.dumps(results, indent=2))
    
    
#     # Define a sequence of victim queries for multi-turn testing (simulating a government impersonation scam conversation)
#     queries = [
#         "I received a call from someone claiming to be a bank official She said I had suspicious fraudulent transactions in my account and requested that I confirm these transactions.",
#         "It happened last week, on July 20, 2025.",
#         "They then transferred the call to a police officer who said that I might have committed a crime.",
#         "I transferred $1500 to their account via bank transfer.",
#         "The bank was HSBC, account number 1234567890, and the contact number was +1-123-456-7890."
#         "They called me again from the same number and threatened legal action if I didn't pay more."
#     ]
    
#     # Dictionary to store multi-turn results for each model (list of responses per model)
#     results = {}

#     logger.info("Starting multi-turn model testing")
#     for model_name in models:
#         logger.info(f"--- Testing model: {model_name} ---")
#         try:
#             # Initialize chatbot with the current model
#             chatbot = PoliceChatbot(model_name=model_name)
            
#             # Initialize list to collect responses for this model
#             model_responses = []
            
#             # Process each query in sequence
#             for i, query in enumerate(queries, 1):
#                 logger.info(f"Processing turn {i} with query: '{query}'")
#                 response = chatbot.process_query(query)
#                 model_responses.append({
#                     "turn": i,
#                     "query": query,
#                     "response": response["response"],
#                     "structured_data": response["structured_data"]
#                 })
#                 logger.info(f"Turn {i} response: {json.dumps(response, indent=2)}")
            
#             # Store the responses for this model
#             results[model_name] = model_responses
            
#             # End conversation to save history
#             chatbot.end_conversation()
#         except LangChainException as e:
#             logger.error(f"LangChain error with model {model_name}: {str(e)}", exc_info=True)
#             results[model_name] = {"error": f"LangChain error: {str(e)}"}
#         except Exception as e:
#             logger.error(f"Unexpected error with model {model_name}: {str(e)}", exc_info=True)
#             results[model_name] = {"error": f"Unexpected error: {str(e)}"}

#     # Print results for all models
#     logger.info("Completed multi-turn model testing")
#     print(json.dumps(results, indent=2))
