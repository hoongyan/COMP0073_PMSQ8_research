# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_ollama import ChatOllama
# from langgraph.graph import StateGraph, END
# from typing import TypedDict, Annotated, Sequence
# import json
# import logging
# from datetime import datetime
# from pathlib import Path
# from filelock import FileLock
# from config.settings import get_settings
# from config.logging_config import setup_logger
# from src.agents.tools import VictimTools
# from config.id_manager import IDManager
# from src.preprocessing.preprocess import VictimProfilePreprocessor
# from src.agents.utils import RecordCounter
# from langchain_core.exceptions import LangChainException
# import csv
# from src.agents.prompt import Prompt

# # Define state
# class ChatbotState(TypedDict):
#     messages: Annotated[Sequence[HumanMessage | AIMessage], "Chat history"]
#     conversation_id: int

# class VictimChatbot:
#     def __init__(self, model_name: str = "qwen2.5:7b", json_file: str = "data/victim_profile/victim_details.json"):
#         """
#         Initialize the VictimChatbot with dynamic victim and scam details.

#         Args:
#             model_name (str): The LLM model name to use.
#             json_file (str): Path to the preprocessed JSON file with victim and scam details.
#         """
#         self.settings = get_settings()
#         self.logger = setup_logger("VictimAgent", self.settings.log.subdirectories["agent"])
#         self.id_manager = IDManager()
#         self.record_counter = RecordCounter()
#         self.model_name = model_name
#         self.conversation_id = self.id_manager.get_next_id()
#         self.json_file = json_file
#         self.conversation_history = []
#         self.victim_tools = VictimTools()
        
#         # Load and select victim/scam details
#         self.records = self._load_records()
#         self.record_index = self.record_counter.get_next_index(len(self.records))
#         self.victim_details, self.scam_details = self._get_current_record()
#         self.prompt_template = self._generate_prompt_template()
        
#         self.workflow = self._build_workflow()
#         self.logger.info(f"Victim chatbot initialized with model: {model_name}, conversation_id: {self.conversation_id}, record_index: {self.record_index}")

#     def _load_records(self) -> list:
#         """Load records from the JSON file."""
#         try:
#             with open(self.json_file, "r", encoding="utf-8") as f:
#                 records = json.load(f)
#             if not records:
#                 self.logger.error("No records found in scam_reports.json")
#                 raise ValueError("No records found in scam_reports.json")
#             return records
#         except Exception as e:
#             self.logger.error(f"Error loading records from {self.json_file}: {str(e)}", exc_info=True)
#             raise

#     def _get_current_record(self) -> tuple:
#         """Get the current victim and scam details based on record_index."""
#         record = self.records[self.record_index]
#         return record["victim_details"], record["scam_details"]

#     @staticmethod
#     def _escape_template_braces(text: str) -> str:
#         """Escape curly braces in text to prevent LangChain from interpreting them as placeholders."""
#         return text.replace("{", "{{").replace("}", "}}")

#     def _generate_prompt_template(self) -> ChatPromptTemplate:
#         if "baseline_victim" not in Prompt.template:
#             self.logger.error("Prompt type 'baseline_victim' not found in Prompt.template")
#             raise ValueError("Prompt type 'baseline_victim' not found")
        
#         # Serialize and escape JSON strings
        
#         victim_details_json = json.dumps(self.victim_details, indent=2)
#         scam_details_json = json.dumps(self.scam_details, indent=2)
#         victim_details_str = self._escape_template_braces(victim_details_json)
#         scam_details_str = self._escape_template_braces(scam_details_json)
        
#         # Format prompt with escaped JSON strings
#         try:
#             prompt_text = Prompt.template["baseline_victim"].format(
#                 victim_details=victim_details_str,
#                 scam_details=scam_details_str
#             )
#         except KeyError as e:
#             self.logger.error(f"Error formatting prompt template: {str(e)}")
#             raise ValueError(f"Missing key in prompt template: {str(e)}")
        
#         self.logger.debug(f"Formatted prompt text: {prompt_text}")
        
#         prompt_template = ChatPromptTemplate.from_messages([
#             ("system", prompt_text),
#             MessagesPlaceholder(variable_name="messages"),
#             ("human", "{user_input}"),
#         ])
#         self.logger.debug("Generated baseline_victim prompt template with dynamic details")
#         return prompt_template


#     def _initialize_llm_and_tools(self):
#         """
#         Initialize the LLM and bind tools for future extensibility.

#         Returns:
#             Tuple: (LLM with bound tools, list of tools).
#         """
#         llm = ChatOllama(
#             model=self.model_name,
#             base_url=self.settings.agents.ollama_base_url,
#         )
#         tools = self.victim_tools.get_tools()  # Currently empty, but bound for future use
#         return llm.bind_tools(tools), tools

#     def _build_workflow(self):
#         """
#         Build the LangGraph workflow with a single node for processing LLM responses.
#         """
#         llm, tools = self._initialize_llm_and_tools()
#         workflow = StateGraph(ChatbotState)
#         workflow.add_node("process_llm", lambda state: self._process_llm(state, llm))
#         workflow.add_edge("process_llm", END)
#         workflow.set_entry_point("process_llm")
#         return workflow.compile()

#     def _process_llm(self, state: ChatbotState, llm):
#         self.logger.debug("Processing LLM response")
#         messages = state["messages"]
        
#         try:
#             prompt = self.prompt_template.format(
#                 messages=messages,
#                 user_input=messages[-1].content
#             )
#         except KeyError as e:
#             self.logger.error(f"KeyError in prompt formatting: {str(e)}")
#             raise ValueError(f"Failed to format prompt due to missing key: {str(e)}")
        
#         try:
#             response = llm.invoke(prompt)
#             if not response.content.strip():
#                 self.logger.error("LLM returned empty response")
#                 raise ValueError("Empty response from LLM")
            
#             # Check for [END_CONVERSATION] in response
#             conversational_response = response.content
#             end_conversation = "[END_CONVERSATION]" in conversational_response
#             if end_conversation:
#                 conversational_response = conversational_response.replace("[END_CONVERSATION]", "").strip()
            
#             return {
#                 "messages": [AIMessage(content=conversational_response)],
#                 "end_conversation": end_conversation
#             }
#         except Exception as e:
#             self.logger.error(f"LLM processing failed for model {self.model_name}: {str(e)}", exc_info=True)
#             fallback_response = "Um, I'm sorry, something went wrong. Can you repeat that or clarify what you need?"
#             return {
#                 "messages": [AIMessage(content=fallback_response)],
#                 "end_conversation": False
#             }

#     def _save_conversation(self):
#         """
#         Save conversation history to CSV.
#         """
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
#                         f.seek(0)
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
#                             "timestamp", "llm_model"
#                         ])
#                     for msg in self.conversation_history:
#                         key = (str(self.conversation_id), msg["role"], msg["content"], msg["timestamp"])
#                         if key not in existing_entries:
#                             index_counter += 1
#                             row = [
#                                 str(index_counter),
#                                 str(self.conversation_id),
#                                 "non_autonomous",
#                                 msg["role"],
#                                 msg["content"],
#                                 msg["timestamp"],
#                                 self.model_name if msg["role"] == "victim" else ""
#                             ]
#                             writer.writerow(row)
#                             existing_entries.add(key)
#                             self.logger.debug(f"Wrote message to CSV: conversation_id={self.conversation_id}, sender={msg['role']}")
#             except Exception as e:
#                 self.logger.error(f"Error writing to CSV: {str(e)}", exc_info=True)

#     def process_query(self, query: str):
#         """
#         Process a query from the police agent and generate a victim response.

#         Args:
#             query (str): The input query from the police agent.

#         Returns:
#             Dict: Response with conversational text, conversation ID, and history.
#         """
#         if not query.strip():
#             self.logger.error("Query cannot be empty")
#             return {"error": "Query cannot be empty"}

#         state = {
#             "messages": [HumanMessage(content=query)],
#             "conversation_id": self.conversation_id
#         }
#         result = self.workflow.invoke(state)
#         conversational_response = result["messages"][-1].content
#         end_conversation = result.get("end_conversation", False)

#         self.conversation_history.append({
#             "role": "police",
#             "content": query,
#             "timestamp": datetime.now().isoformat()
#         })
#         self.conversation_history.append({
#             "role": "victim",
#             "content": conversational_response,
#             "timestamp": datetime.now().isoformat()
#         })

#         self.logger.debug(f"Processed query response: {conversational_response}")
#         response = {
#             "response": conversational_response,
#             "conversation_id": self.conversation_id,
#             "conversation_history": self.conversation_history
#         }
#         if end_conversation:
#             response["end_conversation": True]
#             self.end_conversation()
#         return response

#     def end_conversation(self):
#         """
#         End the conversation and save history to CSV.

#         Returns:
#             Dict: Status and conversation ID.
#         """
#         self._save_conversation()
#         self.logger.info(f"Conversation {self.conversation_id} saved to CSV")
#         return {"status": "Conversation ended", "conversation_id": self.conversation_id}

# if __name__ == "__main__":
    
    
#     settings = get_settings()
#     logger = setup_logger("VictimAgent", settings.log.subdirectories["agent"])
#     json_file = "data/victim_profile/victim_details.json"
    
    
#     # Generate victim_details.json only if it does not exist
#     if not Path(json_file).exists():
#         logger.info(f"JSON file {json_file} not found, running preprocessor")
#         try:
#             preprocessor = VictimProfilePreprocessor()
#             preprocessor.preprocess(output_file=json_file)
#             logger.info(f"Successfully generated {json_file}")
#         except Exception as e:
#             logger.error(f"Failed to run preprocessor: {str(e)}", exc_info=True)
#             raise

#     models = ["qwen2.5:7b", "granite3.2:8b", "mistral:7b"]
#     query = "Can you tell me about any recent scam incidents you’ve experienced?"
#     results = {}
#     num_reinitializations = 3  #adjust number of reinitialisations accordingly

#     logger.info("Starting model testing with multiple reinitializations")
#     for model_name in models:
#         logger.info(f"--- Testing model: {model_name} ---")
        
#         # Reset record counter before testing each model
#         record_counter = RecordCounter()
#         record_counter.reset()
#         logger.info(f"Reset record counter for model {model_name}")
        
#         model_results = []
#         for i in range(num_reinitializations):
#             logger.info(f"Reinitialization {i+1} for model {model_name}")
#             try:
#                 chatbot = VictimChatbot(model_name=model_name)
#                 response = chatbot.process_query(query)
                
#                 # Log victim details to verify prompt changes
#                 logger.info(f"Victim details for reinitialization {i+1}: {json.dumps(chatbot.victim_details, indent=2)}")
#                 model_results.append({
#                     "reinitialization": i + 1,
#                     "record_index": chatbot.record_index,
#                     "response": response
#                 })
#                 logger.info(f"Successfully processed query with model {model_name}, reinitialization {i+1}: {json.dumps(response, indent=2)}")
#                 chatbot.end_conversation()
#             except LangChainException as e:
#                 logger.error(f"LangChain error with model {model_name}, reinitialization {i+1}: {str(e)}", exc_info=True)
#                 model_results.append({
#                     "reinitialization": i + 1,
#                     "error": f"LangChain error: {str(e)}"
#                 })
#             except Exception as e:
#                 logger.error(f"Unexpected error with model {model_name}, reinitialization {i+1}: {str(e)}", exc_info=True)
#                 model_results.append({
#                     "reinitialization": i + 1,
#                     "error": f"Unexpected error: {str(e)}"
#                 })
#         results[model_name] = model_results

#     logger.info("Completed model testing")
#     print(json.dumps(results, indent=2))


# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_ollama import ChatOllama
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages
# from typing import TypedDict, Annotated, Sequence
# import json
# import logging
# from datetime import datetime
# from pathlib import Path
# from filelock import FileLock
# from config.settings import get_settings
# from config.logging_config import setup_logger
# from config.id_manager import IDManager
# from src.agents.utils import RecordCounter
# from src.agents.prompt import Prompt
# from langchain_core.exceptions import LangChainException
# import csv
# from src.preprocessing.preprocess import VictimProfilePreprocessor

# # Define state for the victim chatbot
# class ChatbotState(TypedDict):
#     messages: Annotated[Sequence[AnyMessage], add_messages]
#     conversation_id: int

# class VictimChatbot:
#     def __init__(self, model_name: str = "qwen2.5:7b", json_file: str = "data/victim_profile/victim_details.json"):
#         """
#         Initialize the VictimChatbot with dynamic user profiles, victim details, and scam details.

#         Args:
#             model_name (str): The LLM model name to use (default: "qwen2.5:7b").
#             json_file (str): Path to the JSON file with user profiles and scam details.
#         """
#         self.settings = get_settings()
#         self.logger = setup_logger("VictimAgent", self.settings.log.subdirectories["agent"])
#         self.id_manager = IDManager()
#         self.record_counter = RecordCounter()
#         self.model_name = model_name
#         self.conversation_id = self.id_manager.get_next_id()
#         self.json_file = json_file
#         self.conversation_history = []
#         self.messages: Sequence[AnyMessage] = []

#         # Load and select victim/scam details
#         self.records = self._load_records()
#         self.record_index = self.record_counter.get_next_index(len(self.records))
#         self.user_profile, self.victim_details, self.scam_details = self._get_current_record()

#         # Initialize prompt and workflow
#         self.prompt_template = self._generate_prompt_template()
#         self.llm = self._get_llm()
#         self.workflow = self._build_workflow()

#         self.logger.info(
#             f"Victim chatbot initialized: model={model_name}, "
#             f"conversation_id={self.conversation_id}, record_index={self.record_index}"
#         )

#     def _load_records(self) -> list:
#         """
#         Load user profiles, victim details, and scam details from the JSON file.

#         Returns:
#             list: List of records containing user_profile, victim_details, and scam_details.

#         Raises:
#             ValueError: If the JSON file is empty or invalid.
#         """
#         try:
#             with open(self.json_file, "r", encoding="utf-8") as f:
#                 records = json.load(f)
#             if not records:
#                 self.logger.error(f"No records found in {self.json_file}")
#                 raise ValueError(f"No records found in {self.json_file}")
#             return records
#         except Exception as e:
#             self.logger.error(f"Error loading records from {self.json_file}: {str(e)}", exc_info=True)
#             raise

#     def _get_current_record(self) -> tuple[dict, dict, dict]:
#         """
#         Get the current user profile, victim details, and scam details based on record_index.

#         Returns:
#             tuple: (user_profile, victim_details, scam_details)
#         """
#         record = self.records[self.record_index]
#         return (
#             record.get("user_profile", {}),
#             record.get("victim_details", {}),
#             record.get("scam_details", {})
#         )

#     @staticmethod
#     def _escape_template_braces(text: str) -> str:
#         """
#         Escape curly braces in text to prevent LangChain from interpreting them as placeholders.

#         Args:
#             text (str): Input text to escape.

#         Returns:
#             str: Escaped text with doubled braces.
#         """
#         return text.replace("{", "{{").replace("}", "}}")

#     def _generate_prompt_template(self) -> ChatPromptTemplate:
#         """
#         Generate the ChatPromptTemplate using the baseline_victim prompt.

#         Returns:
#             ChatPromptTemplate: Formatted prompt template with dynamic details.

#         Raises:
#             ValueError: If the prompt template is missing or cannot be formatted.
#         """
#         if "baseline_victim" not in Prompt.template:
#             self.logger.error("Prompt type 'baseline_victim' not found in Prompt.template")
#             raise ValueError("Prompt type 'baseline_victim' not found")

#         # Serialize and escape JSON strings
#         user_profile_json = json.dumps(self.user_profile, indent=2)
#         victim_details_json = json.dumps(self.victim_details, indent=2)
#         scam_details_json = json.dumps(self.scam_details, indent=2)
#         user_profile_str = self._escape_template_braces(user_profile_json)
#         victim_details_str = self._escape_template_braces(victim_details_json)
#         scam_details_str = self._escape_template_braces(scam_details_json)

#         # Format prompt with escaped JSON strings
#         try:
#             prompt_text = Prompt.template["baseline_victim"].format(
#                 user_profile=user_profile_str,
#                 victim_details=victim_details_str,
#                 scam_details=scam_details_str
#             )

#         except KeyError as e:
#             self.logger.error(f"Error formatting prompt template: {str(e)}")
#             raise ValueError(f"Missing key in prompt template: {str(e)}")

#         self.logger.debug(f"Formatted prompt text: {prompt_text[:500]}...")  # Truncate for logging
#         return ChatPromptTemplate.from_messages([
#             ("system", prompt_text),
#             MessagesPlaceholder(variable_name="messages"),
#             ("human", "{user_input}"),
#         ])

#     def _get_llm(self) -> ChatOllama:
#         """
#         Initialize the LLM without tool binding.

#         Returns:
#             ChatOllama: Configured LLM instance.
#         """
#         return ChatOllama(
#             model=self.model_name,
#             base_url=self.settings.agents.ollama_base_url,
#         )

#     def _build_workflow(self):
#         """
#         Build the LangGraph workflow with a single node for processing LLM responses.

#         Returns:
#             Compiled workflow object.
#         """
#         workflow = StateGraph(ChatbotState)
#         workflow.add_node("process_llm", self._process_llm)
#         workflow.add_edge("process_llm", END)
#         workflow.set_entry_point("process_llm")
#         return workflow.compile()

#     def _process_llm(self, state: ChatbotState):
#         """
#         Process the LLM response for the given state.

#         Args:
#             state (ChatbotState): Current chatbot state.

#         Returns:
#             dict: Updated state with response and end_conversation flag.
#         """
#         self.logger.debug("Processing LLM response")
#         messages = state["messages"]

#         try:
#             prompt = self.prompt_template.format(
#                 messages=messages,
#                 user_input=messages[-1].content
#             )
#         except KeyError as e:
#             self.logger.error(f"KeyError in prompt formatting: {str(e)}")
#             raise ValueError(f"Failed to format prompt: {str(e)}")

#         try:
#             response = self.llm.invoke(prompt)
#             if not response.content.strip():
#                 self.logger.error("LLM returned empty response")
#                 raise ValueError("Empty response from LLM")

#             conversational_response = response.content
#             end_conversation = "[END_CONVERSATION]" in conversational_response
#             if end_conversation:
#                 conversational_response = conversational_response.replace("[END_CONVERSATION]", "").strip()

#             return {
#                 "messages": [AIMessage(content=conversational_response)],
#                 "end_conversation": end_conversation
#             }
#         except LangChainException as e:
#             self.logger.error(f"LLM processing failed: {str(e)}", exc_info=True)
#             fallback_response = "Um, I'm sorry, something went wrong. Can you repeat that or clarify what you need?"
#             return {
#                 "messages": [AIMessage(content=fallback_response)],
#                 "end_conversation": False
#             }

#     def _save_conversation(self):
#         """
#         Save conversation history to CSV, aligning with police_agent.py structure.
#         """
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
#                         f.seek(0)
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
#                             "timestamp", "llm_model", "scam_details", "rag_invoked", "target_slot_map"
#                         ])
#                     for msg in self.conversation_history:
#                         key = (str(self.conversation_id), msg["role"], msg["content"], msg["timestamp"])
#                         if key not in existing_entries:
#                             index_counter += 1
#                             row = [
#                                 str(index_counter),
#                                 str(self.conversation_id),
#                                 "non_autonomous",
#                                 msg["role"],
#                                 msg["content"],
#                                 msg["timestamp"],
#                                 self.model_name if msg["role"] == "victim" else "",
#                                 "",  # scam_details empty for victim
#                                 "",  # rag_invoked empty for victim
#                                 ""   # target_slot_map empty for victim
#                             ]
#                             writer.writerow(row)
#                             existing_entries.add(key)
#                             self.logger.debug(
#                                 f"Wrote message to CSV: conversation_id={self.conversation_id}, sender={msg['role']}"
#                             )
#             except Exception as e:
#                 self.logger.error(f"Error writing to CSV: {str(e)}", exc_info=True)

#     def process_query(self, query: str):
#         """
#         Process a query from the police agent and generate a victim response.

#         Args:
#             query (str): The input query from the police agent.

#         Returns:
#             dict: Response with conversational text, conversation ID, history, and end flag.
#         """
#         if not query.strip():
#             self.logger.error("Query cannot be empty")
#             return {"error": "Query cannot be empty"}

#         state = {
#             "messages": self.messages + [HumanMessage(content=query)],
#             "conversation_id": self.conversation_id
#         }
#         result = self.workflow.invoke(state)
#         self.messages = result["messages"]
#         conversational_response = result["messages"][-1].content
#         end_conversation = result.get("end_conversation", False)

#         self.conversation_history.append({
#             "role": "police",
#             "content": query,
#             "timestamp": datetime.now().isoformat()
#         })
#         self.conversation_history.append({
#             "role": "victim",
#             "content": conversational_response,
#             "timestamp": datetime.now().isoformat()
#         })

#         self.logger.debug(f"Processed query response: {conversational_response}")
#         response = {
#             "response": conversational_response,
#             "conversation_id": self.conversation_id,
#             "conversation_history": self.conversation_history,
#             "end_conversation": end_conversation
#         }
#         if end_conversation:
#             self._save_conversation()
#         return response

#     def end_conversation(self):
#         """
#         End the conversation and save history to CSV.

#         Returns:
#             dict: Status and conversation ID.
#         """
#         self._save_conversation()
#         self.logger.info(f"Conversation {self.conversation_id} saved to CSV")
#         return {"status": "Conversation ended", "conversation_id": self.conversation_id}

# if __name__ == "__main__":
#     settings = get_settings()
#     logger = setup_logger("VictimAgent", settings.log.subdirectories["agent"])
#     json_file = "data/victim_profile/victim_details.json"

#     # Generate victim_details.json only if it does not exist
#     if not Path(json_file).exists():
#         logger.info(f"JSON file {json_file} not found, running preprocessor")
#         try:
#             preprocessor = VictimProfilePreprocessor()
#             preprocessor.preprocess(output_file=json_file)
#             logger.info(f"Successfully generated {json_file}")
#         except Exception as e:
#             logger.error(f"Failed to run preprocessor: {str(e)}", exc_info=True)
#             raise

#     models = ["qwen2.5:7b", "granite3.2:8b", "mistral:7b"]
#     query = "Can you tell me about any recent scam incidents you’ve experienced?"
#     results = {}
#     num_reinitializations = 3

#     logger.info("Starting model testing with multiple reinitializations")
#     for model_name in models:
#         logger.info(f"--- Testing model: {model_name} ---")
#         record_counter = RecordCounter()
#         record_counter.reset()
#         model_results = []
#         for i in range(num_reinitializations):
#             logger.info(f"Reinitialization {i+1} for model {model_name}")
#             try:
#                 chatbot = VictimChatbot(model_name=model_name, json_file=json_file)
#                 response = chatbot.process_query(query)
#                 logger.info(f"Victim profile: {json.dumps(chatbot.user_profile, indent=2)}")
#                 model_results.append({
#                     "reinitialization": i + 1,
#                     "record_index": chatbot.record_index,
#                     "response": response
#                 })
#                 logger.info(f"Processed query, reinitialization {i+1}: {json.dumps(response, indent=2)}")
#                 chatbot.end_conversation()
#             except LangChainException as e:
#                 logger.error(f"LangChain error, reinitialization {i+1}: {str(e)}", exc_info=True)
#                 model_results.append({
#                     "reinitialization": i + 1,
#                     "error": f"LangChain error: {str(e)}"
#                 })
#             except Exception as e:
#                 logger.error(f"Unexpected error, reinitialization {i+1}: {str(e)}", exc_info=True)
#                 model_results.append({
#                     "reinitialization": i + 1,
#                     "error": f"Unexpected error: {str(e)}"
#                 })
#         results[model_name] = model_results

#     logger.info("Completed model testing")
#     print(json.dumps(results, indent=2))





# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_ollama import ChatOllama
# from langgraph.graph import StateGraph, END
# from langgraph.graph.message import add_messages
# from typing import TypedDict, Annotated, Sequence, Optional
# import json
# import logging
# from datetime import datetime
# from pathlib import Path
# from filelock import FileLock
# from config.settings import get_settings
# from config.logging_config import setup_logger
# from config.id_manager import IDManager
# from src.agents.utils import RecordCounter
# from src.agents.prompt import Prompt
# from langchain_core.exceptions import LangChainException
# import csv
# from src.preprocessing.preprocess import VictimProfilePreprocessor
# from sentence_transformers import SentenceTransformer, util

# # Define state for the victim chatbot
# class ChatbotState(TypedDict):
#     messages: Annotated[Sequence[AnyMessage], add_messages]
#     # conversation_id: int

# class VictimChatbot:
#     def __init__(
#         self,
#         model_name: str = "qwen2.5:7b",
#         json_file: str = "data/victim_profile/victim_details.json",
#         user_profile: Optional[dict] = None,
#         victim_details: Optional[dict] = None,
#         scam_details: Optional[dict] = None
#     ):
#         """
#         Initialize the VictimChatbot with dynamic user profiles, victim details, and scam details.
#         If profiles/details are provided directly, use them; otherwise, load from JSON and cycle via RecordCounter.

#         Args:
#             model_name (str): The LLM model name to use (default: "qwen2.5:7b").
#             json_file (str): Path to the JSON file with user profiles and scam details (used if no direct params).
#             user_profile (Optional[dict]): Direct user profile dict (overrides JSON loading).
#             victim_details (Optional[dict]): Direct victim details dict (overrides JSON loading).
#             scam_details (Optional[dict]): Direct scam details dict (overrides JSON loading).
#         """
#         self.settings = get_settings()
#         self.logger = setup_logger("VictimAgent", self.settings.log.subdirectories["agent"])
#         # self.id_manager = IDManager()
#         self.record_counter = RecordCounter()
#         self.model_name = model_name
#         # self.conversation_id = self.id_manager.get_next_id()
#         self.json_file = json_file
#         # self.conversation_history: list[dict] = []
#         self.messages: Sequence[AnyMessage] = []

#         # Load or use provided profiles/details
#         if user_profile and victim_details and scam_details:
#             self.user_profile = user_profile
#             self.victim_details = victim_details
#             self.scam_details = scam_details
#             self.record_index = -1  # Indicate direct params (no index)
#             self.logger.debug("Using directly provided profiles/details")
#         else:
#             self.records = self._load_records()
#             self.record_index = self.record_counter.get_next_index(len(self.records))
#             self.user_profile, self.victim_details, self.scam_details = self._get_current_record()

#         # Initialize prompt and workflow
#         self.prompt_template = self._generate_prompt_template()
#         self.llm = self._get_llm()
#         self.workflow = self._build_workflow()
#         self.embedder = SentenceTransformer(self.settings.vector.embedding_model)

#         self.logger.info(
#             f"Victim chatbot initialized: model={model_name}, "
#             f"record_index={self.record_index}"
#         )

#     def _load_records(self) -> list:
#         """
#         Load user profiles, victim details, and scam details from the JSON file.

#         Returns:
#             list: List of records containing user_profile, victim_details, and scam_details.

#         Raises:
#             ValueError: If the JSON file is empty or invalid.
#         """
#         try:
#             with open(self.json_file, "r", encoding="utf-8") as f:
#                 records = json.load(f)
#             if not records:
#                 self.logger.error(f"No records found in {self.json_file}")
#                 raise ValueError(f"No records found in {self.json_file}")
#             return records
#         except Exception as e:
#             self.logger.error(f"Error loading records from {self.json_file}: {str(e)}", exc_info=True)
#             raise

#     def _get_current_record(self) -> tuple[dict, dict, dict]:
#         """
#         Get the current user profile, victim details, and scam details based on record_index.

#         Returns:
#             tuple: (user_profile, victim_details, scam_details)
#         """
#         record = self.records[self.record_index]
#         return (
#             record.get("user_profile", {}),
#             record.get("victim_details", {}),
#             record.get("scam_details", {})
#         )

#     @staticmethod
#     def _escape_template_braces(text: str) -> str:
#         """
#         Escape curly braces in text to prevent LangChain from interpreting them as placeholders.

#         Args:
#             text (str): Input text to escape.

#         Returns:
#             str: Escaped text with doubled braces.
#         """
#         return text.replace("{", "{{").replace("}", "}}")

#     def _generate_prompt_template(self) -> ChatPromptTemplate:
#         """
#         Generate the ChatPromptTemplate using the baseline_victim prompt.

#         Returns:
#             ChatPromptTemplate: Formatted prompt template with dynamic details.

#         Raises:
#             ValueError: If the prompt template is missing or cannot be formatted.
#         """
#         if "baseline_victim" not in Prompt.template:
#             self.logger.error("Prompt type 'baseline_victim' not found in Prompt.template")
#             raise ValueError("Prompt type 'baseline_victim' not found")

#         # Serialize and escape JSON strings
#         user_profile_json = json.dumps(self.user_profile, indent=2)
#         victim_details_json = json.dumps(self.victim_details, indent=2)
#         scam_details_json = json.dumps(self.scam_details, indent=2)
#         user_profile_str = self._escape_template_braces(user_profile_json)
#         victim_details_str = self._escape_template_braces(victim_details_json)
#         scam_details_str = self._escape_template_braces(scam_details_json)

#         # Format prompt with escaped JSON strings
#         try:
#             prompt_text = Prompt.template["baseline_victim"].format(
#                 user_profile=user_profile_str,
#                 victim_details=victim_details_str,
#                 scam_details=scam_details_str
#             )
#         except KeyError as e:
#             self.logger.error(f"Error formatting prompt template: {str(e)}")
#             raise ValueError(f"Missing key in prompt template: {str(e)}")

#         self.logger.debug(f"Formatted prompt text: {prompt_text[:500]}...")  # Truncate for logging
#         return ChatPromptTemplate.from_messages([
#             ("system", prompt_text),
#             MessagesPlaceholder(variable_name="messages"),
#             ("human", "{user_input}"),
#         ])

#     def _get_llm(self) -> ChatOllama:
#         """
#         Initialize the LLM without tool binding.

#         Returns:
#             ChatOllama: Configured LLM instance.
#         """
#         return ChatOllama(
#             model=self.model_name,
#             base_url=self.settings.agents.ollama_base_url,
#         )

#     def _build_workflow(self):
#         """
#         Build the LangGraph workflow with a single node for processing LLM responses.

#         Returns:
#             Compiled workflow object.
#         """
#         workflow = StateGraph(ChatbotState)
#         workflow.add_node("process_llm", self._process_llm)
#         workflow.add_edge("process_llm", END)
#         workflow.set_entry_point("process_llm")
#         return workflow.compile()

#     def _process_llm(self, state: ChatbotState):
#         """
#         Process the LLM response for the given state.

#         Args:
#             state (ChatbotState): Current chatbot state.

#         Returns:
#             dict: Updated state with response and end_conversation flag.
#         """
#         self.logger.debug("Processing LLM response")
#         messages = state["messages"]

#         try:
#             prompt = self.prompt_template.format(
#                 messages=messages,
#                 user_input=messages[-1].content
#             )
#         except KeyError as e:
#             self.logger.error(f"KeyError in prompt formatting: {str(e)}")
#             raise ValueError(f"Failed to format prompt: {str(e)}")

#         try:
#             response = self.llm.invoke(prompt)
#             if not response.content.strip():
#                 self.logger.error("LLM returned empty response")
#                 raise ValueError("Empty response from LLM")

#             conversational_response = response.content
#             end_conversation = "[END_CONVERSATION]" in conversational_response
#             if end_conversation:
#                 conversational_response = conversational_response.replace("[END_CONVERSATION]", "").strip()

#             return {
#                 "messages": [AIMessage(content=conversational_response)],
#                 "end_conversation": end_conversation
#             }
#         except LangChainException as e:
#             self.logger.error(f"LLM processing failed: {str(e)}", exc_info=True)
#             fallback_response = "Um, I'm sorry, something went wrong. Can you repeat that or clarify what you need?"
#             return {
#                 "messages": [AIMessage(content=fallback_response)],
#                 "end_conversation": False
#             }


#     def process_query(self, query: str):
#         """
#         Process a query from the police agent and generate a victim response.

#         Args:
#             query (str): The input query from the police agent.

#         Returns:
#             dict: Response with conversational text, conversation ID, history, and end flag.
#         """
#         if not query.strip():
#             self.logger.error("Query cannot be empty")
#             return {"error": "Query cannot be empty"}

#         state = {
#             "messages": self.messages + [HumanMessage(content=query)],
#             # "conversation_id": self.conversation_id
#         }
#         result = self.workflow.invoke(state)
#         self.messages = result["messages"]
#         conversational_response = result["messages"][-1].content
#         end_conversation = result.get("end_conversation", False)

#         # self.conversation_history.append({
#         #     "role": "police",
#         #     "content": query,
#         #     "timestamp": datetime.now().isoformat()
#         # })
#         # self.conversation_history.append({
#         #     "role": "victim",
#         #     "content": conversational_response,
#         #     "timestamp": datetime.now().isoformat()
#         # })
        
#         self.logger.debug(f"Processed query response: {conversational_response}")
        
#         # New: Handle empty or invalid response
#         if not conversational_response.strip():
#             self.logger.warning("Empty LLM response detected; using fallback")
#             conversational_response = "I think that's all I know. [END_CONVERSATION]"
#             end_conversation = True

#         # # New: Check if police query is a summary (no questions)
#         # if not any(c in query.lower() for c in ["?", "what", "when", "where", "how", "who"]) and not end_conversation:
#         #     self.logger.debug("Police query appears to be a summary; checking if all details shared")
#         #     past_victim_responses = [msg.content for msg in self.messages if isinstance(msg, AIMessage)]
#         #     if len(past_victim_responses) >= 4:  # Align with prompt's minimum 4 turns
#         #         self.logger.info("Turn count >= 4 and no questions; ending conversation")
#         #         conversational_response += " [END_CONVERSATION]"
#         #         end_conversation = True
        
#         # New: Use state (self.messages) for past victim responses
#         past_victim_responses = [msg.content for msg in self.messages if isinstance(msg, AIMessage)]
        
#         # Check for repetition (after workflow, before building response)
#         if not end_conversation and len(past_victim_responses) >= 3:  # >=3 for -3: reliability
#             try:
#                 recent_responses = past_victim_responses[-2:]
#                 embeddings = self.embedder.encode(recent_responses)
                
#                 sim_scores = []
#                 for i in range(1, len(embeddings)):
#                     sim = util.cos_sim(embeddings[-1], embeddings[-1 - i])[0][0].item()
#                     sim_scores.append(sim)
#                 avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0
                
#                 if avg_sim > 0.90:
#                     self.logger.warning(f"High repetition detected (avg_sim={avg_sim:.2f}); forcing end.")
#                     conversational_response += " [END_CONVERSATION]"
#                     end_conversation = True
#                     # Update history to reflect change
#                     # self.conversation_history[-1]["content"] = conversational_response
#             except Exception as e:
#                 self.logger.error(f"Embedding failed: {e}; falling back to string check.")
#                 # Fallback: Exact match on last two
#                 if past_victim_responses[-1] == past_victim_responses[-2]:
#                     self.logger.warning("Exact repetition detected; forcing end.")
#                     conversational_response += " [END_CONVERSATION]"
#                     end_conversation = True
#                     # self.conversation_history[-1]["content"] = conversational_response

#         response = {
#             "response": conversational_response,
#             # "conversation_id": self.conversation_id,
#             # "conversation_history": self.conversation_history,
#             "end_conversation": end_conversation
#         }
        

                
#         # if end_conversation:
#         #     self._save_conversation()
#         return response

#     def reset_state(self):
#         """Reset internal state (messages and history) for reuse in new conversations."""
#         self.messages = []
#         self.conversation_history = []
#         self.logger.debug("VictimChatbot state reset")

#     def end_conversation(self):
#         """
#         End the conversation and save history to CSV.

#         Returns:
#             dict: Status and conversation ID.
#         """
#         # self._save_conversation()
#         self.logger.info(f"Conversation ended (save handled by manager)")
#         return {"status": "Conversation ended"}

# # if __name__ == "__main__":
# #     settings = get_settings()
# #     logger = setup_logger("VictimAgent", settings.log.subdirectories["agent"])
# #     json_file = "data/victim_profile/victim_details.json"

# #     # Generate victim_details.json only if it does not exist
# #     if not Path(json_file).exists():
# #         logger.info(f"JSON file {json_file} not found, running preprocessor")
# #         try:
# #             preprocessor = VictimProfilePreprocessor()
# #             preprocessor.preprocess(output_file=json_file)
# #             logger.info(f"Successfully generated {json_file}")
# #         except Exception as e:
# #             logger.error(f"Failed to run preprocessor: {str(e)}", exc_info=True)
# #             raise

# #     models = ["qwen2.5:7b", "granite3.2:8b", "mistral:7b"]
# #     query = "Can you tell me about any recent scam incidents you’ve experienced?"
# #     results = {}
# #     num_reinitializations = 3

# #     logger.info("Starting model testing with multiple reinitializations")
# #     for model_name in models:
# #         logger.info(f"--- Testing model: {model_name} ---")
# #         record_counter = RecordCounter()
# #         record_counter.reset()
# #         model_results = []
# #         for i in range(num_reinitializations):
# #             logger.info(f"Reinitialization {i+1} for model {model_name}")
# #             try:
# #                 chatbot = VictimChatbot(model_name=model_name, json_file=json_file)
# #                 response = chatbot.process_query(query)
# #                 logger.info(f"Victim profile: {json.dumps(chatbot.user_profile, indent=2)}")
# #                 model_results.append({
# #                     "reinitialization": i + 1,
# #                     "record_index": chatbot.record_index,
# #                     "response": response
# #                 })
# #                 logger.info(f"Processed query, reinitialization {i+1}: {json.dumps(response, indent=2)}")
# #                 chatbot.end_conversation()
# #             except LangChainException as e:
# #                 logger.error(f"LangChain error, reinitialization {i+1}: {str(e)}", exc_info=True)
# #                 model_results.append({
# #                     "reinitialization": i + 1,
# #                     "error": f"LangChain error: {str(e)}"
# #                 })
# #             except Exception as e:
# #                 logger.error(f"Unexpected error, reinitialization {i+1}: {str(e)}", exc_info=True)
# #                 model_results.append({
# #                     "reinitialization": i + 1,
# #                     "error": f"Unexpected error: {str(e)}"
# #                 })
# #         results[model_name] = model_results

# #     logger.info("Completed model testing")
# #     print(json.dumps(results, indent=2))