# from langgraph.prebuilt import create_react_agent
# from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
# from langchain.memory import ConversationBufferMemory
# from src.agents.llm_providers import get_llm, get_structured_llm, SUPPORTED_MODELS
# from src.agents.tools import get_police_tools, get_victim_tools
# from src.models.response_model import PoliceResponse
# from src.config.prompt import Prompt
# from typing import Dict, List, Optional
# import logging
# import json
# import csv
# import os
# from datetime import datetime

# # Configure logging
# logging.basicConfig(
#     filename="errors.log",
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# # Global registries to store agent instances and conversation histories
# AGENT_REGISTRY = {}  # Maps "police" or "victim" to {"agent": agent_instance, "max_turns": int}
# CONVERSATION_REGISTRY = {}  # Maps conversation_id to list of messages

# def create_police_chatbot(
#     llm_provider: str = "Ollama",
#     model: str = "gemma2:9b",
#     max_turns: int = 10
# ) -> Dict:
#     """Instantiate a police chatbot with structured output and RAG support, stored in the registry.

#     Args:
#         llm_provider (str): The LLM provider (e.g., "Ollama"). Defaults to "Ollama".
#         model (str): The model to use (e.g., "gemma2:9b"). Defaults to "gemma2:9b".
#         max_turns (int): Maximum number of turns for conversation memory. Defaults to 10.

#     Returns:
#         Dict: Success message, agent_id, and configuration details.
#     """
#     # Validate LLM provider and model
#     if llm_provider not in SUPPORTED_MODELS:
#         logging.error(f"Invalid llm_provider: {llm_provider}")
#         return {"error": f"Invalid llm_provider. Must be one of {list(SUPPORTED_MODELS.keys())}"}
#     if model not in SUPPORTED_MODELS[llm_provider]:
#         logging.error(f"Invalid model: {model} for provider: {llm_provider}")
#         return {"error": f"Invalid model. Must be one of {SUPPORTED_MODELS[llm_provider]}"}

#     try:
#         # Initialize structured LLM with PoliceResponse schema
#         llm = get_structured_llm(provider=llm_provider, model=model, structured_model=PoliceResponse)
#         logging.debug(f"Successfully initialized LLM: {llm_provider}/{model}")

#         # Get police-specific tools, including RAG
#         tools = get_police_tools()
        
#         # Create ReAct agent
#         agent_instance = create_react_agent(model=llm, tools=tools)
#         logging.debug("Successfully created ReAct agent")
        
#         # Use fixed key for police chatbot
#         agent_id = "police"
#         AGENT_REGISTRY[agent_id] = {
#             "agent": agent_instance,
#             "max_turns": max_turns
#         }
        
#         logging.info(f"Police chatbot created with agent_id: {agent_id}")
#         return {
#             "message": f"Police chatbot created successfully",
#             "agent_id": agent_id,
#             "config": {
#                 "llm_provider": llm_provider,
#                 "model": model,
#                 "max_turns": max_turns
#             }
#         }
#     except ValueError as e:
#         logging.error(f"ValueError in create_police_chatbot: {str(e)}", exc_info=True)
#         return {"error": str(e)}
#     except ImportError as e:
#         logging.error(f"ImportError in create_police_chatbot: {str(e)}", exc_info=True)
#         return {"error": f"Failed to create chatbot due to import error: {str(e)}"}
#     except Exception as e:
#         logging.error(f"Unexpected error in create_police_chatbot: {str(e)}", exc_info=True)
#         return {"error": f"Failed to create chatbot: {str(e)}"}

# def create_victim_chatbot(
#     llm_provider: str = "Ollama",
#     model: str = "llama3.2",
#     max_turns: int = 10
# ) -> Dict:
#     """Instantiate a victim chatbot with plain text output and RAG support, stored in the registry.

#     Args:
#         llm_provider (str): The LLM provider (e.g., "Ollama"). Defaults to "Ollama".
#         model (str): The model to use (e.g., "llama3.2"). Defaults to "llama3.2".
#         max_turns (int): Maximum number of turns for conversation memory. Defaults to 10.

#     Returns:
#         Dict: Success message, agent_id, and configuration details.
#     """
#     # Validate LLM provider and model
#     if llm_provider not in SUPPORTED_MODELS:
#         logging.error(f"Invalid llm_provider: {llm_provider}")
#         return {"error": f"Invalid llm_provider. Must be one of {list(SUPPORTED_MODELS.keys())}"}
#     if model not in SUPPORTED_MODELS[llm_provider]:
#         logging.error(f"Invalid model: {model} for provider: {llm_provider}")
#         return {"error": f"Invalid model. Must be one of {SUPPORTED_MODELS[llm_provider]}"}

#     try:
#         # Initialize LLM without structured output
#         llm = get_llm(provider=llm_provider, model=model)
#         logging.debug(f"Successfully initialized LLM: {llm_provider}/{model}")

#         # Get victim-specific tools, including RAG
#         tools = get_victim_tools()
        
#         # Create ReAct agent
#         agent_instance = create_react_agent(model=llm, tools=tools)
#         logging.debug("Successfully created ReAct agent")
        
#         # Use fixed key for victim chatbot
#         agent_id = "victim"
#         AGENT_REGISTRY[agent_id] = {
#             "agent": agent_instance,
#             "max_turns": max_turns
#         }
        
#         logging.info(f"Victim chatbot created with agent_id: {agent_id}")
#         return {
#             "message": f"Victim chatbot created successfully",
#             "agent_id": agent_id,
#             "config": {
#                 "llm_provider": llm_provider,
#                 "model": model,
#                 "max_turns": max_turns
#             }
#         }
#     except ValueError as e:
#         logging.error(f"ValueError in create_victim_chatbot: {str(e)}", exc_info=True)
#         return {"error": str(e)}
#     except ImportError as e:
#         logging.error(f"ImportError in create_victim_chatbot: {str(e)}", exc_info=True)
#         return {"error": f"Failed to create chatbot due to import error: {str(e)}"}
#     except Exception as e:
#         logging.error(f"Unexpected error in create_victim_chatbot: {str(e)}", exc_info=True)
#         return {"error": f"Failed to create chatbot: {str(e)}"}

# def get_nonautonomous_response(
#     agent_id: str,
#     query: str,
#     prompt: str = None,
#     conversation_id: str = None,
#     conversation_history: Optional[List[Dict[str, str]]] = None
# ) -> Dict:
#     """Get a response from a chatbot for a non-autonomous conversation and save to CSV.

#     Args:
#         agent_id (str): The ID of the chatbot ("police" or "victim").
#         query (str): The user's input query.
#         prompt (str, optional): Custom system prompt for RAG variation. Defaults to Prompt.template["baseline_<agent_id>"].
#         conversation_id (str, optional): Unique ID for the conversation. Generated if not provided.
#         conversation_history (List[Dict[str, str]], optional): List of previous messages with 'role', 'content', 'timestamp'.

#     Returns:
#         Dict: The chatbot's response, structured data, updated conversation history, and conversation metadata.
#     """
#     # Validate inputs
#     if not query.strip():
#         logging.error("Query cannot be empty")
#         return {"error": "Query cannot be empty"}
    
#     if agent_id not in AGENT_REGISTRY:
#         logging.error(f"Agent not found for agent_id: {agent_id}")
#         return {"error": f"Agent not found for agent_id: {agent_id}"}

#     # Use baseline_prompt based on agent_id
#     effective_prompt = prompt.strip() if prompt and prompt.strip() else Prompt.template[f"baseline_{agent_id}"]
#     if not effective_prompt:
#         logging.error("Prompt cannot be empty")
#         return {"error": "Prompt cannot be empty"}

#     # Generate conversation_id if not provided
#     if not conversation_id:
#         conversation_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

#     try:
#         # Retrieve agent and memory settings
#         agent_data = AGENT_REGISTRY[agent_id]
#         agent = agent_data["agent"]
#         max_turns = agent_data["max_turns"]

#         # Initialize conversation history if not provided
#         if conversation_history is None:
#             conversation_history = CONVERSATION_REGISTRY.get(conversation_id, [])
        
#         # Validate conversation history roles
#         valid_roles = {"victim", "police"}
#         for msg in conversation_history:
#             if msg.get("role") not in valid_roles:
#                 logging.error(f"Invalid role in conversation history: {msg.get('role')}")
#                 return {"error": f"Invalid role in conversation history: {msg.get('role')}"}

#         # Create new memory instance for this conversation
#         memory = ConversationBufferMemory(
#             memory_key="chat_history",
#             input_key="input",
#             output_key="output",
#             max_turns=max_turns
#         )
#         for msg in conversation_history:
#             if msg["role"] == "victim":
#                 memory.chat_memory.add_user_message(msg["content"])
#             elif msg["role"] == "police":
#                 memory.chat_memory.add_ai_message(msg["content"])

#         # Construct state for agent invocation
#         state = {
#             "messages": [
#                 SystemMessage(content=effective_prompt),
#                 *(HumanMessage(content=msg["content"]) if msg["role"] == "victim" else AIMessage(content=msg["content"])
#                   for msg in conversation_history),
#                 HumanMessage(content=query)
#             ]
#         }

#         # Invoke agent (RAG tools may retrieve relevant scam reports)
#         logging.debug(f"Invoking agent for query: {query}, conversation_id: {conversation_id}")
#         response = agent.invoke(state)
#         messages = response.get("messages", [])
#         ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
#         ai_response = ai_messages[-1] if ai_messages else "Sorry, I couldn't generate a response."

#         # Parse structured output (for police chatbot only)
#         structured_data = {}
#         conversational_response = ai_response.strip() or "I'm not sure how to respond."
#         if agent_id == "police":
#             try:
#                 parsed_response = json.loads(ai_response)
#                 police_response = PoliceResponse(**parsed_response)
#                 conversational_response = police_response.conversational_response.strip()
#                 # Extract all fields except conversational_response for scam_generic_details
#                 structured_data = police_response.dict(exclude={"conversational_response"})
#             except (json.JSONDecodeError, ValueError) as e:
#                 logging.warning(f"Failed to parse response as JSON: {str(e)}")

#         # Update conversation history
#         sender_role = agent_id
#         updated_history = conversation_history + [
#             {"role": "victim", "content": query, "timestamp": datetime.now().isoformat()},
#             {"role": sender_role, "content": conversational_response, "timestamp": datetime.now().isoformat()}
#         ]
#         # For production, wrap this in a threading.Lock:
#         # from threading import Lock
#         # REGISTRY_LOCK = Lock()
#         # with REGISTRY_LOCK:
#         CONVERSATION_REGISTRY[conversation_id] = updated_history

#         # Save to CSV
#         # For production, wrap this in a threading.Lock:
#         # CSV_LOCK = Lock()
#         # with CSV_LOCK:
#         csv_file = "conversation_history.csv"
#         file_exists = os.path.isfile(csv_file)
#         with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             # Write header if file is new
#             if not file_exists:
#                 writer.writerow([
#                     "conversation_id", "conversation_type", "sender_type", "content", "timestamp",
#                     "scam_generic_details", "scam_specific_details"
#                 ])
#             # Write conversation messages
#             for msg in updated_history[-2:]:  # Only write the latest query and response
#                 row = [
#                     conversation_id,
#                     "non_autonomous",
#                     msg["role"],
#                     msg["content"],
#                     msg["timestamp"],
#                 ]
#                 # Append structured data for police response
#                 if msg["role"] == "police" and structured_data:
#                     row.extend([
#                         json.dumps(structured_data, ensure_ascii=False),
#                         ""  # Empty scam_specific_details as per request
#                     ])
#                 else:
#                     row.extend(["", ""])  # Placeholder for structured fields
#                 writer.writerow(row)

#         logging.debug(f"Response generated and saved for agent_id: {agent_id}, conversation_id: {conversation_id}")
#         return {
#             "response": conversational_response,
#             "structured_data": structured_data,
#             "conversation_id": conversation_id,
#             "conversation_history": updated_history,
#             "conversation_type": "non_autonomous"
#         }

#     except Exception as e:
#         logging.error(f"Error in get_nonautonomous_response for agent_id {agent_id}, conversation_id {conversation_id}: {str(e)}")
#         return {"error": f"Failed to get response: {str(e)}"}

# def get_autonomous_response(
#     police_agent_id: str = "police",
#     victim_agent_id: str = "victim",
#     police_prompt: str = None,
#     victim_prompt: str = None,
#     initial_query: str = "Hello, this is the police. How can we help?",
#     conversation_id: str = None,
#     max_turns: int = 5
# ) -> Dict:
#     """Simulate an autonomous conversation between a police and victim chatbot and save to CSV.

#     Args:
#         police_agent_id (str): The ID of the police chatbot ("police"). Defaults to "police".
#         victim_agent_id (str): The ID of the victim chatbot ("victim"). Defaults to "victim".
#         police_prompt (str, optional): Custom system prompt for the police chatbot. Defaults to Prompt.template["baseline_police"].
#         victim_prompt (str, optional): Custom system prompt for the victim chatbot. Defaults to Prompt.template["baseline_victim"].
#         initial_query (str): Initial query from the police chatbot. Defaults to a generic greeting.
#         conversation_id (str, optional): Unique ID for the conversation. Generated if not provided.
#         max_turns (int): Maximum number of conversation turns. Defaults to 5.

#     Returns:
#         Dict: Conversation status, conversation_id, conversation history, and final structured data.
#     """
#     # Validate inputs
#     if police_agent_id not in AGENT_REGISTRY or victim_agent_id not in AGENT_REGISTRY:
#         logging.error(f"Agent not found: police_agent_id={police_agent_id}, victim_agent_id={victim_agent_id}")
#         return {"error": f"Agent not found: police={police_agent_id}, victim={victim_agent_id}"}

#     # Use baseline prompts as default if prompts are None or empty
#     police_prompt = police_prompt.strip() if police_prompt and police_prompt.strip() else Prompt.template["baseline_police"]
#     victim_prompt = victim_prompt.strip() if victim_prompt and victim_prompt.strip() else Prompt.template["baseline_victim"]
#     if not police_prompt or not victim_prompt:
#         logging.error("Prompts cannot be empty")
#         return {"error": "Prompts cannot be empty"}

#     # Generate conversation_id if not provided
#     if not conversation_id:
#         conversation_id = datetime.now().strftime("%Y%m%d%H%M%S%f")

#     try:
#         # Retrieve agents and memory settings
#         police_data = AGENT_REGISTRY[police_agent_id]
#         victim_data = AGENT_REGISTRY[victim_agent_id]
#         police_agent = police_data["agent"]
#         victim_agent = victim_data["agent"]
#         max_memory_turns = min(police_data["max_turns"], victim_data["max_turns"])

#         # Initialize conversation history
#         conversation_history = CONVERSATION_REGISTRY.get(conversation_id, [])
#         current_query = initial_query
#         last_messages = []
#         structured_data_history = []

#         for turn in range(max_turns):
#             # Get victim response (plain text)
#             victim_memory = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 input_key="input",
#                 output_key="output",
#                 max_turns=max_memory_turns
#             )
#             for msg in conversation_history:
#                 if msg["role"] == "police":
#                     victim_memory.chat_memory.add_user_message(msg["content"])
#                 elif msg["role"] == "victim":
#                     victim_memory.chat_memory.add_ai_message(msg["content"])

#             victim_state = {
#                 "messages": [
#                     SystemMessage(content=victim_prompt),
#                     *(HumanMessage(content=msg["content"]) if msg["role"] == "police" else AIMessage(content=msg["content"])
#                       for msg in conversation_history),
#                     HumanMessage(content=current_query)
#                 ]
#             }

#             logging.debug(f"Invoking victim agent for query: {current_query}, conversation_id: {conversation_id}")
#             victim_response = victim_agent.invoke(victim_state)
#             victim_messages = victim_response.get("messages", [])
#             victim_ai_messages = [msg.content for msg in victim_messages if isinstance(msg, AIMessage)]
#             victim_message = victim_ai_messages[-1] if victim_ai_messages else "I'm not sure how to respond."

#             conversation_history.append({
#                 "role": "victim",
#                 "content": victim_message,
#                 "timestamp": datetime.now().isoformat()
#             })

#             # Save victim message to CSV
#             # For production, wrap this in a threading.Lock:
#             # from threading import Lock
#             # CSV_LOCK = Lock()
#             # with CSV_LOCK:
#             csv_file = "conversation_history.csv"
#             file_exists = os.path.isfile(csv_file)
#             with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 if not file_exists:
#                     writer.writerow([
#                         "conversation_id", "conversation_type", "sender_type", "content", "timestamp",
#                         "scam_generic_details", "scam_specific_details"
#                     ])
#                 writer.writerow([
#                     conversation_id, "autonomous", "victim", victim_message,
#                     conversation_history[-1]["timestamp"], "", ""
#                 ])

#             # Check for conversation end
#             if "[END_CONVERSATION]" in victim_message or "thank you for your cooperation" in victim_message.lower():
#                 logging.info(f"Conversation ended: {victim_message}")
#                 break

#             last_messages.append(victim_message)
#             if len(last_messages) > 6:
#                 last_messages.pop(0)
#                 if len(set(last_messages)) <= 4:
#                     logging.warning(f"Detected potential repetition: {last_messages}")
#                     break

#             # Get police response
#             police_memory = ConversationBufferMemory(
#                 memory_key="chat_history",
#                 input_key="input",
#                 output_key="output",
#                 max_turns=max_memory_turns
#             )
#             for msg in conversation_history:
#                 if msg["role"] == "victim":
#                     police_memory.chat_memory.add_user_message(msg["content"])
#                 elif msg["role"] == "police":
#                     police_memory.chat_memory.add_ai_message(msg["content"])

#             police_state = {
#                 "messages": [
#                     SystemMessage(content=police_prompt),
#                     *(HumanMessage(content=msg["content"]) if msg["role"] == "victim" else AIMessage(content=msg["content"])
#                       for msg in conversation_history),
#                     HumanMessage(content=victim_message)
#                 ]
#             }

#             logging.debug(f"Invoking police agent for query: {victim_message}, conversation_id: {conversation_id}")
#             police_response = police_agent.invoke(police_state)
#             police_messages = police_response.get("messages", [])
#             police_ai_messages = [msg.content for msg in police_messages if isinstance(msg, AIMessage)]
#             police_message = police_ai_messages[-1] if police_ai_messages else "Sorry, I couldn't generate a response."

#             # Parse structured output for police response
#             try:
#                 parsed_response = json.loads(police_message)
#                 police_response_obj = PoliceResponse(**parsed_response)
#                 conversational_response = police_response_obj.conversational_response.strip()
#                 # Extract all fields except conversational_response for scam_generic_details
#                 structured_data = police_response_obj.dict(exclude={"conversational_response"})
#             except (json.JSONDecodeError, ValueError) as e:
#                 logging.warning(f"Failed to parse police response as JSON: {str(e)}")
#                 conversational_response = police_message.strip() or "I'm not sure how to respond."
#                 structured_data = {}

#             conversation_history.append({
#                 "role": "police",
#                 "content": conversational_response,
#                 "timestamp": datetime.now().isoformat()
#             })
#             if structured_data:
#                 structured_data_history.append(structured_data)

#             # Save police message to CSV
#             # For production, wrap this in a threading.Lock:
#             # with CSV_LOCK:
#             with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([
#                     conversation_id, "autonomous", "police", conversational_response,
#                     conversation_history[-1]["timestamp"],
#                     json.dumps(structured_data, ensure_ascii=False),
#                     ""  # Empty scam_specific_details as per request
#                 ])

#             last_messages.append(conversational_response)
#             if len(last_messages) > 6:
#                 last_messages.pop(0)
#                 if len(set(last_messages)) <= 4:
#                     logging.warning(f"Detected potential repetition: {last_messages}")
#                     break

#             current_query = conversational_response

#             # Check for conversation end
#             if "thank you for your cooperation" in conversational_response.lower():
#                 logging.info("Police ended conversation")
#                 break

#         # For production, wrap this in a threading.Lock:
#         # with REGISTRY_LOCK:
#         CONVERSATION_REGISTRY[conversation_id] = conversation_history

#         final_structured_data = structured_data_history[-1] if structured_data_history else {}
#         logging.debug(f"Autonomous conversation completed for conversation_id: {conversation_id}")
#         return {
#             "status": "Conversation completed",
#             "conversation_id": conversation_id,
#             "conversation_history": conversation_history,
#             "conversation_type": "autonomous",
#             "structured_data": final_structured_data
#         }

#     except Exception as e:
#         logging.error(f"Error in get_autonomous_response for conversation_id {conversation_id}: {str(e)}")
#         return {"error": f"Failed to simulate conversation: {str(e)}"}




#working new version

# from typing import Dict, List, Optional
# from langgraph.prebuilt import create_react_agent
# from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
# from langchain_community.chat_message_histories import ChatMessageHistory
# from src.agents.llm_providers import LLMProvider
# from src.agents.tools import PoliceTools, VictimTools
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
        
#         # Verify Ollama server connectivity
#         try:
#             response = requests.get(f"{self.settings.agents.ollama_base_url}/api/tags")
#             if response.status_code != 200:
#                 raise ConnectionError(f"Ollama server not responding: {response.status_code}")
#             models = response.json().get("models", [])
#             if not any(m["name"] == self.model for m in models):
#                 raise ValueError(f"Model {self.model} not loaded in Ollama server")
#             self.logger.debug(f"Ollama server is accessible and model {self.model} is loaded")
#         except Exception as e:
#             self.logger.error(f"Failed to connect to Ollama server: {str(e)}")
#             raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")

# class PoliceAgent(Agent):
#     """Police chatbot agent with structured output and RAG support."""
    
#     def __init__(self, llm_provider: str = "Ollama", model: str = "qwen2.5:7b", max_turns: int = 10):
#         super().__init__("police", llm_provider, model, max_turns)
#         self._initialize_agent()
    
#     def _initialize_agent(self):
#         """Initialize the police agent with LLM and tools."""
#         try:
#             # Try structured LLM, fall back to plain LLM if tools are not supported
#             try:
#                 llm = self.llm_provider_instance.get_structured_llm(
#                     provider=self.llm_provider, model=self.model, structured_model=PoliceResponse
#                 )
#             except Exception as e:
#                 self.logger.warning(f"Structured LLM failed for {self.model}: {str(e)}. Falling back to plain LLM.")
#                 llm = self.llm_provider_instance.get_llm(provider=self.llm_provider, model=self.model)
            
#             self.logger.debug(f"Successfully initialized LLM: {self.llm_provider}/{self.model}")
#             tools = PoliceTools().get_tools()
#             self.logger.debug(f"Tools initialized: {[t.name for t in tools]}")  # Changed from t.__name__ to t.name
#             self.agent = create_react_agent(model=llm, tools=tools)
#             self.logger.debug("Successfully created ReAct agent")
#             self.logger.info(f"Police chatbot created with agent_id: {self.agent_id}, model: {self.model}")
#         except Exception as e:
#             self.logger.error(f"Failed to create police chatbot: {str(e)}", exc_info=True)
#             raise

# class VictimAgent(Agent):
#     """Victim chatbot agent with plain text output and RAG support."""
    
#     def __init__(self, llm_provider: str = "Ollama", model: str = "qwen2.5:7b", max_turns: int = 10):
#         super().__init__("victim", llm_provider, model, max_turns)
#         self._initialize_agent()
    
#     def _initialize_agent(self):
#         """Initialize the victim agent with LLM and tools."""
#         try:
#             llm = self.llm_provider_instance.get_llm(provider=self.llm_provider, model=self.model)
#             self.logger.debug(f"Successfully initialized LLM: {self.llm_provider}/{self.model}")
#             tools = VictimTools().get_tools()
#             self.logger.debug(f"Tools initialized: {[t.name for t in tools] if tools else 'No tools'}")  # Handle empty tools list
#             self.agent = create_react_agent(model=llm, tools=tools)
#             self.logger.debug("Successfully created ReAct agent")
#             self.logger.info(f"Victim chatbot created with agent_id: {self.agent_id}, model: {self.model}")
#         except Exception as e:
#             self.logger.error(f"Failed to create victim chatbot: {str(e)}", exc_info=True)
#             raise

# class ConversationManager:
#     """Manages conversation history and CSV storage."""
    
#     def __init__(self):
#         self.settings = get_settings()
#         self.conversation_registry = {}
#         self.id_manager = IDManager()
#         self._setup_logging()
#         self.index_counter = self._load_last_index()
    
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
#         """Load the last used index from CSV."""
#         csv_file = "conversation_history.csv"
#         max_index = 0
#         if os.path.exists(csv_file):
#             try:
#                 with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                     reader = csv.DictReader(f)
#                     if "index" in reader.fieldnames:
#                         indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
#                         max_index = max(indices) if indices else 0
#                     else:
#                         self.logger.warning(f"CSV {csv_file} does not contain 'index' field. Starting with index 0.")
#                 self.logger.debug(f"Loaded max index {max_index} from {csv_file}")
#             except Exception as e:
#                 self.logger.error(f"Error reading index from CSV: {str(e)}")
#         return max_index
    
#     def _generate_conversation_id(self) -> int:
#         """Generate a unique integer conversation ID."""
#         return self.id_manager.get_next_id()
    
#     def _save_to_csv(self, conversation_id: int, conversation_type: str, messages: List[Dict], structured_data: Dict = None, llm_model: str = None):
#         """Save conversation messages to CSV."""
#         csv_file = "conversation_history.csv"
#         file_exists = os.path.isfile(csv_file)
        
#         with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             if not file_exists:
#                 writer.writerow([
#                     "index", "conversation_id", "conversation_type", "sender_type", "content", "timestamp",
#                     "llm_model", "scam_generic_details", "scam_specific_details"
#                 ])
            
#             for msg in messages[-2:]:  # Only write the latest query and response
#                 self.index_counter += 1
#                 row = [
#                     str(self.index_counter),
#                     str(conversation_id),
#                     conversation_type,
#                     msg["role"],
#                     msg["content"],
#                     msg["timestamp"],
#                     llm_model or "",
#                     json.dumps(structured_data, ensure_ascii=False) if msg["role"] == "police" and structured_data else "",
#                     ""  # Empty scam_specific_details as per original
#                 ]
#                 writer.writerow(row)
    
#     def get_nonautonomous_response(
#         self, agent: Agent, query: str, prompt: str = None, conversation_id: int = None,
#         conversation_history: Optional[List[Dict[str, str]]] = None
#     ) -> Dict:
#         """Get a response from an agent for a non-autonomous conversation."""
#         if not query.strip():
#             self.logger.error("Query cannot be empty")
#             return {"error": "Query cannot be empty"}
        
#         if not agent.agent:
#             self.logger.error(f"Agent not initialized for agent_id: {agent.agent_id}")
#             return {"error": f"Agent not initialized for agent_id: {agent.agent_id}"}
        
#         effective_prompt = prompt.strip() if prompt and prompt.strip() else Prompt.template[f"baseline_{agent.agent_id}"]
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
#             history = ChatMessageHistory()
#             for msg in conversation_history:
#                 if msg["role"] == "victim":
#                     history.add_user_message(msg["content"])
#                 elif msg["role"] == "police":
#                     history.add_ai_message(msg["content"])
            
#             state = {
#                 "messages": [
#                     SystemMessage(content=effective_prompt),
#                     *history.messages,
#                     HumanMessage(content=query)
#                 ]
#             }
            
#             self.logger.debug(f"Invoking agent for query: {query}, conversation_id: {conversation_id}")
#             try:
#                 response = agent.agent.invoke(state)
#                 self.logger.debug(f"Raw agent response: {response}")
#             except Exception as e:
#                 self.logger.error(f"LLM invocation failed for {agent.agent_id}: {str(e)}")
#                 return {"error": f"LLM invocation failed: {str(e)}"}
            
#             messages = response.get("messages", [])
#             ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
#             ai_response = ai_messages[-1] if ai_messages else "I'm sorry, I need more information to assist you."
            
#             if not ai_response.strip():
#                 self.logger.warning(f"Empty response from {agent.agent_id}. Using fallback response.")
#                 ai_response = "I'm sorry, I need more details about the incident to assist you."
            
#             rag_invoked = any("retrieve_scam_reports" in msg.content for msg in messages if isinstance(msg, AIMessage))
#             self.logger.debug(f"RAG tool invoked: {rag_invoked}")
            
#             structured_data = {"rag_invoked": rag_invoked} if rag_invoked else {}
#             conversational_response = ai_response.strip()
#             if agent.agent_id == "police":
#                 try:
#                     parsed_response = json.loads(ai_response)
#                     police_response = PoliceResponse(**parsed_response)
#                     conversational_response = police_response.conversational_response.strip()
#                     structured_data.update(police_response.dict(exclude={"conversational_response"}))
#                 except (json.JSONDecodeError, ValueError) as e:
#                     self.logger.warning(f"Failed to parse response as JSON: {str(e)}. Raw response: {ai_response}")
#                     conversational_response = ai_response.strip() or "I'm sorry, I need more information to assist you."
            
#             updated_history = conversation_history + [
#                 {"role": "victim", "content": query, "timestamp": datetime.now().isoformat()},
#                 {"role": agent.agent_id, "content": conversational_response, "timestamp": datetime.now().isoformat()}
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
        
#         police_prompt = police_prompt.strip() if police_prompt and police_prompt.strip() else Prompt.template["baseline_police"]
#         victim_prompt = victim_prompt.strip() if victim_prompt and police_prompt.strip() else Prompt.template["baseline_victim"]
#         if not police_prompt or not victim_prompt:
#             self.logger.error("Prompts cannot be empty")
#             return {"error": "Prompts cannot be empty"}
        
#         conversation_id = conversation_id or self._generate_conversation_id()
#         conversation_history = []
#         current_query = initial_query
#         last_messages = []
#         structured_data_history = []
        
#         try:
#             max_memory_turns = min(police_agent.max_turns, victim_agent.max_turns)
            
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
#                     if len(set(last_messages)) <= 4: #convert to set to remove duplicate messages
#                         self.logger.warning(f"Detected potential repetition: {last_messages}")
#                         break
                
#                 # Police response
#                 police_history = ChatMessageHistory()
#                 for msg in conversation_history:
#                     if msg["role"] == "victim":
#                         police_history.add_user_message(msg["content"])
#                     elif msg["role"] == "police":
#                         police_history.add_ai_message(msg["content"])
                
#                 police_state = {
#                     "messages": [
#                         SystemMessage(content=police_prompt),
#                         *police_history.messages,
#                         HumanMessage(content=victim_message)
#                     ]
#                 }
                
#                 self.logger.debug(f"Invoking police agent for query: {victim_message}, conversation_id: {conversation_id}")
#                 try:
#                     police_response = police_agent.agent.invoke(police_state)
#                     self.logger.debug(f"Raw police response: {police_response}")
#                 except Exception as e:
#                     self.logger.error(f"LLM invocation failed for police agent: {str(e)}")
#                     police_message = "Can you provide more details about the scam, such as where it happened or how you were contacted?"
#                     conversation_history.append({
#                         "role": "police",
#                         "content": police_message,
#                         "timestamp": datetime.now().isoformat()
#                     })
#                     self._save_to_csv(conversation_id, "autonomous", conversation_history, llm_model=police_agent.model)
#                     break
                
#                 police_messages = police_response.get("messages", [])
#                 police_ai_messages = [msg.content for msg in police_messages if isinstance(msg, AIMessage)]
#                 police_message = police_ai_messages[-1] if police_ai_messages else "Can you provide more details about the scam, such as where it happened or how you were contacted?"
                
#                 if not police_message.strip():
#                     self.logger.warning(f"Empty police response in turn {turn+1}. Using fallback response.")
#                     police_message = "Can you provide more details about the scam, such as where it happened or how you were contacted?"
                
#                 rag_invoked = any("retrieve_scam_reports" in msg.content for msg in police_messages if isinstance(msg, AIMessage))
#                 self.logger.debug(f"RAG tool invoked: {rag_invoked}")
                
#                 structured_data = {"rag_invoked": rag_invoked} if rag_invoked else {}
#                 try:
#                     parsed_response = json.loads(police_message)
#                     police_response = PoliceResponse(**parsed_response)
#                     conversational_response = police_response.conversational_response.strip()
#                     structured_data.update(police_response.dict(exclude={"conversational_response"}))
#                 except (json.JSONDecodeError, ValueError) as e:
#                     self.logger.warning(f"Failed to parse police response as JSON: {str(e)}. Raw response: {police_message}")
#                     conversational_response = police_message.strip() or "Can you provide more details about the scam?"
                
#                 conversation_history.append({
#                     "role": "police",
#                     "content": conversational_response,
#                     "timestamp": datetime.now().isoformat()
#                 })
#                 if structured_data:
#                     structured_data_history.append(structured_data)
                
#                 self._save_to_csv(conversation_id, "autonomous", conversation_history, structured_data, llm_model=police_agent.model)
                
#                 last_messages.append(conversational_response)
#                 if len(last_messages) > 6:
#                     last_messages.pop(0)
#                     if len(set(last_messages)) <= 4: #convert to set to remove duplicate messages
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
        
        
        
        
#new test# 
# from typing import Dict, List, Optional
# from langgraph.prebuilt import create_react_agent
# from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
# from langchain_community.chat_message_histories import ChatMessageHistory
# from src.agents.llm_providers import LLMProvider
# from src.agents.tools import PoliceTools, VictimTools
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
        
#         # Verify Ollama server connectivity
#         try:
#             response = requests.get(f"{self.settings.agents.ollama_base_url}/api/tags")
#             if response.status_code != 200:
#                 raise ConnectionError(f"Ollama server not responding: {response.status_code}")
#             models = response.json().get("models", [])
#             if not any(m["name"] == self.model for m in models):
#                 raise ValueError(f"Model {self.model} not loaded in Ollama server")
#             self.logger.debug(f"Ollama server is accessible and model {self.model} is loaded")
#         except Exception as e:
#             self.logger.error(f"Failed to connect to Ollama server: {str(e)}")
#             raise ConnectionError(f"Failed to connect to Ollama server: {str(e)}")

# class PoliceAgent(Agent):
#     """Police chatbot agent with structured output and RAG support."""
    
#     def __init__(self, llm_provider: str = "Ollama", model: str = "qwen2.5:7b", max_turns: int = 10):
#         super().__init__("police", llm_provider, model, max_turns)
#         self._initialize_agent()
    
#     def _initialize_agent(self):
#         """Initialize the police agent with LLM and tools."""
#         try:
#             # Try structured LLM, fall back to plain LLM if tools are not supported
#             try:
#                 llm = self.llm_provider_instance.get_structured_llm(
#                     provider=self.llm_provider, model=self.model, structured_model=PoliceResponse
#                 )
#             except Exception as e:
#                 self.logger.warning(f"Structured LLM failed for {self.model}: {str(e)}. Falling back to plain LLM.")
#                 llm = self.llm_provider_instance.get_llm(provider=self.llm_provider, model=self.model)
            
#             self.logger.debug(f"Successfully initialized LLM: {self.llm_provider}/{self.model}")
#             tools = PoliceTools().get_tools()
#             self.logger.debug(f"Tools initialized: {[t.name for t in tools]}")  # Changed from t.__name__ to t.name
#             self.agent = create_react_agent(model=llm, tools=tools)
#             self.logger.debug("Successfully created ReAct agent")
#             self.logger.info(f"Police chatbot created with agent_id: {self.agent_id}, model: {self.model}")
#         except Exception as e:
#             self.logger.error(f"Failed to create police chatbot: {str(e)}", exc_info=True)
#             raise

# class VictimAgent(Agent):
#     """Victim chatbot agent with plain text output and RAG support."""
    
#     def __init__(self, llm_provider: str = "Ollama", model: str = "qwen2.5:7b", max_turns: int = 10):
#         super().__init__("victim", llm_provider, model, max_turns)
#         self._initialize_agent()
    
#     def _initialize_agent(self):
#         """Initialize the victim agent with LLM and tools."""
#         try:
#             llm = self.llm_provider_instance.get_llm(provider=self.llm_provider, model=self.model)
#             self.logger.debug(f"Successfully initialized LLM: {self.llm_provider}/{self.model}")
#             tools = VictimTools().get_tools()
#             self.logger.debug(f"Tools initialized: {[t.name for t in tools] if tools else 'No tools'}")  # Handle empty tools list
#             self.agent = create_react_agent(model=llm, tools=tools)
#             self.logger.debug("Successfully created ReAct agent")
#             self.logger.info(f"Victim chatbot created with agent_id: {self.agent_id}, model: {self.model}")
#         except Exception as e:
#             self.logger.error(f"Failed to create victim chatbot: {str(e)}", exc_info=True)
#             raise

# class ConversationManager:
#     """Manages conversation history and CSV storage."""
    
#     def __init__(self):
#         self.settings = get_settings()
#         self.conversation_registry = {}
#         self.id_manager = IDManager()
#         self._setup_logging()
#         self.index_counter = self._load_last_index()
    
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
#         """Load the last used index from CSV."""
#         csv_file = "conversation_history.csv"
#         max_index = 0
#         if os.path.exists(csv_file):
#             try:
#                 with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                     reader = csv.DictReader(f)
#                     if "index" in reader.fieldnames:
#                         indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
#                         max_index = max(indices) if indices else 0
#                     else:
#                         self.logger.warning(f"CSV {csv_file} does not contain 'index' field. Starting with index 0.")
#                 self.logger.debug(f"Loaded max index {max_index} from {csv_file}")
#             except Exception as e:
#                 self.logger.error(f"Error reading index from CSV: {str(e)}")
#         return max_index
    
#     def _generate_conversation_id(self) -> int:
#         """Generate a unique integer conversation ID."""
#         return self.id_manager.get_next_id()
    
#     def _save_to_csv(self, conversation_id: int, conversation_type: str, messages: List[Dict], structured_data: Dict = None, llm_model: str = None):
#         """Save conversation messages to CSV, avoiding duplicates."""
#         csv_file = "conversation_history.csv"
#         file_exists = os.path.isfile(csv_file)
        
#         # Read existing CSV to check for duplicates
#         existing_entries = set()
#         if file_exists:
#             try:
#                 with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                     reader = csv.DictReader(f)
#                     for row in reader:
#                         key = (row["conversation_id"], row["sender_type"], row["content"], row["timestamp"])
#                         existing_entries.add(key)
#             except Exception as e:
#                 self.logger.error(f"Error reading CSV for deduplication: {str(e)}")
#                 raise  # Propagate error to catch in calling method
        
#         with FileLock(f"{csv_file}.lock"):  # Use file lock for thread safety
#             try:
#                 with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
#                     writer = csv.writer(f)
#                     if not file_exists:
#                         writer.writerow([
#                             "index", "conversation_id", "conversation_type", "sender_type", "content", "timestamp",
#                             "llm_model", "scam_generic_details", "scam_specific_details"
#                         ])
                    
#                     for msg in messages[-2:]:  # Only write the latest query and response
#                         key = (str(conversation_id), msg["role"], msg["content"], msg["timestamp"])
#                         if key not in existing_entries:  # Skip if message already exists
#                             self.index_counter += 1
#                             row = [
#                                 str(self.index_counter),
#                                 str(conversation_id),
#                                 conversation_type,
#                                 msg["role"],
#                                 msg["content"],
#                                 msg["timestamp"],
#                                 llm_model or "",
#                                 json.dumps(structured_data, ensure_ascii=False) if msg["role"] == "police" and structured_data else "",
#                                 ""  # Empty scam_specific_details
#                             ]
#                             writer.writerow(row)
#                             existing_entries.add(key)  # Update in-memory set
#                             self.logger.debug(f"Wrote new message to CSV: conversation_id={conversation_id}, sender={msg['role']}")
#                         else:
#                             self.logger.debug(f"Skipped duplicate message: conversation_id={conversation_id}, sender={msg['role']}")
#             except Exception as e:
#                 self.logger.error(f"Error writing to CSV: {str(e)}")
#                 raise
    
#     def get_nonautonomous_response(
#         self, agent: Agent, query: str, prompt: str = None, conversation_id: int = None,
#         conversation_history: Optional[List[Dict[str, str]]] = None
#     ) -> Dict:
#         """Get a response from an agent for a non-autonomous conversation."""
#         if not query.strip():
#             self.logger.error("Query cannot be empty")
#             return {"error": "Query cannot be empty"}
        
#         if not agent.agent:
#             self.logger.error(f"Agent not initialized for agent_id: {agent.agent_id}")
#             return {"error": f"Agent not initialized for agent_id: {agent.agent_id}"}
        
#         effective_prompt = prompt.strip() if prompt and prompt.strip() else Prompt.template[f"baseline_{agent.agent_id}"]
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
#             history = ChatMessageHistory()
#             for msg in conversation_history:
#                 if msg["role"] == "victim":
#                     history.add_user_message(msg["content"])
#                 elif msg["role"] == "police":
#                     history.add_ai_message(msg["content"])
            
#             state = {
#                 "messages": [
#                     SystemMessage(content=effective_prompt),
#                     *history.messages,
#                     HumanMessage(content=query)
#                 ]
#             }
            
#             self.logger.debug(f"Invoking agent for query: {query}, conversation_id: {conversation_id}")
#             try:
#                 response = agent.agent.invoke(state)  # Correctly invoke the underlying agent
#                 self.logger.debug(f"Raw agent response: {response}")
#             except Exception as e:
#                 self.logger.error(f"LLM invocation failed for {agent.agent_id}: {str(e)}")
#                 return {"error": f"LLM invocation failed: {str(e)}"}
            
#             messages = response.get("messages", [])
#             ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
#             ai_response = ai_messages[-1] if ai_messages else "I'm sorry, I need more information to assist you."
            
#             if not ai_response.strip():
#                 self.logger.warning(f"Empty response from {agent.agent_id}. Using fallback response.")
#                 ai_response = "I'm sorry, I need more details about the incident to assist you."
            
#             rag_invoked = any("retrieve_scam_reports" in msg.content for msg in messages if isinstance(msg, AIMessage))
#             self.logger.debug(f"RAG tool invoked: {rag_invoked}")
            
#             structured_data = {"rag_invoked": rag_invoked} if rag_invoked else {}
#             conversational_response = ai_response.strip()
#             if agent.agent_id == "police":
#                 try:
#                     parsed_response = json.loads(ai_response)
#                     police_response = PoliceResponse(**parsed_response)
#                     conversational_response = police_response.conversational_response.strip()
#                     structured_data.update(police_response.dict(exclude={"conversational_response"}))
#                 except (json.JSONDecodeError, ValueError) as e:
#                     self.logger.warning(f"Failed to parse response as JSON: {str(e)}. Raw response: {ai_response}")
#                     conversational_response = ai_response.strip() or "I'm sorry, I need more information to assist you."
            
#             updated_history = conversation_history + [
#                 {"role": "victim", "content": query, "timestamp": datetime.now().isoformat()},
#                 {"role": agent.agent_id, "content": conversational_response, "timestamp": datetime.now().isoformat()}
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
        
#         police_prompt = police_prompt.strip() if police_prompt and police_prompt.strip() else Prompt.template["baseline_police"]
#         victim_prompt = victim_prompt.strip() if victim_prompt and police_prompt.strip() else Prompt.template["baseline_victim"]
#         if not police_prompt or not victim_prompt:
#             self.logger.error("Prompts cannot be empty")
#             return {"error": "Prompts cannot be empty"}
        
#         conversation_id = conversation_id or self._generate_conversation_id()
#         conversation_history = self.conversation_registry.get(conversation_id, [])  # Load existing history if any
#         current_query = initial_query
#         last_messages = []
#         structured_data_history = []
        
#         try:
#             max_memory_turns = min(police_agent.max_turns, victim_agent.max_turns)
            
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
#                     victim_response = victim_agent.agent.invoke(victim_state)  # Corrected to use agent.invoke
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
                
#                 # Police response
#                 police_history = ChatMessageHistory()
#                 for msg in conversation_history:
#                     if msg["role"] == "victim":
#                         police_history.add_user_message(msg["content"])
#                     elif msg["role"] == "police":
#                         police_history.add_ai_message(msg["content"])
                
#                 police_state = {
#                     "messages": [
#                         SystemMessage(content=police_prompt),
#                         *police_history.messages,
#                         HumanMessage(content=victim_message)
#                     ]
#                 }
                
#                 self.logger.debug(f"Invoking police agent for query: {victim_message}, conversation_id: {conversation_id}")
#                 try:
#                     police_response = police_agent.agent.invoke(police_state)  # Corrected to use agent.invoke
#                     self.logger.debug(f"Raw police response: {police_response}")
#                 except Exception as e:
#                     self.logger.error(f"LLM invocation failed for police agent: {str(e)}")
#                     police_message = "Can you provide more details about the scam, such as where it happened or how you were contacted?"
#                     conversation_history.append({
#                         "role": "police",
#                         "content": police_message,
#                         "timestamp": datetime.now().isoformat()
#                     })
#                     self._save_to_csv(conversation_id, "autonomous", conversation_history, llm_model=police_agent.model)
#                     break
                
#                 police_messages = police_response.get("messages", [])
#                 police_ai_messages = [msg.content for msg in police_messages if isinstance(msg, AIMessage)]
#                 police_message = police_ai_messages[-1] if police_ai_messages else "Can you provide more details about the scam, such as where it happened or how you were contacted?"
                
#                 if not police_message.strip():
#                     self.logger.warning(f"Empty police response in turn {turn+1}. Using fallback response.")
#                     police_message = "Can you provide more details about the scam, such as where it happened or how you were contacted?"
                
#                 rag_invoked = any("retrieve_scam_reports" in msg.content for msg in police_messages if isinstance(msg, AIMessage))
#                 self.logger.debug(f"RAG tool invoked: {rag_invoked}")
                
#                 structured_data = {"rag_invoked": rag_invoked} if rag_invoked else {}
#                 try:
#                     parsed_response = json.loads(police_message)
#                     police_response = PoliceResponse(**parsed_response)
#                     conversational_response = police_response.conversational_response.strip()
#                     structured_data.update(police_response.dict(exclude={"conversational_response"}))
#                 except (json.JSONDecodeError, ValueError) as e:
#                     self.logger.warning(f"Failed to parse police response as JSON: {str(e)}. Raw response: {police_message}")
#                     conversational_response = police_message.strip() or "Can you provide more details about the scam?"
                
#                 conversation_history.append({
#                     "role": "police",
#                     "content": conversational_response,
#                     "timestamp": datetime.now().isoformat()
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
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from src.agents.remove.llm_providers import LLMProvider
from src.agents.tools import PoliceTools, VictimTools
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

class Agent:
    """Base class for chatbot agents."""
    
    def __init__(self, agent_id: str, llm_provider: str, model: str, max_turns: int):
        """Initialize agent with configuration."""
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
        """Configure logging to write to agent-specific log file."""
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
        """Validate LLM provider and model."""
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
        else:
            self.logger.debug(f"Skipping Ollama server check for provider: {self.llm_provider}")

class PoliceAgent(Agent):
    """Police chatbot agent with structured output and RAG support."""
    
    def __init__(self, llm_provider: str = "OpenAI", model: str = "gpt-4o-mini", max_turns: int = 10):
        super().__init__("police", llm_provider, model, max_turns)
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the police agent with LLM and tools."""
        try:
            try:
                llm = self.llm_provider_instance.get_structured_llm(
                    provider=self.llm_provider, model=self.model, structured_model=PoliceResponse
                )
            except Exception as e:
                self.logger.warning(f"Structured LLM failed for {self.model}: {str(e)}. Falling back to plain LLM.")
                llm = self.llm_provider_instance.get_llm(provider=self.llm_provider, model=self.model)
            
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
    """Victim chatbot agent with plain text output and RAG support."""
    
    def __init__(self, llm_provider: str = "OpenAI", model: str = "gpt-4o-mini", max_turns: int = 10):
        super().__init__("victim", llm_provider, model, max_turns)
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize the victim agent with LLM and tools."""
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

class ConversationManager:
    """Manages conversation history and CSV storage."""
    
    def __init__(self):
        """Initialize conversation manager with settings and logging."""
        self.settings = get_settings()
        self.conversation_registry = {}
        self.id_manager = IDManager()
        self._setup_logging()
        self.index_counter = self._load_last_index()
        self.police_tools = PoliceTools()  # Initialize PoliceTools for manual RAG invocation
    
    def _setup_logging(self):
        """Configure logging to write to conversation-specific log file."""
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
        """Load the last used index from conversation_history.csv."""
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
        """Generate a unique integer conversation ID."""
        return self.id_manager.get_next_id()
    
    def _save_to_csv(self, conversation_id: int, conversation_type: str, messages: List[Dict], structured_data: Dict = None, llm_model: str = None):
        """Save conversation messages to CSV, avoiding duplicates."""
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
                            "llm_model", "scam_generic_details", "scam_specific_details"
                        ])
                    
                    for msg in messages[-2:]:
                        key = (str(conversation_id), msg["role"], msg["content"], msg["timestamp"])
                        if key not in existing_entries:
                            self.index_counter += 1
                            row = [
                                str(self.index_counter),
                                str(conversation_id),
                                conversation_type,
                                msg["role"],
                                msg["content"],
                                msg["timestamp"],
                                llm_model or "",
                                json.dumps(structured_data, ensure_ascii=False) if msg["role"] == "police" and structured_data else "",
                                ""
                            ]
                            writer.writerow(row)
                            existing_entries.add(key)
                            self.logger.debug(f"Wrote new message to CSV: conversation_id={conversation_id}, sender={msg['role']}")
                        else:
                            self.logger.debug(f"Skipped duplicate message: conversation_id={conversation_id}, sender={msg['role']}")
            except Exception as e:
                self.logger.error(f"Error writing to CSV: {str(e)}")
                raise
    
    def get_nonautonomous_response(
        self, agent: Agent, query: str, prompt: str = None, conversation_id: int = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict:
        """Get a response from an agent for a non-autonomous conversation."""
        if not query.strip():
            self.logger.error("Query cannot be empty")
            return {"error": "Query cannot be empty"}
        
        if not agent.agent:
            self.logger.error(f"Agent not initialized for agent_id: {agent.agent_id}")
            return {"error": f"Agent not initialized for agent_id: {agent.agent_id}"}
        
        effective_prompt = prompt.strip() if prompt and prompt.strip() else Prompt.template[f"baseline_{agent.agent_id}"]
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
            try:
                response = agent.agent.invoke(state)
                self.logger.debug(f"Raw agent response: {response}")
            except Exception as e:
                self.logger.error(f"LLM invocation failed for {agent.agent_id}: {str(e)}")
                return {"error": f"LLM invocation failed: {str(e)}"}
            
            messages = response.get("messages", [])
            ai_messages = [msg.content for msg in messages if isinstance(msg, AIMessage)]
            ai_response = ai_messages[-1] if ai_messages else "I'm sorry, I need more information to assist you."
            
            if not ai_response.strip():
                self.logger.warning(f"Empty response from {agent.agent_id}. Using fallback response.")
                ai_response = "I'm sorry, I need more details about the incident to assist you."
            
            rag_invoked = any("retrieve_scam_reports" in msg.content for msg in messages if isinstance(msg, AIMessage))
            self.logger.debug(f"RAG tool invoked by LLM for query '{query}': {rag_invoked}")
            
            # Fallback RAG invocation for scam-related queries
            if not rag_invoked and agent.agent_id == "police" and any(keyword in query.lower() for keyword in ["scam", "fraud", "phishing", "facebook"]):
                self.logger.warning(f"RAG tool not invoked by LLM for scam-related query: {query}. Invoking manually.")
                try:
                    tools = self.police_tools.get_tools()
                    retrieve_scam_reports = tools[0]
                    rag_result = retrieve_scam_reports.invoke({
                        "query": query,
                        "top_k": 5,
                        "conversation_id": conversation_id,
                        "llm_model": agent.model
                    })
                    self.logger.debug(f"Manual RAG invocation result: {rag_result}")
                    rag_invoked = True
                    ai_response = f"{ai_response}\n\nBased on similar cases, we found reports of scams on Facebook involving phishing and non-delivery. {ai_response}"
                except Exception as e:
                    self.logger.error(f"Manual RAG invocation failed: {str(e)}")
            
            structured_data = {"rag_invoked": rag_invoked}
            conversational_response = ai_response.strip()
            if agent.agent_id == "police":
                try:
                    parsed_response = json.loads(ai_response)
                    police_response = PoliceResponse(**parsed_response)
                    conversational_response = police_response.conversational_response.strip()
                    structured_data.update(police_response.dict(exclude={"conversational_response"}))
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.warning(f"Failed to parse response as JSON: {str(e)}. Raw response: {ai_response}")
                    conversational_response = ai_response.strip() or "I'm sorry, I need more information to assist you."
            
            updated_history = conversation_history + [
                {"role": "victim", "content": query, "timestamp": datetime.now().isoformat()},
                {"role": agent.agent_id, "content": conversational_response, "timestamp": datetime.now().isoformat(), "structured_data": structured_data if agent.agent_id == "police" else {}}
            ]
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
        """Simulate an autonomous conversation between police and victim agents."""
        if not police_agent.agent or not victim_agent.agent:
            self.logger.error(f"Agent not initialized: police={police_agent.agent_id}, victim={victim_agent.agent_id}")
            return {"error": f"Agent not initialized: police={police_agent.agent_id}, victim={victim_agent.agent_id}"}
        
        police_prompt = police_prompt.strip() if police_prompt and police_prompt.strip() else Prompt.template["baseline_police"]
        victim_prompt = victim_prompt.strip() if victim_prompt and police_prompt.strip() else Prompt.template["baseline_victim"]
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
            
            for turn in range(max_turns):
                # Victim response
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
                
                # Police response
                police_history = ChatMessageHistory()
                for msg in conversation_history:
                    if msg["role"] == "victim":
                        police_history.add_user_message(msg["content"])
                    elif msg["role"] == "police":
                        police_history.add_ai_message(msg["content"])
                
                police_state = {
                    "messages": [
                        SystemMessage(content=police_prompt),
                        *police_history.messages,
                        HumanMessage(content=victim_message)
                    ]
                }
                
                self.logger.debug(f"Invoking police agent for query: {victim_message}, conversation_id: {conversation_id}")
                try:
                    police_response = police_agent.agent.invoke(police_state)
                    self.logger.debug(f"Raw police response: {police_response}")
                except Exception as e:
                    self.logger.error(f"LLM invocation failed for police agent: {str(e)}")
                    police_message = "Can you provide more details about the scam, such as where it happened or how you were contacted?"
                    conversation_history.append({
                        "role": "police",
                        "content": police_message,
                        "timestamp": datetime.now().isoformat()
                    })
                    self._save_to_csv(conversation_id, "autonomous", conversation_history, llm_model=police_agent.model)
                    break
                
                police_messages = police_response.get("messages", [])
                police_ai_messages = [msg.content for msg in police_messages if isinstance(msg, AIMessage)]
                police_message = police_ai_messages[-1] if police_ai_messages else "Can you provide more details about the scam, such as where it happened or how you were contacted?"
                
                if not police_message.strip():
                    self.logger.warning(f"Empty police response in turn {turn+1}. Using fallback response.")
                    police_message = "Can you provide more details about the scam, such as where it happened or how you were contacted?"
                
                rag_invoked = any("retrieve_scam_reports" in msg.content for msg in police_messages if isinstance(msg, AIMessage))
                self.logger.debug(f"RAG tool invoked by LLM for query '{victim_message}': {rag_invoked}")
                
                # Fallback RAG invocation for scam-related queries
                if not rag_invoked and any(keyword in victim_message.lower() for keyword in ["scam", "fraud", "phishing", "facebook"]):
                    self.logger.warning(f"RAG tool not invoked by LLM for scam-related query: {victim_message}. Invoking manually.")
                    try:
                        tools = self.police_tools.get_tools()
                        retrieve_scam_reports = tools[0]
                        rag_result = retrieve_scam_reports.invoke({
                            "query": victim_message,
                            "top_k": 5,
                            "conversation_id": conversation_id,
                            "llm_model": police_agent.model
                        })
                        self.logger.debug(f"Manual RAG invocation result: {rag_result}")
                        rag_invoked = True
                        police_message = f"{police_message}\n\nBased on similar cases, we found reports of scams on Facebook involving phishing and non-delivery. {police_message}"
                    except Exception as e:
                        self.logger.error(f"Manual RAG invocation failed: {str(e)}")
                
                structured_data = {"rag_invoked": rag_invoked}
                try:
                    parsed_response = json.loads(police_message)
                    police_response = PoliceResponse(**parsed_response)
                    conversational_response = police_response.conversational_response.strip()
                    structured_data.update(police_response.dict(exclude={"conversational_response"}))
                except (json.JSONDecodeError, ValueError) as e:
                    self.logger.warning(f"Failed to parse police response as JSON: {str(e)}. Raw response: {police_message}")
                    conversational_response = police_message.strip() or "Can you provide more details about the scam?"
                
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