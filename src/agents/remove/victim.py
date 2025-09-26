import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
import json
import logging
from datetime import datetime
from pathlib import Path
from filelock import FileLock
from config.settings import get_settings
from src.agents.tools import VictimTools
from config.id_manager import IDManager
from src.preprocessing.preprocess import VictimProfilePreprocessor
from src.agents.utils import RecordCounter
from langchain_core.exceptions import LangChainException
import csv
from src.agents.prompt import Prompt

# Configure logging
def setup_logging():
    settings = get_settings()
    log_dir = Path(settings.log.directory) / settings.log.subdirectories["agent"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "chatbot.log"
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, mode='a')
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    return logger

logger = setup_logging()

# Define state
class ChatbotState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "Chat history"]
    conversation_id: int

class VictimChatbot:
    def __init__(self, model_name: str = "qwen2.5:7b", json_file: str = "data/victim_profile/victim_details.json"):
        """
        Initialize the VictimChatbot with dynamic victim and scam details.

        Args:
            model_name (str): The LLM model name to use.
            json_file (str): Path to the preprocessed JSON file with victim and scam details.
        """
        self.settings = get_settings()
        self.id_manager = IDManager()
        self.record_counter = RecordCounter()
        self.model_name = model_name
        self.conversation_id = self.id_manager.get_next_id()
        self.json_file = json_file
        self.conversation_history = []
        self.victim_tools = VictimTools()
        
        # Load and select victim/scam details
        self.records = self._load_records()
        self.record_index = self.record_counter.get_next_index(len(self.records))
        self.victim_details, self.scam_details = self._get_current_record()
        self.prompt_template = self._generate_prompt_template()
        
        self.workflow = self._build_workflow()
        logger.info(f"Victim chatbot initialized with model: {model_name}, conversation_id: {self.conversation_id}, record_index: {self.record_index}")

    def _load_records(self) -> list:
        """Load records from the JSON file."""
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not records:
                logger.error("No records found in scam_reports.json")
                raise ValueError("No records found in scam_reports.json")
            return records
        except Exception as e:
            logger.error(f"Error loading records from {self.json_file}: {str(e)}", exc_info=True)
            raise

    def _get_current_record(self) -> tuple:
        """Get the current victim and scam details based on record_index."""
        record = self.records[self.record_index]
        return record["victim_details"], record["scam_details"]

    @staticmethod
    def _escape_template_braces(text: str) -> str:
        """Escape curly braces in text to prevent LangChain from interpreting them as placeholders."""
        return text.replace("{", "{{").replace("}", "}}")

    def _generate_prompt_template(self) -> ChatPromptTemplate:
        if "baseline_victim" not in Prompt.template:
            logger.error("Prompt type 'baseline_victim' not found in Prompt.template")
            raise ValueError("Prompt type 'baseline_victim' not found")
        
        # Serialize and escape JSON strings
        victim_details_json = json.dumps(self.victim_details, indent=2)
        scam_details_json = json.dumps(self.scam_details, indent=2)
        victim_details_str = self._escape_template_braces(victim_details_json)
        scam_details_str = self._escape_template_braces(scam_details_json)
        
        # Format prompt with escaped JSON strings
        try:
            prompt_text = Prompt.template["baseline_victim"].format(
                victim_details=victim_details_str,
                scam_details=scam_details_str
            )
        except KeyError as e:
            logger.error(f"Error formatting prompt template: {str(e)}")
            raise ValueError(f"Missing key in prompt template: {str(e)}")
        
        logger.debug(f"Formatted prompt text: {prompt_text}")
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", prompt_text),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{user_input}"),
        ])
        logger.debug("Generated baseline_victim prompt template with dynamic details")
        return prompt_template


    def _initialize_llm_and_tools(self):
        """
        Initialize the LLM and bind tools for future extensibility.

        Returns:
            Tuple: (LLM with bound tools, list of tools).
        """
        llm = ChatOllama(
            model=self.model_name,
            base_url=self.settings.agents.ollama_base_url,
        )
        tools = self.victim_tools.get_tools()  # Currently empty, but bound for future use
        return llm.bind_tools(tools), tools

    def _build_workflow(self):
        """
        Build the LangGraph workflow with a single node for processing LLM responses.
        """
        llm, tools = self._initialize_llm_and_tools()
        workflow = StateGraph(ChatbotState)
        workflow.add_node("process_llm", lambda state: self._process_llm(state, llm))
        workflow.add_edge("process_llm", END)
        workflow.set_entry_point("process_llm")
        return workflow.compile()

    def _process_llm(self, state: ChatbotState, llm):
        logger.debug("Processing LLM response")
        messages = state["messages"]
        
        try:
            prompt = self.prompt_template.format(
                messages=messages,
                user_input=messages[-1].content
            )
        except KeyError as e:
            logger.error(f"KeyError in prompt formatting: {str(e)}")
            raise ValueError(f"Failed to format prompt due to missing key: {str(e)}")
        
        try:
            response = llm.invoke(prompt)
            if not response.content.strip():
                logger.error("LLM returned empty response")
                raise ValueError("Empty response from LLM")
            
            # Check for [END_CONVERSATION] in response
            conversational_response = response.content
            end_conversation = "[END_CONVERSATION]" in conversational_response
            if end_conversation:
                conversational_response = conversational_response.replace("[END_CONVERSATION]", "").strip()
            
            return {
                "messages": [AIMessage(content=conversational_response)],
                "end_conversation": end_conversation
            }
        except Exception as e:
            logger.error(f"LLM processing failed for model {self.model_name}: {str(e)}", exc_info=True)
            fallback_response = "Um, I'm sorry, something went wrong. Can you repeat that or clarify what you need?"
            return {
                "messages": [AIMessage(content=fallback_response)],
                "end_conversation": False
            }

    def _save_conversation(self):
        """
        Save conversation history to CSV.
        """
        csv_file = "conversation_history.csv"
        file_exists = os.path.isfile(csv_file)
        existing_entries = set()
        index_counter = 0

        if file_exists:
            try:
                with FileLock(f"{csv_file}.lock"):
                    with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
                        reader = csv.DictReader(f)
                        if "index" in reader.fieldnames:
                            indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
                            index_counter = max(indices) if indices else 0
                        f.seek(0)
                        reader = csv.DictReader(f)
                        for row in reader:
                            key = (row["conversation_id"], row["sender_type"], row["content"], row["timestamp"])
                            existing_entries.add(key)
            except Exception as e:
                logger.error(f"Error reading CSV for deduplication: {str(e)}", exc_info=True)
                return

        with FileLock(f"{csv_file}.lock"):
            try:
                with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "index", "conversation_id", "conversation_type", "sender_type", "content",
                            "timestamp", "llm_model"
                        ])
                    for msg in self.conversation_history:
                        key = (str(self.conversation_id), msg["role"], msg["content"], msg["timestamp"])
                        if key not in existing_entries:
                            index_counter += 1
                            row = [
                                str(index_counter),
                                str(self.conversation_id),
                                "non_autonomous",
                                msg["role"],
                                msg["content"],
                                msg["timestamp"],
                                self.model_name if msg["role"] == "victim" else ""
                            ]
                            writer.writerow(row)
                            existing_entries.add(key)
                            logger.debug(f"Wrote message to CSV: conversation_id={self.conversation_id}, sender={msg['role']}")
            except Exception as e:
                logger.error(f"Error writing to CSV: {str(e)}", exc_info=True)

    def process_query(self, query: str):
        """
        Process a query from the police agent and generate a victim response.

        Args:
            query (str): The input query from the police agent.

        Returns:
            Dict: Response with conversational text, conversation ID, and history.
        """
        if not query.strip():
            logger.error("Query cannot be empty")
            return {"error": "Query cannot be empty"}

        state = {
            "messages": [HumanMessage(content=query)],
            "conversation_id": self.conversation_id
        }
        result = self.workflow.invoke(state)
        conversational_response = result["messages"][-1].content
        end_conversation = result.get("end_conversation", False)

        self.conversation_history.append({
            "role": "police",
            "content": query,
            "timestamp": datetime.now().isoformat()
        })
        self.conversation_history.append({
            "role": "victim",
            "content": conversational_response,
            "timestamp": datetime.now().isoformat()
        })

        logger.debug(f"Processed query response: {conversational_response}")
        response = {
            "response": conversational_response,
            "conversation_id": self.conversation_id,
            "conversation_history": self.conversation_history
        }
        if end_conversation:
            response["end_conversation": True]
            self.end_conversation()
        return response

    def end_conversation(self):
        """
        End the conversation and save history to CSV.

        Returns:
            Dict: Status and conversation ID.
        """
        self._save_conversation()
        logger.info(f"Conversation {self.conversation_id} saved to CSV")
        return {"status": "Conversation ended", "conversation_id": self.conversation_id}

if __name__ == "__main__":
    
    json_file = "data/victim_profile/victim_details.json"
    
    # Generate victim_details.json only if it does not exist
    if not Path(json_file).exists():
        logger.info(f"JSON file {json_file} not found, running preprocessor")
        try:
            preprocessor = VictimProfilePreprocessor()
            preprocessor.preprocess(output_file=json_file)
            logger.info(f"Successfully generated {json_file}")
        except Exception as e:
            logger.error(f"Failed to run preprocessor: {str(e)}", exc_info=True)
            raise

    models = ["qwen2.5:7b", "granite3.2:8b", "mistral:7b"]
    query = "Can you tell me about any recent scam incidents you’ve experienced?"
    results = {}
    num_reinitializations = 3  #adjust number of reinitialisations accordingly

    logger.info("Starting model testing with multiple reinitializations")
    for model_name in models:
        logger.info(f"--- Testing model: {model_name} ---")
        
        # Reset record counter before testing each model
        record_counter = RecordCounter()
        record_counter.reset()
        logger.info(f"Reset record counter for model {model_name}")
        
        model_results = []
        for i in range(num_reinitializations):
            logger.info(f"Reinitialization {i+1} for model {model_name}")
            try:
                chatbot = VictimChatbot(model_name=model_name)
                response = chatbot.process_query(query)
                
                # Log victim details to verify prompt changes
                logger.info(f"Victim details for reinitialization {i+1}: {json.dumps(chatbot.victim_details, indent=2)}")
                model_results.append({
                    "reinitialization": i + 1,
                    "record_index": chatbot.record_index,
                    "response": response
                })
                logger.info(f"Successfully processed query with model {model_name}, reinitialization {i+1}: {json.dumps(response, indent=2)}")
                chatbot.end_conversation()
            except LangChainException as e:
                logger.error(f"LangChain error with model {model_name}, reinitialization {i+1}: {str(e)}", exc_info=True)
                model_results.append({
                    "reinitialization": i + 1,
                    "error": f"LangChain error: {str(e)}"
                })
            except Exception as e:
                logger.error(f"Unexpected error with model {model_name}, reinitialization {i+1}: {str(e)}", exc_info=True)
                model_results.append({
                    "reinitialization": i + 1,
                    "error": f"Unexpected error: {str(e)}"
                })
        results[model_name] = model_results

    logger.info("Completed model testing")
    print(json.dumps(results, indent=2))