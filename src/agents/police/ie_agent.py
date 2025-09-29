import sys
import os
import json
from datetime import datetime

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence, Optional, Dict
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from config.settings import get_settings
from config.logging_config import setup_logger
from src.agents.prompt import Prompt
from src.models.response_model import PoliceResponse, PoliceResponseSlots

class ChatbotState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    conversation_id: int
    police_response: dict
    unfilled_slots: Optional[dict]

class IEChatbot:
    """A baseline police AI conversational agent for scam reporting without RAG, user profiling or kb agents.
    
    This agent uses a LangGraph workflow to extract structured information from user queries,
    track unfilled slots, and generate conversational responses."""

    def __init__(self, model_name: str = "qwen2.5:7b", llm_provider: str = "Ollama", temperature: float = 0.0):
        self.settings = get_settings()
        self.logger = setup_logger("Baseline_PoliceAgent", self.settings.log.subdirectories["agent"])
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.temperature = temperature

        self.messages: Sequence[AnyMessage] = []
        self.unfilled_slots: dict = {}
        self.prev_ie_output: Optional[Dict] = None

        
        self.ie_prompt_template = ChatPromptTemplate.from_messages([
            ("system", Prompt.template["baseline_ie_only"]),
            ("system", "Previous Extraction from Last Turn (MUST use as base: Copy all slots unchanged unless explicitly corrected/clarified in the NEW query only):\n{prev_ie_output}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ])

        # Build workflow
        self.workflow = self._build_workflow()
        self.logger.info(f"Baseline police chatbot initialized with model: {model_name}")

    def _get_llm(self, schema: BaseModel):
        """Get a structured LLM instance based on provider."""
        if self.llm_provider == "Ollama":
            return ChatOllama(
                model=self.model_name,
                base_url=self.settings.agents.ollama_base_url,
                format="json",
                temperature=self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": schema.model_json_schema()
                }
            )
        elif self.llm_provider == "OpenAI":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                self.logger.error("OPENAI_API_KEY not found")
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.logger.debug(f"Initializing structured OpenAI LLM with model: {self.model_name}")
            base_llm = ChatOpenAI(
                model=self.model_name,
                api_key=api_key,
                temperature=self.temperature,
            )
            return base_llm.with_structured_output(schema)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _preprocess_ie_output(self, police_response: dict) -> dict:
        """Preprocesses extracted information to normalize platforms, dates, etc."""
        try:
            platforms = ["LAZADA", "SHOPEE", "FACEBOOK", "CAROUSELL", "INSTAGRAM", "WHATSAPP", "SMS", "CALL"]
            
            preprocessed = police_response.copy()
            
            # Extract scam type
            scam_type = preprocessed.get("scam_type", "").upper()
            
            # Preprocessing for approach platform 
            approach = preprocessed.get("scam_approach_platform", "").upper().strip()
            if approach:
                if any(word in approach for word in ["TEXT", "TEXT MESSAGE", "SMS", "MESSAGE"]):
                    approach = "SMS"
                if approach == "SMS" and scam_type == "PHISHING":
                    preprocessed["scam_communication_platform"] = "SMS"
                for known in platforms:
                    if known in approach:
                        approach = known
                        break
                preprocessed["scam_approach_platform"] = approach
            
            comm = preprocessed.get("scam_communication_platform", "").upper().strip()
            if comm:
                if any(word in comm for word in ["TEXT", "TEXT MESSAGE", "SMS"]):
                    comm = "SMS"
                for known in platforms:
                    if known in comm:
                        comm = known
                        break
                preprocessed["scam_communication_platform"] = comm
                if approach not in ["LAZADA", "SHOPEE", "FACEBOOK", "CAROUSELL", "INSTAGRAM"]:
                    preprocessed["scam_moniker"] = ""
            
            bank = preprocessed.get("scam_beneficiary_platform", "").upper()
            if bank in ["UOB", "DBS", "HSBC", "SCB", "MAYBANK", "BOC", "CITIBANK", "CIMB", "GXS", "TRUST"]:
                preprocessed["scam_transaction_type"] = "BANK TRANSFER"
            
            if scam_type in ["GOVERNMENT OFFICIALS IMPERSONATION", "PHISHING"]:
                preprocessed["scam_moniker"] = ""
            
            if scam_type in ["GOVERNMENT OFFICIALS IMPERSONATION", "ECOMMERCE"]:
                preprocessed["scam_url_link"] = ""
            
            if scam_type == "GOVERNMENT OFFICIALS IMPERSONATION" and approach == "SMS":
                preprocessed["scam_type"] = ""
            
            incident_date = preprocessed.get("scam_incident_date", "")
            if incident_date and incident_date.strip():
                try:
                    date_obj = datetime.strptime(incident_date, "%Y-%m-%d")
                    current_year = datetime.now().year
                    if date_obj.year != current_year:
                        new_date = date_obj.replace(year=current_year)
                        preprocessed["scam_incident_date"] = new_date.strftime("%Y-%m-%d")
                except ValueError:
                    pass
            
            url = preprocessed.get("scam_url_link", "").strip()
            invalid_values = {"unknown", "na", "n/a"}
            if url and url.lower() not in invalid_values and not url.startswith("https://"):
                preprocessed["scam_url_link"] = "https://" + url
            
            #If no digits are present in contact no, set to empty
            contact_no = preprocessed.get("scam_contact_no", "").strip()
            if contact_no and not any(char.isdigit() for char in contact_no):
                preprocessed["scam_contact_no"] = ""
                
            #If no digits are present in bank account, set to empty
            bank_account = preprocessed.get("scam_beneficiary_identifier", "").strip()
            if bank_account and not any(char.isdigit() for char in bank_account):
                preprocessed["scam_beneficiary_identifier"] = ""
            
            return preprocessed
        except Exception as e:
            self.logger.error(f"Error in preprocessing: {str(e)}")
            return police_response

    def _build_workflow(self):
        """Build simplified LangGraph workflow (IE agent + slot tracker)."""
        workflow = StateGraph(state_schema=ChatbotState)
        
        workflow.add_node("ie_agent", self.ie_agent)
        workflow.add_node("slot_tracker", self.slot_tracker)
        
        workflow.add_edge(START, "ie_agent")
        workflow.add_edge("ie_agent", "slot_tracker")
        workflow.add_edge("slot_tracker", END)
        
        return workflow.compile()

    def ie_agent(self, state: ChatbotState) -> ChatbotState:
        """Extract structured information using LLM (amended for no RAG)."""
        
        ie_llm = self._get_llm(PoliceResponse)
        messages = state["messages"]
        history = messages[:-1]
        user_input = messages[-1].content
        
        prompt = self.ie_prompt_template.format_messages(
            prev_ie_output=json.dumps(self.prev_ie_output or {}, indent=2),
            history=history,
            user_input=user_input,
            unfilled_slots=json.dumps(self.unfilled_slots)
        )
        self.logger.debug(f"Previous ie_output: {self.prev_ie_output}")

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = ie_llm.invoke(prompt)
                if self.llm_provider == "OpenAI":
                    if not response:
                        self.logger.error("LLM returned empty response")
                        continue
                    police_response = response.model_dump()
                else:
                    if not response.content.strip():
                        self.logger.error("LLM returned empty response")
                        continue
                    police_response = json.loads(response.content)
                
                default_response = PoliceResponse(conversational_response="I'm sorry, I need more information to assist you.").model_dump()
                default_response.update(police_response)
                police_response = default_response
                PoliceResponse(**police_response)
                
                police_response = self._preprocess_ie_output(police_response)
                self.logger.debug(f"Preprocessed PoliceResponse: {police_response}")
                
                return {
                    "messages": [AIMessage(content=json.dumps(police_response))],
                    "police_response": police_response
                }
            except Exception as e:
                self.logger.error(f"LLM attempt {attempt+1} failed: {str(e)}", exc_info=True)
                if attempt == max_retries - 1:
                    fallback_response = PoliceResponse(
                        conversational_response="Sorry, I encountered an error. Can you provide more details about the scam, such as the date, what the caller said, and any contact details provided?",
                        scam_incident_description=messages[-1].content
                    ).model_dump()
                    
                    return {
                        "messages": [AIMessage(content=json.dumps(fallback_response))],
                        "police_response": fallback_response
                    }

    def slot_tracker(self, state: ChatbotState) -> ChatbotState:
        """Tracks unfilled slots based on police response."""
        police_response = state["police_response"]
        
        unfilled = {}
        
        for field_name in PoliceResponseSlots:
            value = police_response.get(field_name)
            if field_name == "scam_amount_lost":
                is_unfilled = value == 0.0 or value is None
            else:
                is_unfilled = value in ("", None, "unknown", "na") or not value
            unfilled[field_name] = is_unfilled
        
        self.logger.debug(f"SlotTracker: unfilled={unfilled}") 
        
        return {"unfilled_slots": unfilled}

    def process_query(self, query: str, conversation_id: int = None) -> dict:
        """Process a user query through the simplified workflow."""
        if not query.strip():
            self.logger.error("Query cannot be empty")
            return {"error": "Query cannot be empty"}

        state = {
            "messages": self.messages + [HumanMessage(content=query)],
            "conversation_id": conversation_id,
            "police_response": {},
        }
        state["unfilled_slots"] = self.unfilled_slots
        
        result = self.workflow.invoke(state)
        self.messages = result["messages"]
        self.unfilled_slots = result.get("unfilled_slots", {})
        
        police_response = result["police_response"]
        prev_ie = police_response.copy()  
        if "conversational_response" in prev_ie:
            del prev_ie["conversational_response"]
        self.prev_ie_output = prev_ie

        conversational_response = police_response.get("conversational_response", "I'm sorry, I need more information to assist you.")

        
        structured_data = police_response.copy()
        structured_data["rag_upsert"] = False
        structured_data["user_profile"] = {}
        structured_data["rag_suggestions"] = []
        structured_data["initial_profile"] = {}
        structured_data["retrieved_strategies"] = []
        structured_data["upserted_strategy"] = {}

        return {
            "response": conversational_response,
            "structured_data": structured_data,
            "conversation_id": conversation_id,
            "rag_invoked": False,  # Always False for this baseline
            "rag_suggestions": []  # Empty for compatibility
        }

    def reset_state(self):
        """Reset internal state for a new conversation."""
        self.messages = []
        self.unfilled_slots = {}
        self.prev_ie_output = None
        self.logger.debug("PoliceChatbot state reset")

    def end_conversation(self, csv_file: Optional[str] = None) -> dict:
        """End the conversation and log completion."""
        self.logger.info("Conversation saved to CSV")
        return {"status": "Conversation ended"}