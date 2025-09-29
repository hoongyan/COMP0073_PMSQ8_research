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
from src.models.response_model import PoliceResponse, RagOutput, PoliceResponseSlots
from src.agents.tools import PoliceTools
from src.agents.prompt import Prompt
from src.agents.utils import build_query_with_history

# Define state
class ChatbotState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    conversation_id: int
    police_response: dict
    rag_suggestions: dict
    rag_results: list
    unfilled_slots: Optional[dict]
    rag_invoked: bool
    user_profile: dict  
    rag_upsert: bool
    
class RAGIEAgent:
    """A baseline police AI conversational agent for scam reporting without user profiling and knowledgebase augmentation.
    
    This agent uses a LangGraph workflow to retrieve similar scam reports (RAG),
    extract structured information from user queries, track unfilled slots, and
    generate conversational responses. It ensures incremental slot filling and
    compatibility with conversation managers for logging. Designed for modularity
    in AI research, focusing on vanilla RAG for benchmark comparisons."""
    
    def __init__(self, model_name: str = "qwen2.5:7b", llm_provider: str = "Ollama", rag_csv_path: str = "rag_invocations.csv", temperature: float = 0.0):
        self.settings = get_settings()
        self.logger = setup_logger("RAG_IE_Agent", self.settings.log.subdirectories["agent"])
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.temperature = temperature

        self.messages: Sequence[AnyMessage] = []
        self.unfilled_slots: dict = {}
        self.prev_ie_output: Optional[Dict] = None
        
        # Initialize tools and agent prompts
        self.police_tools = PoliceTools(rag_csv_path=rag_csv_path)
        self.tools = self.police_tools.get_tools()
        self.rag_prompt_template = ChatPromptTemplate.from_template(Prompt.template["rag_agent"])
        self.ie_prompt_template = ChatPromptTemplate.from_messages([
            ("system", Prompt.template["baseline_police_test2"]),
            ("system", "Previous Extraction from Last Turn (MUST use as base: Copy all slots unchanged unless explicitly corrected/clarified in the NEW query only):\n{prev_ie_output}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ])

        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        self.logger.info(f"Police chatbot initialized with model: {model_name}")
        

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
        """
        Preprocesses extracted information to normalize platforms, dates, and other fields.
        This ensures consistency in scam reporting data, e.g., mapping 'text message' to 'SMS'.
        """
        
        try:

            platforms = ["LAZADA", "SHOPEE", "FACEBOOK", "CAROUSELL", "INSTAGRAM", "WHATSAPP", "SMS", "CALL"]
            
            preprocessed = police_response.copy()
            
            #Extract scam type
            scam_type = preprocessed.get("scam_type", "").upper()
            
            # Preprocessing for approach platform 
            approach = preprocessed.get("scam_approach_platform", "").upper().strip()
            if approach:
                # Normalize variations to "SMS"
                if any(word in approach for word in ["TEXT", "TEXT MESSAGE", "SMS", "MESSAGE"]):
                    approach = "SMS"
                    
                # If SMS is approach and scam_type is PHISHING, ensure communication is also SMS
                if approach == "SMS" and scam_type == "PHISHING":
                    preprocessed["scam_communication_platform"] = "SMS"
                    
                for known in platforms:
                    if known in approach:
                        approach = known
                        break
                        
                preprocessed["scam_approach_platform"] = approach
            
            comm = preprocessed.get("scam_communication_platform", "").upper().strip()
            if comm:
                # Normalize variations to "SMS"
                if any(word in comm for word in ["TEXT", "TEXT MESSAGE", "SMS"]):
                    comm = "SMS"
                
                # Check similarity and map to known platform if close
                for known in platforms:
                    if known in comm:
                        comm = known
                        break
                
                preprocessed["scam_communication_platform"] = comm
                
                # If approach not in e-commerce/social platforms, reset moniker to empty
                if approach not in ["LAZADA", "SHOPEE", "FACEBOOK", "CAROUSELL", "INSTAGRAM"]:
                    preprocessed["scam_moniker"] = ""
                    
                    
    #       Bank beneficiary implies bank transfer (only set if transaction_type is empty)
            bank = preprocessed.get("scam_beneficiary_platform", "").upper()
            if bank in ["UOB", "DBS", "HSBC", "SCB", "MAYBANK", "BOC", "CITIBANK", "CIMB", "GXS", "TRUST"]: # Avoid overwriting existing value
                preprocessed["scam_transaction_type"] = "BANK TRANSFER"
                    
            #GOIS and Phishing does not have moniker 
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
                
            # Preprocess scam_url_link to add https:// if missing
            url = preprocessed.get("scam_url_link", "").strip()
            invalid_values = {"unknown", "na", "n/a"}
            if url and url.lower() not in invalid_values and not url.startswith("https://"):
                preprocessed["scam_url_link"] = "https://" + url
                
            #If no digits are present in contact no, set to empty
            contact_no = preprocessed.get("scam_contact_no", "").strip()
            if contact_no and not any(char.isdigit() for char in contact_no):
                preprocessed["scam_contact_no"] = ""
                
            #If no digits are present in bank account, set to empty
            bank_account = preprocessed.get("scam_bank_account", "").strip()
            if bank_account and not any(char.isdigit() for char in bank_account):
                preprocessed["scam_bank_account"] = ""
            
            return preprocessed

        except Exception as e:
                self.logger.error(f"IE Preprocessing error: {e}")
                return police_response


    def _build_workflow(self):
        """
        Build the LangGraph workflow for the agent's processing pipeline.
        The workflow includes nodes for RAG retrieval, information extraction,
        and slot tracking, ensuring sequential processing of user queries.
        """
        
        workflow = StateGraph(ChatbotState)
        
        workflow.add_node("retrieval_agent", self.retrieval_agent)
        workflow.add_node("ie_agent", self.ie_agent)
        workflow.add_node("slot_tracker", self.slot_tracker)
        
        workflow.add_edge(START, "retrieval_agent")
        workflow.add_edge("retrieval_agent", "ie_agent")
        workflow.add_edge("ie_agent", "slot_tracker")
        workflow.add_edge("slot_tracker", END)
        
        return workflow.compile()
    
    def retrieval_agent(self, state: ChatbotState):
        """
        Retrieve similar scam reports using RAG and generate suggestions.
        This node invokes the RAG tool to fetch reports, then uses an LLM to
        generate suggestions for information extraction based on those reports.
        """
        
        user_query = build_query_with_history(state["messages"][-1].content, [m.content for m in state["messages"][:-1] if isinstance(m, HumanMessage)], max_history=5)
        rag_tool = self.tools[0]  
        
        #RAG tool invocation to retrieve scam reports
        try:
            rag_results = rag_tool.invoke({
                "query": user_query,
                "top_k": 5,
                "conversation_id": state["conversation_id"],
                "llm_model": self.model_name,
            })
            rag_results = json.loads(rag_results)
            self.logger.debug(f"RAG results: {rag_results}")
        except Exception as e:
            self.logger.error(f"RAG invocation failed: {str(e)}", exc_info=True)
            rag_results = []
           
        #RAG suggestions generation based on retrieved scam reports
        max_retries = 3  
        for attempt in range(max_retries):
            try:
                rag_prompt = self.rag_prompt_template.format(rag_results=json.dumps(rag_results))
                rag_llm = self._get_llm(RagOutput)
                rag_response = rag_llm.invoke(rag_prompt)
                if self.llm_provider == "OpenAI":
                    if not rag_response:
                        self.logger.error("Rag LLM returned empty response")
                        rag_suggestions = []
                    else:
                        rag_output = rag_response.model_dump()
                        rag_suggestions = RagOutput(**rag_output).model_dump()
                        self.logger.debug(f"RAG suggestions: {rag_suggestions}")
                else:
                    if not rag_response.content.strip():
                        self.logger.error("Rag LLM returned empty response")
                        rag_suggestions = []
                    else:
                        rag_output = json.loads(rag_response.content)
                        rag_suggestions = RagOutput(**rag_output).model_dump()
                        self.logger.debug(f"RAG suggestions: {rag_suggestions}")
                break  # Success, exit loop
            except Exception as e:
                self.logger.error(f"Rag LLM invocation failed (attempt {attempt+1}): {str(e)}", exc_info=True)
                if attempt == max_retries - 1:
                    rag_suggestions = {}  # Final fallback: empty dict, like original code
            
        return {
            "rag_results": rag_results,
            "rag_suggestions": rag_suggestions,
            "rag_invoked": True
        }

        
    def ie_agent(self, state: ChatbotState):
        """
        Information extraction agent that processes user inputs and generates structured output using the PoliceResponse schema. 
        Includes conversational response for conversational flow.
        """
        
        ie_llm = self._get_llm(PoliceResponse)
        messages = state["messages"]
        history = messages[:-1]
        user_input = messages[-1].content
        rag_suggestions = state.get("rag_suggestions", [])
        rag_invoked = state.get("rag_invoked", False)
        
        prompt = self.ie_prompt_template.format(
            history = history,
            user_input=user_input,
            rag_suggestions=json.dumps(rag_suggestions), 
            unfilled_slots=json.dumps(self.unfilled_slots),
            prev_ie_output=json.dumps(self.prev_ie_output or {}, indent=2),
        )
        self.logger.debug(f"Previous ie_output: {self.prev_ie_output}")

        max_retries = 3
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
                    "police_response": police_response,
                    "rag_invoked": rag_invoked
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
                        "police_response": fallback_response,
                        "rag_invoked": rag_invoked
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
        """
        Process a user query through the full workflow.
    
        Invokes the LangGraph workflow to retrieve, extract, and track slots,
        returning a response dict compatible with conversation managers.
        """
        if not query.strip():
            self.logger.error("Query cannot be empty")
            return {"error": "Query cannot be empty"}

        state = {
            "messages": self.messages + [HumanMessage(content=query)],
            "conversation_id": conversation_id,
            "police_response": {},
            "rag_results": [],
            "rag_suggestions": [],
            "rag_invoked": False,
            "rag_upsert": False, 
            "user_profile": {} 
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

        police_response = result["police_response"]
        rag_invoked = result.get("rag_invoked", False)
        rag_suggestions = result.get("rag_suggestions", [])
        conversational_response = police_response.get("conversational_response", "I'm sorry, I need more information to assist you.")

        
        structured_data = police_response.copy()
        structured_data["rag_upsert"] = False  
        structured_data["user_profile"] = {}
        structured_data["rag_suggestions"] = rag_suggestions
        structured_data["initial_profile"] = {}
        structured_data["retrieved_strategies"] = []
        structured_data["upserted_strategy"] = {}

        return {
            "response": conversational_response,
            "structured_data": structured_data,
            "conversation_id": conversation_id,
            "rag_invoked": rag_invoked,
            "rag_suggestions": rag_suggestions
        }

    def reset_state(self):
        """Reset internal state for a new conversation."""
        self.messages = []
        # self.scam_type = ""
        self.unfilled_slots = {}
        self.prev_ie_output = None
        self.logger.debug("PoliceChatbot state reset")
        

    def end_conversation(self, csv_file: Optional[str] = None) -> dict:
        """End the conversation and log completion. CSV is saved through ConversationManager. """
        
        self.logger.info(f"Conversation saved to CSV")
        return {"status": "Conversation ended"}

