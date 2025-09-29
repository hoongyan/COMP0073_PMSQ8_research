import sys
import os
from datetime import datetime
import json
from pydantic import BaseModel
from typing import TypedDict, List, Dict, Optional, Type, Union

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from src.models.response_model import UserProfile, PoliceResponse, RetrievalOutput, PoliceResponseSlots, RagOutput 
from src.agents.prompt import Prompt
from src.agents.tools import PoliceTools
from src.database.vector_operations import VectorStore 
from src.database.database_operations import DatabaseManager 
from config.settings import get_settings
from config.logging_config import setup_logger
from src.agents.utils import build_query_with_history

LEVEL_TO_NUM = {"low": 0, "high": 1.0, "distressed": 0, "neutral": 1.0}

# State
class GraphState(TypedDict):
    """
    State dictionary for the LangGraph workflow, tracking query processing across nodes.
    """
    query: str
    messages: List[BaseMessage]  # Combined history
    user_profile: Optional[UserProfile]
    retrieval_output: Optional[RetrievalOutput]  # Stores rag output
    ie_output: Optional[PoliceResponse]
    filled_this_turn: List[str]
    prev_turn_data: Dict[str, Union[List[str], Dict[str, bool]]] 
    unfilled_slots: Dict[str, bool]
    conversation_id: int
    metrics: Dict[str, Union[bool, int]]  # {'rag_retrieved': bool, 'rag_upserted': bool, 'turn_count': int}
    initial_profile: Optional[Dict]
    
class ProfileRagIEAgent:
    """
    A baseline police AI chatbot for scam reporting with user profiling but no knowledge base augmentation.
    
    Uses LangGraph for workflow orchestration. Builds on baseline functionality with
    user profiling for research in adaptive AI systems, but without self-augmentation.
    """
    
    def __init__(self, model_name: str = "qwen2.5:7b", llm_provider: str = "Ollama", rag_csv_path: str = "rag_invocations.csv", temperature = 0.0):
        """Initialize the chatbot with LLM model, tools, prompts, and workflow."""
        
        self.settings = get_settings()
        self.logger = setup_logger("Profile_RAG_IE_Agent", self.settings.log.subdirectories["agent"])
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.temperature = temperature
        
        # Initialize tools, database manager, and prompts
        self.police_tools = PoliceTools(rag_csv_path=rag_csv_path)
        db_manager = DatabaseManager()
        self.vector_store = VectorStore(session_factory=db_manager.session_factory)  
        self.user_profile_prompt = ChatPromptTemplate.from_template(Prompt.template["user_profile_test"])
        self.rag_prompt_template = ChatPromptTemplate.from_template(Prompt.template["rag_agent"])
        self.ie_prompt = ChatPromptTemplate.from_messages([
            ("system", Prompt.template["ie_profile_only"]),
            ("system", "Previous Extraction from Last Turn (MUST use as base: Copy all slots unchanged unless explicitly corrected/clarified in the NEW query only):\n{prev_ie_output}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}"),
        ])

        # Build workflow
        self.workflow = self._build_workflow()
        
        self.messages: List[BaseMessage] = []
        self.user_profile: Optional[Dict] = None
        self.turn_count = 0
        self.unfilled_slots: Optional[Dict[str,bool]] = None
        self.initial_profile = None
        self.prev_ie_output: Optional[Dict] = None
        
        # Tunable weights for user_profile_agent 
        self.profile_alpha = 0.5  # Smoothing factor for score blending 
        self.level_threshold = 0.5  # Threshold for binary level mapping
        self.conf_blend_factor = 0.5  # Weight for new conf in blending (default to 0.5 for simple averaging)

        self.logger.info(f"Police chatbot initialized with model: {model_name} and provider: {llm_provider}")

    def _get_llm(self, schema: Optional[Type[BaseModel]] = None):
        """Get the configured LLM instance, optionally with structured output schema."""
        
        if self.llm_provider == "Ollama":
            params = {
                "model": self.model_name,
                "base_url": self.settings.agents.ollama_base_url,
                "format": "json",
                "temperature": self.temperature
            }
            if schema:
                params["response_format"] = {
                    "type": "json_schema",
                    "json_schema": schema.model_json_schema()
                }
            llm = ChatOllama(**params)
        elif self.llm_provider == "OpenAI":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                self.logger.error("OPENAI_API_KEY not found")
                raise ValueError("OPENAI_API_KEY not found in environment")
            base_llm = ChatOpenAI(
                model=self.model_name,
                api_key=api_key,
                temperature = self.temperature
            )
            if schema:
                llm = base_llm.with_structured_output(schema)
            else:
                llm = base_llm
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        return llm
    
    def _preprocess_ie_output(self, ie_output: PoliceResponse) -> Dict:
        """Preprocess IE output before saving/returning. Focuses on specific keys; returns Dict."""
        try:

            platforms = ["LAZADA", "SHOPEE", "FACEBOOK", "CAROUSELL", "INSTAGRAM", "WHATSAPP", "SMS", "CALL"]
            
            preprocessed = ie_output.model_dump()
            
            #Extract scam type
            scam_type = preprocessed.get("scam_type", "").upper()
            
            # Preprocessing for approach platform 
            approach = preprocessed.get("scam_approach_platform", "").upper().strip()
            if approach:
                
                if any(word in approach for word in ["TEXT", "TEXT MESSAGE", "MESSAGE"]):
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
               
                if any(word in comm for word in ["TEXT", "TEXT MESSAGE", "MESSAGE"]):
                    comm = "SMS"
                
                # Check similarity and map to known platform if close
                for known in platforms:
                    if known in comm:
                        comm = known
                        break
                
                preprocessed["scam_communication_platform"] = comm
                
             
                if approach not in ["LAZADA", "SHOPEE", "FACEBOOK", "CAROUSELL", "INSTAGRAM"]:
                    preprocessed["scam_moniker"] = ""
                    
                    
            bank = preprocessed.get("scam_beneficiary_platform", "").upper()
            if bank in ["UOB", "DBS", "HSBC", "SCB", "MAYBANK", "BOC", "CITIBANK", "CIMB", "GXS", "TRUST"]: # Avoid overwriting existing value
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
            bank_account = preprocessed.get("scam_bank_account", "").strip()
            if bank_account and not any(char.isdigit() for char in bank_account):
                preprocessed["scam_bank_account"] = ""
            
            return preprocessed

        except Exception as e:
            self.logger.error(f"IE Preprocessing error: {e}")
            return ie_output.model_dump()

    def _build_workflow(self):
        """Build LangGraph workflow for baseline processing with profiling."""
        
        workflow = StateGraph(GraphState)
        
        workflow.add_node("shift_prev", self.shift_prev_hook)
        workflow.add_node("user_profile", self.user_profile_agent)
        workflow.add_node("retrieval", self.retrieval_agent)
        workflow.add_node("ie", self.ie_agent)
        workflow.add_node("slot_tracker", self.slot_tracker)
        
        workflow.add_edge(START, "shift_prev")
        workflow.add_edge("shift_prev", "user_profile")
        workflow.add_edge("user_profile", "retrieval")
        workflow.add_edge("retrieval", "ie")
        workflow.add_edge("ie", "slot_tracker")
        workflow.add_edge("slot_tracker", END) 
        
        return workflow.compile()

    def shift_prev_hook(self, state: GraphState) -> GraphState:
        """Shift previous turn data at the start of a new turn and increment turn count."""
        
        updates = {}
        
        if state.get("metrics", {}).get("turn_count", 0) > 0:
            ie_output = state.get("ie_output")
            conv_response = state["messages"][-1].content if state["messages"] and isinstance(state["messages"][-1], AIMessage) else ""
            
            prev_data = state.get("prev_turn_data", {}).copy()  #
            prev_data["unfilled_slots"] = {**state.get("unfilled_slots", {})} 
            prev_data["prev_response"] = conv_response
        
            updates["prev_turn_data"] = prev_data
            
        metrics = state["metrics"].copy()  
        metrics["turn_count"] = metrics.get("turn_count", 0) + 1
        updates["metrics"] = metrics
        return updates
    
    def user_profile_agent(self, state: GraphState):  
        """Infer and update user profile from query and history using exponential smoothing. 
        On first turn (initial_profile None), creates new profile and sets initial. This is used for subsequent downstream tasks.
        On later turns, updates running profile by averaging with previous. Only used to track how many turns are required to obtain accurate profile.
        """
        
        up_llm = self._get_llm(UserProfile) 
        extracted_queries = [msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)]
        prompt = self.user_profile_prompt.format(query_history=extracted_queries, query=state["query"])

        # Default profile dict (binary midpoint: score=0.5 for average) 
        default_profile = {
            "tech_literacy": {"score": 0.5, "level": "high", "confidence": 0.5},  # Default to 'high'
            "language_proficiency": {"score": 0.5, "level": "high", "confidence": 0.5},
            "emotional_state": {"score": 0.5, "level": "neutral", "confidence": 0.5}
        }
        
        # Get prior profile, force default if None or not dict
        prior_profile = state.get("user_profile", default_profile)
        if not isinstance(prior_profile, dict):
            self.logger.warning("prior_profile was not a dict; forcing default")
            prior_profile = default_profile
        
        # Get initial profile 
        initial_profile = state.get("initial_profile", None)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = up_llm.invoke(prompt)
                
                if self.llm_provider == "OpenAI":
                    self.logger.debug(f"UserProfileAgent Response: {response.model_dump()}")
                    new_profile = response.model_dump() 
                else:
                    self.logger.debug(f"UserProfileAgent Response: {response.content}")
                    new_profile = json.loads(response.content)
                    
                if not isinstance(new_profile, dict):
                    raise ValueError("Invalid new_profile from LLM")
                
                break  
            except Exception as e:
                self.logger.error(f"UserProfileAgent LLM error (attempt {attempt+1}): {str(e)}")
                if attempt < max_retries - 1:
                    prompt += "\nPrevious output invalid. Output ONLY valid JSON as specified. No extra text."
                else:
                    new_profile = prior_profile  # Reuse prior on final failure (no drift)
        
        updated_profile = {}
        changes = []
        for dim in ['tech_literacy', 'language_proficiency', 'emotional_state']:
            
            # Get prior_data (includes "score" for continuous blending). If missing, use default midpoint value (score =0.5)
            prior_data = prior_profile.get(dim, {"score": 0.5, "level": "high" if dim != "emotional_state" else "neutral", "confidence": 0.5})
            if not isinstance(prior_data, dict):
                prior_data = {"score": 0.5, "level": "high" if dim != "emotional_state" else "neutral", "confidence": 0.5}
            
            # Get new_data from LLM's new inference (map level to score).
            new_data = new_profile.get(dim, {"level": "high" if dim != "emotional_state" else "neutral", "confidence": 0.5})
            if not isinstance(new_data, dict):
                new_data = {"level": "high" if dim != "emotional_state" else "neutral", "confidence": 0.5}
            
            #Extract new levels for each dimension
            new_level = new_data["level"]
            new_conf = new_data["confidence"]
            
            # Compute new level to a score
            new_score = LEVEL_TO_NUM.get(new_level, 0.5)  # 0 for low/distressed, 1 for high/neutral, 0.5 if invalid
            
            # Blend scores using weighted average for score (use confidence as weights)
            prior_score = prior_data["score"]
            total_weight = prior_data["confidence"] + new_conf
            if total_weight > 0:  # Avoid div-by-zero
                conf_weighted_new = (new_conf * new_score) / total_weight if total_weight else new_score
                updated_score = self.profile_alpha * prior_score + (1 - self.profile_alpha) * conf_weighted_new #Set alpha as 0.5 but increase for stability if needed (favour prior profile inferences)
            else:
                updated_score = 0.5  # Rare edge case
            
            # Simple average for confidence: Blend old and new confidence to keep it stable and prevent big jumps if the LLM's guesses vary a lot.
            updated_conf = self.conf_blend_factor * new_conf + (1 - self.conf_blend_factor) * prior_data["confidence"]  # Tunable weighted blend (default to 0.5 for simple average)
            
            # Map updated_score back to binary level (threshold at 0.5)
            if dim == "emotional_state":
                updated_level = "distressed" if updated_score < self.level_threshold else "neutral"
            else:
                updated_level = "low" if updated_score < self.level_threshold else "high"
            
            # Store for next turn
            updated_profile[dim] = {"score": updated_score, "level": updated_level, "confidence": updated_conf}
            self.logger.debug(f"{dim} blending: prior_score={prior_score:.2f} (conf={prior_data['confidence']:.2f}), new_score={new_score:.2f} (conf={new_conf:.2f}) → updated_score={updated_score:.2f}, level={updated_level}")

            #Detect/track changes from initial (if set)
            if initial_profile:
                initial_level = initial_profile[dim]["level"]
                initial_score = initial_profile[dim]["score"]
                if updated_level != initial_level:
                    changes.append(f"{dim}: {initial_level}→{updated_level} (score diff {updated_score - initial_score:.2f}, conf {updated_conf:.2f})")
                    
        is_first_inference = all(prior_data["score"] == 0.5 for prior_data in prior_profile.values())
        
        #Log changes from initial profile if any (for evaluation)
        if changes:
            self.logger.info(f"Turn {state['metrics']['turn_count']}: Profile changes from initial: {'; '.join(changes)}")
        else:
            self.logger.debug(f"Turn {state['metrics']['turn_count']}: No profile changes from initial.")
        
        self.logger.debug(f"User Profile output: {updated_profile}")
        
        updates = {"user_profile": updated_profile}
        
        if is_first_inference:
            updates["initial_profile"] = updated_profile
            self.logger.info(f"Turn {state['metrics']['turn_count']}: First inference - Setting initial profile: {updated_profile}")
        return updates
        
    def retrieval_agent(self, state: GraphState):
        """
        Retrieve similar scam reports only (vanilla RAG); generate suggestions for extraction.
        Uses vanilla tool and vector store for scam reports based on query.
        """
        tool = self.police_tools.get_tools()[0]  # Vanilla tool for scam reports only
        extracted_queries = [msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)]
        user_query = build_query_with_history(state["query"], extracted_queries)

        scam_reports = []  
        rag_retrieved = False
        
        try:
            rag_results = tool.invoke({
                "query": user_query,
                "top_k": 5,
                "conversation_id": state["conversation_id"],
                "llm_model": self.model_name,
            })
            result_dict = json.loads(rag_results) if isinstance(rag_results, str) else rag_results
            scam_reports = result_dict  # Vanilla returns list of scam reports directly
            rag_retrieved = bool(scam_reports)
            
        except Exception as e:
            self.logger.error(f"RAG invocation failed: {str(e)}")
            scam_reports = []
            rag_retrieved = False
        
        max_retries = 3  
        for attempt in range(max_retries):
            try:
                rag_prompt = self.rag_prompt_template.format(rag_results=json.dumps(scam_reports))
                rag_llm = self._get_llm(schema=RagOutput)
                rag_response = rag_llm.invoke(rag_prompt)
                
                if self.llm_provider == "OpenAI":
                    rag_output = rag_response.model_dump()
                else:
                    rag_output_dict = json.loads(rag_response.content)
                    rag_output = RagOutput(**rag_output_dict).model_dump()
                
                retrieval_output = RetrievalOutput(
                    scam_reports=scam_reports,  # Raw scam reports retrieved
                    strategies=[],  # Empty for baseline compatibility
                    rag_suggestions=rag_output
                )

                self.logger.debug(f"RetrievalAgent output: {retrieval_output.model_dump()}")
                break  # Success, exit loop
            except Exception as e:
                self.logger.error(f"RetrievalAgent error (attempt {attempt+1}): {str(e)}")
                if attempt == max_retries - 1:
                    retrieval_output = RetrievalOutput(scam_reports=[], strategies=[])  # Final fallback: empty
            
        return {
            "retrieval_output": retrieval_output,
            "metrics": {**state["metrics"], "rag_retrieved": rag_retrieved}
            }
    
    def ie_agent(self, state: GraphState):
        """
        Extract structured scam info and generate conversational response using LLM.
        Preprocesses output for consistency; uses retrieved suggestions and history.
        """
        
        ie_llm = self._get_llm(schema=PoliceResponse)
        prev_ie = state["prev_turn_data"].get("prev_ie_output", {})
        
        prompt = self.ie_prompt.format(
            history=state["messages"],
            user_input=state["query"],
            user_profile=state.get("initial_profile", state["user_profile"]),
            rag_suggestions=state["retrieval_output"].rag_suggestions,
            unfilled_slots=json.dumps(state["unfilled_slots"]),
            prev_ie_output=json.dumps(prev_ie, indent=2)
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = ie_llm.invoke(prompt)
                if self.llm_provider == "OpenAI":
                    ie_output = response
                else:
                    ie_output_dict = json.loads(response.content)
                    ie_output = PoliceResponse(**ie_output_dict)
            
                self.logger.debug(f"IEAgent output: {ie_output.model_dump()}")
                
                preprocessed_ie_output = self._preprocess_ie_output(ie_output)
                self.logger.debug(f"Preprocessed IE: {preprocessed_ie_output}")
                
                return {"ie_output": preprocessed_ie_output}
            
            except Exception as e:
                self.logger.error(f"IEAgent attempt {attempt+1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    ie_output = PoliceResponse(
                        conversational_response="Sorry, I encountered an error. Can you provide more details about the scam, such as the date, what the caller said, and any contact details provided?",
                        scam_incident_description=state["query"]
                    )
                    preprocessed_ie_output = self._preprocess_ie_output(ie_output)
                    self.logger.debug(f"Preprocessed IE (fallback): {preprocessed_ie_output}")
                    return {"ie_output": preprocessed_ie_output}
                

    def slot_tracker(self, state: GraphState):
        """
        Track unfilled slots and identify those filled this turn.
        """
        
        ie_output = state["ie_output"]
        state["prev_turn_data"]["unfilled_slots"] = state.get("unfilled_slots", {})
        
        # Compute unfilled
        unfilled = {}
        for field_name in PoliceResponseSlots:
            value = ie_output.get(field_name.value)
            is_unfilled = value in ("", 0.0, {}, None, "unknown", "na") or not value
            unfilled[field_name.value] = is_unfilled
        
        # Compute filled_this_turn
        prev_unfilled = state["prev_turn_data"]["unfilled_slots"]
        filled_this_turn = [slot for slot in prev_unfilled if prev_unfilled[slot] and not unfilled.get(slot, True)]
        
        self.logger.debug(f"SlotTracker: unfilled={unfilled}, filled={filled_this_turn}")
        return {
        "unfilled_slots": unfilled,
        "filled_this_turn": filled_this_turn
    }

    def process_query(self, query: str, conversation_id: int = None) -> dict:
        """
        Processes a user query through the workflow, returning response and structured data for conversation manager.
        Appends query to messages, invokes graph, persists updates. Ensures compatibility with external managers.
        """
        if not query.strip():
            self.logger.error("Empty query")
            return {"response": "Query cannot be empty", "structured_data": {}, "rag_invoked": False, "conversation_id": conversation_id}

        # NEW: Print initial_profile at entry
        self.logger.info(f"Entering process_query - Persisted self.initial_profile: {self.initial_profile}")
        self.messages.append(HumanMessage(content=query))
        state = {
            "query": query,
            "messages": self.messages[:-1],
            "user_profile": self.user_profile,
            "retrieval_output": None,
            "ie_output": None,
            "filled_this_turn": [],
            "prev_turn_data": {"unfilled_slots": {}, "prev_response": self.messages[-1].content if self.messages and isinstance(self.messages[-1], AIMessage) else "", "prev_ie_output": self.prev_ie_output or {}},
            "unfilled_slots": {s.value: True for s in PoliceResponseSlots} if self.unfilled_slots is None else self.unfilled_slots.copy(),
            "conversation_id": conversation_id,
            "metrics": {"rag_retrieved": False, "rag_upserted": False, "turn_count": self.turn_count},
            "initial_profile": self.initial_profile,
        }
        
        final_state = self.workflow.invoke(state)
        
        self.user_profile = final_state.get("user_profile")  # Add: Sync class var
        self.initial_profile = final_state.get("initial_profile", self.initial_profile)  # Sync updated initial_profile back to class (fallback to current if not set)
        self.unfilled_slots = final_state["unfilled_slots"]

        ie_out = final_state["ie_output"]
        response = ie_out.get("conversational_response") if ie_out else "Error processing query."
        structured_data = ie_out.copy() if ie_out else {}

        # Add defaults for compatibility (no KB)
        structured_data["rag_upsert"] = False
        structured_data["rag_suggestions"] = final_state["retrieval_output"].rag_suggestions if final_state.get("retrieval_output") else {}
        structured_data["initial_profile"] = final_state.get("initial_profile")  # Add/update this to persist across calls
        structured_data["user_profile"] = self.user_profile
        structured_data["retrieved_strategies"] = []  # Empty for baseline
        structured_data["upserted_strategy"] = {}  # Empty for baseline
        
        self.logger.debug(f"Structured Data before return: {json.dumps(structured_data, indent=2)}")
        
        # Save back for next call
        self.messages.append(AIMessage(content=response))
        self.turn_count = final_state["metrics"]["turn_count"]
        if final_state.get("ie_output"):
            prev_ie = final_state["ie_output"].copy()  
            if "conversational_response" in prev_ie:
                del prev_ie["conversational_response"]  
            self.prev_ie_output = prev_ie
        
        return {
            "response": response,
            "structured_data": structured_data,
            "rag_invoked": final_state["metrics"]["rag_retrieved"],
            "conversation_id": conversation_id
        }

    def reset_state(self):
        """Reset class-level state for a new conversation."""
        
        # self.upsert_count = 0
        self.messages = []
        self.user_profile = None
        self.turn_count = 0
        self.unfilled_slots = None
        self.initial_profile = None
        self.prev_ie_output = None
        self.logger.debug("PoliceChatbot state reset")

    def end_conversation(self):
        """End conversation and reset."""
        
        self.reset_state()
        self.logger.info("Conversation ended")
        return {"status": "Conversation ended"}