import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from typing import TypedDict, List, Dict, Optional, Type, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END, START
from src.models.response_model import UserProfile, PoliceResponse, RetrievalOutput, PoliceResponseSlots, KnowledgeBaseOutput #QuestionSlotMap, ValidatedQuestionSlotMap
from src.agents.prompt import Prompt
from src.agents.tools import PoliceTools
from src.database.vector_operations import VectorStore
from src.database.database_operations import DatabaseManager, CRUDOperations
from src.models.data_model import Strategy
from config.settings import get_settings
from config.logging_config import setup_logger
from src.agents.utils import format_history_as_messages, build_query_with_history
import json
from datetime import datetime
import csv
from pydantic import BaseModel
from copy import deepcopy
import numpy as np
from pydantic import ValidationError
import re


LEVEL_TO_NUM = {"low": 0, "high": 1.0, "distressed": 0, "neutral": 1.0, }
NUM_TO_LEVEL = {0: "low", 1: "high"}  # Aligns with your binary levels
EMO_NUM_TO_LEVEL = {0: "distressed", 1: "neutral"}

# State
class GraphState(TypedDict):
    query: str
    query_history: List[str]
    past_responses: List[str]  # Track AI outputs
    user_profile: Optional[UserProfile]
    templates: Optional[RetrievalOutput]
    ie_output: Optional[PoliceResponse]
    filled_this_turn: List[str]
    prev_turn_data: Dict[str, Union[List[str], Dict[str, bool]]] 
    unfilled_slots: Dict[str, bool]
    conversation_id: int
    metrics: Dict[str, Union[bool, int]]  # {'rag_retrieved': bool, 'rag_upserted': bool, 'turn_count': int}

class SelfAugmentingPoliceChatbot:
    def __init__(self, model_name: str = "qwen2.5:7b", llm_provider: str = "Ollama", rag_csv_path: str = "rag_invocations.csv"):
        self.settings = get_settings()
        self.logger = setup_logger("Augmented_PoliceAgent", self.settings.log.subdirectories["agent"])
        self.model_name = model_name
        self.llm_provider = llm_provider
        # Tools and DB
        self.police_tools = PoliceTools(rag_csv_path=rag_csv_path)
        db_manager = DatabaseManager()
        self.vector_store = VectorStore(db_manager.session_factory)
        self.strategy_crud = CRUDOperations(Strategy, db_manager.session_factory)
        
        # Prompts
        self.user_profile_prompt = ChatPromptTemplate.from_template(Prompt.template["user_profile_test"])
        self.ie_prompt = ChatPromptTemplate.from_messages([
            ("system", Prompt.template["ie"]),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{query}"),
        ])
        self.kb_prompt = ChatPromptTemplate.from_template(Prompt.template["knowledge_base"])
        self.rag_prompt_template = ChatPromptTemplate.from_template(Prompt.template["rag_agent"])


        # Workflow
        self.workflow = self._build_workflow()
        self.upsert_count = 0  # For pruning trigger
        self.logger.info(f"Police chatbot initialized with model: {model_name} and provider: {llm_provider}")
        self.query_history: List[str] = []
        self.past_responses: List[str] = []
        self.user_profile: Optional[Dict] = None
        self.turn_count=0

    
    def _get_llm(self, schema: Optional[Type[BaseModel]] = None):
        """Get configured LLM, optionally with structured output."""
        if self.llm_provider == "Ollama":
            params = {
                "model": self.model_name,
                "base_url": self.settings.agents.ollama_base_url,
                "format": "json"
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
            )
            if schema:
                llm = base_llm.with_structured_output(schema)
            else:
                llm = base_llm
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
        return llm

    def _build_workflow(self):
        """Build LangGraph workflow for augmented processing."""
        workflow = StateGraph(GraphState)
        workflow.add_node("shift_prev", self.shift_prev_hook)
        workflow.add_node("user_profile", self.user_profile_agent)
        workflow.add_node("retrieval", self.retrieval_agent)
        workflow.add_node("ie", self.ie_agent)
        workflow.add_node("slot_tracker", self.slot_tracker)
        workflow.add_node("knowledge_base", self.knowledge_base_agent)
        
        workflow.add_edge(START, "shift_prev")
        workflow.add_edge("shift_prev", "user_profile")
        workflow.add_edge("user_profile", "retrieval")
        workflow.add_edge("retrieval", "ie")
        workflow.add_edge("ie", "slot_tracker")
        workflow.add_edge("slot_tracker", "knowledge_base")
        workflow.add_edge("knowledge_base", END)
        
        return workflow.compile()

    def shift_prev_hook(self, state: GraphState) -> GraphState:
        """Shift data to prev_turn_data at turn start."""
        if state.get("metrics", {}).get("turn_count", 0) > 0:
            ie_output = state.get("ie_output")
            conv_response = state["past_responses"][-1] if state["past_responses"] else ""
            state["prev_turn_data"] = {
                # "target_slots": deepcopy(state.get("target_slots", [])),
                "unfilled_slots": {**state.get("unfilled_slots", {})},
                "prev_response": conv_response
            }
        state["metrics"]["turn_count"] = state["metrics"].get("turn_count", 0) + 1
        return state

    def _extract_prev_response(self, response: str) -> str:
        """Extract full previous conversational response."""
        return response.strip()
    
    
    def user_profile_agent(self, state: GraphState, alpha: float = 0.98) -> GraphState:  # Increased alpha for stability
        """Infer user profile from query and history."""
        up_llm = self._get_llm()  # No schema
        prompt = self.user_profile_prompt.format(query_history=state["query_history"], query=state["query"])
        
        # Default profile dict (binary midpoint: score=0.5 for average)
        default_profile = {
            "tech_literacy": {"score": 0.5, "level": "high", "confidence": 0.5},  # Default to 'high'
            "language_proficiency": {"score": 0.5, "level": "high", "confidence": 0.5},
            "emotional_state": {"score": 0.5, "level": "neutral", "confidence": 0.5}
        }
        
        # Get prior, force default if None or not dict
        prior_profile = state.get("user_profile", default_profile)
        if not isinstance(prior_profile, dict):
            self.logger.warning("prior_profile was not a dict; forcing default")
            prior_profile = default_profile
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = up_llm.invoke(prompt)
                self.logger.debug(f"UserProfileAgent Response: {response.content}")
                new_profile = json.loads(response.content)  # Dict of dicts like {"tech_literacy": {"level": "low", "confidence": 0.8}}
                
                # Safety: If new_profile not dict or missing keys, set defaults
                if not isinstance(new_profile, dict):
                    raise ValueError("Invalid new_profile from LLM")
                
                break  # Success
            except Exception as e:
                self.logger.error(f"UserProfileAgent LLM error (attempt {attempt+1}): {str(e)}")
                if attempt < max_retries - 1:
                    prompt += "\nPrevious output invalid. Output ONLY valid JSON as specified. No extra text."
                else:
                    new_profile = prior_profile  # Reuse prior on final failure (no drift)
        
        updated_profile = {}
        for dim in ['tech_literacy', 'language_proficiency', 'emotional_state']:
            # Safe get for prior_data (includes "score" for continuous blending)
            prior_data = prior_profile.get(dim, {"score": 0.5, "level": "high" if dim != "emotional_state" else "neutral", "confidence": 0.5})
            if not isinstance(prior_data, dict):
                prior_data = {"score": 0.5, "level": "high" if dim != "emotional_state" else "neutral", "confidence": 0.5}
            
            # Safe get for new_data (map level to score)
            new_data = new_profile.get(dim, {"level": "high" if dim != "emotional_state" else "neutral", "confidence": 0.5})
            if not isinstance(new_data, dict):
                new_data = {"level": "high" if dim != "emotional_state" else "neutral", "confidence": 0.5}
            
            new_level = new_data["level"]
            new_conf = new_data["confidence"]
            
            # Compute new_score from level (no LLM score)
            new_score = LEVEL_TO_NUM.get(new_level, 0.5)  # 0 for low/distressed, 1 for high/neutral, 0.5 if invalid
            
            # Weighted average for score (confidence as weight, ProfiLLM-style)
            prior_score = prior_data["score"]
            total_weight = prior_data["confidence"] + new_conf
            if total_weight > 0:  # Avoid div-by-zero
                conf_weighted_new = (new_conf * new_score) / total_weight if total_weight else new_score
                updated_score = alpha * prior_score + (1 - alpha) * conf_weighted_new  # Blend
            else:
                updated_score = 0.5  # Rare edge case
            
            # Weighted average for confidence (simple avg for stability)
            updated_conf = (prior_data["confidence"] + new_conf) / 2
            
            # Map updated_score back to binary level (threshold at 0.5)
            if dim == "emotional_state":
                updated_level = "distressed" if updated_score < 0.5 else "neutral"
            else:
                updated_level = "low" if updated_score < 0.5 else "high"
            
            # Store for next turn
            updated_profile[dim] = {"score": updated_score, "level": updated_level, "confidence": updated_conf}
            self.logger.debug(f"{dim} blending: prior_score={prior_score:.2f} (conf={prior_data['confidence']:.2f}), new_score={new_score:.2f} (conf={new_conf:.2f}) → updated_score={updated_score:.2f}, level={updated_level}")

        state["user_profile"] = updated_profile
        
        self.logger.debug(f"User Profile output: {updated_profile}")
        return state
    
    # def user_profile_agent(self, state: GraphState, alpha: float = 0.98) -> GraphState:  # alpha=0.98 for stability
    #     """Infer user profile from query and history."""
    #     up_llm = self._get_llm()  # No schema
    #     # NEW: Apply exponential decay to history (recent more weighted)
    #     weighted_history = []
    #     decay_factor = 0.8  # Tuneable: 0.8 means oldest ~0.5 weight if long history
    #     for i, q in enumerate(reversed(state["query_history"])):  # Reverse: latest first
    #         weight = decay_factor ** i
    #         weighted_history.append(f"Weighted Query (weight={weight:.2f}): {q}")
    #     weighted_history_str = "\n".join(weighted_history)
        
    #     # UPDATED PROMPT: Added balance for high/neutral cues
    #     prompt = self.user_profile_prompt.format(
    #         query_history=weighted_history_str, 
    #         query=state["query"]
    #     ) + "\nAdditional Guidance: Fluent, error-free sentences indicate 'high' language_proficiency even if neutral. Factual, composed tone without urgency indicates 'neutral' emotional_state. Require consistent errors/repetitions for 'low' language_proficiency. Add examples: Neutral high: 'The transaction occurred on...' → high proficiency, neutral state."

    #     # Default profile dict
    #     default_profile = {
    #         "tech_literacy": {"score": 0.5, "level": "high", "confidence": 0.5},
    #         "language_proficiency": {"score": 0.5, "level": "high", "confidence": 0.5},
    #         "emotional_state": {"score": 0.5, "level": "neutral", "confidence": 0.5}
    #     }
        
    #     # Get prior, force default if invalid
    #     prior_profile = state.get("user_profile", default_profile)
    #     if not isinstance(prior_profile, dict):
    #         self.logger.warning("prior_profile was not a dict; forcing default")
    #         prior_profile = default_profile
        
    #     # NEW: Detect and store initial baseline profile (first non-default inference)
    #     baseline_profile = state.get("baseline_profile", None)  # Persist across turns via state
    #     is_first_inference = all(prior_data["score"] == 0.5 for prior_data in prior_profile.values())  # Check if prior is default
        
    #     max_retries = 5  # Increased retries for robustness
    #     for attempt in range(max_retries):
    #         try:
    #             response = up_llm.invoke(prompt)
    #             self.logger.debug(f"UserProfileAgent Response: {response.content}")
    #             new_profile = json.loads(response.content)
    #             if not isinstance(new_profile, dict):
    #                 raise ValueError("Invalid new_profile from LLM")
    #             break  # Success
    #         except Exception as e:
    #             self.logger.error(f"UserProfileAgent LLM error (attempt {attempt+1}): {str(e)}")
    #             if attempt < max_retries - 1:
    #                 prompt += "\nPrevious output invalid. Output ONLY valid JSON as specified. No extra text."
    #             else:
    #                 new_profile = {dim: default_profile[dim] for dim in default_profile}  # Fallback to default
        
    #     updated_profile = {}
    #     for dim in ['tech_literacy', 'language_proficiency', 'emotional_state']:
    #         prior_data = prior_profile.get(dim, default_profile[dim])
    #         if not isinstance(prior_data, dict):
    #             prior_data = default_profile[dim]
            
    #         new_data = new_profile.get(dim, default_profile[dim])
    #         if not isinstance(new_data, dict):
    #             new_data = default_profile[dim]
            
    #         new_level = new_data["level"]
    #         new_conf = new_data["confidence"]
            
    #         # NEW: Rule-based booster for high/neutral (counter bias)
    #         booster = 0.0
    #         if dim in ['tech_literacy', 'language_proficiency']:
    #             # Simple regex: No common low cues (fragments, repetitions)
    #             low_cues = re.compile(r'\b(I no|what mean|help!|oh no|scared|stupid)\b|(\w+)\s+\1', re.I)  # Errors/repeats
    #             if not low_cues.search(state["query"]):
    #                 booster = 0.1  # +0.1 if no low cues (reinforce high)
    #         new_score = LEVEL_TO_NUM.get(new_level, 0.5) + booster  # Cap at 1.0
    #         new_score = min(1.0, new_score)
            
    #         # CHANGED: For first inference, set directly to new_score (no prior blend)
    #         if is_first_inference:
    #             updated_score = new_score  # Direct set: No pull from default 0.5
    #             updated_conf = new_conf
    #         else:
    #             # Normal blending for subsequent turns
    #             prior_score = prior_data["score"]
    #             if new_conf < 0.5:  # Threshold - ignore low-conf new outputs
    #                 updated_score = prior_score
    #                 updated_conf = prior_data["confidence"]
    #             else:
    #                 total_weight = prior_data["confidence"] + new_conf
    #                 conf_weighted_new = (new_conf * new_score) / total_weight if total_weight else new_score
    #                 updated_score = alpha * prior_score + (1 - alpha) * conf_weighted_new
    #                 updated_conf = (prior_data["confidence"] + new_conf) / 2
            
    #         # NEW: Pull toward baseline if exists (weight initial inference heavily)
    #         if baseline_profile:
    #             baseline_score = baseline_profile[dim]["score"]
    #             baseline_weight = 0.3  # Tuneable: 30% pull toward initial
    #             updated_score = (1 - baseline_weight) * updated_score + baseline_weight * baseline_score
            
    #         # Map back to level
    #         if dim == "emotional_state":
    #             updated_level = "distressed" if updated_score < 0.5 else "neutral"
    #         else:
    #             updated_level = "low" if updated_score < 0.5 else "high"
            
    #         updated_profile[dim] = {"score": updated_score, "level": updated_level, "confidence": updated_conf}
    #         if not is_first_inference:
    #             self.logger.debug(f"{dim} blending: prior_score={prior_score:.2f} (conf={prior_data['confidence']:.2f}), new_score={new_score:.2f} (conf={new_conf:.2f}) → updated_score={updated_score:.2f}, level={updated_level}")
    #         else:
    #             self.logger.debug(f"{dim} first inference: new_score={new_score:.2f} (conf={new_conf:.2f}) → updated_score={updated_score:.2f}, level={updated_level}")
    #                 # NEW: Set baseline if first inference
    #     if is_first_inference:
    #         state["baseline_profile"] = updated_profile  # Persist in state for future turns
        
    #     state["user_profile"] = updated_profile
        
    #     self.logger.debug(f"User Profile output: {updated_profile}")
    #     return state

    def retrieval_agent(self, state: GraphState) -> GraphState:
        self.rag_llm = self._get_structured_llm(RagOutput)
        self.logger.debug(f"Invoking RAG for query: {state['messages'][-1].content}")
        user_query = build_query_with_history(state["messages"][-1].content, [m.content for m in state["messages"][:-1] if isinstance(m, HumanMessage)], max_history=5)
        rag_tool = tools[0]  # retrieve_scam_reports
        try:
            rag_results = rag_tool.invoke({
                "query": user_query,
                "top_k": 3,
                "conversation_id": state["conversation_id"],
                "llm_model": self.model_name,
            })
            rag_results = json.loads(rag_results)
            self.logger.debug(f"RAG results: {rag_results}")
        except Exception as e:
            self.logger.error(f"RAG invocation failed: {str(e)}", exc_info=True)
            rag_results = []
        try:
            rag_prompt = self.rag_prompt_template.format(rag_results=json.dumps(rag_results))
            rag_response = self.rag_llm.invoke(rag_prompt)
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
        except Exception as e:
                self.logger.error(f"Rag LLM invocation failed: {str(e)}", exc_info=True)
                rag_suggestions = []
            
        self.logger.info("RAG invoked successfully")
        return {
            "rag_results": rag_results,
            "rag_suggestions": rag_suggestions,
            "rag_invoked": True
        }

    def ie_agent(self, state: GraphState) -> GraphState:
        """Extract information and generate response."""
        ie_llm = self._get_llm(schema=PoliceResponse)
        history_messages = format_history_as_messages(state["query_history"], state["past_responses"])
        prompt = self.ie_prompt.format(
            messages=history_messages,
            query=state["query"],
            user_profile=state["user_profile"],
            scam_reports=state["templates"].scam_reports,
            strategies=state["templates"].strategies
        )
        try:
            response = ie_llm.invoke(prompt)
            if self.llm_provider == "OpenAI":
                ie_output = response
            else:
                ie_output_dict = json.loads(response.content)
                ie_output = PoliceResponse(**ie_output_dict)
            state["ie_output"] = ie_output
            self.logger.debug(f"IEAgent output: {ie_output.model_dump()}")
        except Exception as e:
            self.logger.error(f"IEAgent error: {str(e)}")
            state["ie_output"] = PoliceResponse(
                conversational_response="I'm sorry, I need more information. Can you provide details?",
                scam_incident_description=state["query"]
            )
        return state

    # def _map_questions_to_slots(self, conversational: str) -> Dict[str, List[str]]:
    #     """Map questions to slots using LLM."""
    #     prompt = self.target_slot_prompt.format(conversational_response=conversational)
    #     llm = self._get_llm(QuestionSlotMap)
        
    #     max_retries = 2
    #     for attempt in range(max_retries):
    #         response = llm.invoke(prompt)
    #         try:
    #             mapped = response.root
    #             issues = self.detect_issues(mapped)
    #             if issues:
    #                 if attempt < max_retries - 1:
    #                     corrective = prompt + f"\nIssues: {', '.join(issues)}. Fix: Full questions, no fragments/empties."
    #                     prompt = corrective
    #                     continue
    #                 else:
    #                     mapped = self.rule_based_fallback(conversational)
    #             break
    #         except Exception as e:
    #             self.logger.error(f"Mapping error (attempt {attempt+1}): {e}")
        
    #     try:
    #         validated = ValidatedQuestionSlotMap(root=mapped)
    #         return validated.root
    #     except ValidationError as e:
    #         self.logger.error(f"Validation failed: {e}. Fallback.")
    #         return self.rule_based_fallback(conversational)

    # def detect_issues(self, mapped_data: dict) -> list[str]:
    #     """Detect issues in mapped data."""
    #     issues = []
    #     for slot, questions in mapped_data.items():
    #         if not questions:
    #             issues.append(f"Empty list for {slot}")
    #         for q in questions:
    #             if len(q.strip().split()) < 3 or re.match(r'^(and|or|if)\b', q.strip().lower()):
    #                 issues.append(f"Fragment in {slot}: {q}")
    #     return issues

    # def rule_based_fallback(self, conversational: str) -> dict:
    #     """Rule-based fallback for question-slot mapping."""
    #     slots = {}
    #     keywords = {  # From your provided code
    #         'scam_incident_date': ['date', 'when'],
    #         # ... (add all keywords as in your code)
    #     }
    #     for slot, kws in keywords.items():
    #         if any(kw in conversational.lower() for kw in kws):
    #             slots[slot] = [conversational]
    #     return slots

    def slot_tracker(self, state: GraphState) -> GraphState:
        """Track slots and map questions."""
        ie_output = state["ie_output"]
        state["prev_turn_data"]["unfilled_slots"] = state.get("unfilled_slots", {})
        
        # # Map questions
        # conversational = ie_output.conversational_response
        # mapped_data = self._map_questions_to_slots(conversational)
        # state["target_slot_map"] = mapped_data
        # state["target_slots"] = list(mapped_data.keys())
        
        # Compute unfilled
        unfilled = {}
        for field_name in PoliceResponseSlots:
            value = getattr(ie_output, field_name.value)
            is_unfilled = value in ("", 0.0, {}, None, "unknown", "na") or not value
            unfilled[field_name.value] = is_unfilled
        state["unfilled_slots"] = unfilled
        
        # Compute filled_this_turn
        prev_unfilled = state["prev_turn_data"]["unfilled_slots"]
        filled_this_turn = [slot for slot in prev_unfilled if prev_unfilled[slot] and not unfilled.get(slot, True)]
        state["filled_this_turn"] = filled_this_turn
        
        self.logger.debug(f"SlotTracker: unfilled={unfilled}, filled={filled_this_turn}")
        return state

    def knowledge_base_agent(self, state: GraphState) -> GraphState:
        """Evaluate and upsert strategies."""
        kb_llm = self._get_llm(schema=KnowledgeBaseOutput)
        filled_this_turn = state["filled_this_turn"]
        # prev_questions = state["prev_turn_data"].get("questions_asked", [])
        # prev_targets = state["prev_turn_data"].get("target_slots", [])
        
        if not filled_this_turn: #or not prev_targets or not prev_questions:
            self.logger.debug("Skipping KB upsert: Insufficient data.")
            return state
        
        prev_response = state["prev_turn_data"].get("prev_response", "")
        if not prev_response:
            self.logger.debug("Skipping KB upsert: No previous response.")
            return state
        
        prompt = self.kb_prompt.format(
            prev_turn_data=state["prev_turn_data"],
            filled_slots_this_turn=filled_this_turn,
            query=state["query"],
            query_history=state["query_history"],
            user_profile=state["user_profile"]
        )
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = kb_llm.invoke(prompt)
                if self.llm_provider == "OpenAI":
                    kb_output = response
                else:
                    kb_output_dict = json.loads(response.content)
                    kb_output = KnowledgeBaseOutput(**kb_output_dict)
                state["kb_outputs"] = kb_output
                self.logger.debug(f"KBAgentOutput: {kb_output}")
                
                num_targets = len(filled_this_turn)
                total_slots = len(PoliceResponseSlots)
                target_score = 0.3 * min(max(num_targets, 0) / total_slots, 1.0)  # Ensures max 0.3, handles num_targets=0

                score = target_score + 0.6 * kb_output.avg_rating() + 0.1 * (1 if state["metrics"]["turn_count"] < 2 else 0)
                self.logger.info(f"Score: {score}")
                if score >= 0.6:
                    similar_df = self.vector_store.hierarchical_strategy_search(state["user_profile"], limit=5) 
                    to_upsert = True
                    for _, row in similar_df.iterrows():
                        if row['profile_distance'] < 0.3 and np.linalg.norm(self.vector_store.model.encode(row['response']) - self.vector_store.model.encode(prev_response)) < 0.3:
                            if score > row['success_score']:
                                self.strategy_crud.update(row['id'], {'success_score': score})
                            to_upsert = False
                            break
                    if to_upsert:
                        strategy_data = {
                            "strategy_type": kb_output.strategy_type,
                            "response": prev_response,
                            "success_score": score,
                            "user_profile": state["user_profile"],  #change to model_dump if pydantic model
                            "profile_embedding": self.vector_store.model.encode(json.dumps(state["user_profile"])),
                            "strategy_embedding": self.vector_store.model.encode(kb_output.strategy_type),
                            "response_embedding": self.vector_store.model.encode(prev_response)
                        }
                        self.strategy_crud.create(strategy_data)  # Pass the dict directly
                        self.upsert_count += 1
                        state["metrics"]["rag_upserted"] = True
                        if self.upsert_count % 50 == 0:
                            self.vector_store.prune_strategies()
                break
            except Exception as e:
                self.logger.error(f"KB agent error (attempt {attempt+1}): {str(e)}")
        return state

    def process_query(self, query: str, conversation_id: int = None) -> dict:
        """Process query and return response dict for manager."""
        if not query.strip():
            self.logger.error("Empty query")
            return {"response": "Query cannot be empty", "structured_data": {}, "rag_invoked": False, "conversation_id": conversation_id}

        state = {
            "query": query,
            "query_history": self.query_history[:],  # Copy accumulated
            "past_responses": self.past_responses[:],
            "user_profile": self.user_profile,
            "templates": None,
            "ie_output": None,
            "filled_this_turn": [],
            "prev_turn_data": {"unfilled_slots": {},"prev_response": ""},
            "unfilled_slots": {s.value: True for s in PoliceResponseSlots},
            "conversation_id": conversation_id,
            "metrics": {"rag_retrieved": False, "rag_upserted": False, "turn_count": self.turn_count}
        }
        state["query_history"].append(query)
        
        
        final_state = self.workflow.invoke(state)
        
        ie_out = final_state["ie_output"]
        response = ie_out.conversational_response if ie_out else "Error processing query."
        structured_data = ie_out.model_dump() if ie_out else {}
        
        # Add rag_upsert from metrics
        structured_data["rag_upsert"] = final_state["metrics"].get("rag_upserted", False)
    
        # Add user_profile from state
        self.user_profile = final_state["user_profile"]
        structured_data["user_profile"] = self.user_profile
        
 
        # structured_data["target_slot_map"] = final_state["target_slot_map"]
        
        # Save back for next call
        self.query_history = final_state["query_history"]  # Updated from workflow
        self.past_responses.append(response)  # Append new response
        self.turn_count = final_state["metrics"]["turn_count"]
        
        return {
            "response": response,
            "structured_data": structured_data,
            "rag_invoked": final_state["metrics"]["rag_retrieved"],
            "conversation_id": conversation_id
        }

    def reset_state(self):
        """Reset internal state for new conversation."""
        self.upsert_count = 0
        self.query_history = []  # Reset history
        self.past_responses = []
        self.user_profile = None
        self.turn_count = 0
        self.logger.debug("PoliceChatbot state reset")

    def end_conversation(self):
        """End conversation and reset."""
        self.reset_state()
        self.logger.info("Conversation ended")
        return {"status": "Conversation ended"}
    
if __name__ == "__main__":
    from config.id_manager import IDManager
    logger = setup_logger("SelfAugmenting_PoliceAgent", get_settings().log.subdirectories["agent"])
    models = [
    ("gpt-4o-mini", "OpenAI"),
    ("qwen2.5:7b", "Ollama"),
    ("granite3.2:8b", "Ollama"),
    ("mistral:7b", "Ollama")
]
    queries = [
        "Help me! Help me! I got an SMS and all my money is gone! I am in trouble!",
        "It happened last week, on July 20, 2025. I feel so stupid! How could I be so stupid!",
        "I don't know! I don't know! All I saw was this SMS telling me that I had outstanding bills and I clicked the link!",
        "$1500 was transferred to their account via bank transfer. I should not have done that at all!",
        "I can't quite remember. What is a bank account number? How do I know where to find it? The contact number was +1-123-456-7890 I think. How do I check? I don't know anymore! I can't do this!",
        "I think the bank account number is HSBC 123456789."
    ]
    
    results = {}
    id_manager = IDManager(csv_file="rag_invocations.csv")  # Initialize ID manager
    
    logger.info("Starting multi-turn model testing (independent turns)")
    for model_name, llm_provider in models:
        logger.info(f"--- Testing model: {model_name} with provider {llm_provider} ---")
        try:
            chatbot = SelfAugmentingPoliceChatbot(model_name=model_name, llm_provider=llm_provider)
            conversation_id = id_manager.get_next_id()
            
            model_responses = []
            
            for i, query in enumerate(queries, 1):
                logger.info(f"Processing turn {i} with query: '{query}'")
                response = chatbot.process_query(query, conversation_id=conversation_id)
                model_responses.append({
                    "turn": i,
                    "query": query,
                    "response": response["response"],
                    "structured_data": response["structured_data"]
                })
                logger.info(f"Turn {i} response: {json.dumps(response, indent=2)}")
            
            results[model_name] = model_responses
            
            chatbot.end_conversation()
        except Exception as e:
            logger.error(f"Error with model {model_name}: {str(e)}", exc_info=True)
            results[model_name] = {"error": f"Error: {str(e)}"}

    logger.info("Completed multi-turn model testing")
    print(json.dumps(results, indent=2))