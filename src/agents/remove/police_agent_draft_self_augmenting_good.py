# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# from typing import Dict, List
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_ollama import ChatOllama
# from langgraph.graph import StateGraph, END
# from config.id_manager import IDManager
# from config.settings import get_settings
# from config.logging_config import setup_logger
# from src.database.database_operations import CRUDOperations, DatabaseManager
# from src.database.vector_operations import VectorStore
# from src.models.data_model import Strategy
# from src.models.response_model import UserProfile, RetrievalOutput, IEOutput, KnowledgeBaseOutput, ProcessedOutput, PipelineState, PoliceResponseSlots
# from src.agents.prompt import Prompt
# from src.agents.tools import PoliceTools
# import json

# class SelfAugmentingRag:
#     """Multi-agent police chatbot for scam reporting with self-augmenting RAG.

#     This class processes victim queries to generate an augmented prompt stored in the pipeline state,
#     without saving conversation history to a CSV file.
#     """
    
#     def __init__(self, model_name: str = "llama3.1:8b"):
#         """Initialize the chatbot with dependencies and workflow."""
#         self.settings = get_settings()
#         self.logger = setup_logger("SelfAugmentingRAG", self.settings.log.subdirectories["agent"])
#         self.model_name = model_name
#         self.id_manager = IDManager()
#         self.conversation_id = self.id_manager.get_next_id()
        
#         # Initialize database and vector store
#         db_manager = DatabaseManager()
#         self.vector_store = VectorStore(db_manager.session_factory)
#         self.crud = CRUDOperations(Strategy, db_manager.session_factory)
        
#         # Initialize tools
#         self.police_tools = PoliceTools()
        
#         # Build LangGraph workflow
#         self.workflow = self._build_workflow()
        
#         self.logger.info(f"Police chatbot initialized: model={model_name}, conversation_id={self.conversation_id}")

#     def _initialize_llm(self, schema: Dict = None) -> ChatOllama:
#         """Initialize LLM with optional JSON schema."""
#         return ChatOllama(
#             model=self.model_name,
#             base_url=self.settings.agents.ollama_base_url,
#             format="json",
#             response_format={"type": "json_schema", "json_schema": schema or {"type": "object"}}
#         )

#     def _build_workflow(self) -> StateGraph:
#         """Build the LangGraph workflow for the multi-agent pipeline."""
#         workflow = StateGraph(PipelineState)
#         workflow.add_node("user_profile", self._user_profile_node)
#         workflow.add_node("retrieval", self._retrieval_node)
#         workflow.add_node("ie", self._ie_node)
#         workflow.add_node("knowledge_base", self._knowledge_base_node)
#         workflow.add_node("orchestrator", self._orchestrator_node)

#         workflow.add_edge("user_profile", "retrieval")
#         workflow.add_edge("retrieval", "ie")
#         workflow.add_edge("ie", "orchestrator")
#         workflow.add_conditional_edges(
#             "ie",
#             lambda state: "knowledge_base" if state.ie_output.targeted_slot and (len(state.ie_output.scam_specific_details) + len(state.ie_output.scam_generic_details)) < len(PoliceResponseSlots) else END,
#             {
#                 "knowledge_base": "knowledge_base",
#                 END: END
#             }
#         )

#         workflow.set_entry_point("user_profile")
#         return workflow.compile()

#     def _user_profile_node(self, state: PipelineState) -> PipelineState:
#         """Infer victim profile from query and history."""
#         self.logger.debug(f"UserProfileAgent processing query: {state.query}")
#         llm = self._initialize_llm(schema=UserProfile.model_json_schema())
#         prompt = ChatPromptTemplate.from_template(Prompt.template["user_profile"])
#         try:
#             response = llm.invoke(prompt.format_prompt(query=state.query, query_history=state.query_history).to_messages())
#             self.logger.debug(f"Raw LLM response: {response.content}")
#             state.user_profile = UserProfile(**json.loads(response.content))
#             self.logger.debug(f"UserProfileAgent output: {state.user_profile}")
#         except Exception as e:
#             self.logger.error(f"UserProfileAgent error: {str(e)}")
#             state.user_profile = UserProfile()
#         return state

#     def _retrieval_node(self, state: PipelineState) -> PipelineState:
#         """Retrieve scam reports and strategies using augmented police tool."""
#         self.logger.debug(f"RetrievalAgent processing query: {state.query}")
#         try:
#             tool = self.police_tools.get_augmented_tools()[0]  # augmented_police_tool
#             results = tool.invoke({
#                 "query": state.query,
#                 "top_k": 5,
#                 "conversation_id": self.conversation_id,
#                 "llm_model": self.model_name
#             })
#             result_dict = json.loads(results) if isinstance(results, str) else results
#             state.templates = RetrievalOutput(
#                 scam_report=[json.dumps(report) for report in result_dict.get("scam_reports", [])],
#                 strategy_type=[{"strategy_type": s["strategy_type"], "question": s["question"]} for s in result_dict.get("strategies", [])]
#             )
#             self.logger.debug(f"RetrievalAgent output: {state.templates}")
#         except Exception as e:
#             self.logger.error(f"RetrievalAgent error: {str(e)}")
#             state.templates = RetrievalOutput(
#                 scam_report=[],
#                 strategy_type=[
#                     {"strategy_type": "ask with simple terms", "question": "Can you describe what happened?"},
#                     {"strategy_type": "empathetic", "question": "I'm sorry to hear that. How did the scammer contact you?"}
#                 ]
#             )
#         return state

#     def _ie_node(self, state: PipelineState) -> PipelineState:
#         """Extract scam details, prioritize slots, and generate questions."""
#         self.logger.debug(f"IEAgent processing query: {state.query}")
#         llm = self._initialize_llm(schema=IEOutput.model_json_schema())
#         unfilled_slots = [
#             s for s in PoliceResponseSlots
#             if s.value not in state.ie_output.scam_specific_details
#             and s.value not in state.ie_output.scam_generic_details
#         ]
#         prompt = ChatPromptTemplate.from_template(Prompt.template["ie"])
#         try:
#             response = llm.invoke(prompt.format_prompt(
#                 query=state.query,
#                 victim_profile=state.user_profile.model_dump(),
#                 scam_reports=state.templates.scam_report,
#                 query_history=state.query_history[-2:],
#                 unfilled_slots=[s.value for s in unfilled_slots]
#             ).to_messages())
#             self.logger.debug(f"Raw LLM response: {response.content}")
#             output = json.loads(response.content)
#             new_generic = {**state.ie_output.scam_generic_details, **output.get("scam_generic_details", {})}
#             new_specific = {**state.ie_output.scam_specific_details, **output.get("scam_specific_details", {})}
#             state.ie_output = IEOutput(
#                 question=output["question"],
#                 scam_generic_details=new_generic,
#                 scam_specific_details=new_specific,
#                 targeted_slot=output["targeted_slot"],
#             )
#             state.query_history.append(f"Query: {state.query}; Response: {state.ie_output.question}")
#             self.logger.debug(f"IEAgent output: {state.ie_output}")
#         except Exception as e:
#             self.logger.error(f"IEAgent error: {str(e)}")
#             state.ie_output = IEOutput(
#                 question="Can you describe what happened?",
#                 targeted_slot=PoliceResponseSlots.scam_type,
#             )
#         return state

#     def _knowledge_base_node(self, state: PipelineState) -> PipelineState:
#         """Evaluate and store successful strategies."""
#         self.logger.debug(f"KnowledgeBaseAgent processing: {state.ie_output}")
#         llm = self._initialize_llm(schema=KnowledgeBaseOutput.model_json_schema())
#         prompt = ChatPromptTemplate.from_template(Prompt.template["knowledge_base"])
#         extracted_strategy = state.ie_output.strategy_type or "ask with simple terms"
        
#         try:
#             response = llm.invoke(prompt.format_prompt(
#                 ie_output=state.ie_output.model_dump(),
#                 query=state.query,
#                 query_history=state.query_history,
#                 victim_profile=state.user_profile.model_dump()
#             ).to_messages())
#             self.logger.debug(f"Raw LLM response: {response.content}")
#             kb_output = KnowledgeBaseOutput(**json.loads(response.content))
            
#             if kb_output.success_score >= 0.7 and kb_output.strategy_match and kb_output.valid:
#                 strategy_data = {
#                     "strategy_type": extracted_strategy,
#                     "success_score": kb_output.success_score,
#                     "scam_type": state.ie_output.scam_generic_details.get("scam_type", "GOVERNMENT OFFICIALS IMPERSONATION"),
#                     "victim_profile": state.user_profile.model_dump(),
#                     "target_slot": state.ie_output.targeted_slot,
#                     "question": state.ie_output.question
#                 }
#                 self.crud.create(strategy_data)
#                 self.logger.debug(f"Stored strategy: {extracted_strategy}")
#             state.ie_output.success_score = kb_output.success_score
#             self.logger.debug(f"KnowledgeBaseAgent output: {kb_output}")
#         except Exception as e:
#             self.logger.error(f"KnowledgeBaseAgent error: {str(e)}")
#         return state

#     def _orchestrator_node(self, state: PipelineState) -> PipelineState:
#         """Consolidate outputs into a processed response."""
#         self.logger.debug(f"OrchestratorAgent consolidating: {state.ie_output}")
#         llm = self._initialize_llm(schema=ProcessedOutput.model_json_schema())
#         prompt = ChatPromptTemplate.from_template(Prompt.template["orchestrator"])
#         try:
#             response = llm.invoke(prompt.format_prompt(
#                 query=state.query,
#                 query_history=state.query_history[-1:],
#                 victim_profile=state.user_profile.model_dump(),
#                 ie_output=state.ie_output.model_dump()
#             ).to_messages())
#             self.logger.debug(f"Raw LLM response: {response.content}")
#             state.output_json = ProcessedOutput(**json.loads(response.content))
#             self.logger.debug(f"OrchestratorAgent output: {state.output_json}")
#         except Exception as e:
#             self.logger.error(f"OrchestratorAgent error: {str(e)}")
#             state.output_json = ProcessedOutput(
#                 question=state.ie_output.question,
#                 user_profile=state.user_profile,
#                 scam_generic_details=state.ie_output.scam_generic_details,
#                 scam_specific_details=state.ie_output.scam_specific_details
#             )
#         return state

#     def process_query(self, query: str) -> Dict:
#         """Process a victim query to generate an augmented prompt."""
#         if not query.strip():
#             self.logger.error("Query cannot be empty")
#             return {"error": "Query cannot be empty", "conversation_id": self.conversation_id}
        
#         if "[END_CONVERSATION]" in query:
#             return self.end_conversation()

#         try:
#             state = PipelineState(
#                 query=query,
#                 query_history=[],
#                 user_profile=UserProfile(),
#                 templates=RetrievalOutput(),
#                 ie_output=IEOutput(targeted_slot=PoliceResponseSlots.scam_type, strategy_type="ask with simple terms"),
#                 output_json=ProcessedOutput(
#                     user_profile=UserProfile(),
#                     scam_generic_details={},
#                     scam_specific_details={}
#                 )
#             )
#             result = self.workflow.invoke(state)

#             # Map to PoliceResponse schema for response
#             police_response = {
#                 "conversational_response": result.ie_output.question,
#                 "scam_incident_date": result.ie_output.scam_generic_details.get("scam_incident_date", ""),
#                 "scam_type": result.ie_output.scam_generic_details.get("scam_type", ""),
#                 "scam_approach_platform": result.ie_output.scam_generic_details.get("scam_approach_platform", ""),
#                 "scam_communication_platform": result.ie_output.scam_generic_details.get("scam_communication_platform", ""),
#                 "scam_transaction_type": result.ie_output.scam_generic_details.get("scam_transaction_type", ""),
#                 "scam_beneficiary_platform": result.ie_output.scam_generic_details.get("scam_beneficiary_platform", ""),
#                 "scam_beneficiary_identifier": result.ie_output.scam_generic_details.get("scam_beneficiary_identifier", ""),
#                 "scam_contact_no": result.ie_output.scam_generic_details.get("scam_contact_no", ""),
#                 "scam_email": result.ie_output.scam_generic_details.get("scam_email", ""),
#                 "scam_moniker": result.ie_output.scam_generic_details.get("scam_moniker", ""),
#                 "scam_url_link": result.ie_output.scam_generic_details.get("scam_url_link", ""),
#                 "scam_amount_lost": float(result.ie_output.scam_generic_details.get("scam_amount_lost", 0.0)),
#                 "scam_incident_description": result.ie_output.scam_generic_details.get("scam_incident_description", ""),
#                 "scam_specific_details": result.ie_output.scam_specific_details,
#                 "rag_invoked": True
#             }

#             self.logger.debug(f"Processed query response: {police_response}")
#             return {
#                 "response": police_response["conversational_response"],
#                 "structured_data": police_response,
#                 "conversation_id": self.conversation_id,
#                 "output_json": result.output_json.model_dump()
#             }
#         except Exception as e:
#             self.logger.error(f"Error processing query: {str(e)}")
#             return {"error": f"Error processing query: {str(e)}", "conversation_id": self.conversation_id}

#     def end_conversation(self) -> Dict:
#         """End the conversation."""
#         self.logger.info(f"Conversation {self.conversation_id} ended")
#         return {"status": "Conversation ended", "conversation_id": self.conversation_id}

# if __name__ == "__main__":
#     self_augmenting_rag = SelfAugmentingRag(model_name="qwen2.5:7b")
#     queries = [
#         "I received a call from someone claiming to be a government official.",
#         "It was last week, and they asked for my bank details.",
#         "They said it was for a tax refund, and I lost $500.",
#         "I think that’s all I know. [END_CONVERSATION]"
#     ]
#     results = []
#     for query in queries:
#         result = self_augmenting_rag.process_query(query)
#         results.append(result)
#         if "status" in result and result["status"] == "Conversation ended":
#             break
    
#     print("Multi-Turn Test Results:")
#     for i, result in enumerate(results):
#         print(json.dumps({
#             "query": queries[i],
#             "conversation_id": result.get("conversation_id"),
#             "response": result.get("response", result.get("status", result.get("error"))),
#             "output_json": result.get("output_json", {}),
#             "state": {
#                 "query": result.get("structured_data", {}).get("query", ""),
#                 "victim_details": result.get("output_json", {}).get("user_profile", {}),
#                 "scam_generic_details": result.get("structured_data", {}).get("scam_generic_details", {}),
#                 "scam_specific_details": result.get("structured_data", {}).get("scam_specific_details", {}),
#                 "targeted_slot": result.get("structured_data", {}).get("targeted_slot", "")
#             }
#         }, indent=2, default=str))




# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# from typing import TypedDict, List, Dict, Optional, Type, Union, get_origin, get_args
# from langchain_core.messages import HumanMessage
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langgraph.graph import StateGraph, END, START
# from src.models.response_model import UserProfile, PoliceResponse, RetrievalOutput, PoliceResponseSlots, KnowledgeBaseOutput
# from src.agents.prompt import Prompt
# from src.agents.tools import PoliceTools
# from src.database.vector_operations import VectorStore
# from src.database.database_operations import DatabaseManager, CRUDOperations
# from src.models.data_model import Strategy
# from config.settings import get_settings
# from config.logging_config import setup_logger
# from config.id_manager import IDManager
# from src.agents.utils import save_conversation_history
# import json
# from datetime import datetime
# import csv
# from pydantic import BaseModel
# from copy import deepcopy

# # State
# class GraphState(TypedDict):
#     query: str
#     query_history: List[str]
#     user_profile: Optional[Dict[str, str]]
#     templates: Optional[Dict]
#     ie_output: Optional[Dict]
#     unfilled_slots: Dict[str, bool]
#     prev_unfilled_slots: Dict[str, bool]
#     conversation_id: int
#     rag_invoked: bool
#     target_slots: List[str]  # From IE agent: slots targeted by questions this turn
#     questions: List[str]  # Parsed questions from ie_output.conversational_response

# class PoliceChatbot:
#     def __init__(self, model_name: str = "qwen2.5:7b"):
#         self.settings = get_settings()
#         self.logger = setup_logger("Augmented_PoliceAgent", self.settings.log.subdirectories["agent"])
#         self.id_manager = IDManager()
#         self.model_name = model_name
#         self.conversation_id = self.id_manager.get_next_id()
#         self.conversation_history = []
        
#         # Tools and DB
#         self.police_tools = PoliceTools()
#         db_manager = DatabaseManager()
#         self.vector_store = VectorStore(db_manager.session_factory)
#         self.strategy_crud = CRUDOperations(Strategy, db_manager.session_factory)
        
#         # Prompts
#         self.user_profile_prompt = ChatPromptTemplate.from_template(Prompt.template["user_profile"])
#         self.ie_prompt = ChatPromptTemplate.from_template(Prompt.template["ie"])
#         self.kb_prompt = ChatPromptTemplate.from_template(Prompt.template["knowledge_base"])
        
#         # Graph
#         self.workflow = self._build_workflow()
#         self.logger.info(f"Police chatbot initialized with model: {model_name}, conversation_id: {self.conversation_id}")

#     def _get_llm(self, schema: Optional[Type[BaseModel]] = None) -> ChatOllama:
#         llm = ChatOllama(
#             model=self.model_name,
#             base_url=self.settings.agents.ollama_base_url,
#             format="json"
#         )
#         if schema:
#             llm = llm.with_structured_output(schema)
#         return llm

#     def _build_workflow(self):
#         workflow = StateGraph(GraphState)
#         workflow.add_node("user_profile", self.user_profile_agent)
#         workflow.add_node("retrieval", self.retrieval_agent)
#         workflow.add_node("ie", self.ie_agent)
#         workflow.add_node("slot_tracker", self.slot_tracker)
#         workflow.add_node("knowledge_base", self.knowledge_base_agent)
        
#         workflow.add_edge(START, "user_profile")
#         workflow.add_edge("user_profile", "retrieval")
#         workflow.add_edge("retrieval", "ie")
#         workflow.add_edge("ie", "slot_tracker")
#         # Parallel: Fan-out from slot_tracker to KB and END
#         workflow.add_edge("slot_tracker", "knowledge_base")
#         workflow.add_edge("slot_tracker", END)
        
#         return workflow.compile()

#     def user_profile_agent(self, state: GraphState) -> GraphState:
#         up_llm = self._get_llm(schema=UserProfile)
#         prompt = self.user_profile_prompt.format(query=state["query"], query_history=state["query_history"])
#         try:
#             profile = up_llm.invoke(prompt).model_dump()
#             self.logger.debug(f"UserProfileAgent output: {profile}")
#         except Exception as e:
#             self.logger.error(f"UserProfileAgent error: {str(e)}")
#             profile = UserProfile().model_dump()
#         return {"user_profile": profile}

#     def retrieval_agent(self, state: GraphState) -> GraphState:
#         try:
#             tool = self.police_tools.get_augmented_tools()[0]
#             results = tool.invoke({
#                 "query": state["query"],
#                 "top_k": 1,
#                 "conversation_id": self.conversation_id,
#                 "llm_model": self.model_name
#             })
#             result_dict = json.loads(results) if isinstance(results, str) else results
#             templates = RetrievalOutput(
#                 scam_reports=[json.dumps(report) for report in result_dict.get("scam_reports", [])],
#                 strategies=[{"strategy_type": s["strategy_type"], "question": s["question"]} for s in result_dict.get("strategies", [])]
#             ).model_dump()  # Fix deprecation
#             self.logger.debug(f"RetrievalAgent output: {templates}")
#             return {"templates": templates, "rag_invoked": True}
#         except Exception as e:
#             self.logger.error(f"RetrievalAgent error: {str(e)}")
#             return {
#                 "templates": RetrievalOutput(
#                     scam_report=[],
#                     strategy_type=[
#                         {"strategy_type": "ask with simple terms", "question": "Can you describe what happened?"},
#                         {"strategy_type": "empathetic", "question": "I'm sorry to hear that. How did the scammer contact you?"}
#                     ]
#                 ).model_dump(),
#                 "rag_invoked": False
#             }

#     def ie_agent(self, state: GraphState) -> GraphState:
#         ie_llm = self._get_llm(schema=PoliceResponse)
#         prompt = self.ie_prompt.format(
#             query=state["query"],
#             query_history=state["query_history"],
#             user_profile=state["user_profile"],
#             scam_reports=state["templates"].get("scam_report", []),  # Singular
#             strategies=state["templates"].get("strategy_type", [])  # Singular
#         )
#         try:
#             ie_output = ie_llm.invoke(prompt).model_dump()
#             self.logger.debug(f"IEAgent output: {ie_output}")
#             return {"ie_output": ie_output}
#         except Exception as e:
#             self.logger.error(f"IEAgent error: {str(e)}")
#             return {
#                 "ie_output": PoliceResponse(
#                     conversational_response="I'm sorry, I need more information. Can you provide details about the scam?",
#                     scam_incident_description=state["query"]
#                 ).model_dump()
#             }



#     def slot_tracker(self, state: GraphState) -> GraphState:
#         ie_output = PoliceResponse(**state["ie_output"])
#         prev = state.get("prev_unfilled_slots", {slot.value: True for slot in PoliceResponseSlots})
#         unfilled = {}
#         for field_name, field in PoliceResponse.model_fields.items():
#             if field_name == "conversational_response":
#                 continue
#             value = getattr(ie_output, field_name)
#             annotation = field.annotation
#             # Determine default based on annotation
#             if annotation == str:
#                 default = ""
#             elif annotation == float:
#                 default = 0.0
#             elif get_origin(annotation) == dict:  # Handles Dict[str, str]
#                 default = {}
#             else:
#                 # Handle Optional or Union types (e.g., Optional[str])
#                 origin = get_origin(annotation)
#                 if origin in (Union, Optional):
#                     args = get_args(annotation)
#                     # If Optional[str] or Union[str, None], default is ""
#                     if str in args:
#                         default = ""
#                     else:
#                         default = None
#                 else:
#                     default = None
#             is_unfilled = value == default
#             unfilled[field_name] = is_unfilled  # Assign to unfilled dictionary
#         self.logger.debug(f"SlotTracker: unfilled_slots={unfilled}")
#         return {"unfilled_slots": unfilled, "prev_unfilled_slots": prev}

#     def knowledge_base_agent(self, state: GraphState) -> GraphState:
#         self.logger.debug(f"KnowledgeBaseAgent processing: {state['ie_output']}")
#         # kb_llm = self._get_llm(schema=List[KnowledgeBaseOutput])  
#         kb_llm = self._get_llm()
        
#         filled_slots_this_turn = [k for k, v in state["prev_unfilled_slots"].items() if v and not state["unfilled_slots"].get(k, False)]
        
#         prompt = self.kb_prompt.format(
#             ie_output=state["ie_output"],
#             query=state["query"],
#             query_history=state["query_history"],
#             user_profile=state["user_profile"],
#             filled_slots_this_turn=filled_slots_this_turn
#         )
#         try:
#             response = kb_llm.invoke(prompt)  # KBOutputs instance
#             self.logger.debug(f"Raw LLM response content: {response.content}")
            
#             if not filled_slots_this_turn:
#                 filled_slots_this_turn = ["unknown"]
#             # Parse content as JSON (handle str)
#             if isinstance(response.content, str):
#                 parsed = json.loads(response.content)
#             else:
#                 parsed = response.content
            
#             # Ensure list (if single, wrap)
#             if not isinstance(parsed, list):
#                 parsed = [parsed]
            
#             # Validate with Pydantic
#             kb_outputs = []
#             for item in parsed:
#                 try:
#                     kb_outputs.append(KnowledgeBaseOutput(**item))
#                 except Exception as val_e:
#                     self.logger.warning(f"Validation error for output item: {str(val_e)} - Item: {item}")
            
#             if not kb_outputs:
#                 self.logger.warning("No valid KB outputs after parsing.")
#                 return {}

            
#             questions = [q.strip() for q in state["ie_output"]["conversational_response"].split('?') if q.strip()] or ["unknown"]
            
#             # Zip with repetition if lengths differ
#             max_len = max(len(kb_outputs), len(questions), len(filled_slots_this_turn))
#             questions = questions * (max_len // len(questions) + 1)[:max_len]
#             filled_slots_this_turn = filled_slots_this_turn * (max_len // len(filled_slots_this_turn) + 1)[:max_len]
            
#             for kb_output, question, target_slot in zip(kb_outputs, questions, filled_slots_this_turn):
#                 if kb_output.success_score >= 0.7 and kb_output.strategy_match and kb_output.valid:
#                     new_strategy = {
#                         "strategy_type": kb_output.strategy_type or "unknown",
#                         "question": question,
#                         "success_score": kb_output.success_score,
#                         "scam_type": state["ie_output"].get("scam_type", ""),
#                         "victim_profile": json.dumps(state["user_profile"]),
#                         "target_slot": target_slot,
#                         "timestamp": datetime.now(),
#                         "embedding": self.vector_store.get_embedding(question)
#                     }
#                     self.strategy_crud.create(new_strategy)
#                     self.logger.info(f"Inserted strategy: {new_strategy['strategy_type']} for question: {question}")
#                     self._log_augmentation(kb_output.model_dump(), state)
#             self.logger.debug(f"KnowledgeBaseAgent outputs: {kb_outputs}")
#         except json.JSONDecodeError as json_e:
#             self.logger.error(f"JSON parse error in KnowledgeBaseAgent: {str(json_e)} - Raw content: {response.content if response else 'None'}")
#         except Exception as e:
#             self.logger.error(f"KnowledgeBaseAgent error: {str(e)}")
#         return {}

#     def _log_augmentation(self, kb_output: Dict, state: GraphState):
#         csv_file = "augmentation_log.csv"
#         file_exists = os.path.isfile(csv_file)
#         with open(csv_file, "a", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=["conversation_id", "timestamp", "user_profile_check", "strategy_match", "success_score", "inserted"])
#             if not file_exists:
#                 writer.writeheader()
#             writer.writerow({
#                 "conversation_id": state["conversation_id"],
#                 "timestamp": datetime.now().isoformat(),
#                 "user_profile_check": all(state["user_profile"].get(k) in [e.value for e in v] for k, v in [
#                     ("age_group", AgeGroup), ("tech_literacy", TechLiteracy),
#                     ("language_proficiency", LanguageProficiency), ("emotional_state", EmotionalState)
#                 ]),
#                 "strategy_match": kb_output["strategy_match"],
#                 "success_score": kb_output["success_score"],
#                 "inserted": kb_output["success_score"] >= 0.7 and kb_output["strategy_match"] and kb_output["valid"]
#             })

#     def process_query(self, query: str, query_history: List[str] = None):
#         state = {
#             "query": query,
#             "query_history": query_history or [],
#             "user_profile": None,
#             "templates": None,
#             "ie_output": None,
#             "unfilled_slots": {},
#             "prev_unfilled_slots": {},
#             "conversation_id": self.conversation_id,
#             "rag_invoked": False
#         }
        
#         final_state = self.workflow.invoke(state)
        
#         ie_out = PoliceResponse(**final_state["ie_output"])
#         response = ie_out.conversational_response
        
#         self.conversation_history.append({"role": "victim", "content": query, "timestamp": datetime.now().isoformat()})
#         self.conversation_history.append({
#             "role": "police",
#             "content": response,
#             "timestamp": datetime.now().isoformat(),
#             "structured_data": {**ie_out.model_dump(), "user_profile": final_state["user_profile"], "templates": final_state["templates"]},
#             "rag_invoked": final_state["rag_invoked"]
#         })
        
#         save_conversation_history(self.conversation_id, self.conversation_history, model_name=self.model_name, logger=self.logger)
        
#         return {"response": response, "structured_data": ie_out.model_dump(), "conversation_id": self.conversation_id}

# if __name__ == "__main__":
#     models = ["qwen2.5:7b", "granite3.2:8b", "mistral:7b"]
#     query = "I received a call from someone claiming to be a government official."
#     results = {}
    
#     for model_name in models:
#         try:
#             chatbot = PoliceChatbot(model_name=model_name)
#             response = chatbot.process_query(query, query_history=["Initial context: User mentioned a suspicious call."])
#             results[model_name] = response
#             print(f"Model {model_name}: {json.dumps(response, indent=2)}")
#         except Exception as e:
#             results[model_name] = {"error": str(e)}
#             print(f"Error with {model_name}: {str(e)}")
    
#     print(json.dumps(results, indent=2))



# src/agents/self_augmenting_police_agent.py
# Major amendments:
# - Used LangGraph for workflow.
# - Added state with prev_turn_data dict for preservation.
# - Pre-hook: Shift current to prev at start of each turn (except first).
# - UserProfileAgent: Infers profile.
# - RetrievalAgent: Uses augmented tool, limits to 1 scam/strategy, fallback to default.
# - IEAgent: Adapts questions based on profile/strategies, updates ie_output, sets target_slots.
# - SlotTracker: Programmatic, checks unfilled based on criteria, shifts to prev.
# - KnowledgeBaseAgent: Runs every turn, evaluates prev_turn_data, computes score with quick_bonus (if turn_count <3), upserts if >=0.7, hierarchical similarity check, prunes every 10 upserts, novelty check via LLM.
# - Added metrics tracking.
# - Process_query handles multi-turn state persistence (but since stateless per call, user must pass history).
# - Added rag_upserts.csv logging for KB upserts.

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# from typing import TypedDict, List, Dict, Optional, Type, Union, get_origin, get_args
# from langchain_core.messages import HumanMessage
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate
# from langgraph.graph import StateGraph, END, START
# from src.models.response_model import UserProfile, PoliceResponse, RetrievalOutput, PoliceResponseSlots, KnowledgeBaseOutput
# from src.agents.prompt import Prompt
# from src.agents.tools import PoliceTools
# from src.database.vector_operations import VectorStore
# from src.database.database_operations import DatabaseManager, CRUDOperations
# from src.models.data_model import Strategy
# from config.settings import get_settings
# from config.logging_config import setup_logger
# from config.id_manager import IDManager
# from src.agents.utils import save_conversation_history
# import json
# from datetime import datetime
# import csv
# from pydantic import BaseModel
# from copy import deepcopy
# import numpy as np

# class KBList(BaseModel):
#     items: List[KnowledgeBaseOutput]
    
# # State
# class GraphState(TypedDict):
#     query: str
#     query_history: List[str]
#     user_profile: Optional[UserProfile]
#     templates: Optional[RetrievalOutput]
#     ie_output: Optional[PoliceResponse]
#     target_slots: List[str]
#     prev_turn_data: Dict[str, Union[List[str], Dict[str, bool]]]  # {'target_slots': [], 'unfilled_slots': {}, 'questions_asked': []}
#     unfilled_slots: Dict[str, bool]
#     conversation_id: str
#     metrics: Dict[str, Union[bool, int]]  # {'rag_retrieved': bool, 'rag_upserted': bool, 'turn_count': int}

# class PoliceChatbot:
#     def __init__(self, model_name: str = "qwen2.5:7b"):
#         self.settings = get_settings()
#         self.logger = setup_logger("Augmented_PoliceAgent", self.settings.log.subdirectories["agent"])
#         self.id_manager = IDManager()
#         self.model_name = model_name
#         self.conversation_id = str(self.id_manager.get_next_id())  # Use str for consistency
#         self.conversation_history = []
        
#         # Tools and DB
#         self.police_tools = PoliceTools()
#         db_manager = DatabaseManager()
#         self.vector_store = VectorStore(db_manager.session_factory)
#         self.strategy_crud = CRUDOperations(Strategy, db_manager.session_factory)
        
#         # Prompts
#         self.user_profile_prompt = ChatPromptTemplate.from_template(Prompt.template["user_profile"])
#         self.ie_prompt = ChatPromptTemplate.from_template(Prompt.template["ie"])
#         self.kb_prompt = ChatPromptTemplate.from_template(Prompt.template["knowledge_base"])
        
#         # Graph
#         self.workflow = self._build_workflow()
#         self.logger.info(f"Police chatbot initialized with model: {model_name}, conversation_id: {self.conversation_id}")
        
#         #trigger pruning
#         self.upsert_count = 0

#     def _get_llm(self, schema: Optional[Type[BaseModel]] = None) -> ChatOllama:
#         llm = ChatOllama(
#             model=self.model_name,
#             base_url=self.settings.agents.ollama_base_url,
#             format="json"
#         )
#         if schema:
#             llm = llm.with_structured_output(schema)
#         return llm

#     def _build_workflow(self):
#         workflow = StateGraph(GraphState)
#         workflow.add_node("shift_prev", self.shift_prev_hook)
#         workflow.add_node("user_profile", self.user_profile_agent)
#         workflow.add_node("retrieval", self.retrieval_agent)
#         workflow.add_node("ie", self.ie_agent)
#         workflow.add_node("slot_tracker", self.slot_tracker)
#         workflow.add_node("knowledge_base", self.knowledge_base_agent)
        
#         workflow.add_edge(START, "shift_prev")
#         workflow.add_edge("shift_prev", "user_profile")
#         workflow.add_edge("user_profile", "retrieval")
#         workflow.add_edge("retrieval", "ie")
#         workflow.add_edge("ie", "slot_tracker")
#         workflow.add_edge("slot_tracker", "knowledge_base")
#         workflow.add_edge("knowledge_base", END)
        
#         return workflow.compile()

#     def shift_prev_hook(self, state: GraphState) -> GraphState:
#         """Shift current data to prev_turn_data at start of turn."""
#         if state.get("metrics", {}).get("turn_count", 0) > 0:  # Skip on first turn
#             ie_output = state.get("ie_output")
#             if ie_output:
#                 if isinstance(ie_output, dict):
#                     conv_response = ie_output.get("conversational_response", "")
#                 else:
#                     conv_response = ie_output.conversational_response if hasattr(ie_output, 'conversational_response') else ""
#             else:
#                 conv_response = ""
            
#             state["prev_turn_data"] = {
#                 "target_slots": deepcopy(state.get("target_slots", [])),
#                 "unfilled_slots": deepcopy(state.get("unfilled_slots", {})),
#                 "questions_asked": self._extract_questions(conv_response)
#             }
#         state["metrics"]["turn_count"] = state["metrics"].get("turn_count", 0) + 1
#         return state

#     def _extract_questions(self, response: str) -> List[str]:
#         """Programmatically extract questions from conversational_response."""
#         return [q.strip() + "?" for q in response.split('?') if q.strip()]  # Simple split and re-add '?'

#     def user_profile_agent(self, state: GraphState) -> GraphState:
#         up_llm = self._get_llm(schema=UserProfile)
#         prompt = self.user_profile_prompt.format(query=state["query"], query_history=state["query_history"])
#         try:
#             profile = up_llm.invoke(prompt)
#             state["user_profile"] = profile
#             self.logger.debug(f"UserProfileAgent output: {profile.model_dump()}")
#         except Exception as e:
#             self.logger.error(f"UserProfileAgent error: {str(e)}")
#             state["user_profile"] = UserProfile()
#         return state

#     def retrieval_agent(self, state: GraphState) -> GraphState:
#         try:
#             tool = self.police_tools.get_augmented_tools()[0]
#             results = tool.invoke({
#                 "query": state["query"],
#                 "user_profile": state["user_profile"].model_dump(),
#                 "top_k": 1,  # Limit to 1 as per design
#                 "conversation_id": int(state["conversation_id"]),
#                 "llm_model": self.model_name
#             })
#             result_dict = json.loads(results) if isinstance(results, str) else results
#             templates = RetrievalOutput(
#                 scam_reports=result_dict.get("scam_reports", [])[:1],
#                 strategies=result_dict.get("strategies", [])[:1]
#             )
#             state["templates"] = templates
#             state["metrics"]["rag_retrieved"] = True
#             self.logger.debug(f"RetrievalAgent output: {templates.dict()}")
#             return state
#         except Exception as e:
#             self.logger.error(f"RetrievalAgent error: {str(e)}")
#             state["templates"] = RetrievalOutput(
#                 scam_reports=[],
#                 strategies=[{"strategy_type": "neutral", "question": "Can you tell me more?"}]
#             )
#             state["metrics"]["rag_retrieved"] = False
#             return state

#     def ie_agent(self, state: GraphState) -> GraphState:
#         ie_llm = self._get_llm(schema=PoliceResponse)
#         unfilled = [k for k, v in state.get("unfilled_slots", {}).items() if v]
#         prompt = self.ie_prompt.format(
#             query=state["query"],
#             query_history=state["query_history"],
#             user_profile=state["user_profile"].model_dump(),
#             scam_reports=state["templates"].scam_reports,
#             strategies=state["templates"].strategies
#         )
#         try:
#             ie_output = ie_llm.invoke(prompt)
#             state["ie_output"] = ie_output
#             state["target_slots"] = ie_output.target_slots  # From structured output
#             self.logger.debug(f"IEAgent output: {ie_output.model_dump()}")
#             return state
#         except Exception as e:
#             self.logger.error(f"IEAgent error: {str(e)}")
#             default_output = PoliceResponse(
#                 conversational_response="I'm sorry, I need more information. Can you provide details about the scam?",
#                 scam_incident_description=state["query"],
#                 target_slots=[]
#             )
#             state["ie_output"] = default_output
#             state["target_slots"] = []
#             return state

#     def slot_tracker(self, state: GraphState) -> GraphState:
#         ie_output = state["ie_output"]
#         state["prev_turn_data"]["unfilled_slots"] = state.get("unfilled_slots", {})  # Shift current to prev
#         unfilled = {}
#         for field_name in PoliceResponseSlots:
#             value = getattr(ie_output, field_name.value)
#             is_unfilled = (
#                 value in ("", 0.0, {}, None, "unknown", "na") or
#                 (isinstance(value, dict) and not value) or
#                 (isinstance(value, list) and not value)
#             )
#             unfilled[field_name.value] = is_unfilled
#         state["unfilled_slots"] = unfilled
#         self.logger.debug(f"SlotTracker: unfilled_slots={unfilled}")
#         return state

#     def knowledge_base_agent(self, state: GraphState) -> GraphState:
#         kb_llm = self._get_llm(schema=KBList)
#         filled_this_turn = [k for k, v in state["prev_turn_data"]["unfilled_slots"].items() if v and not state["unfilled_slots"].get(k, False)]
#         prev_questions = state["prev_turn_data"].get("questions_asked", [])
#         prev_targets = state["prev_turn_data"].get("target_slots", [])
        
#         if not filled_this_turn or not prev_targets or not prev_questions:
#                 self.logger.debug("Skipping KB upsert: No filled slots or no prior questions/targets.")
#                 return state  # Skip if nothing to evaluate
            
#         prompt = self.kb_prompt.format(
#             prev_turn_data=state["prev_turn_data"],  # Already dict, no change
#             filled_slots_this_turn=filled_this_turn,
#             query=state["query"],
#             query_history=state["query_history"],
#             user_profile=state["user_profile"].model_dump()
#         )
#         for attempt in range(2):  # Retry once on failure
#             try:
#                 response = kb_llm.invoke(prompt)
#                 kb_outputs = response.items if hasattr(response, 'items') else []
#                 if not isinstance(kb_outputs, list):
#                     kb_outputs = [kb_outputs]
#                 state["kb_outputs"] = kb_outputs

#                 for i, kb_output in enumerate(kb_outputs):
#                     # Per-strategy score 
#                     num_targets = len(prev_targets)
#                     score = 0.6 * (1 / num_targets) + \
#                             0.3 * kb_output.strategy_match + \
#                             0.1 * (1 if state["metrics"]["turn_count"] < 3 else 0)
#                     if score >= 0.7:
#                         # Use filled_this_turn[i] for target_slot if needed
#                         similar_df = self.vector_store.hierarchical_strategy_search(state["user_profile"].model_dump(), limit=5)
#                         to_upsert = True
#                         for _, row in similar_df.iterrows():
#                             if row['profile_distance'] < 0.25 and np.linalg.norm(np.array(self.vector_store.embedder.encode(row['question'])) - np.array(self.vector_store.embedder.encode(prev_questions[0]))) < 0.2:
#                                 if score > row['success_score']:
#                                     self.strategy_crud.update(row['id'], {'success_score': score})
#                                     to_upsert = False
#                                 else:
#                                     to_upsert = False
#                                 break
#                         if to_upsert:
#                             new_strategy = Strategy(
#                                 strategy_type=kb_output.strategy_type,
#                                 question=prev_questions[0],  # Adjust if multi questions
#                                 success_score=score,
#                                 scam_type=state["ie_output"].scam_type,
#                                 user_profile=state["user_profile"].model_dump(),
#                                 target_slot=state["prev_turn_data"]["target_slots"][i] if i < len(state["prev_turn_data"]["target_slots"]) else state["prev_turn_data"]["target_slots"][0],
#                                 profile_embedding=self.vector_store.embedder.encode(json.dumps(state["user_profile"].model_dump())),
#                                 strategy_embedding=self.vector_store.embedder.encode(kb_output.strategy_type),
#                                 question_embedding=self.vector_store.embedder.encode(prev_questions[0])
#                             )
#                             self.strategy_crud.create(new_strategy.model_dump(exclude={'id'}))
#                             self.upsert_count += 1
#                             state["metrics"]["rag_upserted"] = True
#                             if self.upsert_count % 20 == 0:
#                                 self.vector_store.prune_strategies()
#                     self._log_upsert(state["conversation_id"], score, kb_output.model_dump())
#             except Exception as e:
#                 self.logger.error(f"KnowledgeBaseAgent error: {str(e)}")
#             return state

#     def _log_upsert(self, conv_id: str, score: float, kb_output: Dict):
#         # Log to rag_upserts.csv - implement similar to save_conversation_history
#         pass  # Placeholder; add CSV logging with params for variants later

#     def process_query(self, query: str, query_history: Optional[List[str]] = None, prev_state: Optional[GraphState] = None):
#         state = prev_state or {
#             "query": query,
#             "query_history": query_history or [],
#             "user_profile": None,
#             "templates": None,
#             "ie_output": None,
#             "target_slots": [],
#             "prev_turn_data": {"target_slots": [], "unfilled_slots": {}, "questions_asked": []},
#             "unfilled_slots": {s.value: True for s in PoliceResponseSlots},
#             "conversation_id": self.conversation_id,
#             "metrics": {"rag_retrieved": False, "rag_upserted": False, "turn_count": 0}
#         }
#         state["query"] = query
#         state["query_history"].append(query)
        
#         final_state = self.workflow.invoke(state)
        
#         ie_out = final_state["ie_output"]
#         response = ie_out.conversational_response
        
#         self.conversation_history.append({"role": "victim", "content": query, "timestamp": datetime.now().isoformat()})
#         self.conversation_history.append({
#             "role": "police",
#             "content": response,
#             "timestamp": datetime.now().isoformat(),
#             "structured_data": ie_out.model_dump(),
#             "rag_invoked": final_state["metrics"]["rag_retrieved"]
#         })
        
#         save_conversation_history(self.conversation_id, self.conversation_history, model_name=self.model_name, logger=self.logger)
#         final_state["user_profile"] = final_state.get("user_profile").model_dump() if final_state.get("user_profile") else None
#         final_state["templates"] = final_state.get("templates").model_dump() if final_state.get("templates") else None
#         final_state["ie_output"] = final_state.get("ie_output").model_dump() if final_state.get("ie_output") else None
#         final_state["kb_outputs"] = [kb.model_dump() for kb in final_state.get("kb_outputs", [])] if final_state.get("kb_outputs") else []
#         return {"response": response, "structured_data": ie_out.model_dump(), "conversation_id": self.conversation_id, "state": final_state}

# # __main__ test: Adapt to multi-turn by passing prev_state.
# if __name__ == "__main__":
#     # models = ["qwen2.5:7b", "granite3.2:8b", "mistral:7b"]
#     models = ["qwen2.5:7b"]
#     query = "I received a call from someone claiming to be a government official."
#     results = {}
    
#     #single turn test for multiple models
#     for model_name in models:
#         try:
#             chatbot = PoliceChatbot(model_name=model_name)
#             response = chatbot.process_query(query, query_history=["Initial context: User mentioned a suspicious call."])
#             results[model_name] = response
#             print(f"Model {model_name}: {json.dumps(response, indent=2)}")
#         except Exception as e:
#             results[model_name] = {"error": str(e)}
#             print(f"Error with {model_name}: {str(e)}")
    
#     print(json.dumps(results, indent=2))
    
    
#     #Multi-turn test with one model
#     print("\n=== Multi-turn Conversation Test ===")
#     for model_name in models:
#         chatbot = PoliceChatbot(model_name=model_name)

#         # Simulate a multi-turn conversation
#         queries = [
#             "I received a call from someone claiming to be a government official.",
#             "It happened on 2025-02-22 and I lost 500 dollars.",
#             "The scammer's phone number was +123456789 and they asked for my bank details.",
#             "I think that's all the details I have."
#         ]

#         prev_state = None
#         conversation_responses = []

#         for i, q in enumerate(queries, 1):
#             print(f"\nTurn {i}: User Query: {q}")
#             response_data = chatbot.process_query(q, prev_state=prev_state)
#             conversation_responses.append(response_data["response"])
#             prev_state = response_data["state"]  # Pass the state for the next turn
#             print(f"Police Response: {response_data['response']}")

#         print("\nFull Conversation Responses:")
#         print(json.dumps(conversation_responses, indent=2))
        
        
        
        
        
        
        
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# from typing import TypedDict, List, Dict, Optional, Type, Union, get_origin, get_args
# from langchain_core.messages import HumanMessage
# from langchain_ollama import ChatOllama
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langgraph.graph import StateGraph, END, START
# from src.models.response_model import UserProfile, PoliceResponse, RetrievalOutput, PoliceResponseSlots, KnowledgeBaseOutput, QuestionSlotMap, ValidatedQuestionSlotMap
# from src.agents.prompt import Prompt
# from src.agents.tools import PoliceTools
# from src.database.vector_operations import VectorStore
# from src.database.database_operations import DatabaseManager, CRUDOperations
# from src.models.data_model import Strategy
# from config.settings import get_settings
# from config.logging_config import setup_logger
# from config.id_manager import IDManager
# from src.agents.utils import format_history_as_messages, build_query_with_history
# import json
# from datetime import datetime
# import csv
# from pydantic import BaseModel
# from copy import deepcopy
# import numpy as np
# from pydantic import ValidationError
# import re

# class KBList(BaseModel):
#     items: List[KnowledgeBaseOutput]
    
# # State
# class GraphState(TypedDict):
#     query: str
#     query_history: List[str]
#     past_responses: List[str]  # Track AI outputs
#     user_profile: Optional[UserProfile]
#     templates: Optional[RetrievalOutput]
#     ie_output: Optional[PoliceResponse]
#     target_slots: List[str]
#     target_slot_map: Dict[str, List[str]]
#     filled_this_turn: List[str]
#     prev_turn_data: Dict[str, Union[List[str], Dict[str, bool]]]  # {'target_slots': [], 'unfilled_slots': {}, 'questions_asked': []}
#     unfilled_slots: Dict[str, bool]
#     conversation_id: str
#     metrics: Dict[str, Union[bool, int]]  # {'rag_retrieved': bool, 'rag_upserted': bool, 'turn_count': int}

# class PoliceChatbot:
#     def __init__(self, model_name: str = "qwen2.5:7b"):
#         self.settings = get_settings()
#         self.logger = setup_logger("Augmented_PoliceAgent", self.settings.log.subdirectories["agent"])
#         self.id_manager = IDManager()
#         self.model_name = model_name
#         self.conversation_id = str(self.id_manager.get_next_id())  # Use str for consistency
#         self.conversation_history = []
        
#         # Tools and DB
#         self.police_tools = PoliceTools()
#         db_manager = DatabaseManager()
#         self.vector_store = VectorStore(db_manager.session_factory)
#         self.strategy_crud = CRUDOperations(Strategy, db_manager.session_factory)
        
#         # Prompts
#         self.user_profile_prompt = ChatPromptTemplate.from_template(Prompt.template["user_profile"])
#         self.ie_prompt = ChatPromptTemplate.from_messages([
#             ("system", Prompt.template["ie"]),  # Your custom IE prompt as system (with {user_profile}, {strategies}, etc.)
#             MessagesPlaceholder(variable_name="messages"),  # For history
#             ("human", "{query}"),  # Current user input
#         ])
#         self.target_slot_prompt = ChatPromptTemplate.from_template(Prompt.template["target_slot_agent"])
#         self.kb_prompt = ChatPromptTemplate.from_template(Prompt.template["knowledge_base"])

#         # Graph
#         self.workflow = self._build_workflow()
#         self.logger.info(f"Police chatbot initialized with model: {model_name}, conversation_id: {self.conversation_id}")
        
#         #trigger pruning
#         self.upsert_count = 0

#     def _get_llm(self, schema: Optional[Type[BaseModel]] = None) -> ChatOllama:
#         llm = ChatOllama(
#             model=self.model_name,
#             base_url=self.settings.agents.ollama_base_url,
#             format="json"
#         )
#         if schema:
#             llm = llm.with_structured_output(schema)
#         return llm

#     def _build_workflow(self):
#         workflow = StateGraph(GraphState)
#         workflow.add_node("shift_prev", self.shift_prev_hook)
#         workflow.add_node("user_profile", self.user_profile_agent)
#         workflow.add_node("retrieval", self.retrieval_agent)
#         workflow.add_node("ie", self.ie_agent)
#         workflow.add_node("slot_tracker", self.slot_tracker)
#         workflow.add_node("knowledge_base", self.knowledge_base_agent)
        
#         workflow.add_edge(START, "shift_prev")
#         workflow.add_edge("shift_prev", "user_profile")
#         workflow.add_edge("user_profile", "retrieval")
#         workflow.add_edge("retrieval", "ie")
#         workflow.add_edge("ie", "slot_tracker")
#         workflow.add_edge("slot_tracker", "knowledge_base")
#         workflow.add_edge("knowledge_base", END)
        
#         return workflow.compile()

#     def shift_prev_hook(self, state: GraphState) -> GraphState:
#         """Shift current data to prev_turn_data at start of turn."""
#         if state.get("metrics", {}).get("turn_count", 0) > 0:  # Skip on first turn
#             ie_output = state.get("ie_output")
#             if ie_output:
#                 if isinstance(ie_output, dict):
#                     conv_response = ie_output.get("conversational_response", "")
#                 else:
#                     conv_response = ie_output.conversational_response if hasattr(ie_output, 'conversational_response') else ""
#             else:
#                 conv_response = ""
            
#             state["prev_turn_data"] = {
#                 "target_slots": deepcopy(state.get("target_slots", [])),
#                 "unfilled_slots": deepcopy(state.get("unfilled_slots", {})),
#                 "questions_asked": self._extract_questions(conv_response)
#             }
#         state["metrics"]["turn_count"] = state["metrics"].get("turn_count", 0) + 1
#         return state

#     def _extract_questions(self, response: str) -> List[str]:
#         """Programmatically extract questions from conversational_response."""
#         return [q.strip() + "?" for q in response.split('?') if q.strip()]  # Simple split and re-add '?'

#     def user_profile_agent(self, state: GraphState) -> GraphState:
#         up_llm = self._get_llm(schema=UserProfile)
#         prompt = self.user_profile_prompt.format(query=state["query"], query_history=state["query_history"])
#         try:
#             profile = up_llm.invoke(prompt)
#             state["user_profile"] = profile
#             self.logger.debug(f"UserProfileAgent output: {profile.model_dump()}")
#         except Exception as e:
#             self.logger.error(f"UserProfileAgent error: {str(e)}")
#             state["user_profile"] = UserProfile()
#         return state

#     def retrieval_agent(self, state: GraphState) -> GraphState:
#         try:
#             tool = self.police_tools.get_augmented_tools()[0]
#             user_query = build_query_with_history(state["query"], state["query_history"], max_history=None, max_tokens=2000)
#             self.logger.debug(f"Built RAG query: {user_query[:100]}...")
#             results = tool.invoke({
#                 "query": user_query,
#                 "user_profile": state["user_profile"].model_dump(),
#                 "top_k": 1,  # Limit to 1 as per design
#                 "conversation_id": int(state["conversation_id"]),
#                 "llm_model": self.model_name
#             })
#             result_dict = json.loads(results) if isinstance(results, str) else results
#             templates = RetrievalOutput(
#                 scam_reports=result_dict.get("scam_reports", [])[:1],
#                 strategies=result_dict.get("strategies", [])[:1]
#             )
#             state["templates"] = templates
#             state["metrics"]["rag_retrieved"] = True
#             self.logger.debug(f"RetrievalAgent output: {templates.model_dump()}")
#             return state
#         except Exception as e:
#             self.logger.error(f"RetrievalAgent error: {str(e)}")
#             state["templates"] = RetrievalOutput(
#                 scam_reports=[],
#                 strategies=[{"strategy_type": "neutral", "question": "Can you tell me more?"}]
#             )
#             state["metrics"]["rag_retrieved"] = False
#             return state

#     def ie_agent(self, state: GraphState) -> GraphState:
#         ie_llm = self._get_llm(schema=PoliceResponse)
#         unfilled = [k for k, v in state.get("unfilled_slots", {}).items() if v]
        
#         history_messages = format_history_as_messages(state["query_history"], state["past_responses"])  # Add past_responses to state
#         prompt = self.ie_prompt.format(
#             messages=history_messages,  # Inserted via placeholder
#             query=state["query"],  # Current input
#             query_history=state["query_history"],  # Keep if your "ie" template still uses it
#             user_profile=state["user_profile"].model_dump(),
#             scam_reports=state["templates"].scam_reports,
#             strategies=state["templates"].strategies
#         )
#         try:
#             ie_output = ie_llm.invoke(prompt)
#             state["ie_output"] = ie_output
#             state["target_slots"] = ie_output.target_slots  # From structured output
#             self.logger.debug(f"IEAgent output: {ie_output.model_dump()}")
#             return state
#         except Exception as e:
#             self.logger.error(f"IEAgent error: {str(e)}")
#             default_output = PoliceResponse(
#                 conversational_response="I'm sorry, I need more information. Can you provide details about the scam?",
#                 scam_incident_description=state["query"],
#                 target_slots=[]
#             )
#             state["ie_output"] = default_output
#             state["target_slots"] = []
#             return state

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
    
#     def _map_questions_to_slots(self, conversational: str) -> Dict[str, List[str]]:
#         prompt = self.target_slot_prompt.format(conversational_response=conversational)  # Your target_slot_agent prompt
#         llm = self._get_llm(QuestionSlotMap)  # Keep this; assumes QuestionSlotMap is your schema
        
#         max_retries = 2
#         for attempt in range(max_retries):
#             response = llm.invoke(prompt)
#             self.logger.debug(f"Mapping LLM response (attempt {attempt+1}): {response.model_dump_json(indent=2)}")  # Log the structured data
            
#             if not response.root:  # Check if the dict is empty (equivalent to empty content)
#                 self.logger.warning("Empty mapping response. Retrying...")
#                 continue
            
#             try:
#                 mapped = response.root  # Directly access the root dict (no json.loads needed)
#                 issues = self.detect_issues(mapped)  # Assuming this is defined elsewhere (from vanilla code)
#                 if issues:
#                     if attempt < max_retries - 1:
#                         corrective = prompt + f"\nIssues: {', '.join(issues)}. Fix: Use full questions, no fragments/empties."
#                         prompt = corrective
#                         continue
#                     else:
#                         self.logger.warning("Max retries; using fallback")
#                         mapped = self.rule_based_fallback(conversational)  # Assuming this is defined
#                 break
#             except Exception as e:  # Broader catch since no JSONDecodeError now
#                 self.logger.error(f"Mapping error: {e}. Retrying...")
#                 continue
        
#         # Validate final (using ValidatedQuestionSlotMap to enforce no empty lists)
#         try:
#             validated = ValidatedQuestionSlotMap(root=mapped)
#             return validated.root
#         except ValidationError as e:
#             self.logger.error(f"Validation failed: {e}. Fallback.")
#             return self.rule_based_fallback(conversational)

#     def slot_tracker(self, state: GraphState) -> GraphState:
#         ie_output = state["ie_output"]
#         # Shift unfilled to prev (existing)
#         state["prev_turn_data"]["unfilled_slots"] = state.get("unfilled_slots", {})
        
#         # New: Map questions to slots first (LLM sub-agent)
#         conversational = ie_output.conversational_response
#         mapped_data = self._map_questions_to_slots(conversational)  # New helper below
#         state["target_slot_map"] = mapped_data
#         state["target_slots"] = list(mapped_data.keys())  # Derived from keys
        
#         # Existing: Compute current unfilled
#         unfilled = {}
#         for field_name in PoliceResponseSlots:
#             value = getattr(ie_output, field_name.value)
#             is_unfilled = (
#                 value in ("", 0.0, {}, None, "unknown", "na") or
#                 (isinstance(value, dict) and not value) or
#                 (isinstance(value, list) and not value)
#             )
#             unfilled[field_name.value] = is_unfilled
#         state["unfilled_slots"] = unfilled
        
#         # Existing + New: Compute filled_this_turn + Reconcile with mapped targets
#         prev_unfilled = state["prev_turn_data"]["unfilled_slots"]
#         filled_this_turn = [slot for slot, was_unfilled in prev_unfilled.items() if was_unfilled and not unfilled.get(slot, True)]
#         state["filled_this_turn"] = filled_this_turn
        
#         # Reconcile: Check if mapped targets were filled (log mismatches)
#         missed_targets = [slot for slot in state["target_slots"] if unfilled.get(slot, True)]
#         if missed_targets:
#             self.logger.warning(f"Targets not filled: {missed_targets}")
#             # Optional: Adjust for next (e.g., flag in metrics)
        
#         self.logger.debug(f"SlotTracker: target_slots={state['target_slots']}, unfilled={unfilled}, filled_this_turn={filled_this_turn}")
#         return state

#     def knowledge_base_agent(self, state: GraphState) -> GraphState:
#         kb_llm = self._get_llm(schema=KBList)
#         filled_this_turn = [k for k, v in state["prev_turn_data"]["unfilled_slots"].items() if v and not state["unfilled_slots"].get(k, False)]
#         prev_questions = state["prev_turn_data"].get("questions_asked", [])
#         prev_targets = state["prev_turn_data"].get("target_slots", [])
        
#         if not filled_this_turn or not prev_targets or not prev_questions:
#                 self.logger.debug("Skipping KB upsert: No filled slots or no prior questions/targets.")
#                 return state  # Skip if nothing to evaluate
            
#         prompt = self.kb_prompt.format(
#             prev_turn_data=state["prev_turn_data"],  # Already dict, no change
#             filled_slots_this_turn=filled_this_turn,
#             query=state["query"],
#             query_history=state["query_history"],
#             user_profile=state["user_profile"].model_dump()
#         )
#         for attempt in range(2):  # Retry once on failure
#             try:
#                 response = kb_llm.invoke(prompt)
#                 kb_outputs = response.items if hasattr(response, 'items') else []
#                 if not isinstance(kb_outputs, list):
#                     kb_outputs = [kb_outputs]
#                 state["kb_outputs"] = kb_outputs

#                 for i, kb_output in enumerate(kb_outputs):
#                     # Per-strategy score 
#                     num_targets = len(prev_targets)
#                     score = 0.6 * (1 / num_targets) + \
#                             0.3 * kb_output.strategy_match + \
#                             0.1 * (1 if state["metrics"]["turn_count"] < 3 else 0)
#                     if score >= 0.7:
#                         # Use filled_this_turn[i] for target_slot if needed
#                         similar_df = self.vector_store.hierarchical_strategy_search(state["user_profile"].model_dump(), limit=5)
#                         to_upsert = True
#                         for _, row in similar_df.iterrows():
#                             if row['profile_distance'] < 0.25 and np.linalg.norm(np.array(self.vector_store.embedder.encode(row['question'])) - np.array(self.vector_store.embedder.encode(prev_questions[0]))) < 0.2:
#                                 if score > row['success_score']:
#                                     self.strategy_crud.update(row['id'], {'success_score': score})
#                                     to_upsert = False
#                                 else:
#                                     to_upsert = False
#                                 break
#                         if to_upsert:
#                             new_strategy = Strategy(
#                                 strategy_type=kb_output.strategy_type,
#                                 question=prev_questions[0],  # Adjust if multi questions
#                                 success_score=score,
#                                 scam_type=state["ie_output"].scam_type,
#                                 user_profile=state["user_profile"].model_dump(),
#                                 target_slot=state["prev_turn_data"]["target_slots"][i] if i < len(state["prev_turn_data"]["target_slots"]) else state["prev_turn_data"]["target_slots"][0],
#                                 profile_embedding=self.vector_store.embedder.encode(json.dumps(state["user_profile"].model_dump())),
#                                 strategy_embedding=self.vector_store.embedder.encode(kb_output.strategy_type),
#                                 question_embedding=self.vector_store.embedder.encode(prev_questions[0])
#                             )
#                             self.strategy_crud.create(new_strategy.model_dump(exclude={'id'}))
#                             self.upsert_count += 1
#                             state["metrics"]["rag_upserted"] = True
#                             if self.upsert_count % 20 == 0:
#                                 self.vector_store.prune_strategies()
#                     self._log_upsert(state["conversation_id"], score, kb_output.model_dump())
#             except Exception as e:
#                 self.logger.error(f"KnowledgeBaseAgent error: {str(e)}")
#             return state

#     def _log_upsert(self, conv_id: str, score: float, kb_output: Dict):
#         # Log to rag_upserts.csv - implement similar to save_conversation_history
#         pass  # Placeholder; add CSV logging with params for variants later

#     def process_query(self, query: str, query_history: Optional[List[str]] = None, prev_state: Optional[GraphState] = None):
#         state = prev_state or {
#             "query": query,
#             "query_history": query_history or [],
#             "past_responses": [],  # Initialize as empty list to avoid KeyError in ie_agent
#             "user_profile": None,
#             "templates": None,
#             "ie_output": None,
#             "target_slots": [],
#             "target_slot_map": {},  # Initialize as empty dict for slot_tracker
#             "filled_this_turn": [],  # Initialize as empty list for slot_tracker and knowledge_base_agent
#             "prev_turn_data": {"target_slots": [], "unfilled_slots": {}, "questions_asked": []},
#             "unfilled_slots": {s.value: True for s in PoliceResponseSlots},
#             "conversation_id": self.conversation_id,
#             "metrics": {"rag_retrieved": False, "rag_upserted": False, "turn_count": 0}
#         }
#         state["query"] = query
#         state["query_history"].append(query)
        
#         final_state = self.workflow.invoke(state)
        
#         ie_out = final_state["ie_output"]
#         response = ie_out.conversational_response
        
#         self.conversation_history.append({
#             "role": "victim", 
#             "content": query, 
#             "timestamp": datetime.now().isoformat(),
#             "autonomous":False})
#         self.conversation_history.append({
#             "role": "police",
#             "content": ie_out.conversational_response,
#             "timestamp": datetime.now().isoformat(),
#             "structured_data": ie_out.model_dump(),
#             "rag_invoked": final_state["metrics"]["rag_retrieved"],
#             "autonomous":False
#         })
        
#         save_conversation_history(self.conversation_id, self.conversation_history, model_name=self.model_name, logger=self.logger)
#         final_state["user_profile"] = final_state.get("user_profile").model_dump() if final_state.get("user_profile") else None
#         final_state["templates"] = final_state.get("templates").model_dump() if final_state.get("templates") else None
#         final_state["ie_output"] = final_state.get("ie_output").model_dump() if final_state.get("ie_output") else None
#         final_state["kb_outputs"] = [kb.model_dump() for kb in final_state.get("kb_outputs", [])] if final_state.get("kb_outputs") else []
#         final_state["past_responses"].append(ie_out.conversational_response)  # Assuming state has past_responses = [] initially
#         return {"response": response, "structured_data": ie_out.model_dump(), "conversation_id": self.conversation_id, "state": final_state}
    
#     def end_conversation(self):
#         save_conversation_history(
#             conversation_id=self.conversation_id,
#             history=self.conversation_history,
#             csv_file="conversation_history.csv",  # Same as vanilla
#             model_name=self.model_name,
#             logger=self.logger
#         )
#         self.logger.info(f"Conversation {self.conversation_id} saved to CSV")
#         return {"status": "Conversation ended", "conversation_id": self.conversation_id}

# # __main__ test: Adapt to multi-turn by passing prev_state.
# if __name__ == "__main__":
#     # models = ["qwen2.5:7b", "granite3.2:8b", "mistral:7b"]
#     models = ["qwen2.5:7b"]
#     query = "I received a call from someone claiming to be a government official."
#     results = {}
    
#     #single turn test for multiple models
#     for model_name in models:
#         try:
#             chatbot = PoliceChatbot(model_name=model_name)
#             response = chatbot.process_query(query, query_history=["Initial context: User mentioned a suspicious call."])
#             results[model_name] = response
#             print(f"Model {model_name}: {json.dumps(response, indent=2)}")
#         except Exception as e:
#             results[model_name] = {"error": str(e)}
#             print(f"Error with {model_name}: {str(e)}")
    
#     print(json.dumps(results, indent=2))
    
    
#     #Multi-turn test with one model
#     print("\n=== Multi-turn Conversation Test ===")
#     for model_name in models:
#         chatbot = PoliceChatbot(model_name=model_name)

#         # Simulate a multi-turn conversation
#         queries = [
#             "I received a call from someone claiming to be a government official.",
#             "The incident happened on 22 Jul 2025 and I lost 500 dollars.",
#             "The scammer's phone number was 12345678 and they asked for my bank details. I subsequently transferred money to CIMB 1234567890",
#             "I think that's all the details I have."
#         ]

#         prev_state = None
#         conversation_responses = []

#         for i, q in enumerate(queries, 1):
#             print(f"\nTurn {i}: User Query: {q}")
#             response_data = chatbot.process_query(q, prev_state=prev_state)
#             conversation_responses.append(response_data["response"])
#             prev_state = response_data["state"]  # Pass the state for the next turn
#             print(f"Police Response: {response_data['response']}")

#         print("\nFull Conversation Responses:")
#         print(json.dumps(conversation_responses, indent=2))







#previous kb agent in working code 


# def knowledge_base_agent(self, state: GraphState) -> GraphState:
    #     """Evaluate and upsert strategies."""
    #     kb_llm = self._get_llm(schema=KnowledgeBaseOutput)
    #     filled_this_turn = state["filled_this_turn"]

        
    #     if not filled_this_turn: #or not prev_targets or not prev_questions:
    #         self.logger.debug("Skipping KB upsert: Insufficient data.")
    #         return state
        
    #     prev_response = state["prev_turn_data"].get("prev_response", "")
    #     if not prev_response:
    #         self.logger.debug("Skipping KB upsert: No previous response.")
    #         return state
        
    #     prompt = self.kb_prompt.format(
    #         prev_turn_data=state["prev_turn_data"],
    #         filled_slots_this_turn=filled_this_turn,
    #         query=state["query"],
    #         query_history=state["query_history"],
    #         user_profile=state["user_profile"]
    #     )
    #     max_retries = 2
    #     for attempt in range(max_retries):
    #         try:
    #             response = kb_llm.invoke(prompt)
    #             if self.llm_provider == "OpenAI":
    #                 kb_output = response
    #             else:
    #                 kb_output_dict = json.loads(response.content)
    #                 kb_output = KnowledgeBaseOutput(**kb_output_dict)
    #             state["kb_outputs"] = kb_output
                
                
    #             # self.logger.debug(f"KBAgentOutput: {kb_output}")
                
    #             # num_targets = len(filled_this_turn)
    #             # total_slots = len(PoliceResponseSlots)
    #             # target_score = 0.3 * min(max(num_targets, 0) / total_slots, 1.0)  # Ensures max 0.3, handles num_targets=0

    #             # score = target_score + 0.6 * kb_output.avg_rating() + 0.1 * (1 if state["metrics"]["turn_count"] < 2 else 0)
    #             # NEW: Dynamic success score with confidence (Task 2)
               
    #             # Use initial_profile for stable conf (anti-drift)
    #             profile_for_conf = state.get("initial_profile", state["user_profile"])  # Fallback to current if no initial (rare)
    #             conf_values = [profile_for_conf[dim]["confidence"] for dim in profile_for_conf]
    #             avg_conf = np.mean(conf_values) if conf_values else 0.5  # Avg of dims
    #             self.logger.debug(f"Avg confidence (from initial): {avg_conf:.2f}")

    #             # Optional: Log if current differs (for eval)
    #             current_conf = np.mean([state["user_profile"][dim]["confidence"] for dim in state["user_profile"]])
    #             if abs(avg_conf - current_conf) > 0.1:
    #                 self.logger.info(f"Conf drift noted: Initial avg {avg_conf:.2f} vs Current {current_conf:.2f}")

    #             avg_rating = kb_output.avg_rating()  # Your method (normalize 1-5 to 0-1 if not already)
    #             num_filled = len(filled_this_turn)
    #             total_slots = len(PoliceResponseSlots)
    #             slots_norm = min(num_filled / total_slots, 1.0) if total_slots > 0 else 0  # Normalized 0-1

    #             conf_threshold = 0.7  # Class var? Tuneable
    #             if avg_conf < conf_threshold:
    #                 score = 0.8 * slots_norm + 0.2 * avg_rating  # Prioritize slots
    #                 self.logger.info(f"Low conf ({avg_conf:.2f} < {conf_threshold}): Score = 0.8*{slots_norm:.2f} + 0.2*{avg_rating:.2f} = {score:.2f}")
    #             else:
    #                 score = 0.3 * slots_norm + 0.7 * avg_rating  # Prioritize ratings
    #                 self.logger.info(f"High conf ({avg_conf:.2f} >= {conf_threshold}): Score = 0.3*{slots_norm:.2f} + 0.7*{avg_rating:.2f} = {score:.2f}")

    #             self.logger.info(f"Score: {score}")

    #             if score >= 0.6:
    #                 similar_df = self.vector_store.hierarchical_strategy_search(state["user_profile"], limit=5) 
    #                 to_upsert = True
    #                 for _, row in similar_df.iterrows():
    #                     if row['profile_distance'] < 0.3 and np.linalg.norm(self.vector_store.model.encode(row['response']) - self.vector_store.model.encode(prev_response)) < 0.3:
    #                         if score > row['success_score']:
    #                             self.strategy_crud.update(row['id'], {'success_score': score})
    #                         to_upsert = False
    #                         break
    #                 if to_upsert:
                        
    #                     strategy_data = {
    #                         "strategy_type": kb_output.strategy_type,
    #                         "response": prev_response,
    #                         "success_score": score,
    #                         "user_profile": state["user_profile"],  #change to model_dump if pydantic model
    #                         "profile_embedding": self.vector_store.model.encode(json.dumps(state["user_profile"])),
    #                         "strategy_embedding": self.vector_store.model.encode(kb_output.strategy_type),
    #                         "response_embedding": self.vector_store.model.encode(prev_response)
    #                     }
    #                     self.strategy_crud.create(strategy_data)  # Pass the dict directly
    #                     self.upsert_count += 1
    #                     state["metrics"]["rag_upserted"] = True
    #                     if self.upsert_count % 50 == 0:
    #                         self.vector_store.prune_strategies()
    #             break
    #         except Exception as e:
    #             self.logger.error(f"KB agent error (attempt {attempt+1}): {str(e)}")
    #     return state