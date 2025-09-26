import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, AnyMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence, Optional
from pydantic import BaseModel, ValidationError
import json
import logging
import csv
import os
from datetime import datetime
from pathlib import Path
from filelock import FileLock
from config.settings import get_settings
from config.logging_config import setup_logger
from src.database.vector_operations import VectorStore
from src.database.database_operations import DatabaseManager
from src.models.data_model import ScamReport
from src.models.response_model import PoliceResponse, QuestionSlotMap, ValidatedQuestionSlotMap
from src.agents.tools import PoliceTools
from config.id_manager import IDManager
from langchain_core.exceptions import LangChainException
from src.agents.prompt import Prompt
from src.agents.utils import build_query_with_history
import re

# Define state
class ChatbotState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    conversation_id: int
    police_response: dict
    rag_results: list
    rag_invoked: bool
    user_profile: dict  # Or Optional[Dict] if it might be None
    rag_upsert: bool

class PoliceChatbot:
    def __init__(self, model_name: str = "qwen2.5:7b", rag_csv_path: str = "rag_invocations.csv"):
        self.settings = get_settings()
        self.logger = setup_logger("VanillaRag_PoliceAgent", self.settings.log.subdirectories["agent"])
        # self.id_manager = IDManager()
        self.model_name = model_name

        self.messages: Sequence[AnyMessage] = []
        self.scam_type: str = ""
        
        # Initialize tools and chatbot prompt
        self.police_tools = PoliceTools(rag_csv_path=rag_csv_path)
        self.tools = self.police_tools.get_tools()
        
        self.main_prompt_template = self._generate_prompt_template()
        self.llm = self._get_structured_llm(PoliceResponse)
        self.target_slot_prompt = ChatPromptTemplate.from_template(Prompt.template["target_slot_agent"])
        
        # Build LangGraph workflow
        self.workflow = self._build_workflow()
        
        self.logger.info(f"Police chatbot initialized with model: {model_name}")

    def _generate_prompt_template(self) -> ChatPromptTemplate:
        """
        Generate the ChatPromptTemplate using the baseline_police prompt from Prompt class.
        """
        if "baseline_police" not in Prompt.template:
            self.logger.error("Prompt type 'baseline_police' not found in Prompt.template")
            raise ValueError("Prompt type 'baseline_police' not found")
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", Prompt.template["baseline_police"]),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{user_input}"),
        ])
        self.logger.debug("Generated baseline_police prompt template")
        return prompt_template
    
    def _get_structured_llm(self, schema: BaseModel) -> ChatOllama:
        return ChatOllama(
            model=self.model_name,
            base_url=self.settings.agents.ollama_base_url,
            format="json",
            response_format={
                "type": "json_schema",
                "json_schema": schema.model_json_schema()
            }
        )
    
    def _build_workflow(self):
        workflow = StateGraph(ChatbotState)
        workflow.add_node("invoke_rag", lambda state: self._invoke_rag(state, self.tools))
        workflow.add_node("process_llm", lambda state: self._process_llm(state))
        # workflow.add_node("map_target_slots", lambda state: self._map_target_slots(state))
        workflow.add_edge("invoke_rag", "process_llm")
        workflow.add_edge("process_llm", END)
        # workflow.add_edge("process_llm", "map_target_slots")
        # workflow.add_edge("map_target_slots", END)
        workflow.set_entry_point("invoke_rag")
        return workflow.compile()

    def _invoke_rag(self, state: ChatbotState, tools):
        self.logger.debug(f"Invoking RAG for query: {state['messages'][-1].content}")
        user_query = build_query_with_history(state["messages"][-1].content, [m.content for m in state["messages"][:-1] if isinstance(m, HumanMessage)], max_history=5)
        rag_tool = tools[0]  # retrieve_scam_reports
        try:
            rag_results = rag_tool.invoke({
                "query": user_query,
                "top_k": 1,
                "conversation_id": state["conversation_id"],
                "llm_model": self.model_name,
            })
            rag_results = json.loads(rag_results)
            self.logger.debug(f"RAG results: {rag_results}")
        except Exception as e:
            self.logger.error(f"RAG invocation failed: {str(e)}", exc_info=True)
            rag_results = []
        self.logger.info("RAG invoked successfully")
        return {
            "rag_results": rag_results,
            "rag_invoked": True
        }

    def _process_llm(self, state: ChatbotState):
        self.logger.debug("Processing LLM response")
        messages = state["messages"]
        rag_results = state.get("rag_results", [])
        rag_invoked = state.get("rag_invoked", False)
        
        prompt = self.main_prompt_template.format(
            messages=messages,
            user_input=messages[-1].content,
            rag_results=json.dumps(rag_results)
        )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                if not response.content.strip():
                    self.logger.error("LLM returned empty response")
                    continue
                police_response = json.loads(response.content)
                default_response = {
                    "conversational_response": "I'm sorry, I need more information to assist you.",
                    "scam_incident_date": "",
                    "scam_type": "",
                    "scam_approach_platform": "",
                    "scam_communication_platform": "",
                    "scam_transaction_type": "",
                    "scam_beneficiary_platform": "",
                    "scam_beneficiary_identifier": "",
                    "scam_contact_no": "",
                    "scam_email": "",
                    "scam_moniker": "",
                    "scam_url_link": "",
                    "scam_amount_lost": 0.0,
                    "scam_incident_description": "",
                }
                default_response.update(police_response)
                police_response = default_response
                PoliceResponse(**police_response)
                # Persist scam_type if newly identified and non-empty
                if police_response["scam_type"] and not self.scam_type:
                    self.scam_type = police_response["scam_type"]
                self.logger.debug(f"Validated PoliceResponse: {police_response}")
                
                
                return {
                    "messages": [AIMessage(content=json.dumps(police_response))],
                    "police_response": police_response,
                    "rag_invoked": rag_invoked
                }
            except Exception as e:
                self.logger.error(f"LLM attempt {attempt+1} failed: {str(e)}", exc_info=True)
                if attempt == max_retries - 1:
                    fallback_response = {
                        "conversational_response": "Sorry, I encountered an error. Can you provide more details about the scam, such as the date, what the caller said, and any contact details provided?",
                        "scam_incident_date": "",
                        "scam_type": "",
                        "scam_approach_platform": "",
                        "scam_communication_platform":"",
                        "scam_transaction_type": "",
                        "scam_beneficiary_platform": "",
                        "scam_beneficiary_identifier": "",
                        "scam_contact_no": "",
                        "scam_email": "",
                        "scam_moniker": "",
                        "scam_url_link": "",
                        "scam_amount_lost": 0.0,
                        "scam_incident_description": messages[-1].content,
                    }
                    return {
                        "messages": [AIMessage(content=json.dumps(fallback_response))],
                        "police_response": fallback_response,
                        "rag_invoked": rag_invoked,
                        "rag_upsert": False,
                        "user_profile_inferred": {}  
                    }

    # def detect_issues(self, mapped_data: dict) -> list[str]:
    #     issues = []
    #     for slot, questions in mapped_data.items():
    #         if not questions:
    #             issues.append(f"Empty list for {slot}")
    #         for q in questions:
    #             if len(q.strip().split()) < 3 or re.match(r'^(and|or|if)\b', q.strip().lower()):
    #                 issues.append(f"Fragment detected in {slot}: {q}")
    #     return issues
    

    # def rule_based_fallback(self,conversational: str) -> dict:
    #     slots = {}
    #     keywords = {
    #         'scam_incident_date': ['date', 'when'],
    #         'scam_amount_lost': ['amount', 'lost', 'paid', 'money', 'dollars', 'sum', 'total', 'cost', 'transferred'],
    #         'scam_type': ['type', 'kind', 'scam', 'government', 'ecommerce', 'phishing'],
    #         'scam_approach_platform': ['approached', 'contacted', 'reached'],
    #         'scam_communication_platform': ['communicated', 'talked', 'messaged', 'called'],
    #         'scam_transaction_type': ['transaction', 'payment', 'transfer', 'method', 'type', 'bank'],
    #         'scam_beneficiary_platform': ['beneficiary', 'sent to', 'paid to', 'account', 'bank'],
    #         'scam_beneficiary_identifier': ['identifier', 'account number'],
    #         'scam_contact_no': ['contact', 'phone', 'number'],
    #         'scam_email': ['email', 'gmail', 'yahoo', 'outlook'],
    #         'scam_moniker': ['name', 'alias', 'moniker', 'called themselves', 'username', 'handle', 'nickname'],
    #         'scam_url_link': ['url', 'link', 'website', 'site','clicked', 'visited', 'domain'],
    #         'scam_incident_description': ['description', 'happened', 'details', 'story', 'explain', 'what occurred', 'incident', 'how']
    #     }
    #     for slot, kws in keywords.items():
    #         if any(kw in conversational.lower() for kw in kws):
    #             slots[slot] = [conversational]  # Duplicate full as fallback
    #     return slots

    # def _map_target_slots(self, state: ChatbotState):
    #     police_response = state["police_response"] 
    #     conversational = police_response["conversational_response"]
    #     prompt = self.target_slot_prompt.format(conversational_response=conversational)
    #     llm = self._get_structured_llm(QuestionSlotMap)
        
    #     max_retries = 2
    #     for attempt in range(max_retries):
    #         response = llm.invoke(prompt)
    #         self.logger.debug(f"LLM raw response for mapping (attempt {attempt+1}): {response.content}")
            
    #         if not response.content.strip():
    #             self.logger.error("Empty LLM response for mapping. Retrying..." if attempt < max_retries-1 else "Max retries reached.")
    #             continue
            
    #         try:
    #             mapped = json.loads(response.content)
    #             issues = self.detect_issues(mapped)
    #             self.logger.debug(f"Mapping issues: {issues}")
                
    #             if issues:
    #                 if attempt < max_retries - 1:
    #                     corrective_prompt = prompt + f"\nPrevious output had issues: {', '.join(issues)}. Fix by extracting full questions and mapping to relevant slots without empties."
    #                     prompt = corrective_prompt  # Update for next invoke
    #                     continue
    #                 else:
    #                     self.logger.warning("Max retries; using fallback.")
    #                     mapped = self.rule_based_fallback(conversational)
    #             break  # Success
    #         except json.JSONDecodeError as e:
    #             self.logger.error(f"JSON decode error: {e}. Retrying...")
    #             continue
    #         except Exception as e:
    #             self.logger.error(f"Mapping error: {e}")
    #             mapped = self.rule_based_fallback(conversational)
    #             break

    #     # Final validation
    #     try:
    #         validated = ValidatedQuestionSlotMap(root=mapped)
    #         police_response["target_slot_map"] = validated.root
    #         police_response["target_slots"] = list(validated.root.keys())
    #     except ValidationError as e:
    #         self.logger.error(f"Validation failed: {e}. Using fallback.")
    #         police_response["target_slot_map"] = self.rule_based_fallback(conversational)
    #         police_response["target_slots"] = list(police_response["target_slot_map"].keys())

    #     return {"police_response": police_response}
        

    def process_query(self, query: str, conversation_id: int = None) -> dict:
        if not query.strip():
            self.logger.error("Query cannot be empty")
            return {"error": "Query cannot be empty"}

        state = {
            "messages": self.messages + [HumanMessage(content=query)],
            "conversation_id": conversation_id,
            "police_response": {},
            "rag_results": [],
            "rag_invoked": False,
            "rag_upsert": False, # Default False
            "user_profile": {} # Default empty
        }
        result = self.workflow.invoke(state)
        self.messages = result["messages"]
        police_response = result["police_response"]
        rag_invoked = result.get("rag_invoked", False)
        conversational_response = police_response.get("conversational_response", "I'm sorry, I need more information to assist you.")
        user_profile = result.get("user_profile", {})
        rag_upsert = result.get("rag_upsert", False)
        
        structured_data = police_response.copy()
        structured_data["user_profile"] = user_profile
        structured_data["rag_upsert"] = rag_upsert


        self.logger.debug(f"Processed query response: {conversational_response}")
        return {
            "response": conversational_response,
            "structured_data": structured_data,
            "conversation_id": conversation_id,
            # "conversation_history": self.conversation_history,
            "rag_invoked": rag_invoked
        }

    def reset_state(self):
        """Reset internal state (messages, history, scam_type) for reuse in new conversations."""
        self.messages = []
        self.conversation_history = []
        self.scam_type = ""
        self.logger.debug("PoliceChatbot state reset")

    def end_conversation(self, csv_file: Optional[str] = None) -> dict:
        # self._save_conversation(csv_file)
        self.logger.info(f"Conversation saved to CSV")
        return {"status": "Conversation ended"}


if __name__ == "__main__":
    from config.id_manager import IDManager
    logger = setup_logger("VanillaRag_PoliceAgent", get_settings().log.subdirectories["agent"])
    models = ["qwen2.5:7b", "granite3.2:8b", "mistral:7b"]
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
    for model_name in models:
        logger.info(f"--- Testing model: {model_name} ---")
        try:
            chatbot = PoliceChatbot(model_name=model_name)
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