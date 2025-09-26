import sys
import os
import json
import re

from langchain_core.messages import HumanMessage, AIMessage, AnyMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.exceptions import LangChainException
from sentence_transformers import SentenceTransformer, util
from typing import TypedDict, Annotated, Sequence, Optional
from pydantic import ValidationError
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from config.settings import get_settings
from config.logging_config import setup_logger
from src.agents.utils import RecordCounter
from src.agents.prompt import Prompt
from src.models.response_model import VictimResponse


# Define state 
class ChatbotState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]
    end_conversation: bool = False

class VictimChatbot:
    def __init__(
        self,
        model_name: str = "qwen2.5:7b",
        llm_provider: str = "Ollama",
        json_file: str = "data/victim_profile/victim_details.json",
        user_profile: Optional[dict] = None,
        scam_details: Optional[dict] = None,
        profile_id: Optional[int] = None,
        temperature: float = 0.0):
        """
        Initialize the VictimChatbot with dynamic user profiles, victim details, and scam details.
        If profiles/details are provided directly, use them; otherwise, load from JSON and cycle via RecordCounter.
        """
        self.settings = get_settings()
        self.logger = setup_logger("VictimAgent", self.settings.log.subdirectories["agent"])
        self.record_counter = RecordCounter()
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.json_file = json_file
        self.messages: Sequence[AnyMessage] = []

        # Load or use provided profiles/details
        if user_profile and scam_details:
            self.user_profile = user_profile
            self.scam_details = scam_details
            self.profile_id = profile_id
            self.record_index = -1 
            self.logger.debug("Using directly provided profiles/details")
        else:
            self.records = self._load_records()
            if profile_id is not None:
                for i, rec in enumerate(self.records):
                    if rec.get("profile_id") == profile_id:
                        self.record_index = i
                        break
                else:
                    raise ValueError(f"Profile ID {profile_id} not found in {json_file}")
            else:
                self.record_index = self.record_counter.get_next_index(len(self.records))
            self.user_profile, self.scam_details = self._get_current_record()
            self.profile_id = self.records[self.record_index].get("profile_id")  
            if self.profile_id is None:
                self.profile_id = self.record_index  
                self.logger.warning(f"No profile_id in record {self.record_index}; using index as fallback: {self.profile_id}")

        # Initialize prompt and workflow
        self.prompt_template = self._generate_prompt_template()
        self.llm = self._get_llm()
        self.workflow = self._build_workflow()
        self.embedder = SentenceTransformer(self.settings.vector.embedding_model)

        self.logger.info(
            f"Victim chatbot initialized: model={model_name}, "
            f"record_index={self.record_index}"
        )

    def _load_records(self) -> list:
        """
        Load user profiles, victim details, and scam details from the JSON file.
        """
        try:
            with open(self.json_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            if not records:
                self.logger.error(f"No records found in {self.json_file}")
                raise ValueError(f"No records found in {self.json_file}")
            return records
        except Exception as e:
            self.logger.error(f"Error loading records from {self.json_file}: {str(e)}", exc_info=True)
            raise

    def _get_current_record(self) -> tuple[dict, dict, dict]:
        """
        Get the current user profile, victim details, and scam details based on record_index.
        """
        record = self.records[self.record_index]
        return (
            record.get("user_profile", {}),
            # record.get("victim_details", {}),
            record.get("scam_details", {})
        )

    @staticmethod
    def _escape_template_braces(text: str) -> str:
        """
        Escape curly braces in text to prevent LangChain from interpreting them as placeholders.
        """
        return text.replace("{", "{{").replace("}", "}}")

    def _generate_prompt_template(self) -> ChatPromptTemplate:
        """
        Create a prompt template using the "baseline_victim" prompt, filled with profile and scam details.
        """
        if "baseline_victim" not in Prompt.template:
            self.logger.error("Prompt type 'baseline_victim' not found in Prompt.template")
            raise ValueError("Prompt type 'baseline_victim' not found")

        user_profile_json = json.dumps(self.user_profile, indent=2)
        scam_details_json = json.dumps(self.scam_details, indent=2)
        user_profile_str = self._escape_template_braces(user_profile_json)
        scam_details_str = self._escape_template_braces(scam_details_json)

        try:
            prompt_text = Prompt.template["baseline_victim"].format(
                user_profile=user_profile_str,
                scam_details=scam_details_str
            )
        except KeyError as e:
            self.logger.error(f"Error formatting prompt template: {str(e)}")
            raise ValueError(f"Missing key in prompt template: {str(e)}")

        self.logger.debug(f"Formatted prompt text: {prompt_text[:500]}...")  # Truncate for logging

        return ChatPromptTemplate.from_messages([
            SystemMessage(content=prompt_text),
            MessagesPlaceholder(variable_name="messages"),
            ("human", "{user_input}"),
        ])

    
    def _get_llm(self):
        """
        Initialize the LLM without tool binding. Creates an LLM instance configured for structured output.
        """
        if self.llm_provider == "Ollama":
            return ChatOllama(
                model=self.model_name,
                base_url=self.settings.agents.ollama_base_url,
                format="json",
                temperature = self.temperature,
                response_format={
                    "type": "json_schema",
                    "json_schema": VictimResponse.model_json_schema()
                }
            )
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
            return base_llm.with_structured_output(VictimResponse)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")

    def _build_workflow(self):
        """
        Build the LangGraph workflow with a single node for processing LLM responses.
        """
        workflow = StateGraph(ChatbotState)
        workflow.add_node("process_llm", self._process_llm)
        workflow.add_edge("process_llm", END)
        workflow.set_entry_point("process_llm")
        return workflow.compile()

    def _process_llm(self, state: ChatbotState):
        """
        Run the LLM to generate a victim response. Handles retries, parsing, cleaning,
        and checks for repetition or end conditions to decide if conversation should end.
        """
        self.logger.debug("Processing LLM response")
        past_victim_responses = [msg.content for msg in state["messages"] if isinstance(msg, AIMessage)]
        turn_count = len(past_victim_responses) + 1

        try:
            prompt = self.prompt_template.format(
                messages=state["messages"],
                user_input=f"{state['messages'][-1].content}\n[Current turn: {turn_count}. Remember to set end_conversation true if conditions met.]"
            )
        except KeyError as e:
            self.logger.error(f"KeyError in prompt formatting: {str(e)}")
            raise ValueError(f"Failed to format prompt: {str(e)}")

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                if self.llm_provider == "OpenAI":
                    victim_response = response
                else:
                    raw_response = response.content
                    self.logger.debug(f"Raw LLM response (attempt {attempt+1}): {raw_response}")
                
                    if not raw_response.strip():
                        self.logger.error(f"Empty response (attempt {attempt+1})")
                        continue

                    parsed_response = json.loads(raw_response)
                    victim_response = VictimResponse(**parsed_response)
                self.logger.debug(f"End conversation from JSON: {victim_response.end_conversation}")
                
                conversational_response = victim_response.conversational_response
                end_conversation = victim_response.end_conversation

                # Strip AI prefixes and forbidden terms
                cleaned_response = re.sub(r'^(AI:\s*)+', '', conversational_response).strip()
                forbidden_terms = ["thinking", "<thinking>", "[thinking]", "[END_CONVERSATION]"]
                for term in forbidden_terms:
                    cleaned_response = re.sub(rf'(?i){re.escape(term)}.*?(?=\s|$)', '', cleaned_response).strip()
                
                if not cleaned_response.strip():
                    self.logger.warning(f"Cleaned response empty after filtering (attempt {attempt+1}); continuing")
                    continue

                conversational_response = cleaned_response

                if "[END_CONVERSATION]" in conversational_response:
                    conversational_response = conversational_response.replace("[END_CONVERSATION]", "").strip()
                    end_conversation = True  
                    self.logger.debug("String [END] found; forcing True")

                # Check for closing phrases and OR with JSON bool
                closing_phrases = [
                    r"that's all I recall", r"that's all I know", r"that's all I remember", r"that's everything",
                    r"that’s all I can recall", r"that’s all I know", r"that’s all I remember", r"that’s everything",
                    r"I believe that’s everything", r"I think that’s everything", r"nothing more", r"that's it"
                ]
                closing_pattern = r'|'.join(r'\b' + re.escape(phrase) + r'\b' for phrase in closing_phrases)
                if re.search(closing_pattern, conversational_response, re.IGNORECASE):
                    self.logger.debug(f"Closing phrase detected: {conversational_response}")
                    end_conversation = True  # OR with JSON

                # Repetition check to force end if high similarity
                if not end_conversation and len(past_victim_responses) >= 5:
                    try:
                        recent_responses = past_victim_responses[-2:] + [conversational_response]  # Include current for check
                        embeddings = self.embedder.encode(recent_responses)
                        sim_scores = [util.cos_sim(embeddings[-1], embeddings[-1 - i])[0][0].item() for i in range(1, len(embeddings))]
                        avg_sim = sum(sim_scores) / len(sim_scores) if sim_scores else 0
                        if avg_sim > 0.90:
                            self.logger.warning(f"High repetition detected (avg_sim={avg_sim:.2f}); forcing end.")
                            if not re.search(closing_pattern, conversational_response, re.IGNORECASE):
                                conversational_response += " I think that's all I know."
                            end_conversation = True
                    except Exception as e:
                        self.logger.error(f"Embedding failed: {e}; falling back to string check.")
                        if len(past_victim_responses) >= 2 and past_victim_responses[-1] == past_victim_responses[-2]:
                            self.logger.warning("Exact repetition detected; forcing end.")
                            if not re.search(closing_pattern, conversational_response, re.IGNORECASE):
                                conversational_response += " I think that's all I know."
                            end_conversation = True

                # Force end if police query is summary-like and turn >=5
                query = state["messages"][-1].content.lower()
                if not end_conversation and (
                    not any(q in query for q in ["?", "what", "when", "where", "how", "who", "can you", "tell me", "provide", "did you", "any other"]) or
                    "proceed to submit" in query or "if you're satisfied" in query
                ):
                    if turn_count >= 5:
                        self.logger.info(f"Turn {turn_count} >=3 and summary query; forcing end")
                        if not re.search(closing_pattern, conversational_response, re.IGNORECASE):
                            conversational_response += " I think that's all I know."
                        end_conversation = True

                self.logger.debug(f"Final end conversation after checks: {end_conversation}")
                self.logger.debug(f"Returning end_conversation: {end_conversation}")
                return {
                    "messages": [AIMessage(content=conversational_response)],
                    "end_conversation": end_conversation
                }
            except (json.JSONDecodeError, ValidationError) as e:
                self.logger.error(f"Parse/Validation failed (attempt {attempt+1}): {str(e)}")
                continue

        self.logger.error("Max retries reached; using fallback")
        conversational_response = "Um, I'm sorry, something went wrong. Can you repeat that?"
        end_conversation = False
        return {
            "messages": [AIMessage(content=conversational_response)],
            "end_conversation": end_conversation
        }
    def process_query(self, query: str) -> VictimResponse:
        """
        Take a query from the police agent, run it through the workflow, and return the victim's response.

        Returns:
            VictimResponse: Structured response with conversational text and end flag.
        """
        if not query.strip():
            self.logger.error("Query cannot be empty")
            return VictimResponse(conversational_response="Query cannot be empty", end_conversation=False)

        state = {
            "messages": self.messages + [HumanMessage(content=query)]
        }
        result = self.workflow.invoke(state)
        self.messages = result["messages"]
        conversational_response = result["messages"][-1].content
        end_conversation = result.get("end_conversation", False)

        self.logger.debug(f"Processed query response: {conversational_response}, end_conversation: {end_conversation}")
        
        return VictimResponse(
            conversational_response=conversational_response,
            end_conversation=end_conversation
        )

    def reset_state(self):
        """Reset internal state (messages) for reuse in new conversations."""
        self.messages = []
        self.logger.debug("VictimChatbot state reset")

    def end_conversation(self):
        """
        End the conversation and reset state.
        """
        self.reset_state()
        self.logger.info(f"Conversation ended (save handled by manager)")
        return {"status": "Conversation ended"}

if __name__ == "__main__":
    settings = get_settings()
    logger = setup_logger("VictimAgent", settings.log.subdirectories["agent"])
    json_file = "data/victim_profile/victim_details.json"

    models = [
        ("gpt-4o-mini", "OpenAI"),
        ("qwen2.5:7b", "Ollama"),
        ("granite3.2:8b", "Ollama"),
        ("mistral:7b", "Ollama")
    ]
    query = "Can you tell me about any recent scam incidents you’ve experienced?"
    results = {}
    num_reinitializations = 3

    logger.info("Starting model testing with multiple reinitializations")
    for model_name, llm_provider in models:
        logger.info(f"--- Testing model: {model_name} ---")
        record_counter = RecordCounter()
        record_counter.reset()
        model_results = []
        for i in range(num_reinitializations):
            logger.info(f"Reinitialization {i+1} for model {model_name}")
            try:
                chatbot = VictimChatbot(model_name=model_name, llm_provider = llm_provider, json_file=json_file)
                response = chatbot.process_query(query)
                logger.info(f"Victim profile: {json.dumps(chatbot.user_profile, indent=2)}")
                model_results.append({
                    "reinitialization": i + 1,
                    "record_index": chatbot.record_index,
                    "response": response.conversational_response,
                    "end_conversation": response.end_conversation
                })
                logger.info(f"Processed query, reinitialization {i+1}: conversational_response={response.conversational_response}, end_conversation={response.end_conversation}")
                chatbot.end_conversation()
            except LangChainException as e:
                logger.error(f"LangChain error, reinitialization {i+1}: {str(e)}", exc_info=True)
                model_results.append({
                    "reinitialization": i + 1,
                    "error": f"LangChain error: {str(e)}"
                })
            except Exception as e:
                logger.error(f"Unexpected error, reinitialization {i+1}: {str(e)}", exc_info=True)
                model_results.append({
                    "reinitialization": i + 1,
                    "error": f"Unexpected error: {str(e)}"
                })
        results[model_name] = model_results

    logger.info("Completed model testing")
    print(json.dumps(results, indent=2))