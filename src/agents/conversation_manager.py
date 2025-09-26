import sys
import os

from datetime import datetime
import json
import csv
from enum import Enum
from typing import List, Dict, Optional, Union, Tuple
from filelock import FileLock
from pathlib import Path
import itertools

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config.settings import get_settings
from config.logging_config import setup_logger
from config.id_manager import IDManager
from src.agents.utils import RecordCounter
from src.models.response_model import PoliceResponseSlots
from src.agents.police.rag_ie_agent import RAGIEAgent
from src.agents.police.ie_agent import IEChatbot
from src.agents.police.profile_rag_ie_agent import ProfileRagIEAgent
from src.agents.police.profile_rag_ie_kb_agent import ProfileRAGIEKBAgent 
from src.agents.victim.victim_agent import VictimChatbot

class ConversationMode(Enum):
    AUTONOMOUS = "autonomous"
    NONAUTONOMOUS = "nonautonomous"

class ConversationType(Enum):
    IE = "ie"
    RAG_IE = "rag_ie"
    PROFILE_RAG_IE = "profile_rag_ie"
    PROFILE_RAG_IE_KB = "profile_rag_ie_kb" 

class ConversationManager:
    """
    Manages conversations in autonomous (police-victim) or non-autonomous (human-police) modes.
    Handles logging, end conditions, batch simulations, and error resilience.
    """
    def __init__(
        self,
        mode: ConversationMode,
        conversation_type: ConversationType = ConversationType.RAG_IE,
        police_model_name: str = "qwen2.5:7b",
        police_llm_provider: str = "Ollama",
        victim_model_name: Optional[str] = None, #defaults to police_model_name
        victim_llm_provider: Optional[str]=None,
        max_turns: int = 10,
        json_file: str = "data/victim_profile/victim_details.json"
    ):
        """
        Initialize the manager.

        Args:
            mode (ConversationMode): Autonomous or non-autonomous.
            conversation_type (ConversationType): Vanilla RAG or self-augmenting (for path selection).
            model_name (str): LLM model for agents.
            max_turns (int): Max turns for autonomous mode.
            json_file (str): Victim profiles JSON (for autonomous batch if no direct profiles).
        """
        self.settings = get_settings()
        self.logger = setup_logger("ConversationManager", self.settings.log.subdirectories["agent"])
        self.conversation_type = conversation_type
        self.mode = mode
        # self.history_csv_path = self._get_history_path()
        self.history_csv_path, self.id_file, self.rag_csv_path = self._get_paths()
        self.id_manager = IDManager(csv_file = self.history_csv_path, id_file=self.id_file)

        self.police_model_name = police_model_name
        self.police_llm_provider = police_llm_provider
        self.victim_model_name = victim_model_name
        self.victim_llm_provider = victim_llm_provider
        self.max_turns = max_turns
        self.json_file = json_file
        self.conversation_id: Optional[int] = None
        self.conversation_history: List[Dict] = []
        self.turn_count: int = 0
        self.police_chatbot: Optional[Union[RAGIEAgent, IEChatbot, ProfileRagIEAgent, ProfileRAGIEKBAgent]] = None
        self.victim_chatbot: Optional[VictimChatbot] = None
        self.profile_id: Optional[int] = None
        self.turn_number=0

        self._ensure_csv_headers()
    
    def _get_paths(self) -> Tuple[str, str, str]:
        """Get CSV history path, ID file path, and RAG invocation CSV path from settings based on mode/type."""
        if self.conversation_type == ConversationType.IE:
            if self.mode == ConversationMode.AUTONOMOUS:
                history_path = self.settings.data.conversation_history_ie_autonomous  
                id_file = self.settings.data.id_file_ie_autonomous
                rag_path = ""  
            else:
                history_path = self.settings.data.conversation_history_ie_nonautonomous
                id_file = self.settings.data.id_file_ie_nonautonomous
                rag_path = ""
        elif self.conversation_type == ConversationType.RAG_IE:
            if self.mode == ConversationMode.AUTONOMOUS:
                history_path = self.settings.data.conversation_history_rag_ie_autonomous  
                id_file = self.settings.data.id_file_rag_ie_autonomous
                rag_path = self.settings.data.rag_invocation_rag_ie_autonomous
            else:
                history_path = self.settings.data.conversation_history_rag_ie_nonautonomous
                id_file = self.settings.data.id_file_rag_ie_nonautonomous
                rag_path = self.settings.data.rag_invocation_rag_ie_nonautonomous
        elif self.conversation_type == ConversationType.PROFILE_RAG_IE:  
            if self.mode == ConversationMode.AUTONOMOUS:
                history_path = self.settings.data.conversation_history_profile_rag_ie_autonomous
                id_file = self.settings.data.id_file_profile_rag_ie_autonomous
                rag_path = self.settings.data.rag_invocation_profile_rag_ie_autonomous
            else:
                history_path = self.settings.data.conversation_history_profile_rag_ie_nonautonomous
                id_file = self.settings.data.id_file_profile_rag_ie_nonautonomous
                rag_path = self.settings.data.rag_invocation_profile_rag_ie_nonautonomous
        elif self.conversation_type == ConversationType.PROFILE_RAG_IE_KB:
            if self.mode == ConversationMode.AUTONOMOUS:
                history_path = self.settings.data.conversation_history_profile_rag_ie_kb_autonomous
                id_file = self.settings.data.id_file_profile_rag_ie_kb_autonomous
                rag_path = self.settings.data.rag_invocation_profile_rag_ie_kb_autonomous
            else:
                history_path = self.settings.data.conversation_history_profile_rag_ie_kb_nonautonomous
                id_file = self.settings.data.id_file_profile_rag_ie_kb_nonautonomous
                rag_path = self.settings.data.rag_invocation_profile_rag_ie_kb_nonautonomous
        else:
            raise ValueError(f"Invalid conversation_type: {self.conversation_type}")
        
        self.logger.debug(f"Resolved paths: history={history_path}, id_file={id_file}, rag={rag_path}")
        return history_path, id_file, rag_path
    
    def _ensure_csv_headers(self):
        """Ensure CSV has headers; create if not exists."""
        path = Path(self.history_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)  # Create directories if needed
        file_exists = path.exists()

        if not file_exists:
            with open(self.history_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "index", "conversation_id", "conversation_type", "sender_type", "content",
                    "timestamp","police_llm_model", "victim_llm_model", "turn_count", "profile_id", "scam_details", 
                    "rag_invoked", "rag_suggestions", "rag_upsert", 
                    "user_profile", "initial_profile",  "retrieved_strategies", "upserted_strategy", 
                     "communication_appropriateness_rating"
                ])
            self.logger.debug(f"Created CSV with headers: {self.history_csv_path}")

    def start_conversation(
        self,
        user_profile: Optional[Dict] = None,
        scam_details: Optional[Dict] = None,
        profile_id: Optional[int] = None #Optional manual profile_id for nonautonomous (or override autonomous)
    ) -> int:
        """Start a new conversation, initialize agents, return ID."""
        
        #Additional check to refresh state
        if self.conversation_id is not None:
            self.logger.debug(f"Ending existing conversation {self.conversation_id} before starting new one")
            self.force_end_conversation()
        
        self.conversation_id = self.id_manager.get_next_id()
        self.conversation_history = []
        self.turn_count = 0
        self.profile_id = profile_id
        
        if self.conversation_type == ConversationType.RAG_IE:
            self.police_chatbot = RAGIEAgent(model_name=self.police_model_name, llm_provider = self.police_llm_provider, rag_csv_path=self.rag_csv_path)
        elif self.conversation_type == ConversationType.PROFILE_RAG_IE_KB:
            self.police_chatbot = ProfileRAGIEKBAgent(model_name=self.police_model_name, llm_provider=self.police_llm_provider, rag_csv_path=self.rag_csv_path)
        elif self.conversation_type == ConversationType.PROFILE_RAG_IE: 
            self.police_chatbot = ProfileRagIEAgent(model_name=self.police_model_name, llm_provider=self.police_llm_provider, rag_csv_path=self.rag_csv_path)
        elif self.conversation_type == ConversationType.IE:
            self.police_chatbot = IEChatbot(model_name=self.police_model_name, llm_provider=self.police_llm_provider)  # No rag_csv_path needed
        else:
            raise ValueError(f"Unsupported conversation_type: {self.conversation_type}")
        

        if self.mode == ConversationMode.AUTONOMOUS:
            effective_victim_model = self.victim_model_name or self.police_model_name #in case victim model is not set, will default to same model as police model
            self.victim_chatbot = VictimChatbot(
                model_name=effective_victim_model,
                llm_provider=self.victim_llm_provider,
                json_file=self.json_file,
                user_profile=user_profile,
                scam_details=scam_details,
                profile_id=profile_id
            )
            # Override with victim's profile_id if available (for autonomous)
            if self.victim_chatbot is not None and self.victim_chatbot.profile_id is not None:
                self.profile_id = self.victim_chatbot.profile_id
        
        self.logger.info(f"Started conversation {self.conversation_id} in {self.mode.value} mode")
        return self.conversation_id

    def process_turn(self, input_query: str) -> Dict:
        """Process one turn: Handle input, get response, log, check end."""
        if self.conversation_id is None:
            raise ValueError("Conversation not started")

        try:
            if self.mode == ConversationMode.AUTONOMOUS:
                # Victim "input" is from previous police response; start with initial query if turn 0
                if self.turn_count == 0:
                    victim_response = self.victim_chatbot.process_query(input_query)  # Initial police query to victim
                else:
                    victim_response = self.victim_chatbot.process_query(input_query)  # input_query is police response
                
                # Log victim response FIRST, before checking end
                self._log_turn("victim", victim_response.conversational_response)
                
                police_response = self.police_chatbot.process_query(victim_response.conversational_response, conversation_id=self.conversation_id)
                self._log_turn("police", police_response["response"], structured_data=police_response["structured_data"], rag_invoked=police_response.get("rag_invoked", False), rag_suggestions=police_response.get("structured_data", {}).get("rag_suggestions", {}))
                self._save_partial_history()
                self.turn_count += 1
                
                # Now check if end
                if victim_response.end_conversation or self.turn_count >= self.max_turns:
                    self.end_conversation()
                    return {"end": True, "response": "Conversation ended", "history": self.conversation_history}
                

                return police_response
            else:  # Non-autonomous: Human input to police
                police_response = self.police_chatbot.process_query(input_query, conversation_id=self.conversation_id)
                self._log_turn("human", input_query)  # Log human as "victim" role for consistency
                self._log_turn("police", police_response["response"], structured_data=police_response["structured_data"], rag_invoked=police_response.get("rag_invoked", False), rag_suggestions=police_response.get("structured_data", {}).get("rag_suggestions", {}))
                self._save_partial_history()  
                self.turn_count += 1
                
                return police_response
        except Exception as e:
            self.logger.error(f"Turn processing error: {str(e)}", exc_info=True)
            self._log_turn("system_error", f"Error: {str(e)}")
            self._save_partial_history()  # Persist on error
            raise

    def _log_turn(self, role: str, content: str, structured_data: Optional[Dict] = None, rag_invoked: bool = False, rag_suggestions: Optional[list] = None):
        """Log a single turn to history list (saved later)."""
        timestamp = datetime.now().isoformat()
        entry = {
            "role": role,
            "content": content,
            "timestamp": timestamp
        }
        if role == "police":
            structured_data = {k: v for k, v in (structured_data or {}).items() if k != "conversational_response"}
            entry["structured_data"] = structured_data
            entry["rag_invoked"] = rag_invoked
            entry["rag_suggestions"] = rag_suggestions or {}
        self.conversation_history.append(entry)

    def _save_partial_history(self):
        """Save current history incrementally to CSV."""
        file_exists = Path(self.history_csv_path).exists()
        existing_entries = set()
        index_counter = 0

        if file_exists:
            with FileLock(f"{self.history_csv_path}.lock"):
                with open(self.history_csv_path, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if "index" in reader.fieldnames:
                        indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
                        index_counter = max(indices) if indices else 0
                    f.seek(0)
                    reader = csv.DictReader(f)
                    for row in reader:
                        key = (row["conversation_id"], row["sender_type"], row["content"], row["timestamp"])
                        existing_entries.add(key)

        with FileLock(f"{self.history_csv_path}.lock"):
            with open(self.history_csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for msg in self.conversation_history:
                    key = (str(self.conversation_id), msg["role"], msg["content"], msg["timestamp"])
                    if key not in existing_entries:
                        index_counter += 1
                        structured_data = msg.get("structured_data", {})
                        scam_slots = [slot.value for slot in PoliceResponseSlots]  # ['scam_incident_date', 'scam_type', ...]
                        filtered_structured_data = {
                            k: v for k, v in (structured_data or {}).items()
                            if k in scam_slots  # ONLY scam slots
}
                        
                        self.logger.debug(f"Logging row with profile_id: {self.profile_id}")
                        

                        row = [
                            str(index_counter),
                            str(self.conversation_id),
                            self.mode.value,
                            msg["role"],
                            msg["content"],
                            msg["timestamp"],
                            self.police_model_name,  
                            self.victim_model_name if self.victim_model_name else "",  
                            str(self.turn_count),
                            str(self.profile_id) if self.profile_id is not None else "",  
                            json.dumps(filtered_structured_data, ensure_ascii=False) if msg["role"] == "police" and structured_data else "",
                            str(msg.get("rag_invoked", False)).lower() if msg["role"] == "police" else "",
                            json.dumps(msg.get("rag_suggestions", {}), ensure_ascii=False) if msg["role"] == "police" else "", 
                            str(structured_data.get("rag_upsert", False)).lower() if msg["role"] == "police" else "",
                            json.dumps(structured_data.get("user_profile", {}), ensure_ascii=False) if msg["role"] == "police" else "",
                            json.dumps(structured_data.get("initial_profile", {}), ensure_ascii=False) if msg["role"] == "police" else "",  
                            json.dumps(structured_data.get("retrieved_strategies", []), ensure_ascii=False) if msg["role"] == "police" else "",  
                            json.dumps(structured_data.get("upserted_strategy", {}), ensure_ascii=False) if msg["role"] == "police" else "",  
                            "" # placeholder for communication_appropriateness_rating
                        ]
                        writer.writerow(row)
                        existing_entries.add(key)
                self.logger.debug(f"Partial history saved for conversation {self.conversation_id}")

    def end_conversation(self):
        """End and save full history, reset agents."""
        self._save_partial_history()  
        if self.police_chatbot:
            self.police_chatbot.reset_state()
        if self.victim_chatbot:
            self.victim_chatbot.reset_state()
        self.logger.info(f"Ended conversation {self.conversation_id}")
        
        old_conversation_id = self.conversation_id  # Optional, if you want to return it
        self.conversation_id = None
        self.conversation_history = []
        self.turn_count = 0
        self.police_chatbot = None
        self.victim_chatbot = None
        self.profile_id = None  
        
        return {"status": "Conversation ended"}
        
    def force_end_conversation(self) -> Dict:
        """
        Forcefully end the current conversation, save history, and reset for a new chat.
        Suitable for GUI 'New Chat' button.
        Returns:
            Dict: Status and conversation ID of the ended conversation.
        """
        if self.conversation_id is None:
            self.logger.warning("No active conversation to end")
            return {"status": "No active conversation", "conversation_id": None}
        
        self.logger.debug(f"Forcing end of conversation {self.conversation_id}")
        self._save_partial_history()  # Save current history
        if self.police_chatbot:
            self.police_chatbot.reset_state()
    
        if self.victim_chatbot:
            self.victim_chatbot.reset_state()
  
        self.logger.info(f"Ended conversation {self.conversation_id}")
        
        # Reset manager state for new conversation
        old_conversation_id = self.conversation_id
        self.conversation_id = None
        self.conversation_history = []
        self.turn_count = 0
        self.police_chatbot = None
        self.victim_chatbot = None
        
        return {"status": "Conversation ended", "conversation_id": old_conversation_id}

    def run_autonomous_simulation(self, initial_query: str, user_profile: Dict, scam_details: Dict, profile_id: Optional[int] = None) -> List[Dict]:
        if self.conversation_id is None: 
            self.start_conversation(user_profile, scam_details, profile_id=profile_id)
        history = []
        current_query = initial_query
        while self.turn_count < self.max_turns:
            response = self.process_turn(current_query)
            history.append(response)
            if response.get("end"):
                break
            current_query = response["response"]
        self.end_conversation()
        return history
    
    def batch_run_autonomous(
        self,
        n: int,
        initial_queries: List[str],
        police_models: List[Tuple[str, str]],
        victim_models: Optional[List[Tuple[str, str]]] = None,
        profiles: Optional[List[Dict]] = None
    ) -> Dict:
        """Batch run N autonomous simulations across all combinations of models, tracking inputs.
        
        Args:
            n (int): Number of simulations per model pair.
            initial_queries (List[str]): List of initial queries (cycled if fewer than n).
            police_models (List[str]): List of police model names.
            victim_models (Optional[List[str]]): List of victim model names (defaults to police_models).
            profiles (Optional[List[Dict]]): List of profile dicts with 'user_profile', 'victim_details', 'scam_details'.
                If None, loads from self.json_file and cycles via RecordCounter.

        Returns:
            Dict: Results keyed by 'police_model_victim_model', with list of simulations including history and record used.
        """
        if victim_models is None:
            victim_models = police_models  # Default to same as police

        results = {}
        record_counter = RecordCounter()
        
        # Load records if no profiles provided
        if profiles is None:
            records = json.loads(Path(self.json_file).read_text())
        else:
            records = profiles  # Use provided profiles
            record_counter = None  # No cycling needed if provided

        # Use itertools.product for all combinations (Cartesian product)
        for police_tuple, victim_tuple in itertools.product(police_models, victim_models or police_models):
            police_model, police_provider = police_tuple
            victim_model, victim_provider = victim_tuple
            self.police_model_name = police_model
            self.police_llm_provider = police_provider
            self.victim_model_name = victim_model
            self.victim_llm_provider = victim_provider
            key = f"{police_model}_{victim_model}"
            model_results = []
            for i in range(n):
                if profiles is None:
                    if self.conversation_id is not None:  
                        self.force_end_conversation()
                    record_index = record_counter.get_next_index(len(records))
                    record = records[record_index]
                else:
                    record = records[i % len(records)]  # Cycle provided profiles if fewer than n
                
                history = self.run_autonomous_simulation(
                    initial_queries[i % len(initial_queries)],  
                    record["user_profile"],
                    record["scam_details"],
                    profile_id=record.get("profile_id")
                )
                model_results.append({
                    "simulation": i + 1,
                    "history": history,
                    "record_used": record  # Track what was provided
                })
            results[key] = model_results
        return results
    
if __name__ == "__main__":
    # # Test: Non-autonomous mode for Vanilla RAG
    # manager_vanilla = ConversationManager(
    #     mode=ConversationMode.NONAUTONOMOUS,
    #     conversation_type=ConversationType.VANILLA_RAG,
    #     # Add your model params if needed, e.g., police_model_name="qwen2.5:7b", etc.
    # )
    # manager_vanilla.logger.debug(f"Starting non-autonomous vanilla RAG test, CSV path: {manager_vanilla.history_csv_path}")
    # try:
    #     # Start with a specific profile_id
    #     profile_id = 42  # Example ID to test
    #     manager_vanilla.start_conversation(profile_id=profile_id)
        
    #     resp1 = manager_vanilla.process_turn("I have just been scammed. Someone posed as an MOM officer and called me.")
    #     manager_vanilla.logger.debug(f"Turn 1 response: {resp1}")
    #     print("Non-Autonomous Vanilla Turn 1:", resp1)
        
    #     resp2 = manager_vanilla.process_turn("He said I was involved in illegal activities and asked me to transfer money to HSBC 123456789.")
    #     manager_vanilla.logger.debug(f"Turn 2 response: {resp2}")
    #     print("Non-Autonomous Vanilla Turn 2:", resp2)
        
    #     # Simulate GUI "New Chat" button
    #     manager_vanilla.force_end_conversation()
    #     manager_vanilla.logger.debug("Non-autonomous vanilla test completed, history saved")
        
    #     # Check if profile_id is recorded in CSV
    #     csv_path = manager_vanilla.history_csv_path
    #     import csv
    #     from pathlib import Path
    #     found = False
    #     if Path(csv_path).exists():
    #         with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
    #             reader = csv.DictReader(f)
    #             for row in reader:
    #                 if row.get("profile_id") == str(profile_id):
    #                     found = True
    #                     break
    #     print(f"Profile ID {profile_id} recorded in Vanilla CSV: {found}")
        
    #     # Start a new conversation to test reset
    #     manager_vanilla.start_conversation(profile_id=profile_id)
    #     resp3 = manager_vanilla.process_turn("I was involved in a phishing scam last week on 20 Jul 2025.")
    #     manager_vanilla.logger.debug(f"New conversation turn 1 response: {resp3}")
    #     print("New Conversation Vanilla Turn 1:", resp3)
    #     manager_vanilla.force_end_conversation()
        
    #     # Check again for the new conversation
    #     found_new = False
    #     with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
    #         reader = csv.DictReader(f)
    #         for row in reader:
    #             if row.get("profile_id") == str(profile_id):
    #                 found_new = True
    #                 break
    #     print(f"Profile ID {profile_id} recorded in new Vanilla conversation: {found_new}")
    # except Exception as e:
    #     manager_vanilla.logger.error(f"Test block error: {str(e)}", exc_info=True)
    #     manager_vanilla._save_partial_history()
    #     raise

    # # Test: Non-autonomous mode for Self-Augmenting RAG
    # manager_self = ConversationManager(
    #     mode=ConversationMode.NONAUTONOMOUS,
    #     conversation_type=ConversationType.SELF_AUGMENTING,
    #     # Add your model params if needed
    # )
    # manager_self.logger.debug(f"Starting non-autonomous self-augmenting test, CSV path: {manager_self.history_csv_path}")
    # try:
    #     # Start with the same profile_id
    #     profile_id = 42  # Example ID to test
    #     manager_self.start_conversation(profile_id=profile_id)
        
    #     resp1 = manager_self.process_turn("I have just been scammed. Someone posed as an MOM officer and called me.")
    #     manager_self.logger.debug(f"Turn 1 response: {resp1}")
    #     print("Non-Autonomous Self-Aug Turn 1:", resp1)
        
    #     resp2 = manager_self.process_turn("He said I was involved in illegal activities and asked me to transfer money to HSBC 123456789.")
    #     manager_self.logger.debug(f"Turn 2 response: {resp2}")
    #     print("Non-Autonomous Self-Aug Turn 2:", resp2)
        
    #     # Simulate GUI "New Chat" button
    #     manager_self.force_end_conversation()
    #     manager_self.logger.debug("Non-autonomous self-aug test completed, history saved")
        
    #     # Check if profile_id is recorded in CSV
    #     csv_path = manager_self.history_csv_path
    #     found = False
    #     if Path(csv_path).exists():
    #         with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
    #             reader = csv.DictReader(f)
    #             for row in reader:
    #                 if row.get("profile_id") == str(profile_id):
    #                     found = True
    #                     break
    #     print(f"Profile ID {profile_id} recorded in Self-Aug CSV: {found}")
        
    #     # Start a new conversation to test reset
    #     manager_self.start_conversation(profile_id=profile_id)
    #     resp3 = manager_self.process_turn("I was involved in a phishing scam last week on 20 Jul 2025.")
    #     manager_self.logger.debug(f"New conversation turn 1 response: {resp3}")
    #     print("New Conversation Self-Aug Turn 1:", resp3)
    #     manager_self.force_end_conversation()
        
    #     # Check again for the new conversation
    #     found_new = False
    #     with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
    #         reader = csv.DictReader(f)
    #         for row in reader:
    #             if row.get("profile_id") == str(profile_id):
    #                 found_new = True
    #                 break
    #     print(f"Profile ID {profile_id} recorded in new Self-Aug conversation: {found_new}")
    # except Exception as e:
    #     manager_self.logger.error(f"Test block error: {str(e)}", exc_info=True)
    #     manager_self._save_partial_history()
    #     raise

    
    # # Test: Autonomous mode for Baseline IE
    # manager = ConversationManager(
    #     mode=ConversationMode.AUTONOMOUS,
    #     conversation_type=ConversationType.IE,
    #     # police_model_name="qwen2.5:7b",
    #     # victim_model_name="qwen2.5:7b",
    #     max_turns=15,
    #     json_file="data/victim_profile/victim_details_human_eval.json"
    # )

    # #autonomous batch simulation test
    # try:
    #     # Load profiles from JSON for test 
    #     with open(manager.json_file, "r") as f:
    #         test_profiles = json.loads(f.read())[:]  
        
    #     police_models = [
    #             ("gpt-4o-mini", "OpenAI"),
    #             ("qwen2.5:7b", "Ollama"),
    #             ("granite3.2:8b", "Ollama"),
    #             ("mistral:7b", "Ollama")
    #         ] 

    #     victim_models = [("gpt-4o-mini", "OpenAI")]
    #     batch_results = manager.batch_run_autonomous(
    #         n=12,
    #         initial_queries=["Hi, I am a police AI assistant. How can I help you?"],
    #         police_models=police_models,
    #         victim_models=victim_models, 
    #         profiles=test_profiles  # Provide specific profiles to track
    #     )
    #     print("Batch Test Results:", json.dumps(batch_results, indent=2))
    # except Exception as e:
    #     manager.logger.error(f"Batch test error: {str(e)}", exc_info=True)
    #     raise


    # # Test: Autonomous mode
    # manager = ConversationManager(
    #     mode=ConversationMode.AUTONOMOUS,
    #     conversation_type=ConversationType.RAG_IE,
    #     # police_model_name="qwen2.5:7b",
    #     # victim_model_name="qwen2.5:7b",
    #     max_turns=15,
    #     json_file="data/victim_profile/victim_details_human_eval.json"
    # )

    # #autonomous batch simulation test
    # try:
    #     # Load profiles from JSON for test 
    #     with open(manager.json_file, "r") as f:
    #         test_profiles = json.loads(f.read())[:]  
        
    #     police_models = [
    #             ("qwen2.5:7b", "Ollama"),
    #             ("gpt-4o-mini", "OpenAI"),
    #             ("granite3.2:8b", "Ollama"),
    #             ("mistral:7b", "Ollama")
    #         ] 

    #     victim_models = [("gpt-4o-mini", "OpenAI")]
    #     batch_results = manager.batch_run_autonomous(
    #         n=12,
    #         initial_queries=["Hi, I am a police AI assistant. How can I help you?"],
    #         police_models=police_models,
    #         victim_models=victim_models, 
    #         profiles=test_profiles  # Provide specific profiles to track
    #     )
    #     print("Batch Test Results:", json.dumps(batch_results, indent=2))
    # except Exception as e:
    #     manager.logger.error(f"Batch test error: {str(e)}", exc_info=True)
    #     raise
    

    # #Test profile_rag_ie agent integration (with user profiling but no KB augmentation)
    # manager = ConversationManager(
    #     mode=ConversationMode.AUTONOMOUS,
    #     conversation_type=ConversationType.PROFILE_RAG_IE,  # Use the new type to test paths/agent
    #     # police_model_name="qwen2.5:7b",
    #     # victim_model_name="qwen2.5:7b",
    #     max_turns=15,
    #     json_file="data/victim_profile/victim_details_human_eval.json"
    # )
    
    # # Autonomous batch simulation test
    # try:
    #     # Load profiles from JSON for test 
    #     with open(manager.json_file, "r") as f:
    #         test_profiles = json.loads(f.read())[:]  
        
    #     # Define initial queries (expand for varied victim responses)
    #     queries = [
    #         "Hi, I am a police AI assistant. How can I help you?",
    #     ]
        
    #     police_models = [
    #             ("qwen2.5:7b", "Ollama"),
    #             ("gpt-4o-mini", "OpenAI"),
    #             ("granite3.2:8b", "Ollama"),
    #             ("mistral:7b", "Ollama")
    #         ] 

    #     victim_models = [("gpt-4o-mini", "OpenAI")]
        
    #     # Run batch simulation with Cartesian product
    #     batch_results = manager.batch_run_autonomous(
    #         n=12,  
    #         initial_queries=queries,
    #         police_models=police_models,
    #         victim_models=victim_models,
    #         profiles=test_profiles  # Provide specific profiles to track
    #     )
    #     print("Profile RAG IE Batch Results (Cartesian):", json.dumps(batch_results, indent=2))
    # except Exception as e:
    #     manager.logger.error(f"Batch test error for PROFILE_RAG_IE: {str(e)}", exc_info=True)
    #     raise
    
    
    # New: Test self-augmenting agent integration
    manager = ConversationManager(
        mode=ConversationMode.AUTONOMOUS,
        conversation_type=ConversationType.PROFILE_RAG_IE_KB,  # Ensure this is set to use the right paths/agent
        # police_model_name="qwen2.5:7b",
        # victim_model_name="qwen2.5:7b",
        max_turns=15,
        json_file="data/victim_profile/victim_details.json"
    )
    
    # Load test profiles (adjust slice for more variety)
    with open(manager.json_file, "r") as f:
        test_profiles = json.loads(f.read())[:]  

    # Define initial queries (expand for varied victim responses)
    queries = [
        "Hi, I am a police AI assistant. How can I help you?",
    ]
    
    police_models = [
            ("qwen2.5:7b", "Ollama"),
            ("gpt-4o-mini", "OpenAI"),
            ("granite3.2:8b", "Ollama"),
            ("mistral:7b", "Ollama")
        ] 

    victim_models = [("gpt-4o-mini", "OpenAI")] #change back to gpt-4o for dynamic rendering
    
    # Run batch simulation with Cartesian product
    batch_results = manager.batch_run_autonomous(
        n=12,  
        initial_queries=queries,
        police_models=police_models,
        victim_models=victim_models,
        profiles=test_profiles
    )

    # Output results
    print("Batch Results:", json.dumps(batch_results, indent=2))