import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..','..')))


from typing import List, Dict, Optional
from src.agents.remove.police_agent_good_draft_vanilla import PoliceChatbot
from src.agents.victim_agent.victim_agent import VictimChatbot
from config.id_manager import IDManager
from config.settings import get_settings
import logging
import csv
import os
from datetime import datetime
from pathlib import Path
from filelock import FileLock
from src.models.response_model import PoliceResponse
import json

class ConversationManager:
    def __init__(self):
        self.settings = get_settings()
        self.id_manager = IDManager()
        self._setup_logging()
        self.index_counter = self._load_last_index()

    def _setup_logging(self):
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

    def _save_to_csv(self, conversation_id: int, conversation_type: str, messages: List[Dict]):
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
                return

        with FileLock(f"{csv_file}.lock"):
            try:
                with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow([
                            "index", "conversation_id", "conversation_type", "sender_type", "content",
                            "timestamp", "llm_model", "scam_generic_details", "scam_specific_details"
                        ])
                    for msg in messages:
                        key = (str(conversation_id), msg["role"], msg["content"], msg["timestamp"])
                        if key not in existing_entries:
                            self.index_counter += 1
                            structured_data = msg.get("structured_data", {})
                            row = [
                                str(self.index_counter),
                                str(conversation_id),
                                conversation_type,
                                msg["role"],
                                msg["content"],
                                msg["timestamp"],
                                msg.get("llm_model", ""),
                                json.dumps(structured_data, ensure_ascii=False) if msg["role"] == "police" and structured_data else "",
                                ""
                            ]
                            writer.writerow(row)
                            existing_entries.add(key)
                            self.logger.debug(f"Wrote message to CSV: conversation_id={conversation_id}, sender={msg['role']}")
            except Exception as e:
                self.logger.error(f"Error writing to CSV: {str(e)}")

    def run_autonomous_conversation(
        self,
        police_model: str = "llama3.1:8b",
        victim_model: str = "llama3.1:8b",
        max_turns: int = 5,
        initial_query: str = "Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?"
    ) -> Dict:
        conversation_id = self.id_manager.get_next_id()
        conversation_history = []
        current_query = initial_query
        last_messages = []
        police_chatbot = PoliceChatbot(model_name=police_model)
        victim_chatbot = VictimChatbot(model_name=victim_model)

        try:
            for turn in range(max_turns):
                # Victim response
                victim_response = victim_chatbot.process_query(current_query, conversation_history)
                victim_message = victim_response["response"]
                conversation_history.append({
                    "role": "victim",
                    "content": victim_message,
                    "timestamp": datetime.now().isoformat(),
                    "llm_model": victim_model
                })
                self.logger.debug(f"Victim response: {victim_message}, conversation_id: {conversation_id}")

                if "[END_CONVERSATION]" in victim_message or "thank you for your cooperation" in victim_message.lower():
                    self.logger.info(f"Conversation ended by victim: {victim_message}")
                    break

                last_messages.append(victim_message)
                if len(last_messages) > 6:
                    last_messages.pop(0)
                    if len(set(last_messages)) <= 4:
                        self.logger.warning(f"Detected potential repetition: {last_messages}")
                        break

                # Police response
                police_response = police_chatbot.process_query(victim_message, conversation_id)
                police_message = police_response["response"]
                structured_data = police_response["structured_data"]
                conversation_history.append({
                    "role": "police",
                    "content": police_message,
                    "timestamp": datetime.now().isoformat(),
                    "llm_model": police_model,
                    "structured_data": {k: v for k, v in structured_data.items() if k != "conversational_response"}
                })
                self.logger.debug(f"Police response: {police_message}, conversation_id: {conversation_id}")

                last_messages.append(police_message)
                if len(last_messages) > 6:
                    last_messages.pop(0)
                    if len(set(last_messages)) <= 4:
                        self.logger.warning(f"Detected potential repetition: {last_messages}")
                        break

                current_query = police_message

                if "thank you for your cooperation" in police_message.lower():
                    self.logger.info("Police ended conversation")
                    break

            # Save the entire conversation to CSV
            self._save_to_csv(conversation_id, "autonomous", conversation_history)
            self.logger.debug(f"Autonomous conversation completed for conversation_id: {conversation_id}")
            return {
                "status": "Conversation completed",
                "conversation_id": conversation_id,
                "conversation_history": conversation_history,
                "conversation_type": "autonomous",
                "police_model": police_model,
                "victim_model": victim_model
            }

        except Exception as e:
            self.logger.error(f"Error in autonomous conversation for conversation_id {conversation_id}: {str(e)}")
            self._save_to_csv(conversation_id, "autonomous", conversation_history)
            return {"error": f"Failed to simulate conversation: {str(e)}"}

if __name__ == "__main__":
    manager = ConversationManager()
    result = manager.run_autonomous_conversation()
    print(json.dumps(result, indent=2))
    
    
