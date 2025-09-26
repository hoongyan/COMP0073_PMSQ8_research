import unittest
import os
import sys
import json
import csv
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents.remove.baseline_agent import PoliceAgent, VictimAgent, ConversationManager
from src.models.response_model import PoliceResponse
from config.id_manager import IDManager
from config.settings import get_settings
import logging

# Configure logging for tests
def setup_logging():
    settings = get_settings()
    log_dir = Path(settings.log.directory) / settings.log.subdirectories["tests"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "baseline_test.log"
    
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode='a'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

class TestBaselineAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        self.conversation_manager = ConversationManager()
        self.police_agent = None
        self.victim_agent = None
        self.id_manager = IDManager()
        self.conversation_id = self.id_manager.get_next_id()
        self.csv_file = "conversation_history.csv"
        self.id_file = "last_conversation_id.txt"
        self.models = ["llama3.2", "qwen2.5:7b", "mistral:7b"]
        
        # Clean up files before tests
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)
        if os.path.exists(self.id_file):
            os.remove(self.id_file)
    
    def tearDown(self):
        """Clean up after tests."""
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)
        if os.path.exists(self.id_file):
            os.remove(self.id_file)
    
    def test_create_police_agent(self):
        """Test PoliceAgent creation for multiple models."""
        for model in self.models:
            with self.subTest(model=model):
                logger.debug(f"Testing PoliceAgent creation with model {model}")
                try:
                    self.police_agent = PoliceAgent(llm_provider="Ollama", model=model, max_turns=10)
                    self.assertIsNotNone(self.police_agent.agent, f"Police agent creation failed for model {model}")
                    self.assertEqual(self.police_agent.agent_id, "police")
                    self.assertEqual(self.police_agent.model, model)
                    logger.info(f"PoliceAgent created successfully with model {model}")
                except Exception as e:
                    logger.error(f"Failed to create PoliceAgent with model {model}: {str(e)}")
                    self.fail(f"Failed to create PoliceAgent with model {model}: {str(e)}")
    
    def test_create_victim_agent(self):
        """Test VictimAgent creation for multiple models."""
        for model in self.models:
            with self.subTest(model=model):
                logger.debug(f"Testing VictimAgent creation with model {model}")
                try:
                    self.victim_agent = VictimAgent(llm_provider="Ollama", model=model, max_turns=10)
                    self.assertIsNotNone(self.victim_agent.agent, f"Victim agent creation failed for model {model}")
                    self.assertEqual(self.victim_agent.agent_id, "victim")
                    self.assertEqual(self.victim_agent.model, model)
                    logger.info(f"VictimAgent created successfully with model {model}")
                except Exception as e:
                    logger.error(f"Failed to create VictimAgent with model {model}: {str(e)}")
                    self.fail(f"Failed to create VictimAgent with model {model}: {str(e)}")
    
    def test_non_autonomous_response(self):
        """Test non-autonomous response generation for multiple models."""
        for model in self.models:
            with self.subTest(model=model):
                logger.debug(f"Testing non-autonomous response with model {model}")
                self.police_agent = PoliceAgent(llm_provider="Ollama", model=model, max_turns=10)
                query = "I received a suspicious email claiming to be from my bank."
                
                response = self.conversation_manager.get_nonautonomous_response(
                    agent=self.police_agent,
                    query=query,
                    conversation_id=self.conversation_id,
                    conversation_history=[]
                )
                
                self.assertNotIn("error", response, f"Non-autonomous response failed for model {model}: {response.get('error')}")
                self.assertTrue(response["response"], "Response content is empty")
                self.assertEqual(response["conversation_id"], self.conversation_id)
                self.assertEqual(len(response["conversation_history"]), 2)
                self.assertEqual(response["conversation_type"], "non_autonomous")
                
                # Verify CSV output
                self.assertTrue(os.path.exists(self.csv_file), "Conversation CSV not created")
                with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    self.assertGreaterEqual(len(rows), 2, "Conversation not saved to CSV")
                    self.assertEqual(int(rows[0]["conversation_id"]), self.conversation_id)
                    self.assertEqual(rows[0]["conversation_type"], "non_autonomous")
                    self.assertEqual(rows[0]["sender_type"], "victim")
                    self.assertEqual(rows[0]["police_model"], model)
                    self.assertEqual(rows[0]["victim_model"], "")
                    self.assertEqual(rows[1]["sender_type"], "police")
                    self.assertEqual(rows[1]["police_model"], model)
                    self.assertEqual(rows[1]["victim_model"], "")
                
                # Verify ID persistence
                self.assertTrue(os.path.exists(self.id_file), "ID file not created")
                with open(self.id_file, "r") as f:
                    last_id = int(f.read().strip())
                    self.assertEqual(last_id, self.conversation_id)
                
                logger.info(f"Non-autonomous response test passed for model {model}")
                self.conversation_id = self.id_manager.get_next_id()  # Increment for next subtest
    
    def test_autonomous_response(self):
        """Test autonomous conversation simulation for multiple models."""
        for model in self.models:
            with self.subTest(model=model):
                logger.debug(f"Testing autonomous response with model {model}")
                self.police_agent = PoliceAgent(llm_provider="Ollama", model=model, max_turns=10)
                self.victim_agent = VictimAgent(llm_provider="Ollama", model=model, max_turns=10)
                
                response = self.conversation_manager.get_autonomous_response(
                    police_agent=self.police_agent,
                    victim_agent=self.victim_agent,
                    initial_query="Hello, this is the police. How can we help?",
                    conversation_id=self.conversation_id,
                    max_turns=5
                )
                
                self.assertNotIn("error", response, f"Autonomous response failed for model {model}: {response.get('error')}")
                self.assertEqual(response["conversation_id"], self.conversation_id)
                self.assertEqual(response["conversation_type"], "autonomous")
                self.assertTrue(response["conversation_history"], "Conversation history is empty")
                self.assertEqual(response["status"], "Conversation completed")
                self.assertEqual(response["police_model"], model)
                self.assertEqual(response["victim_model"], model)
                
                # Verify CSV output
                self.assertTrue(os.path.exists(self.csv_file), "Conversation CSV not created")
                with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    self.assertGreaterEqual(len(rows), 2, "Conversation not saved to CSV")
                    self.assertEqual(int(rows[0]["conversation_id"]), self.conversation_id)
                    self.assertEqual(rows[0]["conversation_type"], "autonomous")
                    self.assertEqual(rows[0]["police_model"], model)
                    self.assertEqual(rows[0]["victim_model"], model)
                
                # Verify ID persistence
                self.assertTrue(os.path.exists(self.id_file), "ID file not created")
                with open(self.id_file, "r") as f:
                    last_id = int(f.read().strip())
                    self.assertEqual(last_id, self.conversation_id)
                
                logger.info(f"Autonomous response test passed for model {model}")
                self.conversation_id = self.id_manager.get_next_id()  # Increment for next subtest

if __name__ == "__main__":
    unittest.main()