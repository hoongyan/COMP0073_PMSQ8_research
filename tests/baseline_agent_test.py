# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.agents.baseline.baseline_agent import create_police_chatbot, create_victim_chatbot, get_nonautonomous_response, get_autonomous_response
# import json

# # Test 1: Non-autonomous conversation
# print("=== Testing Non-Autonomous Conversation ===")
# result = create_police_chatbot(llm_provider="Ollama", model="qwen2.5:7b", max_turns=10)
# print(result)
# police_agent_id = result["agent_id"]
# print(f"Created police chatbot with agent_id: {police_agent_id}")

# response = get_nonautonomous_response(
#     agent_id=police_agent_id,
#     query="I received a suspicious email claiming to be from my bank.",
#     # prompt="Identify phishing patterns and extract details for a police report.",
#     conversation_id="conv_001",
#     conversation_history=[]
# )
# print("Non-Autonomous Response:", response["response"])
# print("Structured Data:", json.dumps(response["structured_data"], indent=2))
# print("Conversation History:", json.dumps(response["conversation_history"], indent=2))

# # Test 2: Autonomous conversation
# print("\n=== Testing Autonomous Conversation ===")
# victim_result = create_victim_chatbot(llm_provider="Ollama", model="qwen2.5:7b", max_turns=10)
# victim_agent_id = victim_result["agent_id"]
# print(f"Created victim chatbot with agent_id: {victim_agent_id}")

# autonomous_response = get_autonomous_response(
#     police_agent_id=police_agent_id,
#     victim_agent_id=victim_agent_id,
#     # police_prompt="Identify phishing patterns and extract details for a police report.",
#     # victim_prompt="Simulate a victim reporting a phishing scam with details.",
#     initial_query="Hello, this is the police. How can we help?",
#     conversation_id="conv_002",
#     max_turns=10
# )
# print("Autonomous Status:", autonomous_response["status"])
# print("Conversation History:", json.dumps(autonomous_response["conversation_history"], indent=2))
# print("Structured Data:", json.dumps(autonomous_response["structured_data"], indent=2))




# #working new version
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.agents.baseline.baseline_agent import PoliceAgent, VictimAgent, ConversationManager
# from src.agents.tools import PoliceTools
# import json
# import logging
# from pathlib import Path
# from config.settings import get_settings

# # Configure logging
# def setup_logging():
#     settings = get_settings()
#     log_dir = Path(settings.log.directory) / "tests"
#     log_dir.mkdir(parents=True, exist_ok=True)
#     log_file = log_dir / "test_chatbots.log"
    
#     logging.basicConfig(
#         level=logging.DEBUG,
#         format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#         handlers=[
#             logging.FileHandler(log_file, mode='a'),
#             logging.StreamHandler()
#         ]
#     )
#     return logging.getLogger(__name__)

# logger = setup_logging()

# def test_chatbot_creation():
#     logger.info("=== Testing Chatbot Creation ===")
#     print("=== Testing Chatbot Creation ===")
#     try:
#         police_agent = PoliceAgent(llm_provider="Ollama", model="qwen2.5:7b", max_turns=10)
#         logger.info(f"Created police chatbot with agent_id: {police_agent.agent_id}")
#         print(f"Created police chatbot with agent_id: {police_agent.agent_id}")
        
#         victim_agent = VictimAgent(llm_provider="Ollama", model="qwen2.5:7b", max_turns=10)
#         logger.info(f"Created victim chatbot with agent_id: {victim_agent.agent_id}")
#         print(f"Created victim chatbot with agent_id: {victim_agent.agent_id}")
        
#         return police_agent, victim_agent
#     except Exception as e:
#         logger.error(f"Error in chatbot creation: {str(e)}", exc_info=True)
#         print(f"Error in chatbot creation: {str(e)}")
#         return None, None

# def test_rag_tool():
#     logger.info("=== Testing RAG Tool ===")
#     print("=== Testing RAG Tool ===")
#     try:
#         police_tools = PoliceTools()
#         tools = police_tools.get_tools()
#         retrieve_scam_reports = tools[0]
#         result = retrieve_scam_reports.invoke("Facebook scam", top_k=5, conversation_id=0, llm_model="qwen2.5:7b")
#         logger.info(f"RAG tool test result: {result}")
#         print(f"RAG tool test result: {result}")
#     except Exception as e:
#         logger.error(f"Error in RAG tool test: {str(e)}", exc_info=True)
#         print(f"Error in RAG tool test: {str(e)}")

# def test_non_autonomous_response(police_agent, victim_agent):
#     logger.info("=== Testing Non-Autonomous Conversation ===")
#     print("=== Testing Non-Autonomous Conversation ===")
#     conversation_manager = ConversationManager()
#     query = "I received a suspicious email claiming to be from my bank."
    
#     # Test PoliceAgent
#     try:
#         response = conversation_manager.get_nonautonomous_response(
#             agent=police_agent,
#             query=query,
#             conversation_id=None,
#             conversation_history=[]
#         )
#         if "error" in response:
#             logger.error(f"Non-autonomous response failed for PoliceAgent: {response['error']}")
#             print(f"Non-autonomous response failed for PoliceAgent: {response['error']}")
#         else:
#             logger.info("Non-autonomous response succeeded for PoliceAgent")
#             print("PoliceAgent Response:", response["response"])
#             print("Structured Data:", json.dumps(response["structured_data"], indent=2))
#             print("Conversation History:", json.dumps(response["conversation_history"], indent=2))
#     except Exception as e:
#         logger.error(f"Error in non-autonomous test for PoliceAgent: {str(e)}", exc_info=True)
#         print(f"Error in non-autonomous test for PoliceAgent: {str(e)}")
    
#     # Test VictimAgent
#     try:
#         response = conversation_manager.get_nonautonomous_response(
#             agent=victim_agent,
#             query="Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?",
#             conversation_id=None,
#             conversation_history=[]
#         )
#         if "error" in response:
#             logger.error(f"Non-autonomous response failed for VictimAgent: {response['error']}")
#             print(f"Non-autonomous response failed for VictimAgent: {response['error']}")
#         else:
#             logger.info("Non-autonomous response succeeded for VictimAgent")
#             print("VictimAgent Response:", response["response"])
#             print("Conversation History:", json.dumps(response["conversation_history"], indent=2))
#     except Exception as e:
#         logger.error(f"Error in non-autonomous test for VictimAgent: {str(e)}", exc_info=True)
#         print(f"Error in non-autonomous test for VictimAgent: {str(e)}")

# def test_autonomous_conversation(police_agent, victim_agent):
#     logger.info("=== Testing Autonomous Conversation ===")
#     print("=== Testing Autonomous Conversation ===")
#     conversation_manager = ConversationManager()
#     try:
#         response = conversation_manager.get_autonomous_response(
#             police_agent=police_agent,
#             victim_agent=victim_agent,
#             initial_query="Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?",
#             conversation_id=None,
#             max_turns=10
#         )
#         if "error" in response:
#             logger.error(f"Autonomous simulation failed: {response['error']}")
#             print(f"Autonomous simulation failed: {response['error']}")
#         else:
#             logger.info("Autonomous simulation succeeded")
#             print("Autonomous Status:", response["status"])
#             print("Conversation History:", json.dumps(response["conversation_history"], indent=2))
#             print("Structured Data:", json.dumps(response["structured_data"], indent=2))
#     except Exception as e:
#         logger.error(f"Error in autonomous test: {str(e)}", exc_info=True)
#         print(f"Error in autonomous test: {str(e)}")

# if __name__ == "__main__":
#     # Ensure Ollama server is running
#     try:
#         import requests
#         response = requests.get("http://localhost:11434/api/tags")
#         if response.status_code != 200:
#             logger.error("Ollama server not responding")
#             print("Ollama server not responding")
#             sys.exit(1)
#         models = response.json().get("models", [])
#         if not any(m["name"] == "qwen2.5:7b" for m in models):
#             logger.error("Model qwen2.5:7b not loaded in Ollama server")
#             print("Model qwen2.5:7b not loaded in Ollama server")
#             sys.exit(1)
#     except Exception as e:
#         logger.error(f"Failed to connect to Ollama server: {str(e)}")
#         print(f"Failed to connect to Ollama server: {str(e)}")
#         sys.exit(1)
    
#     police_agent, victim_agent = test_chatbot_creation()
#     if police_agent and victim_agent:
#         test_rag_tool()
#         test_non_autonomous_response(police_agent, victim_agent)
#         test_autonomous_conversation(police_agent, victim_agent)



import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents.remove.baseline_agent import PoliceAgent, VictimAgent, ConversationManager
from src.agents.tools import PoliceTools
import json
import logging
from pathlib import Path
from config.settings import get_settings
import csv
from filelock import FileLock
import requests

def setup_logging():
    settings = get_settings()
    log_dir = Path(settings.log.directory) / "tests"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "test_chatbots.log"
    
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

def clear_csv(csv_file="conversation_history.csv"):
    """Clear the CSV file to avoid interference between tests."""
    with FileLock(f"{csv_file}.lock"):
        try:
            with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "index", "conversation_id", "conversation_type", "sender_type", "content", "timestamp",
                    "llm_model", "scam_generic_details", "scam_specific_details"
                ])
            logger.info(f"Cleared CSV file: {csv_file}")
        except Exception as e:
            logger.error(f"Failed to clear CSV file: {str(e)}", exc_info=True)
            raise

def count_csv_entries(csv_file="conversation_history.csv"):
    """Count the number of entries in the CSV, excluding the header."""
    with FileLock(f"{csv_file}.lock"):
        try:
            if not os.path.exists(csv_file):
                return 0
            with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                return sum(1 for row in reader)
        except Exception as e:
            logger.error(f"Failed to count CSV entries: {str(e)}", exc_info=True)
            raise

def count_rag_invocations(csv_file="rag_invocations.csv"):
    """Count the number of entries in the RAG invocations CSV, excluding the header."""
    with FileLock(f"{csv_file}.lock"):
        try:
            if not os.path.exists(csv_file):
                return 0
            with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                return sum(1 for row in reader)
        except Exception as e:
            logger.error(f"Failed to count RAG invocations CSV entries: {str(e)}", exc_info=True)
            raise

def test_chatbot_creation():
    logger.info("=== Testing Chatbot Creation ===")
    print("=== Testing Chatbot Creation ===")
    try:
        police_agent = PoliceAgent(llm_provider="Ollama", model="llama3.1:8b", max_turns=10)
        logger.info(f"Created police chatbot with agent_id: {police_agent.agent_id}")
        print(f"Created police chatbot with agent_id: {police_agent.agent_id}")
        
        victim_agent = VictimAgent(llm_provider="Ollama", model="llama3.1:8b", max_turns=10)
        logger.info(f"Created victim chatbot with agent_id: {victim_agent.agent_id}")
        print(f"Created victim chatbot with agent_id: {victim_agent.agent_id}")
        
        return police_agent, victim_agent
    except Exception as e:
        logger.error(f"Error in chatbot creation: {str(e)}", exc_info=True)
        print(f"Error in chatbot creation: {str(e)}")
        return None, None

def test_rag_tool():
    logger.info("=== Testing RAG Tool ===")
    print("=== Testing RAG Tool ===")
    csv_file = "rag_invocations.csv"
    try:
        initial_count = count_rag_invocations(csv_file)
        police_tools = PoliceTools()
        tools = police_tools.get_tools()
        retrieve_scam_reports = tools[0]
        result = retrieve_scam_reports.invoke({
            "query": "Facebook scam",
            "top_k": 5,
            "conversation_id": 0,
            "llm_model": "llama3.1:8b"
        })
        final_count = count_rag_invocations(csv_file)
        
        logger.info(f"RAG tool test result: {result}")
        print(f"RAG tool test result: {result}")
        if final_count != initial_count + 1:
            logger.error(f"RAG invocation not logged: initial_count={initial_count}, final_count={final_count}")
            raise AssertionError(f"Expected 1 new entry in {csv_file}, got {final_count - initial_count}")
        logger.info(f"RAG invocations CSV updated: {final_count} entries")
        print(f"RAG invocations CSV updated: {final_count} entries")
    except Exception as e:
        logger.error(f"Error in RAG tool test: {str(e)}", exc_info=True)
        print(f"Error in RAG tool test: {str(e)}")
        raise

def test_non_autonomous_response(police_agent, victim_agent):
    logger.info("=== Testing Non-Autonomous Conversation ===")
    print("=== Testing Non-Autonomous Conversation ===")
    conversation_manager = ConversationManager()
    csv_file = "conversation_history.csv"
    try:
        clear_csv(csv_file)
    except Exception as e:
        logger.error(f"Failed to clear CSV: {str(e)}", exc_info=True)
        print(f"Failed to clear CSV: {str(e)}")
        return
    
    # Test PoliceAgent with query to trigger RAG
    query = "Search for scam reports related to Facebook phishing attempts."
    try:
        initial_count = count_csv_entries(csv_file)
        rag_initial_count = count_rag_invocations("rag_invocations.csv")
        response = conversation_manager.get_nonautonomous_response(
            agent=police_agent,
            query=query,
            conversation_id=None,
            conversation_history=[]
        )
        final_count = count_csv_entries(csv_file)
        rag_final_count = count_rag_invocations("rag_invocations.csv")
        
        if "error" in response:
            logger.error(f"Non-autonomous response failed for PoliceAgent: {response['error']}")
            print(f"Non-autonomous response failed for PoliceAgent: {response['error']}")
        else:
            logger.info("Non-autonomous response succeeded for PoliceAgent")
            print("PoliceAgent Response:", response["response"])
            print("Structured Data:", json.dumps(response["structured_data"], indent=2))
            print("Conversation History:", json.dumps(response["conversation_history"], indent=2))
            assert len(response["conversation_history"]) == 2, f"Expected 2 messages in history, got {len(response['conversation_history'])}"
            assert final_count == initial_count + 2, f"Expected 2 new CSV entries, got {final_count - initial_count}"
            if response["structured_data"].get("rag_invoked", False):
                assert rag_final_count > rag_initial_count, f"Expected RAG invocation to be logged, got {rag_final_count - rag_initial_count} entries"
            else:
                logger.warning("RAG tool not invoked for PoliceAgent response")
    except Exception as e:
        logger.error(f"Error in non-autonomous test for PoliceAgent: {str(e)}", exc_info=True)
        print(f"Error in non-autonomous test for PoliceAgent: {str(e)}")
    
    # Test VictimAgent
    try:
        initial_count = count_csv_entries(csv_file)
        response = conversation_manager.get_nonautonomous_response(
            agent=victim_agent,
            query="Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?",
            conversation_id=None,
            conversation_history=[]
        )
        final_count = count_csv_entries(csv_file)
        
        if "error" in response:
            logger.error(f"Non-autonomous response failed for VictimAgent: {response['error']}")
            print(f"Non-autonomous response failed for VictimAgent: {response['error']}")
        else:
            logger.info("Non-autonomous response succeeded for VictimAgent")
            print("VictimAgent Response:", response["response"])
            print("Conversation History:", json.dumps(response["conversation_history"], indent=2))
            assert len(response["conversation_history"]) == 2, f"Expected 2 messages in history, got {len(response['conversation_history'])}"
            assert final_count == initial_count + 2, f"Expected 2 new CSV entries, got {final_count - initial_count}"
    except Exception as e:
        logger.error(f"Error in non-autonomous test for VictimAgent: {str(e)}", exc_info=True)
        print(f"Error in non-autonomous test for VictimAgent: {str(e)}")

def test_autonomous_conversation(police_agent, victim_agent):
    logger.info("=== Testing Autonomous Conversation ===")
    print("=== Testing Autonomous Conversation ===")
    conversation_manager = ConversationManager()
    csv_file = "conversation_history.csv"
    
    try:
        initial_count = count_csv_entries(csv_file)
        rag_initial_count = count_rag_invocations("rag_invocations.csv")
        response = conversation_manager.get_autonomous_response(
            police_agent=police_agent,
            victim_agent=victim_agent,
            initial_query="Search for scam reports related to Facebook phishing attempts.",
            conversation_id=None,
            max_turns=3
        )
        final_count = count_csv_entries(csv_file)
        rag_final_count = count_rag_invocations("rag_invocations.csv")
        
        if "error" in response:
            logger.error(f"Autonomous simulation failed: {response['error']}")
            print(f"Autonomous simulation failed: {response['error']}")
        else:
            logger.info("Autonomous simulation succeeded")
            print("Autonomous Status:", response["status"])
            print("Conversation History:", json.dumps(response["conversation_history"], indent=2))
            print("Structured Data:", json.dumps(response["structured_data"], indent=2))
            history_length = len(response["conversation_history"])
            assert history_length <= 6, f"Expected up to 6 messages (3 turns), got {history_length}"
            assert final_count >= initial_count + history_length, f"Expected at least {history_length} new CSV entries, got {final_count - initial_count}"
            with FileLock(f"{csv_file}.lock"):
                with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    entries = {(row["conversation_id"], row["sender_type"], row["content"], row["timestamp"]) for row in reader}
                    assert len(entries) == final_count, f"Found duplicates in CSV: {final_count} entries but {len(entries)} unique"
            if any("rag_invoked" in msg.get("structured_data", {}) and msg["structured_data"]["rag_invoked"] for msg in response["conversation_history"] if msg["role"] == "police"):
                assert rag_final_count > rag_initial_count, f"Expected RAG invocation to be logged, got {rag_final_count - rag_initial_count} entries"
            else:
                logger.warning("RAG tool not invoked during autonomous conversation")
    except Exception as e:
        logger.error(f"Error in autonomous test: {str(e)}", exc_info=True)
        print(f"Error in autonomous test: {str(e)}")

if __name__ == "__main__":
    # try:
    #     response = requests.get("http://localhost:11434/api/tags")
    #     if response.status_code != 200:
    #         logger.error("Ollama server not responding")
    #         print("Ollama server not responding")
    #         sys.exit(1)
    #     models = response.json().get("models", [])
    #     if not any(m["name"] == "qwen2.5:7b" for m in models):
    #         logger.error("Model qwen2.5:7b not loaded in Ollama server")
    #         print("Model qwen2.5:7b not loaded in Ollama server")
    #         sys.exit(1)
    # except Exception as e:
    #     logger.error(f"Failed to connect to Ollama server: {str(e)}")
    #     print(f"Failed to connect to Ollama server: {str(e)}")
    #     sys.exit(1)
    
    clear_csv()
    police_agent, victim_agent = test_chatbot_creation()
    if police_agent and victim_agent:
        test_rag_tool()
        test_non_autonomous_response(police_agent, victim_agent)
        test_autonomous_conversation(police_agent, victim_agent)