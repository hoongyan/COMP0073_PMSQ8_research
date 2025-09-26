# import sys
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.agents.baseline.baseline_agent_langgraph import PoliceAgent, VictimAgent, ConversationManager
# from src.agents.tools2 import PoliceTools
# import json
# import logging
# from pathlib import Path
# from config.settings import get_settings
# import csv
# from filelock import FileLock
# import requests
# from pydantic import ValidationError

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

# def clear_csv(csv_file="conversation_history.csv"):
#     """Clear the CSV file to avoid interference between tests."""
#     with FileLock(f"{csv_file}.lock"):
#         try:
#             with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([
#                     "index", "conversation_id", "conversation_type", "sender_type", 
#                     "content", "timestamp", "llm_model", "scam_generic_details"
#                 ])
#             logger.info(f"Cleared CSV file: {csv_file}")
#         except Exception as e:
#             logger.error(f"Failed to clear CSV file: {str(e)}", exc_info=True)
#             raise

# def count_csv_entries(csv_file="conversation_history.csv"):
#     """Count the number of entries in the CSV, excluding the header."""
#     with FileLock(f"{csv_file}.lock"):
#         try:
#             if not os.path.exists(csv_file):
#                 return 0
#             with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                 reader = csv.reader(f)
#                 next(reader)  # Skip header
#                 return sum(1 for row in reader)
#         except Exception as e:
#             logger.error(f"Failed to count CSV entries: {str(e)}", exc_info=True)
#             raise

# def count_rag_invocations(csv_file="rag_invocations.csv"):
#     """Count the number of entries in the RAG invocations CSV, excluding the header."""
#     with FileLock(f"{csv_file}.lock"):
#         try:
#             if not os.path.exists(csv_file):
#                 return 0
#             with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                 reader = csv.reader(f)
#                 next(reader)  # Skip header
#                 return sum(1 for row in reader)
#         except Exception as e:
#             logger.error(f"Failed to count RAG invocations CSV entries: {str(e)}", exc_info=True)
#             raise

# def test_chatbot_creation():
#     logger.info("=== Testing Chatbot Creation ===")
#     print("=== Testing Chatbot Creation ===")
#     llm_configs = [
#         {"provider": "OpenAI", "model": "gpt-4o-mini"},
#         {"provider": "Ollama", "model": "mistral:7b"},
#         {"provider": "Ollama", "model": "qwen2.5:7b"}
#     ]
#     agents_dict = {}
    
#     for config in llm_configs:
#         provider, model = config["provider"], config["model"]
#         try:
#             police_agent = PoliceAgent(llm_provider=provider, model=model, max_turns=10)
#             logger.info(f"Created police chatbot ({provider}/{model}) with agent_id: {police_agent.agent_id}")
#             print(f"Created police chatbot ({provider}/{model}) with agent_id: {police_agent.agent_id}")
            
#             victim_agent = VictimAgent(llm_provider=provider, model=model, max_turns=10)
#             logger.info(f"Created victim chatbot ({provider}/{model}) with agent_id: {victim_agent.agent_id}")
#             print(f"Created victim chatbot ({provider}/{model}) with agent_id: {victim_agent.agent_id}")
            
#             agents_dict[f"{provider}/{model}"] = (police_agent, victim_agent)
#         except Exception as e:
#             logger.error(f"Error in chatbot creation for {provider}/{model}: {str(e)}", exc_info=True)
#             print(f"Error in chatbot creation for {provider}/{model}: {str(e)}")
    
#     return agents_dict if agents_dict else None

# def test_rag_tool():
#     logger.info("=== Testing RAG Tool ===")
#     print("=== Testing RAG Tool ===")
#     csv_file = "rag_invocations.csv"
#     llm_configs = [
#         {"provider": "OpenAI", "model": "gpt-4o-mini"},
#         {"provider": "Ollama", "model": "mistral:7b"},
#         {"provider": "Ollama", "model": "qwen2.5:7b"}
#     ]
    
#     for config in llm_configs:
#         provider, model = config["provider"], config["model"]
#         try:
#             initial_count = count_rag_invocations(csv_file)
#             police_tools = PoliceTools()
#             tools = police_tools.get_tools()
#             retrieve_scam_reports = tools[0]
#             result = retrieve_scam_reports.invoke({
#                 "query": "Facebook scam",
#                 "top_k": 5,
#                 "conversation_id": 0,
#                 "llm_model": model
#             })
#             final_count = count_rag_invocations(csv_file)
            
#             logger.info(f"RAG tool test result for {provider}/{model}: {result}")
#             print(f"RAG tool test result for {provider}/{model}: {result}")
#             assert final_count == initial_count + 1, f"Expected 1 new entry in {csv_file}, got {final_count - initial_count} for {provider}/{model}"
#             logger.info(f"RAG invocations CSV updated for {provider}/{model}: {final_count} entries")
#             print(f"RAG invocations CSV updated for {provider}/{model}: {final_count} entries")
#         except Exception as e:
#             logger.error(f"Error in RAG tool test for {provider}/{model}: {str(e)}", exc_info=True)
#             print(f"Error in RAG tool test for {provider}/{model}: {str(e)}")
#             raise

# def test_non_autonomous_response(agents_dict):
#     logger.info("=== Testing Non-Autonomous Conversation ===")
#     print("=== Testing Non-Autonomous Conversation ===")
#     conversation_manager = ConversationManager()
#     csv_file = "conversation_history.csv"
#     try:
#         clear_csv(csv_file)
#     except Exception as e:
#         logger.error(f"Failed to clear CSV: {str(e)}", exc_info=True)
#         print(f"Failed to clear CSV: {str(e)}")
#         return
    
#     queries = [
#         "Search for scam reports related to Facebook phishing attempts.",
#         "I received a suspicious email claiming to be from my bank."
#     ]
    
#     for provider_model, (police_agent, victim_agent) in agents_dict.items():
#         logger.info(f"Testing {provider_model} provider")
#         print(f"Testing {provider_model} provider")
        
#         for query in queries:
#             try:
#                 initial_count = count_csv_entries(csv_file)
#                 rag_initial_count = count_rag_invocations("rag_invocations.csv")
#                 response = conversation_manager.get_nonautonomous_response(
#                     agent=police_agent,
#                     query=query,
#                     conversation_id=None,
#                     conversation_history=[]
#                 )
#                 final_count = count_csv_entries(csv_file)
#                 rag_final_count = count_rag_invocations("rag_invocations.csv")
                
#                 if "error" in response:
#                     logger.error(f"Non-autonomous response failed for PoliceAgent ({provider_model}): {response['error']}")
#                     print(f"Non-autonomous response failed for PoliceAgent ({provider_model}): {response['error']}")
#                 else:
#                     logger.info(f"Non-autonomous response succeeded for PoliceAgent ({provider_model})")
#                     print(f"PoliceAgent Response ({provider_model}):", response["response"])
#                     print(f"Structured Data ({provider_model}):", json.dumps(response["structured_data"], indent=2))
#                     print(f"Conversation History ({provider_model}):", json.dumps(response["conversation_history"], indent=2))
                    
#                     assert response["structured_data"].get("rag_invoked", False), f"RAG tool not invoked for PoliceAgent ({provider_model}) with query: {query}"
#                     assert rag_final_count > rag_initial_count, f"Expected RAG invocation to be logged, got {rag_final_count - rag_initial_count} entries"
#                     assert "scam_type" in response["structured_data"], f"Structured data missing scam_type for {provider_model}"
#                     assert len(response["conversation_history"]) == 2, f"Expected 2 messages in history, got {len(response['conversation_history'])}"
#                     assert final_count == initial_count + 2, f"Expected 2 new CSV entries, got {final_count - initial_count}"
#                     assert "details" in response["response"].lower(), f"Response not conversational for {provider_model}: {response['response']}"
                    
#                     with FileLock(f"{csv_file}.lock"):
#                         with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                             reader = csv.DictReader(f)
#                             assert "scam_specific_details" not in reader.fieldnames, "scam_specific_details column should not exist in CSV"
#                             assert "scam_generic_details" in reader.fieldnames, "scam_generic_details column missing in CSV"
#                             for row in reader:
#                                 if row["sender_type"] == "police" and row["scam_generic_details"]:
#                                     assert json.loads(row["scam_generic_details"]).get("rag_invoked", False), f"RAG invocation not logged in CSV for {provider_model}"
#             except Exception as e:
#                 logger.error(f"Error in non-autonomous test for PoliceAgent ({provider_model}): {str(e)}", exc_info=True)
#                 print(f"Error in non-autonomous test for PoliceAgent ({provider_model}): {str(e)}")
        
#         try:
#             initial_count = count_csv_entries(csv_file)
#             response = conversation_manager.get_nonautonomous_response(
#                 agent=victim_agent,
#                 query="Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?",
#                 conversation_id=None,
#                 conversation_history=[]
#             )
#             final_count = count_csv_entries(csv_file)
            
#             if "error" in response:
#                 logger.error(f"Non-autonomous response failed for VictimAgent ({provider_model}): {response['error']}")
#                 print(f"Non-autonomous response failed for VictimAgent ({provider_model}): {response['error']}")
#             else:
#                 logger.info(f"Non-autonomous response succeeded for VictimAgent ({provider_model})")
#                 print(f"VictimAgent Response ({provider_model}):", response["response"])
#                 print(f"Conversation History ({provider_model}):", json.dumps(response["conversation_history"], indent=2))
#                 assert len(response["conversation_history"]) == 2, f"Expected 2 messages in history, got {len(response['conversation_history'])}"
#                 assert final_count == initial_count + 2, f"Expected 2 new CSV entries, got {final_count - initial_count}"
#                 assert not response["structured_data"], f"VictimAgent ({provider_model}) should not return structured data"
#                 assert any(word in response["response"].lower() for word in ["scam", "upset", "facebook", "email"]), f"Victim response not conversational for {provider_model}: {response['response']}"
#         except Exception as e:
#             logger.error(f"Error in non-autonomous test for VictimAgent ({provider_model}): {str(e)}", exc_info=True)
#             print(f"Error in non-autonomous test for VictimAgent ({provider_model}): {str(e)}")

# def test_autonomous_conversation(agents_dict):
#     logger.info("=== Testing Autonomous Conversation ===")
#     print("=== Testing Autonomous Conversation ===")
#     conversation_manager = ConversationManager()
#     csv_file = "conversation_history.csv"
    
#     for provider_model, (police_agent, victim_agent) in agents_dict.items():
#         logger.info(f"Testing autonomous conversation for {provider_model} provider")
#         print(f"Testing autonomous conversation for {provider_model} provider")
        
#         try:
#             initial_count = count_csv_entries(csv_file)
#             rag_initial_count = count_rag_invocations("rag_invocations.csv")
#             response = conversation_manager.get_autonomous_response(
#                 police_agent=police_agent,
#                 victim_agent=victim_agent,
#                 initial_query="Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?",
#                 conversation_id=None,
#                 max_turns=3
#             )
#             final_count = count_csv_entries(csv_file)
#             rag_final_count = count_rag_invocations("rag_invocations.csv")
            
#             if "error" in response:
#                 logger.error(f"Autonomous simulation failed for {provider_model}: {response['error']}")
#                 print(f"Autonomous simulation failed for {provider_model}: {response['error']}")
#             else:
#                 logger.info(f"Autonomous simulation succeeded for {provider_model}")
#                 print(f"Autonomous Status ({provider_model}):", response["status"])
#                 print(f"Conversation History ({provider_model}):", json.dumps(response["conversation_history"], indent=2))
#                 print(f"Structured Data ({provider_model}):", json.dumps(response["structured_data"], indent=2))
                
#                 history_length = len(response["conversation_history"])
#                 assert history_length <= 6, f"Expected up to 6 messages (3 turns), got {history_length} for {provider_model}"
#                 assert final_count >= initial_count + history_length, f"Expected at least {history_length} new CSV entries, got {final_count - initial_count} for {provider_model}"
                
#                 police_messages = [msg for msg in response["conversation_history"] if msg["role"] == "police"]
#                 for msg in police_messages:
#                     assert msg.get("structured_data", {}).get("rag_invoked", False), f"RAG tool not invoked for police message in {provider_model}: {msg['content']}"
#                     assert "details" in msg["content"].lower(), f"Police response not conversational for {provider_model}: {msg['content']}"
                
#                 with FileLock(f"{csv_file}.lock"):
#                     with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                         reader = csv.DictReader(f)
#                         assert "scam_specific_details" not in reader.fieldnames, "scam_specific_details column should not exist in CSV"
#                         assert "scam_generic_details" in reader.fieldnames, "scam_generic_details column missing in CSV"
#                         entries = {(row["conversation_id"], row["sender_type"], row["content"], row["timestamp"]) for row in reader}
#                         assert len(entries) == final_count, f"Found duplicates in CSV: {final_count} entries but {len(entries)} unique for {provider_model}"
                
#                 assert rag_final_count >= rag_initial_count + len(police_messages), f"Expected at least {len(police_messages)} RAG invocations, got {rag_final_count - rag_initial_count} for {provider_model}"
#         except Exception as e:
#             logger.error(f"Error in autonomous test for {provider_model}: {str(e)}", exc_info=True)
#             print(f"Error in autonomous test for {provider_model}: {str(e)}")

# if __name__ == "__main__":
#     clear_csv()
#     agents_dict = test_chatbot_creation()
#     if agents_dict:
#         test_rag_tool()
#         test_non_autonomous_response(agents_dict)
#         # test_autonomous_conversation(agents_dict)



# import sys
# import os
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.agents.baseline.baseline_agent_langgraph import PoliceAgent, VictimAgent, ConversationManager
# from src.agents.tools2 import PoliceTools
# import json
# import logging
# from pathlib import Path
# from config.settings import get_settings
# import csv
# from filelock import FileLock
# import requests
# from pydantic import ValidationError

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

# def clear_csv(csv_file="conversation_history.csv"):
#     """Clear the CSV file to avoid interference between tests."""
#     with FileLock(f"{csv_file}.lock"):
#         try:
#             with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow([
#                     "index", "conversation_id", "conversation_type", "sender_type", 
#                     "content", "timestamp", "llm_model", "scam_generic_details"
#                 ])
#             logger.info(f"Cleared CSV file: {csv_file}")
#         except Exception as e:
#             logger.error(f"Failed to clear CSV file: {str(e)}", exc_info=True)
#             raise

# def count_csv_entries(csv_file="conversation_history.csv"):
#     """Count the number of entries in the CSV, excluding the header."""
#     with FileLock(f"{csv_file}.lock"):
#         try:
#             if not os.path.exists(csv_file):
#                 return 0
#             with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                 reader = csv.reader(f)
#                 next(reader)  # Skip header
#                 return sum(1 for row in reader)
#         except Exception as e:
#             logger.error(f"Failed to count CSV entries: {str(e)}", exc_info=True)
#             raise

# def count_rag_invocations(csv_file="rag_invocations.csv"):
#     """Count the number of entries in the RAG invocations CSV, excluding the header."""
#     with FileLock(f"{csv_file}.lock"):
#         try:
#             if not os.path.exists(csv_file):
#                 return 0
#             with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                 reader = csv.reader(f)
#                 next(reader)  # Skip header
#                 return sum(1 for row in reader)
#         except Exception as e:
#             logger.error(f"Failed to count RAG invocations CSV entries: {str(e)}", exc_info=True)
#             raise

# def test_chatbot_creation():
#     logger.info("=== Testing Chatbot Creation ===")
#     print("=== Testing Chatbot Creation ===")
#     llm_configs = [
#         {"provider": "OpenAI", "model": "gpt-4o-mini"},
#         {"provider": "Ollama", "model": "mistral:7b"},
#         {"provider": "Ollama", "model": "qwen2.5:7b"}
#     ]
#     agents_dict = {}
    
#     for config in llm_configs:
#         provider, model = config["provider"], config["model"]
#         try:
#             police_agent = PoliceAgent(llm_provider=provider, model=model, max_turns=10)
#             logger.info(f"Created police chatbot ({provider}/{model}) with agent_id: {police_agent.agent_id}")
#             print(f"Created police chatbot ({provider}/{model}) with agent_id: {police_agent.agent_id}")
            
#             victim_agent = VictimAgent(llm_provider=provider, model=model, max_turns=10)
#             logger.info(f"Created victim chatbot ({provider}/{model}) with agent_id: {victim_agent.agent_id}")
#             print(f"Created victim chatbot ({provider}/{model}) with agent_id: {victim_agent.agent_id}")
            
#             agents_dict[f"{provider}/{model}"] = (police_agent, victim_agent)
#         except Exception as e:
#             logger.error(f"Error in chatbot creation for {provider}/{model}: {str(e)}", exc_info=True)
#             print(f"Error in chatbot creation for {provider}/{model}: {str(e)}")
    
#     return agents_dict if agents_dict else None

# def test_rag_tool():
#     logger.info("=== Testing RAG Tool ===")
#     print("=== Testing RAG Tool ===")
#     csv_file = "rag_invocations.csv"
#     llm_configs = [
#         {"provider": "OpenAI", "model": "gpt-4o-mini"},
#         {"provider": "Ollama", "model": "mistral:7b"},
#         {"provider": "Ollama", "model": "qwen2.5:7b"}
#     ]
    
#     for config in llm_configs:
#         provider, model = config["provider"], config["model"]
#         try:
#             initial_count = count_rag_invocations(csv_file)
#             police_tools = PoliceTools()
#             tools = police_tools.get_tools()
#             retrieve_scam_reports = tools[0]
#             result = retrieve_scam_reports.invoke({
#                 "query": "Facebook scam",
#                 "top_k": 5,
#                 "conversation_id": 0,
#                 "llm_model": model
#             })
#             final_count = count_rag_invocations(csv_file)
            
#             logger.info(f"RAG tool test result for {provider}/{model}: {result}")
#             print(f"RAG tool test result for {provider}/{model}: {result}")
#             assert final_count == initial_count + 1, f"Expected 1 new entry in {csv_file}, got {final_count - initial_count} for {provider}/{model}"
#             logger.info(f"RAG invocations CSV updated for {provider}/{model}: {final_count} entries")
#             print(f"RAG invocations CSV updated for {provider}/{model}: {final_count} entries")
#         except Exception as e:
#             logger.error(f"Error in RAG tool test for {provider}/{model}: {str(e)}", exc_info=True)
#             print(f"Error in RAG tool test for {provider}/{model}: {str(e)}")
#             raise

# def test_non_autonomous_response(agents_dict):
#     logger.info("=== Testing Non-Autonomous Conversation ===")
#     print("=== Testing Non-Autonomous Conversation ===")
#     conversation_manager = ConversationManager()
#     csv_file = "conversation_history.csv"
#     try:
#         clear_csv(csv_file)
#     except Exception as e:
#         logger.error(f"Failed to clear CSV: {str(e)}", exc_info=True)
#         print(f"Failed to clear CSV: {str(e)}")
#         return
    
#     queries = [
#         "I received a suspicious email claiming to be from my bank.",
#         "Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?"
#     ]
    
#     for provider_model, (police_agent, victim_agent) in agents_dict.items():
#         logger.info(f"Testing {provider_model} provider")
#         print(f"Testing {provider_model} provider")
        
#         for query in queries:
#             try:
#                 initial_count = count_csv_entries(csv_file)
#                 rag_initial_count = count_rag_invocations("rag_invocations.csv")
#                 response = conversation_manager.get_nonautonomous_response(
#                     agent=police_agent if query.startswith("I received") else victim_agent,
#                     query=query,
#                     conversation_id=None,
#                     conversation_history=[]
#                 )
#                 final_count = count_csv_entries(csv_file)
#                 rag_final_count = count_rag_invocations("rag_invocations.csv")
                
#                 if "error" in response:
#                     logger.error(f"Non-autonomous response failed for {'PoliceAgent' if query.startswith('I received') else 'VictimAgent'} ({provider_model}): {response['error']}")
#                     print(f"Non-autonomous response failed for {'PoliceAgent' if query.startswith('I received') else 'VictimAgent'} ({provider_model}): {response['error']}")
#                 else:
#                     logger.info(f"Non-autonomous response succeeded for {'PoliceAgent' if query.startswith('I received') else 'VictimAgent'} ({provider_model})")
#                     print(f"{'PoliceAgent' if query.startswith('I received') else 'VictimAgent'} Response ({provider_model}):", response["response"])
#                     print(f"Conversation History ({provider_model}):", json.dumps(response["conversation_history"], indent=2))
                    
#                     if query.startswith("I received"):
#                         assert response["structured_data"].get("rag_invoked", False), f"RAG tool not invoked for PoliceAgent ({provider_model}) with query: {query}"
#                         assert "scam_type" in response["structured_data"], f"Structured data missing scam_type for {provider_model}"
#                         assert response["structured_data"]["scam_incident_description"] == query, f"Structured data does not match victim input for {provider_model}"
#                         assert rag_final_count > rag_initial_count, f"Expected RAG invocation to be logged, got {rag_final_count - rag_initial_count} entries"
#                     else:
#                         assert not response["structured_data"], f"VictimAgent ({provider_model}) should not return structured data"
#                         assert "[END_CONVERSATION]" not in response["response"], f"Victim response ended prematurely for {provider_model}"
#                         assert any(word in response["response"].lower() for word in ["scam", "upset", "facebook", "email"]), f"Victim response not conversational for {provider_model}: {response['response']}"
                    
#                     assert len(response["conversation_history"]) == 2, f"Expected 2 messages in history, got {len(response['conversation_history'])}"
#                     assert final_count == initial_count + 2, f"Expected 2 new CSV entries, got {final_count - initial_count}"
                    
#                     with FileLock(f"{csv_file}.lock"):
#                         with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                             reader = csv.DictReader(f)
#                             assert "scam_generic_details" in reader.fieldnames, "scam_generic_details column missing in CSV"
#                             for row in reader:
#                                 if row["sender_type"] == "police" and row["scam_generic_details"]:
#                                     assert json.loads(row["scam_generic_details"]).get("rag_invoked", False), f"RAG invocation not logged in CSV for {provider_model}"
#             except Exception as e:
#                 logger.error(f"Error in non-autonomous test for {'PoliceAgent' if query.startswith('I received') else 'VictimAgent'} ({provider_model}): {str(e)}", exc_info=True)
#                 print(f"Error in non-autonomous test for {'PoliceAgent' if query.startswith('I received') else 'VictimAgent'} ({provider_model}): {str(e)}")
        
#         try:
#             initial_count = count_csv_entries(csv_file)
#             rag_initial_count = count_rag_invocations("rag_invocations.csv")
#             victim_response = conversation_manager.get_nonautonomous_response(
#                 agent=victim_agent,
#                 query="Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?",
#                 conversation_id=None,
#                 conversation_history=[]
#             )
#             if "error" in victim_response:
#                 logger.error(f"Non-autonomous response failed for VictimAgent ({provider_model}): {victim_response['error']}")
#                 print(f"Non-autonomous response failed for VictimAgent ({provider_model}): {victim_response['error']}")
#                 continue
            
#             logger.info(f"VictimAgent Response ({provider_model}): {victim_response['response']}")
#             print(f"VictimAgent Response ({provider_model}):", victim_response["response"])
#             print(f"Conversation History ({provider_model}):", json.dumps(victim_response["conversation_history"], indent=2))
            
#             assert "[END_CONVERSATION]" not in victim_response["response"], f"Victim response ended prematurely for {provider_model}"
#             assert any(word in victim_response["response"].lower() for word in ["scam", "upset", "facebook", "email"]), f"Victim response not conversational for {provider_model}: {victim_response['response']}"
            
#             conversation_id = victim_response["conversation_id"]
#             conversation_history = victim_response["conversation_history"]
            
#             police_response = conversation_manager.get_nonautonomous_response(
#                 agent=police_agent,
#                 query=victim_response["response"],
#                 conversation_id=conversation_id,
#                 conversation_history=conversation_history
#             )
#             final_count = count_csv_entries(csv_file)
#             rag_final_count = count_rag_invocations("rag_invocations.csv")
            
#             if "error" in police_response:
#                 logger.error(f"Non-autonomous response failed for PoliceAgent ({provider_model}): {police_response['error']}")
#                 print(f"Non-autonomous response failed for PoliceAgent ({provider_model}): {police_response['error']}")
#             else:
#                 logger.info(f"PoliceAgent Response ({provider_model}): {police_response['response']}")
#                 print(f"PoliceAgent Response ({provider_model}):", police_response["response"])
#                 print(f"Structured Data ({provider_model}):", json.dumps(police_response["structured_data"], indent=2))
#                 print(f"Conversation History ({provider_model}):", json.dumps(police_response["conversation_history"], indent=2))
                
#                 assert len(police_response["conversation_history"]) == 3, f"Expected 3 messages in history, got {len(police_response['conversation_history'])}"
#                 assert police_response["structured_data"].get("rag_invoked", False), f"RAG tool not invoked for PoliceAgent ({provider_model}) with query: {victim_response['response']}"
#                 assert "scam_type" in police_response["structured_data"], f"Structured data missing scam_type for {provider_model}"
#                 assert police_response["structured_data"]["scam_incident_description"] == victim_response["response"], f"Structured data does not match victim input for {provider_model}"
#                 assert "details" in police_response["response"].lower(), f"Police response not conversational for {provider_model}: {police_response['response']}"
#                 assert final_count >= initial_count + 4, f"Expected at least 4 new CSV entries, got {final_count - initial_count}"
#                 assert rag_final_count > rag_initial_count, f"Expected RAG invocation to be logged, got {rag_final_count - rag_initial_count} entries"
                
#                 with FileLock(f"{csv_file}.lock"):
#                     with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
#                         reader = csv.DictReader(f)
#                         assert "scam_generic_details" in reader.fieldnames, "scam_generic_details column missing in CSV"
#                         for row in reader:
#                             if row["sender_type"] == "police" and row["scam_generic_details"]:
#                                 assert json.loads(row["scam_generic_details"]).get("rag_invoked", False), f"RAG invocation not logged in CSV for {provider_model}"
#         except Exception as e:
#             logger.error(f"Error in non-autonomous test for PoliceAgent ({provider_model}): {str(e)}", exc_info=True)
#             print(f"Error in non-autonomous test for PoliceAgent ({provider_model}): {str(e)}")

# if __name__ == "__main__":
#     clear_csv()
#     agents_dict = test_chatbot_creation()
#     if agents_dict:
#         test_rag_tool()
#         test_non_autonomous_response(agents_dict)
#         # test_autonomous_conversation(agents_dict)



import sys
import os
import hashlib
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents.remove.baseline_agent_langgraph import PoliceAgent, VictimAgent, ConversationManager
from src.agents.remove.tools2 import PoliceTools
import json
import logging
from pathlib import Path
from config.settings import get_settings
import csv
from filelock import FileLock
import requests
from pydantic import ValidationError

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
    with FileLock(f"{csv_file}.lock"):
        try:
            with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "index", "conversation_id", "conversation_type", "sender_type", 
                    "content", "timestamp", "llm_model", "scam_generic_details"
                ])
            logger.info(f"Cleared CSV file: {csv_file}")
        except Exception as e:
            logger.error(f"Failed to clear CSV file: {str(e)}", exc_info=True)
            raise

def count_csv_entries(csv_file="conversation_history.csv"):
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
    llm_configs = [
        {"provider": "Ollama", "model": "llama3.2:latest"},
        {"provider": "Ollama", "model": "mistral:7b"},
        {"provider": "Ollama", "model": "qwen2.5:7b"}
    ]
    agents_dict = {}
    
    for config in llm_configs:
        provider, model = config["provider"], config["model"]
        try:
            police_agent = PoliceAgent(llm_provider=provider, model=model, max_turns=10)
            logger.info(f"Created police chatbot ({provider}/{model}) with agent_id: {police_agent.agent_id}")
            print(f"Created police chatbot ({provider}/{model}) with agent_id: {police_agent.agent_id}")
            
            victim_agent = VictimAgent(llm_provider=provider, model=model, max_turns=10)
            logger.info(f"Created victim chatbot ({provider}/{model}) with agent_id: {victim_agent.agent_id}")
            print(f"Created victim chatbot ({provider}/{model}) with agent_id: {victim_agent.agent_id}")
            
            agents_dict[f"{provider}/{model}/police"] = police_agent
            agents_dict[f"{provider}/{model}/victim"] = victim_agent
        except Exception as e:
            logger.error(f"Error in chatbot creation for {provider}/{model}: {str(e)}", exc_info=True)
            print(f"Error in chatbot creation for {provider}/{model}: {str(e)}")
    
    return agents_dict if agents_dict else None

def test_rag_tool():
    logger.info("=== Testing RAG Tool ===")
    print("=== Testing RAG Tool ===")
    csv_file = "rag_invocations.csv"
    llm_configs = [
        {"provider": "Ollama", "model": "llama3.2:latest"},
        {"provider": "Ollama", "model": "mistral:7b"},
        {"provider": "Ollama", "model": "qwen2.5:7b"}
    ]
    
    for config in llm_configs:
        provider, model = config["provider"], config["model"]
        try:
            initial_count = count_rag_invocations(csv_file)
            police_tools = PoliceTools()
            tools = police_tools.get_tools()
            retrieve_scam_reports = tools[0]
            result = retrieve_scam_reports.invoke({
                "query": "scam",
                "top_k": 5,
                "conversation_id": 0,
                "llm_model": model
            })
            final_count = count_rag_invocations(csv_file)
            
            logger.info(f"RAG tool test result for {provider}/{model}: {result}")
            print(f"RAG tool test result for {provider}/{model}: {result}")
            assert final_count == initial_count + 1, f"Expected 1 new entry in {csv_file}, got {final_count - initial_count} for {provider}/{model}"
            logger.info(f"RAG invocations CSV updated for {provider}/{model}: {final_count} entries")
            print(f"RAG invocations CSV updated for {provider}/{model}: {final_count} entries")
        except Exception as e:
            logger.error(f"Error in RAG tool test for {provider}/{model}: {str(e)}", exc_info=True)
            print(f"Error in RAG tool test for {provider}/{model}: {str(e)}")
            raise

def test_non_autonomous_response(agents_dict):
    logger.info("=== Testing Non-Autonomous Conversation ===")
    print("=== Testing Non-Autonomous Conversation ===")
    conversation_manager = ConversationManager()
    queries = [
        "I received a suspicious SMS claiming to be from my bank.",
        "Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?",
        "I got a WhatsApp call from a person claiming to be a MAS official on 2025-01-15 that cost me $1000."
    ]
    
    # Track conversation IDs to avoid duplicates
    conversation_ids = {}
    
    for provider, model in [
        ("Ollama", "llama3.2:latest"),
        ("Ollama", "mistral:7b"),
        ("Ollama", "qwen2.5:7b")
    ]:
        print(f"Testing {provider}/{model} provider")
        police_agent = agents_dict.get(f"{provider}/{model}/police")
        victim_agent = agents_dict.get(f"{provider}/{model}/victim")
        
        for query in queries:
            # Generate or reuse conversation ID based on query content
            query_hash = hashlib.md5(query.encode()).hexdigest()
            conversation_id = conversation_ids.get(query_hash)
            
            # Get police response
            initial_count = count_csv_entries("conversation_history.csv")
            police_response = conversation_manager.get_nonautonomous_response(
                agent=police_agent,
                query=query,
                conversation_id=conversation_id
            )
            conversation_id = police_response["conversation_id"]
            conversation_ids[query_hash] = conversation_id
            
            print(f"PoliceAgent Response ({provider}/{model}): {police_response['response']}")
            print(f"Structured Data ({provider}/{model}): {police_response['structured_data']}")
            print(f"Conversation History ({provider}/{model}): {police_response['conversation_history']}")
            
            # Assert history length for direct victim query
            assert len(police_response["conversation_history"]) == 2, (
                f"Expected exactly 2 messages in history for direct query, got {len(police_response['conversation_history'])}"
            )
            # Assert structured output contains victim input
            if "SMS claiming to be from my bank" in query:
                assert police_response["structured_data"]["scam_type"] == "PHISHING", (
                    f"Expected scam_type 'PHISHING' for SMS query, got {police_response['structured_data']['scam_type']}"
                )
                assert police_response["structured_data"]["scam_approach_platform"] == "SMS", (
                    f"Expected scam_approach_platform 'SMS', got {police_response['structured_data']['scam_approach_platform']}"
                )
            elif "WhatsApp call" in query:
                assert police_response["structured_data"]["scam_type"] == "GOVERNMENT IMPERSONATION", (
                    f"Expected scam_type 'GOVERNMENT IMPERSONATION' for WhatsApp query, got {police_response['structured_data']['scam_type']}"
                )
                assert police_response["structured_data"]["scam_incident_date"] == "2025-01-15", (
                    f"Expected scam_incident_date '2025-01-15', got {police_response['structured_data']['scam_incident_date']}"
                )
                assert police_response["structured_data"]["scam_amount_lost"] == 1000.0, (
                    f"Expected scam_amount_lost 1000.0, got {police_response['structured_data']['scam_amount_lost']}"
                )
            
            # Assert RAG invocation
            assert police_response["structured_data"]["rag_invoked"], "RAG was not invoked for police response"
            
            # Get victim response if police initiated
            if query == "Hello, this is the police. Can you tell me about any recent scam incidents you’ve experienced?":
                victim_response = conversation_manager.get_nonautonomous_response(
                    agent=victim_agent,
                    query=police_response["response"],
                    conversation_id=conversation_id
                )
                print(f"VictimAgent Response ({provider}/{model}): {victim_response['response']}")
                print(f"Conversation History ({provider}/{model}): {victim_response['conversation_history']}")
                
                # Get follow-up police response
                follow_up_response = conversation_manager.get_nonautonomous_response(
                    agent=police_agent,
                    query=victim_response["response"],
                    conversation_id=conversation_id
                )
                print(f"PoliceAgent Response ({provider}/{model}): {follow_up_response['response']}")
                print(f"Structured Data ({provider}/{model}): {follow_up_response['structured_data']}")
                print(f"Conversation History ({provider}/{model}): {follow_up_response['conversation_history']}")
                
                # Assert history length for follow-up
                assert len(follow_up_response["conversation_history"]) == 3, (
                    f"Expected exactly 3 messages in history for follow-up, got {len(follow_up_response['conversation_history'])}"
                )
                
                # Assert structured output reflects victim response
                victim_content = victim_response["response"].lower()
                if "facebook" in victim_content and "ticket" in victim_content:
                    assert follow_up_response["structured_data"]["scam_type"] == "ECOMMERCE", (
                        f"Expected scam_type 'ECOMMERCE' for Facebook ticket scam, got {follow_up_response['structured_data']['scam_type']}"
                    )
                    assert follow_up_response["structured_data"]["scam_approach_platform"] == "FACEBOOK", (
                        f"Expected scam_approach_platform 'FACEBOOK', got {follow_up_response['structured_data']['scam_approach_platform']}"
                    )
                
                # Assert RAG invocation
                assert follow_up_response["structured_data"]["rag_invoked"], "RAG was not invoked for follow-up police response"
                
                # Assert CSV entry count increased
                final_count = count_csv_entries("conversation_history.csv")
                assert final_count > initial_count, (
                    f"Expected new entries in conversation_history.csv, got {final_count - initial_count}"
                )

if __name__ == "__main__":
    clear_csv()
    agents_dict = test_chatbot_creation()
    if agents_dict:
        test_rag_tool()
        test_non_autonomous_response(agents_dict)
