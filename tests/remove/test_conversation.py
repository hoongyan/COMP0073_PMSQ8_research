
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agents.remove.baseline_police import ConversationManager
import json
import csv

def test_conversation():
    manager = ConversationManager(llm_model="llama3.1:8b")
    query = "I received a suspicious SMS claiming to be from my bank."
    response = manager.get_nonautonomous_response(query)
    print(f"Query: {query}")
    print(f"Police Response: {response['response']}")
    print(f"Structured Data: {json.dumps(response['structured_data'], indent=2)}")
    print(f"Conversation History: {json.dumps(response['conversation_history'], indent=2)}")

if __name__ == "__main__":
    csv_file = "conversation_history.csv"
    with open(csv_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "conversation_id", "conversation_type", "sender_type", "content", "timestamp", "llm_model", "scam_generic_details"])
    test_conversation()