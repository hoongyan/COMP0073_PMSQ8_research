
# import unittest
# import json
# import os
# import csv
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.agents.baseline.baseline_police import PoliceChatbot
# from src.models.response_model import PoliceResponse

# class TestPoliceChatbot(unittest.TestCase):
#     def setUp(self):
#         self.chatbot = PoliceChatbot(model_name="llama3.1:8b")
#         self.conversation_id = self.chatbot.conversation_id
#         self.csv_file = "conversation_history.csv"
#         if os.path.exists(self.csv_file):
#             os.remove(self.csv_file)  # Clean up before test

#     def test_single_turn_conversation(self):
#         query = "I got scammed on Facebook buying a Taylor Swift concert ticket for $450."
#         response = self.chatbot.process_query(query)
#         self.chatbot.end_conversation()

#         # Validate response structure
#         self.assertIn("response", response)
#         self.assertIn("structured_data", response)
#         self.assertIn("conversation_id", response)
#         self.assertIn("conversation_history", response)

#         # Validate PoliceResponse model
#         structured_data = response["structured_data"]
#         try:
#             PoliceResponse(**structured_data)
#             self.assertTrue(structured_data.get("rag_invoked", False), "RAG should be invoked for scam-related query")
#             self.assertEqual(structured_data["scam_approach_platform"], "FACEBOOK", "Platform should be standardized to FACEBOOK")
#             self.assertEqual(structured_data["scam_type"], "ECOMMERCE", "Scam type should be ECOMMERCE")
#             self.assertAlmostEqual(structured_data["scam_amount_lost"], 450.0, places=2, msg="Amount lost should be 450.0")
#         except Exception as e:
#             self.fail(f"Structured data does not conform to PoliceResponse model: {str(e)}")

#         # Validate CSV storage
#         with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             rows = list(reader)
#             self.assertGreaterEqual(len(rows), 2, "CSV should contain at least query and response")
#             self.assertEqual(rows[0]["conversation_id"], str(self.conversation_id))
#             self.assertEqual(rows[0]["sender_type"], "victim")
#             self.assertEqual(rows[0]["content"], query)
#             self.assertEqual(rows[1]["sender_type"], "police")
#             self.assertEqual(rows[1]["conversation_type"], "non_autonomous")
#             self.assertEqual(rows[1]["llm_model"], "llama3.1:8b")
#             self.assertEqual(rows[1]["scam_specific_details"], "")
#             self.assertNotEqual(rows[1]["scam_generic_details"], "", "Scam generic details should be populated")

#     def test_multi_turn_conversation(self):
#         queries = [
#             "I got scammed on Facebook buying a concert ticket.",
#             "It was for Taylor Swift, paid $450 to a guy named wilkinsonthomas.",
#             "I sent the money via bank transfer to CIMB on Feb 22, 2025."
#         ]
#         responses = []
#         for query in queries:
#             responses.append(self.chatbot.process_query(query))
#         self.chatbot.end_conversation()

#         # Validate structured data progression
#         for i, response in enumerate(responses):
#             structured_data = response["structured_data"]
#             try:
#                 PoliceResponse(**structured_data)
#                 self.assertTrue(structured_data.get("rag_invoked", False), f"RAG should be invoked for query {i+1}")
#             except Exception as e:
#                 self.fail(f"Structured data does not conform to PoliceResponse model in turn {i+1}: {str(e)}")

#         # Validate final structured data
#         final_structured_data = responses[-1]["structured_data"]
#         self.assertEqual(final_structured_data["scam_approach_platform"], "FACEBOOK")
#         self.assertEqual(final_structured_data["scam_type"], "ECOMMERCE")
#         self.assertAlmostEqual(final_structured_data["scam_amount_lost"], 450.0, places=2)
#         self.assertEqual(final_structured_data["scam_moniker"], "wilkinsonthomas")
#         self.assertEqual(final_structured_data["scam_transaction_type"], "BANK TRANSFER")
#         self.assertEqual(final_structured_data["scam_beneficiary_platform"], "CIMB")
#         self.assertEqual(final_structured_data["scam_incident_date"], "2025-02-22")

#         # Validate CSV storage
#         with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
#             reader = csv.DictReader(f)
#             rows = list(reader)
#             self.assertEqual(len(rows), 6, "CSV should contain 3 queries and 3 responses")
#             for i, query in enumerate(queries):
#                 self.assertEqual(rows[i*2]["content"], query, f"Query {i+1} should match")
#                 self.assertEqual(rows[i*2]["sender_type"], "victim")
#                 self.assertEqual(rows[i*2+1]["sender_type"], "police")
#                 self.assertEqual(rows[i*2+1]["scam_specific_details"], "")

# if __name__ == "__main__":
#     unittest.main()


import unittest
import json
import os
import csv
from src.agents.remove.police_agent_good_draft_vanilla import PoliceChatbot
from src.agents.remove.conversation_manager import ConversationManager
from src.models.response_model import PoliceResponse

class TestPoliceChatbot(unittest.TestCase):
    def setUp(self):
        self.models = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b", "deepseek-r1:7b"]
        self.csv_file = "conversation_history.csv"
        if os.path.exists(self.csv_file):
            os.remove(self.csv_file)  # Clean up before test

    def test_single_turn_conversation(self):
        for model in self.models:
            with self.subTest(model=model):
                chatbot = PoliceChatbot(model_name=model)
                conversation_id = chatbot.id_manager.get_next_id()
                query = "I received an SMS claiming to be from a bank."
                response = chatbot.process_query(query, conversation_id)
                chatbot.end_conversation()

                # Validate response structure
                self.assertIn("response", response)
                self.assertIn("structured_data", response)
                self.assertIn("conversation_id", response)

                # Validate PoliceResponse model
                structured_data = response["structured_data"]
                try:
                    PoliceResponse(**structured_data)
                    self.assertTrue(structured_data.get("rag_invoked", False), f"RAG should be invoked for scam-related query with {model}")
                    self.assertEqual(structured_data["scam_approach_platform"], "SMS", f"Platform should be SMS with {model}")
                    self.assertEqual(structured_data["scam_type"], "PHISHING", f"Scam type should be PHISHING with {model}")
                    self.assertEqual(structured_data["scam_communication_platform"], "SMS", f"Communication platform should be SMS with {model}")
                    self.assertEqual(structured_data["scam_transaction_type"], "", f"Transaction type should be empty with {model}")
                    self.assertNotEqual(structured_data["conversational_response"], "", f"Conversational response should be non-empty with {model}")
                    self.assertEqual(structured_data["scam_incident_description"], query, f"Incident description should match query with {model}")
                    self.assertIn("scam_specific_details", structured_data, f"Specific details should be present with {model}")
                    required_fields = [
                        "scam_incident_date", "scam_communication_platform", "scam_transaction_type",
                        "scam_beneficiary_platform", "scam_beneficiary_identifier", "scam_contact_no",
                        "scam_email", "scam_moniker", "scam_url_link", "scam_amount_lost"
                    ]
                    for field in required_fields:
                        self.assertIn(field, structured_data, f"{field} should be present with {model}")
                except Exception as e:
                    self.fail(f"Structured data does not conform to PoliceResponse model with {model}: {str(e)}")

                # Validate CSV storage
                with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    self.assertGreaterEqual(len(rows), 2, f"CSV should contain at least query and response with {model}")
                    self.assertEqual(rows[0]["conversation_id"], str(conversation_id))
                    self.assertEqual(rows[0]["sender_type"], "victim")
                    self.assertEqual(rows[0]["content"], query)
                    self.assertEqual(rows[1]["sender_type"], "police")
                    self.assertEqual(rows[1]["conversation_type"], "non_autonomous")
                    self.assertEqual(rows[1]["llm_model"], model)
                    self.assertEqual(rows[1]["scam_specific_details"], "")
                    self.assertNotEqual(rows[1]["scam_generic_details"], "", f"Scam generic details should be populated with {model}")
                    try:
                        json.loads(rows[1]["scam_generic_details"])
                    except json.JSONDecodeError:
                        self.fail(f"scam_generic_details should be valid JSON with {model}")

    def test_multi_turn_conversation(self):
        for model in self.models:
            with self.subTest(model=model):
                chatbot = PoliceChatbot(model_name=model)
                conversation_id = chatbot.id_manager.get_next_id()
                queries = [
                    "I got scammed on Facebook buying a concert ticket.",
                    "It was for Taylor Swift, paid $450 to a guy named wilkinsonthomas.",
                    "I sent the money via bank transfer to CIMB on Feb 22, 2025."
                ]
                responses = []
                for query in queries:
                    responses.append(chatbot.process_query(query, conversation_id))
                chatbot.end_conversation()

                # Validate structured data progression
                for i, response in enumerate(responses):
                    structured_data = response["structured_data"]
                    try:
                        PoliceResponse(**structured_data)
                        self.assertTrue(structured_data.get("rag_invoked", False), f"RAG should be invoked for query {i+1} with {model}")
                        self.assertNotEqual(structured_data["conversational_response"], "", f"Conversational response should be non-empty in turn {i+1} with {model}")
                        self.assertIn("scam_incident_description", structured_data, f"Incident description should be present in turn {i+1} with {model}")
                        self.assertIn("scam_specific_details", structured_data, f"Specific details should be present in turn {i+1} with {model}")
                    except Exception as e:
                        self.fail(f"Structured data does not conform to PoliceResponse model in turn {i+1} with {model}: {str(e)}")

                # Validate final structured data
                final_structured_data = responses[-1]["structured_data"]
                self.assertEqual(final_structured_data["scam_approach_platform"], "FACEBOOK", f"Platform should be FACEBOOK with {model}")
                self.assertEqual(final_structured_data["scam_type"], "ECOMMERCE", f"Scam type should be ECOMMERCE with {model}")
                self.assertAlmostEqual(final_structured_data["scam_amount_lost"], 450.0, places=2, msg=f"Amount lost should be 450.0 with {model}")
                self.assertEqual(final_structured_data["scam_moniker"], "wilkinsonthomas", f"Moniker should be wilkinsonthomas with {model}")
                self.assertEqual(final_structured_data["scam_transaction_type"], "BANK TRANSFER", f"Transaction type should be BANK TRANSFER with {model}")
                self.assertEqual(final_structured_data["scam_beneficiary_platform"], "CIMB", f"Beneficiary platform should be CIMB with {model}")
                self.assertEqual(final_structured_data["scam_incident_date"], "2025-02-22", f"Incident date should be 2025-02-22 with {model}")
                self.assertNotEqual(final_structured_data["conversational_response"], "", f"Conversational response should be non-empty with {model}")

                # Validate CSV storage
                with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    self.assertEqual(len(rows), 6, f"CSV should contain 3 queries and 3 responses with {model}")
                    for i, query in enumerate(queries):
                        self.assertEqual(rows[i*2]["content"], query, f"Query {i+1} should match with {model}")
                        self.assertEqual(rows[i*2]["sender_type"], "victim")
                        self.assertEqual(rows[i*2+1]["sender_type"], "police")
                        self.assertEqual(rows[i*2+1]["scam_specific_details"], "")
                        self.assertNotEqual(rows[i*2+1]["scam_generic_details"], "", f"Scam generic details should be populated with {model}")
                        try:
                            json.loads(rows[i*2+1]["scam_generic_details"])
                        except json.JSONDecodeError:
                            self.fail(f"scam_generic_details should be valid JSON with {model}")

    def test_autonomous_conversation(self):
        for model in self.models:
            with self.subTest(model=model):
                manager = ConversationManager()
                result = manager.run_autonomous_conversation(police_model=model, victim_model=model, max_turns=6)
                self.assertIn("status", result)
                self.assertEqual(result["status"], "Conversation completed")
                self.assertIn("conversation_id", result)
                self.assertIn("conversation_history", result)
                self.assertGreaterEqual(len(result["conversation_history"]), 2, f"Conversation should have at least one turn with {model}")

                conversation_id = result["conversation_id"]
                with open(self.csv_file, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = [row for row in reader if row["conversation_id"] == str(conversation_id)]
                    self.assertGreaterEqual(len(rows), 2, f"CSV should contain at least one turn with {model}")
                    for i, row in enumerate(rows):
                        self.assertEqual(row["conversation_id"], str(conversation_id))
                        self.assertEqual(row["conversation_type"], "autonomous")
                        self.assertIn(row["sender_type"], ["victim", "police"])
                        if row["sender_type"] == "police":
                            self.assertEqual(row["llm_model"], model)
                            self.assertEqual(row["scam_specific_details"], "")
                            self.assertNotEqual(row["scam_generic_details"], "", f"Scam generic details should be populated for police with {model}")
                            try:
                                json.loads(row["scam_generic_details"])
                            except json.JSONDecodeError:
                                self.fail(f"scam_generic_details should be valid JSON with {model}")
                        else:
                            self.assertEqual(row["llm_model"], model)
                            self.assertEqual(row["scam_generic_details"], "")

if __name__ == "__main__":
    unittest.main()
