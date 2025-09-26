# baseline_police.py
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser
from src.models.response_model import PoliceResponse
from src.agents.prompts3 import police_prompt
from src.agents.remove.tools3 import PoliceTools
from src.agents.remove.llm_providers3 import LLMProvider
from config.settings import get_settings
import json
import csv
import os
from datetime import datetime
from pathlib import Path
import logging
import hashlib
import re

class ConversationManager:
    def __init__(self, llm_model="llama3.1:8b"):
        self.settings = get_settings()
        self.llm_model = llm_model
        self._setup_logging()
        self.index_counter = self._load_last_index()
        self.tools = PoliceTools().get_tools()
        llm_provider = LLMProvider()
        self.llm = llm_provider.get_structured_llm(
            provider="Ollama",
            model=llm_model,
            structured_model=PoliceResponse
        )
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=police_prompt
        )
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        self.parser = JsonOutputParser(pydantic_object=PoliceResponse)

    def _setup_logging(self):
        log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "conversation_manager.log"
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, mode='a')]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized to {log_file}")

    def _load_last_index(self) -> int:
        csv_file = "conversation_history.csv"
        max_index = 0
        if os.path.exists(csv_file):
            try:
                with open(csv_file, mode="r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    indices = [int(row["index"]) for row in reader if row["index"].isdigit()]
                    max_index = max(indices) if indices else 0
            except Exception as e:
                self.logger.error(f"Error reading index from CSV: {str(e)}")
        return max_index

    def _log_to_csv(self, conversation_id: str, sender_type: str, content: str, structured_data: dict = None):
        csv_file = "conversation_history.csv"
        try:
            with open(csv_file, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                self.index_counter += 1
                writer.writerow([
                    str(self.index_counter),
                    conversation_id,
                    "non_autonomous",
                    sender_type,
                    content,
                    datetime.now().isoformat(),
                    self.llm_model,
                    json.dumps(structured_data, ensure_ascii=False) if structured_data else ""
                ])
        except Exception as e:
            self.logger.error(f"Failed to write to CSV: {str(e)}")

    def _process_response(self, victim_input: str, llm_output: str, rag_results: str) -> dict:
        try:
            llm_output = llm_output.strip()
            if llm_output.startswith("```json") and llm_output.endswith("```"):
                llm_output = llm_output[7:-3].strip()
            structured_data = json.loads(llm_output)
            structured_data["rag_invoked"] = True
            return self.parser.parse(json.dumps(structured_data))
        except (json.JSONDecodeError, ValueError, Exception) as e:
            self.logger.warning(f"Failed to parse LLM output: {str(e)}. Using fallback.")
            structured_data = {
                "conversational_response": "I'm sorry to hear about your experience. Can you provide more details, such as the date or payment details?",
                "scam_incident_date": "",
                "scam_type": "UNKNOWN",
                "scam_approach_platform": "UNKNOWN",
                "scam_communication_platform": "UNKNOWN",
                "scam_transaction_type": "",
                "scam_beneficiary_platform": "",
                "scam_beneficiary_identifier": "",
                "scam_contact_no": "",
                "scam_email": "",
                "scam_moniker": "",
                "scam_url_link": "",
                "scam_amount_lost": 0.0,
                "scam_incident_description": victim_input,
                "scam_specific_details": {},
                "rag_invoked": True
            }

            victim_lower = victim_input.lower()
            if "sms" in victim_lower:
                structured_data.update({
                    "scam_type": "PHISHING",
                    "scam_approach_platform": "SMS",
                    "scam_communication_platform": "SMS",
                    "scam_specific_details": {"scam_subcategory": "SMS PHISHING"}
                })
            elif "facebook" in victim_lower and "ticket" in victim_lower:
                structured_data.update({
                    "scam_type": "ECOMMERCE",
                    "scam_approach_platform": "FACEBOOK",
                    "scam_communication_platform": "FACEBOOK",
                    "scam_specific_details": {
                        "scam_subcategory": "FAILURE TO DELIVER GOODS AND SERVICES",
                        "scam_item_involved": "TICKET",
                        "scam_item_type": "TICKETS"
                    }
                })
            elif "government" in victim_lower or "mas" in victim_lower:
                structured_data.update({
                    "scam_type": "GOVERNMENT IMPERSONATION",
                    "scam_approach_platform": "WHATSAPP" if "whatsapp" in victim_lower else "UNKNOWN",
                    "scam_communication_platform": "WHATSAPP" if "whatsapp" in victim_lower else "UNKNOWN",
                    "scam_specific_details": {
                        "scam_subcategory": "GOVERNMENT SCAM",
                        "scam_impersonation_type": "GOVERNMENT AGENCY",
                        "scam_first_impersonated_entity": "MAS" if "mas" in victim_lower else ""
                    }
                })

            amount_match = re.search(r'\$\d+(\.\d{2})?', victim_lower)
            if amount_match:
                structured_data["scam_amount_lost"] = float(amount_match.group(0).replace('$', ''))

            date_match = re.search(r'\b202[0-5]-\d{2}-\d{2}\b', victim_input)
            if date_match:
                structured_data["scam_incident_date"] = date_match.group(0)

            try:
                rag_data = json.loads(rag_results) if rag_results else []
                if rag_data and isinstance(rag_data, list) and rag_data[0]:
                    first_result = rag_data[0]
                    structured_data["conversational_response"] = (
                        f"I'm sorry to hear about your experience. Based on our records, we found a similar scam involving "
                        f"{first_result.get('scam_type', 'unknown')} on {first_result.get('scam_approach_platform', 'unknown')}. "
                        f"Please provide more details to assist with the investigation."
                    )
                    if structured_data["scam_type"] == "UNKNOWN":
                        structured_data["scam_type"] = first_result.get("scam_type", "UNKNOWN")
                    if structured_data["scam_approach_platform"] == "UNKNOWN":
                        structured_data["scam_approach_platform"] = first_result.get("scam_approach_platform", "UNKNOWN")
                    if structured_data["scam_communication_platform"] == "UNKNOWN":
                        structured_data["scam_communication_platform"] = first_result.get("scam_communication_platform", "UNKNOWN")
                    if not structured_data["scam_specific_details"]:
                        structured_data["scam_specific_details"] = first_result.get("scam_specific_details", {})
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse RAG results")

            return structured_data  # Rely on LLM schema enforcement

    def get_nonautonomous_response(self, query: str, conversation_id: str = None, conversation_history: list = None):
        if conversation_id is None:
            conversation_id = hashlib.md5(query.encode()).hexdigest()[:8]
        if conversation_history is None:
            conversation_history = []

        self._log_to_csv(conversation_id, "victim", query)
        response = self.executor.invoke({
            "input": query,
            "conversation_id": conversation_id,
            "agent_scratchpad": [
                AIMessage(content=json.dumps({
                    "tool": "retrieve_scam_reports",
                    "tool_input": {
                        "query": query,
                        "top_k": 5,
                        "conversation_id": conversation_id,
                        "llm_model": "llama3.2:latest"
                    }
                }))
            ]
        })

        llm_output = response.get("output", "{}")
        rag_results = response.get("intermediate_steps", [({}, "")])[0][1]
        structured_response = self._process_response(query, llm_output, rag_results)

        conversation_history.append({"role": "victim", "content": query})
        conversation_history.append({"role": "police", "content": structured_response["conversational_response"]})
        self._log_to_csv(conversation_id, "police", structured_response["conversational_response"], structured_response)

        return {
            "response": structured_response["conversational_response"],
            "structured_data": structured_response,
            "conversation_id": conversation_id,
            "conversation_history": conversation_history
        }