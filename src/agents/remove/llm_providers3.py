# llm_providers2.py
from langchain_ollama import ChatOllama
from config.settings import get_settings
from pydantic import BaseModel
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import json
import re
from src.agents.remove.tools3 import PoliceTools

load_dotenv()

SUPPORTED_MODELS = {
    "Ollama": ["llama3.2:latest", "llama3.1:8b", "mistral:7b", "qwen2.5:7b", "phi3:3.8b"]
}

class LLMProvider:
    def __init__(self):
        self.settings = get_settings()
        self._setup_logging()

    def _setup_logging(self):
        log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "llm_providers.log"
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file, mode='a')]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Logging initialized to {log_file}")

    def get_supported_models(self):
        return SUPPORTED_MODELS

    def _fallback_parse_response(self, raw_response: str, structured_model: type[BaseModel]) -> dict:
        self.logger.warning(f"Fallback parsing triggered for raw response: {raw_response}")
        structured_data = {
            "conversational_response": "I'm sorry, I need more details about the incident to assist you.",
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
            "scam_incident_description": raw_response.strip() or "",
            "scam_specific_details": {},
            "rag_invoked": True
        }

        raw_lower = raw_response.lower()
        if "sms" in raw_lower:
            structured_data.update({
                "scam_type": "PHISHING",
                "scam_approach_platform": "SMS",
                "scam_communication_platform": "SMS",
                "scam_specific_details": {"scam_subcategory": "SMS PHISHING"}
            })
        elif "facebook" in raw_lower and "ticket" in raw_lower:
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
        elif "government" in raw_lower or "mas" in raw_lower:
            structured_data.update({
                "scam_type": "GOVERNMENT IMPERSONATION",
                "scam_approach_platform": "WHATSAPP" if "whatsapp" in raw_lower else "UNKNOWN",
                "scam_communication_platform": "WHATSAPP" if "whatsapp" in raw_lower else "UNKNOWN",
                "scam_specific_details": {
                    "scam_subcategory": "GOVERNMENT SCAM",
                    "scam_impersonation_type": "GOVERNMENT AGENCY",
                    "scam_first_impersonated_entity": "MAS" if "mas" in raw_lower else ""
                }
            })

        amount_match = re.search(r'\$\d+(\.\d{2})?', raw_response)
        if amount_match:
            structured_data["scam_amount_lost"] = float(amount_match.group(0).replace('$', ''))

        date_match = re.search(r'\b202[0-5]-\d{2}-\d{2}\b', raw_response)
        if date_match:
            structured_data["scam_incident_date"] = date_match.group(0)

        return structured_data

    def get_structured_llm(self, provider: str = "Ollama", model: str = "llama3.2:latest", structured_model: type[BaseModel] = None):
        if provider not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown provider: {provider}. Must be one of {list(SUPPORTED_MODELS.keys())}")
        if model not in SUPPORTED_MODELS[provider]:
            raise ValueError(f"Model '{model}' not supported for provider '{provider}'. Must be one of {SUPPORTED_MODELS[provider]}")
        if structured_model is None or not issubclass(structured_model, BaseModel):
            raise ValueError("structured_model must be a valid Pydantic BaseModel subclass")

        try:
            police_tools = PoliceTools()
            tools = police_tools.get_tools()
            base_url = self.settings.agents.ollama_base_url
            self.logger.debug(f"Initializing structured Ollama LLM with model: {model}, base_url: {base_url}")
            llm = ChatOllama(
                model=model,
                base_url=base_url,
                format=structured_model.model_json_schema(),  # Restore previous schema enforcement
                temperature=0.0,
                num_ctx=4096,
                max_retries=3
            )
            return llm.bind_tools(tools)  # Bind RAG tool
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {str(e)}")
            raise