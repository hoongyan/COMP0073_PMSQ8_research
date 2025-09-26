# from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
# from config.settings import get_settings
# from pydantic import BaseModel, ValidationError
# from typing import Type, Optional
# import logging
# from pathlib import Path
# import os
# from dotenv import load_dotenv
# import json
# import re
# from src.agents.tools2 import RetrieveScamReportsArgs, PoliceTools

# # Load environment variables
# load_dotenv()

# # Access API keys from environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# SUPPORTED_MODELS = {
#     "Ollama": ["llama3.2", "llama3.1:8b", "qwen:7b", "deepseek-r1:7b", "mistral:7b", "qwen2.5:7b", "phi3:3.8b"],
#     "OpenAI": ["gpt-4o-mini", "gpt-4", "gpt-4o-mini-2024-07-18"],
# }

# class LLMProvider:
#     """Manages LLM initialization and configuration with structured output enforcement."""
    
#     def __init__(self):
#         """Initialize with settings and logging."""
#         self.settings = get_settings()
#         self._setup_logging()
#         self.structured_success_count = 0
#         self.structured_failure_count = 0
    
#     def _setup_logging(self):
#         """Configure logging to write to LLM-specific log file."""
#         log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
#         log_dir.mkdir(parents=True, exist_ok=True)
#         log_file = log_dir / "llm_providers.log"
        
#         logging.basicConfig(
#             level=logging.DEBUG,
#             format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
#             handlers=[
#                 logging.FileHandler(log_file, mode='a'),
#                 logging.StreamHandler()
#             ]
#         )
#         self.logger = logging.getLogger(__name__)
#         self.logger.info(f"Logging initialized to {log_file}")
    
#     def get_supported_models(self):
#         """Returns the mapping of LLM providers to their supported models."""
#         return SUPPORTED_MODELS
    
#     def _fallback_parse_response(self, raw_response: str, structured_model: Type[BaseModel], rag_results: Optional[str] = None) -> dict:
#         """Fallback parser to extract structured data from non-JSON responses."""
#         self.logger.warning(f"Attempting fallback parsing for raw response: {raw_response}")
#         try:
#             structured_data = {
#                 "conversational_response": raw_response.strip() or "I'm sorry, I need more details about the incident to assist you.",
#                 "scam_incident_date": "",
#                 "scam_type": "UNKNOWN",
#                 "scam_approach_platform": "UNKNOWN",
#                 "scam_communication_platform": "UNKNOWN",
#                 "scam_transaction_type": "UNKNOWN",
#                 "scam_beneficiary_platform": "UNKNOWN",
#                 "scam_beneficiary_identifier": "UNKNOWN",
#                 "scam_contact_no": "UNKNOWN",
#                 "scam_email": "",
#                 "scam_moniker": "UNKNOWN",
#                 "scam_url_link": "",
#                 "scam_amount_lost": 0.0,
#                 "scam_incident_description": raw_response.strip() or "",
#                 "scam_specific_details": {}
#             }
            
#             # Extract from RAG results if available
#             if rag_results:
#                 try:
#                     rag_data = json.loads(rag_results)
#                     if isinstance(rag_data, list) and rag_data:
#                         first_result = rag_data[0]
#                         structured_data.update({
#                             "scam_incident_date": first_result.get("scam_incident_date", ""),
#                             "scam_type": first_result.get("scam_type", "UNKNOWN"),
#                             "scam_approach_platform": first_result.get("scam_approach_platform", "UNKNOWN"),
#                             "scam_communication_platform": first_result.get("scam_communication_platform", "UNKNOWN"),
#                             "scam_transaction_type": first_result.get("scam_transaction_type", "UNKNOWN"),
#                             "scam_beneficiary_platform": first_result.get("scam_beneficiary_platform", "UNKNOWN"),
#                             "scam_beneficiary_identifier": first_result.get("scam_beneficiary_identifier", "UNKNOWN"),
#                             "scam_contact_no": first_result.get("scam_contact_no", "UNKNOWN"),
#                             "scam_email": first_result.get("scam_email", ""),
#                             "scam_moniker": first_result.get("scam_moniker", "UNKNOWN"),
#                             "scam_url_link": first_result.get("scam_url_link", ""),
#                             "scam_amount_lost": first_result.get("scam_amount_lost", 0.0),
#                             "scam_incident_description": first_result.get("scam_incident_description", raw_response.strip() or ""),
#                             "scam_specific_details": first_result.get("scam_specific_details", {})
#                         })
#                         structured_data["conversational_response"] = (
#                             f"I'm sorry to hear about your experience. Based on our records, we found a similar scam involving "
#                             f"{first_result.get('scam_type', 'unknown')} on {first_result.get('scam_approach_platform', 'unknown')}. "
#                             f"Please provide more details about your incident to assist with the investigation."
#                         )
#                 except json.JSONDecodeError:
#                     self.logger.warning("Failed to parse RAG results in fallback parsing")
            
#             # Extract from raw response if no RAG results
#             if "phishing" in raw_response.lower():
#                 structured_data["scam_type"] = "PHISHING"
#             elif "ecommerce" in raw_response.lower() or "non-delivery" in raw_response.lower():
#                 structured_data["scam_type"] = "ECOMMERCE"
#             elif "government" in raw_response.lower():
#                 structured_data["scam_type"] = "GOVERNMENT IMPERSONATION"
            
#             platforms = ["Facebook", "WhatsApp", "Instagram", "Email", "Call", "SMS"]
#             for platform in platforms:
#                 if platform.lower() in raw_response.lower():
#                     structured_data["scam_approach_platform"] = platform.upper()
#                     structured_data["scam_communication_platform"] = platform.upper()
            
#             amount_match = re.search(r'\$\d+(\.\d{2})?', raw_response)
#             if amount_match:
#                 structured_data["scam_amount_lost"] = float(amount_match.group(0).replace('$', ''))
            
#             account_match = re.search(r'\b\d{6,12}\b', raw_response)
#             if account_match:
#                 structured_data["scam_beneficiary_identifier"] = account_match.group(0)
            
#             moniker_match = re.search(r'\b[A-Za-z0-9_]+\b', raw_response)
#             if moniker_match and len(moniker_match.group(0)) > 3:
#                 structured_data["scam_moniker"] = moniker_match.group(0)
            
#             structured_model(**structured_data)
#             self.structured_success_count += 1
#             self.logger.info("Fallback parsing succeeded")
#             return structured_data
#         except ValidationError as e:
#             self.structured_failure_count += 1
#             self.logger.error(f"Fallback parsing failed: {str(e)}")
#             return structured_data
    
#     def get_llm(self, provider: str = "Ollama", model: str = "mistral:7b"):
#         """Returns an LLM instance based on the provider and model."""
#         if provider not in SUPPORTED_MODELS:
#             self.logger.error(f"Unknown provider: {provider}")
#             raise ValueError(f"Unknown provider: {provider}. Must be one of {list(SUPPORTED_MODELS.keys())}")
        
#         if model not in SUPPORTED_MODELS[provider]:
#             self.logger.error(f"Model '{model}' not supported for provider '{provider}'")
#             raise ValueError(f"Model '{model}' not supported for provider '{provider}'. Must be one of {SUPPORTED_MODELS[provider]}")
        
#         try:
#             if provider == "Ollama":
#                 base_url = self.settings.agents.ollama_base_url
#                 self.logger.debug(f"Initializing Ollama LLM with model: {model}, base_url: {base_url}")
#                 return ChatOllama(model=model, base_url=base_url)
            
#             elif provider == "OpenAI":
#                 if not OPENAI_API_KEY:
#                     self.logger.error("OPENAI_API_KEY not found")
#                     raise ValueError("OPENAI_API_KEY not found in environment")
#                 self.logger.debug(f"Initializing OpenAI LLM with model: {model}")
#                 return ChatOpenAI(model=model, api_key=OPENAI_API_KEY)
        
#         except Exception as e:
#             self.logger.error(f"Failed to initialize LLM for provider {provider}, model {model}: {str(e)}")
#             raise
    
#     def get_structured_llm(self, provider: str = "Ollama", model: str = "mistral:7b", structured_model: Type[BaseModel] = None):
#         if provider not in SUPPORTED_MODELS:
#             self.logger.error(f"Unknown provider: {provider}")
#             raise ValueError(f"Unknown provider: {provider}. Must be one of {list(SUPPORTED_MODELS.keys())}")
        
#         if model not in SUPPORTED_MODELS[provider]:
#             self.logger.error(f"Model '{model}' not supported for provider '{provider}'")
#             raise ValueError(f"Model '{model}' not supported for provider '{provider}'. Must be one of {SUPPORTED_MODELS[provider]}")
        
#         if structured_model is None or not issubclass(structured_model, BaseModel):
#             self.logger.error("Invalid structured_model")
#             raise ValueError("structured_model must be a valid Pydantic BaseModel subclass")
        
#         try:
#             # Get the retrieve_scam_reports tool from PoliceTools
#             police_tools = PoliceTools()
#             tools = police_tools.get_tools()
            
#             if provider == "Ollama":
#                 base_url = self.settings.agents.ollama_base_url
#                 self.logger.debug(f"Initializing structured Ollama LLM with model: {model}, base_url: {base_url}")
#                 llm = ChatOllama(
#                     model=model,
#                     base_url=base_url,
#                     format="json"
#                 )
#                 return llm.bind_tools(tools, tool_choice="auto").with_fallbacks([
#                     ChatOllama(model=model, base_url=base_url).with_config(
#                         {"post_process": lambda x: self._fallback_parse_response(x, structured_model)}
#                     )
#                 ])
            
#             elif provider == "OpenAI":
#                 if not OPENAI_API_KEY:
#                     self.logger.error("OPENAI_API_KEY not found")
#                     raise ValueError("OPENAI_API_KEY not found in environment")
#                 self.logger.debug(f"Initializing structured OpenAI LLM with model: {model}")
#                 llm = ChatOpenAI(
#                     model=model,
#                     api_key=OPENAI_API_KEY,
#                     model_kwargs={
#                         "response_format": {
#                             "type": "json_schema",
#                             "json_schema": {
#                                 "name": "structured_output",
#                                 "schema": structured_model.model_json_schema(),
#                                 "strict": True
#                             }
#                         }
#                     }
#                 )
#                 return llm.bind_tools(tools, tool_choice="auto").with_config(
#                     {"post_process": lambda x: self._fallback_parse_response(x, structured_model)}
#                 )
        
#         except Exception as e:
#             self.logger.error(f"Failed to initialize structured LLM for provider {provider}, model {model}: {str(e)}")
#             raise
    
#     def get_structured_output_metrics(self) -> dict:
#         """Return metrics on structured output success/failure rates."""
#         total = self.structured_success_count + self.structured_failure_count
#         success_rate = (self.structured_success_count / total * 100) if total > 0 else 0
#         return {
#             "success_count": self.structured_success_count,
#             "failure_count": self.structured_failure_count,
#             "success_rate": success_rate
#         }

from langchain_ollama import ChatOllama
from config.settings import get_settings
from pydantic import BaseModel, ValidationError
from typing import Type, Optional
import logging
from pathlib import Path
import os
from dotenv import load_dotenv
import json
import re
from src.agents.remove.tools2 import PoliceTools

load_dotenv()

SUPPORTED_MODELS = {
    "Ollama": ["llama3.2:latest", "llama3.1:8b", "mistral:7b", "qwen2.5:7b", "phi3:3.8b"]
}

class LLMProvider:
    def __init__(self):
        self.settings = get_settings()
        self._setup_logging()
        self.structured_success_count = 0
        self.structured_failure_count = 0
    
    def _setup_logging(self):
        log_dir = Path(self.settings.log.directory) / self.settings.log.subdirectories["agent"]
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "llm_providers.log"
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
    
    def get_supported_models(self):
        return SUPPORTED_MODELS
    
    def _fallback_parse_response(self, raw_response: str, structured_model: Type[BaseModel], rag_results: Optional[str] = None) -> dict:
        self.logger.warning(f"Fallback parsing triggered for raw response: {raw_response}")
        structured_data = {
            "conversational_response": "I'm sorry, I need more details about the incident to assist you.",
            "scam_incident_date": "",
            "scam_type": "UNKNOWN",
            "scam_approach_platform": "UNKNOWN",
            "scam_communication_platform": "UNKNOWN",
            "scam_transaction_type": "UNKNOWN",
            "scam_beneficiary_platform": "UNKNOWN",
            "scam_beneficiary_identifier": "UNKNOWN",
            "scam_contact_no": "UNKNOWN",
            "scam_email": "",
            "scam_moniker": "UNKNOWN",
            "scam_url_link": "",
            "scam_amount_lost": 0.0,
            "scam_incident_description": raw_response.strip() or "",
            "scam_specific_details": {}
        }
        
        raw_lower = raw_response.lower()
        if "phishing" in raw_lower or "sms" in raw_lower:
            structured_data["scam_type"] = "PHISHING"
            structured_data["scam_approach_platform"] = "SMS"
            structured_data["scam_communication_platform"] = "SMS"
        elif "ecommerce" in raw_lower or "non-delivery" in raw_lower or "ticket" in raw_lower:
            structured_data["scam_type"] = "ECOMMERCE"
        elif "government" in raw_lower:
            structured_data["scam_type"] = "GOVERNMENT IMPERSONATION"
        
        platforms = ["Facebook", "WhatsApp", "Instagram", "SMS", "Call"]
        for platform in platforms:
            if platform.lower() in raw_lower:
                structured_data["scam_approach_platform"] = platform.upper()
                structured_data["scam_communication_platform"] = platform.upper()
        
        amount_match = re.search(r'\$\d+(\.\d{2})?', raw_response)
        if amount_match:
            structured_data["scam_amount_lost"] = float(amount_match.group(0).replace('$', ''))
        
        account_match = re.search(r'\b\d{6,12}\b', raw_response)
        if account_match:
            structured_data["scam_beneficiary_identifier"] = account_match.group(0)
        
        moniker_match = re.search(r'\b[A-Za-z0-9_]+\b', raw_response)
        if moniker_match and len(moniker_match.group(0)) > 3:
            structured_data["scam_moniker"] = moniker_match.group(0)
        
        try:
            structured_model(**structured_data)
            self.structured_success_count += 1
            self.logger.info("Fallback parsing succeeded")
        except ValidationError as e:
            self.structured_failure_count += 1
            self.logger.error(f"Fallback parsing failed: {str(e)}")
        return structured_data
    
    def get_llm(self, provider: str = "Ollama", model: str = "llama3.2"):
        if provider not in SUPPORTED_MODELS:
            self.logger.error(f"Unknown provider: {provider}")
            raise ValueError(f"Unknown provider: {provider}. Must be one of {list(SUPPORTED_MODELS.keys())}")
        
        if model not in SUPPORTED_MODELS[provider]:
            self.logger.error(f"Model '{model}' not supported for provider '{provider}'")
            raise ValueError(f"Model '{model}' not supported for provider '{provider}'. Must be one of {SUPPORTED_MODELS[provider]}")
        
        try:
            base_url = self.settings.agents.ollama_base_url
            self.logger.debug(f"Initializing Ollama LLM with model: {model}, base_url: {base_url}")
            return ChatOllama(
                model=model,
                base_url=base_url,
                temperature=0.0,
                num_ctx=4096,
                max_retries=3
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM for provider {provider}, model {model}: {str(e)}")
            raise
    
    def get_structured_llm(self, provider: str = "Ollama", model: str = "llama3.2", structured_model: Type[BaseModel] = None):
        if provider not in SUPPORTED_MODELS:
            self.logger.error(f"Unknown provider: {provider}")
            raise ValueError(f"Unknown provider: {provider}. Must be one of {list(SUPPORTED_MODELS.keys())}")
        
        if model not in SUPPORTED_MODELS[provider]:
            self.logger.error(f"Model '{model}' not supported for provider '{provider}'")
            raise ValueError(f"Model '{model}' not supported for provider '{provider}'. Must be one of {SUPPORTED_MODELS[provider]}")
        
        if structured_model is None or not issubclass(structured_model, BaseModel):
            self.logger.error("Invalid structured_model")
            raise ValueError("structured_model must be a valid Pydantic BaseModel subclass")
        
        try:
            police_tools = PoliceTools()
            tools = police_tools.get_tools()
            base_url = self.settings.agents.ollama_base_url
            self.logger.debug(f"Initializing structured Ollama LLM with model: {model}, base_url: {base_url}")
            llm = ChatOllama(
                model=model,
                base_url=base_url,
                format="json",
                temperature=0.0,
                num_ctx=4096,
                max_retries=3
            )
            return llm.bind_tools(tools)  # Removed strict=True
        except Exception as e:
            self.logger.error(f"Failed to initialize structured LLM for provider {provider}, model {model}: {str(e)}")
            raise
    
    def get_structured_output_metrics(self) -> dict:
        total = self.structured_success_count + self.structured_failure_count
        success_rate = (self.structured_success_count / total * 100) if total > 0 else 0
        return {
            "success_count": self.structured_success_count,
            "failure_count": self.structured_failure_count,
            "success_rate": success_rate
        }