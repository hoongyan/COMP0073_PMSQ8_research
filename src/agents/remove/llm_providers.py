# from langchain_ollama import ChatOllama
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# import os
# from dotenv import load_dotenv
# from pydantic import BaseModel
# from typing import Type

# # Load environment variables from .env file
# load_dotenv()

# # Access API keys from environment variables
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# SUPPORTED_MODELS = {
#     "Ollama": ["llama3.2", "llama3.1:8b", "qwen:7b", "deepseek-r1:7b", "mistral:7b", "gemma2:9b", "qwen2.5:7b", "phi3:3.8b"],
#     "OpenAI": ["gpt-4o-mini", "gpt-4", "gpt-4o-mini-2024-07-18"],
#     # "Anthropic": ["claude-3-5-sonnet", "claude-3-7-sonnet"]
# }

# def get_supported_models():
#     """Returns the mapping of LLM providers to their supported models."""
#     return SUPPORTED_MODELS

# def get_llm(provider: str = "Ollama", model: str = "llama3.2"):
#     """Returns an LLM instance based on the provider and model."""
#     if provider not in SUPPORTED_MODELS:
#         raise ValueError(f"Unknown provider: {provider}. Must be one of {list(SUPPORTED_MODELS.keys())}")
    
#     if model not in SUPPORTED_MODELS[provider]:
#         raise ValueError(f"Model '{model}' not supported for provider '{provider}'. Must be one of {SUPPORTED_MODELS[provider]}")
    
#     if provider == "Ollama":
#         base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
#         return ChatOllama(model=model, base_url=base_url)
    
#     elif provider == "OpenAI":
#         if not OPENAI_API_KEY:
#             raise ValueError("OPENAI_API_KEY not found in environment or config.py")
#         return ChatOpenAI(model=model, api_key=OPENAI_API_KEY)
    
#     # elif provider == "Anthropic":
#     #     if not ANTHROPIC_API_KEY:
#     #         raise ValueError("ANTHROPIC_API_KEY not found in environment or config.py")
#     #     return ChatAnthropic(model=model, api_key=ANTHROPIC_API_KEY)
    
# def get_structured_llm(provider: str = "Ollama", model: str = "llama3.2", structured_model: Type[BaseModel] = None):
#     """Returns an LLM instance configured for structured output using a specified Pydantic model."""
#     if provider not in SUPPORTED_MODELS:
#         raise ValueError(f"Unknown provider: {provider}. Must be one of {list(SUPPORTED_MODELS.keys())}")
    
#     if model not in SUPPORTED_MODELS[provider]:
#         raise ValueError(f"Model '{model}' not supported for provider '{provider}'. Must be one of {SUPPORTED_MODELS[provider]}")
    
#     if structured_model is None or not issubclass(structured_model, BaseModel):
#         raise ValueError("structured_model must be a valid Pydantic BaseModel subclass")

#     if provider == "Ollama":
#         base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
#         return ChatOllama(
#             model=model,
#             base_url=base_url,
#             format=structured_model.model_json_schema()
#         )
    
#     elif provider == "OpenAI":
#         if not OPENAI_API_KEY:
#             raise ValueError("OPENAI_API_KEY not found in environment or config.py")
#         return ChatOpenAI(
#                 model=model,
#                 api_key=OPENAI_API_KEY,
#                 response_format={
#                     "type": "json_schema",
#                     "json_schema": {
#                         "name": "structured_output",
#                         "schema": structured_model.model_json_schema(),
#                         "strict": False
#                     }
#                 }
#             )
    
#     # elif provider == "Anthropic":
#     #     if not ANTHROPIC_API_KEY:
#     #         raise ValueError("ANTHROPIC_API_KEY not found in environment or config.py")
#     #         llm = ChatAnthropic(model=model, api_key=ANTHROPIC_API_KEY)
#     #         structured_llm = llm.with_structured_output(structured_model, method="json_schema")
#     #         logging.debug(f"Successfully created structured LLM for Anthropic with model {model}")
#     #         return structured_llm



from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from config.settings import get_settings
from pydantic import BaseModel
from typing import Type
import logging
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SUPPORTED_MODELS = {
    "Ollama": ["llama3.2", "llama3.1:8b", "qwen:7b", "deepseek-r1:7b", "mistral:7b", "gemma2:9b", "qwen2.5:7b", "phi3:3.8b"],
    "OpenAI": ["gpt-4o-mini", "gpt-4", "gpt-4o-mini-2024-07-18"],
}

class LLMProvider:
    """Manages LLM initialization and configuration."""
    
    def __init__(self):
        """Initialize with settings and logging."""
        self.settings = get_settings()
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging to write to LLM-specific log file."""
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
        """Returns the mapping of LLM providers to their supported models."""
        return SUPPORTED_MODELS
    
    def get_llm(self, provider: str = "Ollama", model: str = "llama3.2"):
        """Returns an LLM instance based on the provider and model."""
        if provider not in SUPPORTED_MODELS:
            self.logger.error(f"Unknown provider: {provider}")
            raise ValueError(f"Unknown provider: {provider}. Must be one of {list(SUPPORTED_MODELS.keys())}")
        
        if model not in SUPPORTED_MODELS[provider]:
            self.logger.error(f"Model '{model}' not supported for provider '{provider}'")
            raise ValueError(f"Model '{model}' not supported for provider '{provider}'. Must be one of {SUPPORTED_MODELS[provider]}")
        
        try:
            if provider == "Ollama":
                base_url = self.settings.agents.ollama_base_url
                self.logger.debug(f"Initializing Ollama LLM with model: {model}, base_url: {base_url}")
                return ChatOllama(model=model, base_url=base_url)
            
            elif provider == "OpenAI":
                if not OPENAI_API_KEY:
                    self.logger.error("OPENAI_API_KEY not found")
                    raise ValueError("OPENAI_API_KEY not found in environment")
                self.logger.debug(f"Initializing OpenAI LLM with model: {model}")
                return ChatOpenAI(model=model, api_key=OPENAI_API_KEY)
        
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM for provider {provider}, model {model}: {str(e)}")
            raise
    
    def get_structured_llm(self, provider: str = "Ollama", model: str = "llama3.2", structured_model: Type[BaseModel] = None):
        """Returns an LLM instance configured for structured output."""
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
            if provider == "Ollama":
                base_url = self.settings.agents.ollama_base_url
                self.logger.debug(f"Initializing structured Ollama LLM with model: {model}, base_url: {base_url}")
                return ChatOllama(
                    model=model,
                    base_url=base_url,
                    format=structured_model.model_json_schema()
                )
            
            elif provider == "OpenAI":
                if not OPENAI_API_KEY:
                    self.logger.error("OPENAI_API_KEY not found")
                    raise ValueError("OPENAI_API_KEY not found in environment")
                self.logger.debug(f"Initializing structured OpenAI LLM with model: {model}")
                return ChatOpenAI(
                    model=model,
                    api_key=OPENAI_API_KEY,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "structured_output",
                            "schema": structured_model.model_json_schema(),
                            "strict": False
                        }
                    }
                )
        
        except Exception as e:
            self.logger.error(f"Failed to initialize structured LLM for provider {provider}, model {model}: {str(e)}")
            raise