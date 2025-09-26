from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, List, Any
from enum import Enum

class PoliceResponse(BaseModel):
    """Response structure for police AI conversational agent"""
    conversational_response: str = Field(default="",
        description="Natural language response summarizing the scam report or providing guidance."
    )
    scam_incident_date: str = Field(
        default="",
        description="Date when the scam incident occurred, in YYYY-MM-DD format."
    )
    scam_type: str = Field(
        default="",
        description="Type of scam (e.g., phishing, ecommerce, government officials impersonation)."
    )
    scam_approach_platform: str = Field(
        default="",
        description="Platform where the scam was initiated (e.g., Instagram, Facebook, WhatsApp, Call)."
    )
    scam_communication_platform: str = Field(
        default="",
        description="Platform used for communication during the scam (e.g., WhatsApp, email)."
    )
    scam_transaction_type: str = Field(
        default="",
        description="Type of transaction involved, if any (e.g., bank transfer, cryptocurrency)."
    )
    scam_beneficiary_platform: str = Field(
        default="",
        description="Bank name where the scammer received funds or benefits (e.g., HSBC, GXS)."
    )
    scam_beneficiary_identifier: str = Field(
        default="",
        description="Identifier for the scammer’s account (e.g., bank account number)."
    )
    scam_contact_no: str = Field(
        default="",
        description="Phone number or contact details used by the scammer, if available."
    )
    scam_email: str = Field(
        default="",
        description="Email used by the scammer, if available."
    )
    scam_moniker: str = Field(
        default="",
        description="Alias or name used by the scammer's online profile, if known."
    )
    scam_url_link: str = Field(
        default="",
        description="URLs used by the scammer, if available."
    )
    scam_amount_lost: float = Field(
        default=0.0,
        description="Monetary amount lost due to the scam, in the reported currency."
    )
    scam_incident_description: str = Field(
        default="",
        description="Detailed description of the scam incident provided by the victim."
    )

    @field_validator('scam_amount_lost', mode='before')
    @classmethod
    def parse_scam_amount(cls, v):
        """
        Preprocess 'scam_amount_lost' to handle 'NA' or similar strings and None.
        - If 'NA' (case-insensitive) or 'unknown', convert to 0.0.
        - If None, convert to 0.0.
        - Otherwise, coerce to float or raise ValueError.
        """
        if v is None or v == '':
            return 0.0
        if isinstance(v, str):
            if v.upper() in ('NA', 'N/A') or v.lower() == 'unknown':
                return 0.0
            try:
                return float(v)
            except ValueError:
                raise ValueError(f"Invalid value for scam_amount_lost: '{v}'. Must be a number or 'NA'.")
        return v

    @field_validator(
        'scam_incident_date',
        'scam_beneficiary_identifier',
        'scam_contact_no',
        'scam_email',
        'scam_url_link',
        mode='before'
    )
    @classmethod
    def parse_string_fields(cls, v, info):
        """
        Preprocess string fields to handle None.
        - If None, convert to empty string ("").
        - Otherwise, return as-is (Pydantic will validate as string).
        """
        return "" if v is None else v

    
class PoliceResponseSlots(str, Enum):
    scam_incident_date = "scam_incident_date"
    scam_type = "scam_type"
    scam_approach_platform = "scam_approach_platform"
    scam_communication_platform = "scam_communication_platform"
    scam_transaction_type = "scam_transaction_type"
    scam_beneficiary_platform = "scam_beneficiary_platform"
    scam_beneficiary_identifier = "scam_beneficiary_identifier"
    scam_contact_no = "scam_contact_no"
    scam_email = "scam_email"
    scam_moniker = "scam_moniker"
    scam_url_link = "scam_url_link"
    scam_amount_lost = "scam_amount_lost"
    scam_incident_description = "scam_incident_description"
    

class TechLiteracy(str, Enum):
    """Levels for technical literacy"""
    low = "low"
    high = "high"

class LanguageProficiency(str, Enum):
    """Levels for language proficiency"""
    low = "low"
    high = "high"

class EmotionalState(str, Enum):
    """Levels for emotional state"""
    distressed = "distressed"
    neutral = "neutral"
    
class ProfileDimension(BaseModel):
    """Pydantic class for profile dimension with level and confidence. To be used for each indicator defined in UserProfile pydantic class"""
    level: str = Field(default="", description="Categorical level")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in level inferred (0-1) for profile indicator.")

class UserProfile(BaseModel):
    """Pydantic class for user profile agent output"""
    tech_literacy: ProfileDimension = Field(default=ProfileDimension(level=""),  description="Inferred technology proficiency based on user input.")
    language_proficiency: ProfileDimension = Field(default=ProfileDimension(level=""),description="Inferred language proficiency based on user input.")
    emotional_state: ProfileDimension = Field(default=ProfileDimension(level=""), description="Inferred emotional state based on user input.")  
    
class KnowledgeBaseOutput(BaseModel):
    """Pydantic class for knowledgebase agent output"""
    strategy_type: Optional[str] = Field(default=None, description="Extracted strategy type for logging")
    language_proficiency: int = Field(..., ge=1, le=5, description="Rating for alignment of communication strategy with victim's language proficiency (1–5)")
    emotional_state: int = Field(..., ge=1, le=5, description="Rating for alignment of communication strategy with victim's emotional state (1–5)")
    tech_literacy: int = Field(..., ge=1, le=5, description="Rating for alignment of communication strategy with victim's tech literacy (1–5)")
    valid: bool = Field(..., description="Whether the strategy is deemed suitable for the victim's profile")
    def avg_rating(self) -> float:
        """Method to compute average rating produced by knowledgebase agent on communication strategy alignment with inferred victim profile."""
        ratings = [self.language_proficiency, self.emotional_state, self.tech_literacy]
        if not ratings:
            return 0.0
        avg = sum(ratings) / len(ratings) if ratings else 0.0
        return (avg-1)/4 #normalize: (avg-min)/(max-min)
    
class RagOutput (BaseModel):
    """Pydantic class for retrieval agent's rag_suggestion output structure."""
    scam_type: str = Field(default="", description="Scam type extracted from rag results")
    scam_details: List[str] = Field(default_factory=list, description="List of scam details to extract")
    
    
class RetrievalOutput(BaseModel):
    """Pydantic class for final retrieval output for profile_rag_ie_kb agent only. Includes strategies retrieved separately without processing from rag_agent."""
    scam_reports: List[Dict[str, Any]] = Field(default_factory=list, description="List of scam reports retrieved from database")
    strategies: List[Dict[str, Any]] = Field(default_factory=list, description="Available questioning strategies with type and question")
    rag_suggestions: Dict = Field(default_factory=dict, description="Processed suggestions from RAG LLM on scam_reports")


class VictimResponse(BaseModel):
    """Pydantic class for victim agent."""
    conversational_response: str = Field(default="", description="Conversational response of victim chatbot")
    end_conversation: bool = Field(default=False, description="Flag to signal end of the conversation")
    