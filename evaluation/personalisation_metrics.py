import pandas as pd
import json
from typing import Dict, Optional
import os
import sys
from dotenv import load_dotenv  # Add this import to load .env files

load_dotenv()  # Load environment variables from .env file if present

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.settings import get_settings  # Import settings for centralized paths
from config.id_manager import IDManager  # Reuse IDManager for incremental indexing
from pathlib import Path
import csv
from filelock import FileLock  # For safe appending

class AppropriatenessRatings(BaseModel):
    reasoning: str = Field(..., description="Step-by-step reasoning for the ratings")
    language_proficiency: int = Field(..., ge=1, le=5, description="Rating for alignment with victim's language proficiency (1–5)")
    emotional_state: int = Field(..., ge=1, le=5, description="Rating for alignment with victim's emotional state (1–5)")
    tech_literacy: int = Field(..., ge=1, le=5, description="Rating for alignment with victim's tech literacy (1–5)")
    valid: bool = Field(..., description="Whether the response is deemed appropriate overall (e.g., avg rating >= 3)")

    def avg_rating(self) -> float:
        """Compute average rating, normalized to 0-1 scale like KB agent."""
        ratings = [self.language_proficiency, self.emotional_state, self.tech_literacy]
        if not ratings:
            return 0.0
        avg = sum(ratings) / len(ratings)
        return (avg - 1) / 4  # Normalize: (avg - min) / (max - min)

class EvaluationAgent:
    def __init__(self, model_name: str = "qwen2.5:7b", llm_provider: str = "Ollama", temperature: float = 0.0):
        self.model_name = model_name
        self.llm_provider = llm_provider
        self.temperature = temperature
        self.judge_prompt = ChatPromptTemplate.from_template(self._get_judge_prompt_template())
        self.llm = self._get_llm(schema=AppropriatenessRatings)

    def _get_judge_prompt_template(self) -> str:
        return """
Victim Persona (Ground Truth):
- Tech Literacy: {tech_literacy} (low: avoid any jargon, simplify asks; high: use relevant terms like 'platform' or 'URL' without over-explaining)
- Language Proficiency: {language_proficiency} (low: basic vocab but allow complext structures; high: allow complex structures/vocab but not overly formal)
- Emotional State: {emotional_state} (distressed: strong empathy/reassurance first, e.g., 'I understand this is scary', 'This is helpful'; neutral: calm/professional, minimal empathy)

Evaluation Criteria (Rate 1-5 per dimension, 1=poor fit/major mismatch, 3=average/pragmatic fit, 5=excellent/near-perfect alignment. Avoid extremes unless justified):
- Tech Literacy: For low, rate 3-4 if no jargon and asks are basic (e.g., 2 if assumes 'URL' knowledge); for high, rate 3-4 if uses some terms appropriately (e.g., 5 if leverages details well, 2 if too simplistic).
- Language Proficiency: For low, rate 4-5 for mostly simple, rate 3 if using normal vocabulary (e.g. rate 4 for "Could you describe the incident?" or "Could you please fill in the specific details"). Long sentences should be rated 4-5 if police response is trying to ask step by step or even provided examples to help victim understand (even with complex sentence structure or more than 20 words); for high, rate 3-4 for professional/complex (e.g., 5 if matches fluency, 2 if too basic).
- Emotional State: For distressed, rate 4-5 if starts with clear empathy/reassurance (e.g., 3 if minimal). Rate 4-5 if acknowledgement of victim's assistance is empathetic (e.g. "It's helpful" or "This is crucial information"); for neutral, rate 3-4 if calm/factual (e.g., 5 if perfectly balanced, 2 if overly empathetic).
- Valid: True if average rating >=3, else False.

Chain-of-Thought:
1. Read victim's message and police response; note profile type (low/high).
2. Rate tech_literacy (1-5) with justification, referencing examples.
3. Rate language_proficiency (1-5) with justification, checking vocabulary. Do not penalize complex sentence structure or long sentences.
4. Rate emotional_state (1-5) with justification, checking for holistic balance (e.g., empathy boosting other scores for low profiles).
5. Compute average; set valid=True if avg >=3; adjust for pragmatism (e.g., no 1s/5s without strong evidence).
6. Compile reasoning summary, explaining any offsets for balance.

Output JSON matching schema: reasoning (str), ratings (1-5 ints), valid (bool).

Victim's Previous Message: {victim_message}
Police Response: {police_response}
"""

    def _get_llm(self, schema: Optional[BaseModel] = None):
        if self.llm_provider == "Ollama":
            params = {"model": self.model_name, "format": "json", "temperature": self.temperature}
            if schema:
                params["response_format"] = {"type": "json_schema", "json_schema": schema.model_json_schema()}
            return ChatOllama(**params)
        elif self.llm_provider == "OpenAI":
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required")
            base_llm = ChatOpenAI(model=self.model_name, api_key=api_key, temperature=self.temperature)
            return base_llm.with_structured_output(schema) if schema else base_llm
        else:
            raise ValueError(f"Unsupported provider: {self.llm_provider}")

    def evaluate_response(self, victim_msg: str, police_response: str, persona: Dict[str, str]) -> Dict:
        prompt_input = {
            "victim_message": victim_msg,
            "police_response": police_response,
            "tech_literacy": persona.get("tech_literacy", "unknown"),
            "language_proficiency": persona.get("language_proficiency", "unknown"),
            "emotional_state": persona.get("emotional_state", "unknown")
        }
        chain = self.judge_prompt | self.llm
        response = chain.invoke(prompt_input)
        if self.llm_provider == "OpenAI":
            output = response.model_dump()
        else:
            output = json.loads(response.content)
        return output

class EvaluationProcessor:
    def __init__(self, csv_path: str, json_path: str, output_path: str, id_file: str):
        self.csv_path = csv_path
        self.json_path = json_path
        self.output_path = output_path
        self.id_file = id_file
        self.lock_file = f"{output_path}.lock"  # Lock for safe appending
        self.id_manager = IDManager(csv_file=self.output_path, id_file=self.id_file)  # Reuse IDManager for incremental index
        self.ground_truth_profiles = self._load_ground_truth()
        self._ensure_csv_headers()

    def _load_ground_truth(self) -> Dict[int, Dict[str, str]]:
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        return {item["profile_id"]: item["user_profile"] for item in data}

    def _ensure_csv_headers(self):
        """Ensure output CSV has headers; create if not exists."""
        path = Path(self.output_path)
        path.parent.mkdir(parents=True, exist_ok=True)  # Create directories if needed
        file_exists = path.exists()

        if not file_exists:
            with open(self.output_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "index", "conversation_id", "profile_id","police_llm_model", "turn_count", "victim_message", "police_response",
                    "reasoning", "language_proficiency", "emotional_state", "tech_literacy",
                    "valid", "avg_rating_normalized", "timestamp"
                ])
            print(f"Created CSV with headers: {self.output_path}")

    def process(self, agent: EvaluationAgent):
        df = pd.read_csv(self.csv_path)
        grouped = df.groupby("conversation_id")

        # Load existing evaluations to avoid duplicates (use set of (conv_id, turn_count) for quick check)
        processed_pairs = set()
        if Path(self.output_path).exists():
            existing_df = pd.read_csv(self.output_path)
            for _, row in existing_df.iterrows():
                processed_pairs.add((row["conversation_id"], row["turn_count"]))

        new_count = 0

        for conv_id, group in grouped:
            profile_id = group["profile_id"].iloc[0]  # Assume consistent per conv
            persona = self.ground_truth_profiles.get(profile_id)
            if not persona:
                print(f"Warning: No persona for profile_id {profile_id}. Skipping conv {conv_id}")
                continue

            for i, row in group.iterrows():
                if row["sender_type"] == "police":
                    turn_count = row["turn_count"]
                    if (conv_id, turn_count) in processed_pairs:
                        print(f"Skipping already evaluated: conv {conv_id}, turn {turn_count}")
                        continue

                    # Get previous victim message (immediate prior row)
                    prev_idx = i - 1
                    # victim_msg = group.loc[prev_idx, "content"] if prev_idx >= 0 and group.loc[prev_idx, "sender_type"] == "victim" else "No prior message"

                    if prev_idx >= 0:
                        prev_sender = group.loc[prev_idx, "sender_type"]
                        if prev_sender in ["human", "victim"]:
                            victim_msg = group.loc[prev_idx, "content"]
                        else:
                            victim_msg = "No prior message"  # Or log a warning if it's unexpected
                    else:
                        victim_msg = "No prior message"
                        
                        
                    eval_output = agent.evaluate_response(victim_msg, row["content"], persona)
                    # Compute avg for output
                    avg_normalized = AppropriatenessRatings(**eval_output).avg_rating()

                    # Get next incremental index using IDManager
                    eval_index = self.id_manager.get_next_id()

                    # Prepare row as list in header order
                    csv_row = [
                        eval_index,
                        conv_id,
                        profile_id,
                        row["police_llm_model"],
                        turn_count,
                        victim_msg,
                        row["content"],
                        eval_output["reasoning"],
                        eval_output["language_proficiency"],
                        eval_output["emotional_state"],
                        eval_output["tech_literacy"],
                        eval_output["valid"],
                        avg_normalized,
                        row["timestamp"]
                    ]

                    # Append incrementally with lock
                    with FileLock(self.lock_file):
                        with open(self.output_path, mode="a", newline="", encoding="utf-8") as f:
                            writer = csv.writer(f)
                            writer.writerow(csv_row)

                    new_count += 1
                    print(f"Appended new evaluation for conv {conv_id}, turn {turn_count}")

        if new_count == 0:
            print("No new evaluations to append.")

if __name__ == "__main__":
    settings = get_settings()  # Use centralized settings

    # Example paths (customize based on your needs; add to DataSettings if centralizing further)
    csv_path = "simulations/phase_2/profile_rag_ie_kb/autonomous_conversation_history.csv"
    # csv_path = "simulations/phase_1/vanilla_rag/nonautonomous_conversation_history.csv"
    # csv_path = "simulations/phase_1/self_augmenting/nonautonomous_conversation_history.csv"
    
    # json_path = settings.data.victim_details_human_eval_json  # Reuse from settings if applicable
    json_path = settings.data.victim_details_json # Reuse from settings if applicable
    
    output_path = "evaluation/results/communication_appropriateness/phase_2/evaluated_autonomous_conversations_profile_rag_ie_kb.csv"
    # output_path = "evaluation/results/communication_appropriateness/phase_1/evaluated_nonautonomous_conversations_profile_rag_ie_kb.csv"
    # output_path = "evaluation/results/communication_appropriateness/phase_1/evaluated_nonautonomous_conversations_rag_ie.csv"
    id_file = "evaluation/results/communication_appropriateness/phase_2/profile_rag_ie_kb_last_eval_id.txt"  # Simple text file for last ID
    # id_file = "evaluation/results/communication_appropriateness/phase_1/rag_ie_last_eval_id.txt"  # Simple text file for last ID

    model_name = "gpt-4o-mini"
    llm_provider = "OpenAI"

    agent = EvaluationAgent(model_name=model_name, llm_provider=llm_provider)
    processor = EvaluationProcessor(csv_path=csv_path, json_path=json_path, output_path=output_path, id_file=id_file)
    processor.process(agent)
    
    
#     You are an expert AI judge evaluating how appropriate a police chatbot's response is to a scam victim's persona during a reporting conversation.

# Victim Persona (Ground Truth):
# - Tech Literacy: {tech_literacy} (low: use simple tech terms; high: can handle details)
# - Language Proficiency: {language_proficiency} (low: simple, clear sentences; high: more complex okay)
# - Emotional State: {emotional_state} (distressed: highly empathetic/reassuring; neutral: professional/factual)

# Evaluation Criteria (Rate 1-5 per dimension, 1=poor fit, 3=average fit (neutral), 5=excellent fit):
# - Tech Literacy: Avoid jargon for low; explain if needed. For persons with high tech literacy, use appropriate technical terms.
# - Language Proficiency: Short, simple sentences or vocabulary for low; use average or more complex vocabulary or sentence structures for high.
# - Emotional State: Empathetic/supportive for distressed; calm or straightforward for neutral.
# - Valid: True if average rating >=3, else False.

# Chain-of-Thought:
# 1. Read victim's message and police chatbot's message.
# 2. Rate tech_literacy (1-5) with justification.
# 3. Rate language_proficiency (1-5) with justification.
# 4. Rate emotional_state (1-5) with justification.
# 5. Compute average; set valid=True if avg >=3.
# 6. Compile reasoning summary.

# Output JSON matching schema: reasoning (str), ratings (1-5 ints), valid (bool).

# Victim's Previous Message: {victim_message}
# Police Response: {police_response}

# You are an expert AI judge evaluating how appropriate a police chatbot's response is to a scam victim's persona during a reporting conversation. Mimic human evaluators: Be pragmatic and holistic—ratings should cluster around 3-4 unless clearly poor/excellent. For low profiles, empathy can offset mild complexity; for high, expect balanced sophistication without excess.


# - Language Proficiency: For low, rate 4-5 for mostly simple, rate 3 if using normal vocabulary (e.g. rate 4 for "Could you describe the incident?"). Do not penalise for complex or long sentence structure (at least 3 even if more than 20 words). 
# Long sentences should be rated 4-5 if police response is trying to ask step by step or even provided examples to help victim understand (even with complex sentence structure); for high, rate 3-4 for professional/complex (e.g., 5 if matches fluency, 2 if too basic).