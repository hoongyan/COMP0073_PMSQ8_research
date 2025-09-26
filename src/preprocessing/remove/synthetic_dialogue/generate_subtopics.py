import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import ast
from rouge_score import rouge_scorer
import re

# Configuration
MODEL_NAME = "gpt-4o-mini"  # Use ChatGPT-4-mini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
SEED_DATA_DIR = os.path.join(SCRIPT_DIR, "seed_data")  # Point to seed_data subdirectory
TOPICS_FILE = os.path.join(SEED_DATA_DIR, "topics.txt")
SUBTOPICS_FILE = os.path.join(SEED_DATA_DIR, "subtopics.json")
OUTPUT_FILE = os.path.join(SEED_DATA_DIR, "subtopics_updated.json")
N_NEW_SUB_TOPICS = 10  # Number of new subtopics per topic
ROUGE_THRESHOLD = 0.5  # Threshold for filtering similar subtopics
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure API key is set

# Create output directory if it doesn't exist
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# Initialize ROUGE scorer for filtering duplicates
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# Tailored prompt for scam-related subtopics (adapted from DiaSynth)
generate_subtopics_prompt = """
You are an expert on fraud and scams, informed by fraud psychology principles (e.g., urgency, authority impersonation, emotional manipulation, Self-Control Theory, Cialdini’s persuasion principles, Elaboration Likelihood Model).
Generate {n_sub_topics} diverse sub-topics for the given topic related to scams or law enforcement interactions.
Each sub-topic should reflect specific scam tactics, victim vulnerabilities (e.g., impulsivity, trust, fear of authority, emotional distress), or officer actions (e.g., evidence collection, victim support).
Ensure sub-topics are unique and avoid overlap with the following existing sub-topics: {existing_subtopics}.
Return the responses as a Python list, with each sub-topic enclosed in double quotes.
DO NOT FORGET to close the list with ']'.

Use the following examples as a reference:

## Example 1
Topic: Ecommerce Scam
Existing Subtopics: ["Fake online reviews and testimonials", "Flash sale and limited-time offer scams", "Secure payment badge fraud"]
Response: ["Misleading product descriptions", "Fake customer support chatbots", "Unauthorized subscription traps"]

## Example 2
Topic: Fraud Reporting
Existing Subtopics: ["Filing a scam report with police", "Gathering transaction details for evidence"]
Response: ["Submitting scam evidence to federal agencies", "Coordinating with cybercrime task forces", "Documenting phone scam interactions"]

Your response should start with '[' and end with ']'.
"""

def load_topics(topics_file: str) -> List[str]:
    """Load topics from a text file or use defaults."""
    if os.path.exists(topics_file):
        with open(topics_file, 'r') as f:
            return [line.strip().replace("scam prevention education", "scam prevention") for line in f.read().splitlines()]
    return [
        "ecommerce scam",
        "government impersonation scam",
        "phishing scam",
        "romance scam",
        "fraud reporting",
        "victim support",
        "scam prevention"
    ]

def load_existing_subtopics(subtopics_file: str) -> Dict[str, List[str]]:
    """Load existing subtopics from a JSON file."""
    if os.path.exists(subtopics_file):
        with open(subtopics_file, 'r') as f:
            return json.load(f)
    return {}

def filter_repeated_subtopics(new_subtopics: List[str], existing_subtopics: List[str]) -> List[str]:
    """Filter out new subtopics that are too similar to existing or other new subtopics using ROUGE-L."""
    unique_subtopics = []
    for subtopic in new_subtopics:
        if all(scorer.score(subtopic, existing)['rougeL'].fmeasure < ROUGE_THRESHOLD for existing in existing_subtopics + unique_subtopics):
            unique_subtopics.append(subtopic)
    return unique_subtopics

def generate_subtopics(topic: str, n_sub_topics: int, existing_subtopics: List[str]) -> List[str]:
    """Generate new sub-topics for a given topic using gpt-4o-mini, avoiding overlap with existing ones."""
    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.1, max_tokens=2048)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", generate_subtopics_prompt),
        ("user", "Topic: {topic}")
    ])
    chain = prompt_template | llm
    try:
        output = chain.invoke({"n_sub_topics": n_sub_topics, "existing_subtopics": existing_subtopics, "topic": topic}).content
        start_idx, end_idx = output.find('['), output.find(']')
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Invalid list format in LLM output")
        subtopics = ast.literal_eval(output[start_idx:end_idx+1])[:n_sub_topics]
        subtopics = [t for t in subtopics if isinstance(t, str)]
        return filter_repeated_subtopics(subtopics, existing_subtopics)
    except Exception as e:
        print(f"Error generating subtopics for {topic}: {e}, retrying with adjusted parameters")
        try:
            output = chain.invoke({"n_sub_topics": n_sub_topics, "existing_subtopics": existing_subtopics, "topic": topic, "max_tokens": 4096}).content
            start_idx, end_idx = output.find('['), output.find(']')
            subtopics = ast.literal_eval(output[start_idx:end_idx+1])[:n_sub_topics]
            subtopics = [t for t in subtopics if isinstance(t, str)]
            return filter_repeated_subtopics(subtopics, existing_subtopics)
        except Exception as e:
            print(f"Retry failed for {topic}: {e}, using fallback")
            return [f"{topic} new scenario {i+1}" for i in range(n_sub_topics)]

def validate_subtopics(subtopics: List[str], topic: str) -> List[str]:
    """Validate subtopics for relevance to fraud psychology or policing principles."""
    scam_keywords = ["scam", "fraud", "impersonation", "phishing", "urgency", "trust", "emotional", "authority"]
    policing_keywords = ["evidence", "support", "empathy", "counseling", "recovery", "education", "reporting"]
    return [st for st in subtopics if any(kw in st.lower() for kw in scam_keywords + policing_keywords)]

def main():
    # Load topics and existing subtopics
    topics = load_topics(TOPICS_FILE)
    topic_subtopics = load_existing_subtopics(SUBTOPICS_FILE)
    
    # Generate new subtopics
    for topic in topics:
        print(f"Generating new subtopics for: {topic}")
        existing_subtopics = topic_subtopics.get(topic, [])
        new_subtopics = generate_subtopics(topic, N_NEW_SUB_TOPICS, existing_subtopics)
        new_subtopics = validate_subtopics(new_subtopics, topic)
        topic_subtopics[topic] = existing_subtopics + new_subtopics
        print(f"New subtopics for {topic}: {new_subtopics}")
    
    # Save updated subtopics
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(topic_subtopics, f, indent=2)
    print(f"Updated subtopics saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()