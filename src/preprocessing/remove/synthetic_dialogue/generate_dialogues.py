import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import random
import uuid

# Configuration
MODEL_NAME = "gpt-4o-mini"  # Use ChatGPT-4-mini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
SEED_DATA_DIR = os.path.join(SCRIPT_DIR, "seed_data")
SUBTOPICS_FILE = os.path.join(SEED_DATA_DIR, "subtopics_updated.json")
PERSONAS_FILE = os.path.join(SEED_DATA_DIR, "personas_updated.jsonl")
DIALOGUES_FILE = os.path.join(SEED_DATA_DIR, "dialogues.jsonl")
N_DIALOGUES_PER_SUBTOPIC = 7  # Number of dialogues per subtopic
N_TURNS = 6  # Number of turns per dialogue
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure API key is set

# Verify seed data directory and files
if not os.path.exists(SEED_DATA_DIR):
    raise FileNotFoundError(f"Seed data directory not found at {SEED_DATA_DIR}.")
if not os.path.exists(SUBTOPICS_FILE):
    raise FileNotFoundError(f"Subtopics file not found at {SUBTOPICS_FILE}.")
if not os.path.exists(PERSONAS_FILE):
    raise FileNotFoundError(f"Personas file not found at {PERSONAS_FILE}.")

def load_subtopics(subtopics_file: str) -> Dict[str, List[str]]:
    """Load subtopics from a JSON file."""
    try:
        with open(subtopics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Error loading subtopics from {subtopics_file}: {e}")

def load_personas(personas_file: str) -> Dict[str, Dict[str, List[str]]]:
    """Load personas, grouping by topic, subtopic, and role."""
    personas_by_subtopic = {}
    with open(personas_file, 'r') as f:
        for line in f:
            try:
                persona_data = json.loads(line.strip())
                topic = persona_data["topic"]
                subtopic = persona_data["subtopic"]
                role = persona_data["role"]
                persona = persona_data["persona"]
                if topic not in personas_by_subtopic:
                    personas_by_subtopic[topic] = {}
                if subtopic not in personas_by_subtopic[topic]:
                    personas_by_subtopic[topic][subtopic] = {"victim": [], "officer": []}
                personas_by_subtopic[topic][subtopic][role].append(persona)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line in {personas_file}")
    return personas_by_subtopic

def generate_dialogue_prompt(subtopic: str, victim_persona: str, officer_persona: str, n_turns: int) -> str:
    """Generate a prompt for dialogue generation with escaped JSON."""
    prompt = f"""
You are tasked with generating a dialogue between a victim and a police officer about a fraud incident related to the subtopic '{subtopic}'. The dialogue should consist of {n_turns} turns, alternating between the victim and the officer, starting with the victim. Use chain-of-thought reasoning to ensure coherence and relevance. For each turn, provide a reasoning step explaining why the persona responds in that way.

The victim persona should reflect fraud psychology vulnerabilities (e.g., impulsivity, trust in authority, emotional distress, low tech literacy) and be likely to report the fraud. The officer persona should reflect trauma-informed policing principles (e.g., empathy, clear communication, procedural efficiency) and confirm the victim's first name, scam topic, and amount lost during the conversation.

Output format:
{{
  "dialogue": [
    {{"speaker": "victim", "text": "...", "reasoning": "..."}},
    {{"speaker": "officer", "text": "...", "reasoning": "..."}},
    ...
  ]
}}

Example:
Subtopic: Fake online reviews and testimonials
Victim Persona: A 65-year-old retiree who trusts online reviews and skips reading terms due to low tech literacy
Officer Persona: A compassionate female officer who reassures scam victims with clear, supportive communication
{{
  "dialogue": [
    {{"speaker": "victim", "text": "I bought a gadget online because it had great reviews, but it never arrived...", "reasoning": "The victim's low tech literacy and trust in reviews led to their scam vulnerability."}},
    {{"speaker": "officer", "text": "I’m so sorry to hear that, sir/madam. Can I have your first name, and how much did you lose?", "reasoning": "The officer uses empathy and confirms key details (name, scam type, amount lost)."}},
    {{"speaker": "victim", "text": "My name is John, and I lost $1200. I feel so foolish!", "reasoning": "The victim confirms the scam type and amount, expressing emotional distress."}},
    {{"speaker": "officer", "text": "You’re not foolish, John. Scammers are skilled at deception. Can you share the website’s name?", "reasoning": "The officer reassures the victim and gathers evidence."}},
    {{"speaker": "victim", "text": "It was some site called TechBargains. I didn’t check it closely.", "reasoning": "The victim’s low tech literacy prevented thorough verification."}},
    {{"speaker": "officer", "text": "Thank you, John. We’ll investigate. Here’s how to secure your accounts...", "reasoning": "The officer provides actionable advice, maintaining a supportive tone."}}
  ]
}}
"""
    return prompt

def generate_dialogue(subtopic: str, topic: str, victim_persona: str, officer_persona: str, vic_first_name: str, amount_lost: str, used_pairs: set) -> Dict:
    """Generate a dialogue between a victim and an officer."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment")
    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.4, max_tokens=4096)
    prompt = generate_dialogue_prompt(subtopic, victim_persona, officer_persona, N_TURNS)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("user", f"Subtopic: {subtopic}\nVictim Persona: {victim_persona}\nOfficer Persona: {officer_persona}")
    ])
    chain = prompt_template | llm
    try:
        output = chain.invoke({"subtopic": subtopic}).content
        start_idx, end_idx = output.find('{'), output.rfind('}')
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Invalid JSON format in LLM output")
        dialogue_data = json.loads(output[start_idx:end_idx+1])
        dialogue = {
            "topic": topic,
            "subtopic": subtopic,
            "vic_first_name": vic_first_name,
            "amount_lost": amount_lost,
            "victim_persona": victim_persona,
            "officer_persona": officer_persona,
            "dialogue": dialogue_data["dialogue"]
        }
        return dialogue
    except Exception as e:
        print(f"Error generating dialogue for {subtopic}: {e}, using fallback")
        return {
            "topic": topic,
            "subtopic": subtopic,
            "vic_first_name": vic_first_name,
            "amount_lost": amount_lost,
            "victim_persona": victim_persona,
            "officer_persona": officer_persona,
            "dialogue": [
                {"speaker": "victim", "text": f"I was scammed related to {subtopic}.", "reasoning": "Fallback due to error."},
                {"speaker": "officer", "text": f"Thank you for reporting, {vic_first_name}. How much did you lose?", "reasoning": "Fallback to confirm details."},
                {"speaker": "victim", "text": f"I lost {amount_lost}. I feel terrible.", "reasoning": "Fallback expressing distress."},
                {"speaker": "officer", "text": f"I understand, {vic_first_name}. Can you describe the scam?", "reasoning": "Fallback to gather details."},
                {"speaker": "victim", "text": f"It was related to {subtopic}. I don’t know what to do.", "reasoning": "Fallback showing confusion."},
                {"speaker": "officer", "text": f"We’ll help you, {vic_first_name}. Let’s secure your accounts.", "reasoning": "Fallback offering support."}
            ]
        }

def main():
    # Load subtopics and personas
    subtopics = load_subtopics(SUBTOPICS_FILE)
    personas_by_subtopic = load_personas(PERSONAS_FILE)
    
    # Track used persona pairs
    used_pairs = set()
    
    # Generate dialogues
    with open(DIALOGUES_FILE, 'w') as f:
        for topic, subtopic_list in subtopics.items():
            for subtopic in subtopic_list:
                print(f"Generating dialogues for subtopic: {subtopic} (topic: {topic})")
                victim_personas = personas_by_subtopic.get(topic, {}).get(subtopic, {}).get("victim", [])
                officer_personas = personas_by_subtopic.get(topic, {}).get(subtopic, {}).get("officer", [])
                if not victim_personas or not officer_personas:
                    print(f"Skipping {subtopic}: insufficient personas (victims: {len(victim_personas)}, officers: {len(officer_personas)})")
                    continue
                
                # Generate N_DIALOGUES_PER_SUBTOPIC dialogues
                for _ in range(N_DIALOGUES_PER_SUBTOPIC):
                    # Select random victim and officer personas
                    available_victims = [p for p in victim_personas if (p, subtopic) not in used_pairs]
                    available_officers = [p for p in officer_personas if (p, subtopic) not in used_pairs]
                    if not available_victims or not available_officers:
                        print(f"Insufficient unique personas for {subtopic}, using fallback")
                        victim_persona = random.choice(victim_personas)
                        officer_persona = random.choice(officer_personas)
                    else:
                        victim_persona = random.choice(available_victims)
                        officer_persona = random.choice(available_officers)
                    used_pairs.add((victim_persona, subtopic))
                    used_pairs.add((officer_persona, subtopic))
                    
                    # Generate random victim first name and amount lost
                    vic_first_name = random.choice(["Sarah", "John", "Emma", "Michael", "Lisa", "David"])
                    amount_lost = f"${random.randint(100, 5000)}"
                    
                    # Generate dialogue
                    dialogue = generate_dialogue(subtopic, topic, victim_persona, officer_persona, vic_first_name, amount_lost, used_pairs)
                    f.write(json.dumps(dialogue) + "\n")
                    print(f"Generated dialogue for {subtopic}: {vic_first_name}, {amount_lost}")

    print(f"Dialogues saved to {DIALOGUES_FILE}")

if __name__ == "__main__":
    main()
