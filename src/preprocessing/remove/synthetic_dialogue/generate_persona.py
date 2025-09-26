# import json
# import os
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from typing import List, Dict
# import ast
# from rouge_score import rouge_scorer
# import re

# # Configuration
# MODEL_NAME = "gpt-4o-mini"  # Use ChatGPT-4-mini
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
# SEED_DATA_DIR = os.path.join(SCRIPT_DIR, "seed_data")
# SUBTOPICS_FILE = os.path.join(SEED_DATA_DIR, "subtopics_updated.json")
# PERSONA_SRC_FILE = os.path.join(SEED_DATA_DIR, "persona.jsonl")  # Source personas
# PERSONAS_FILE = os.path.join(SEED_DATA_DIR, "personas_updated.jsonl")  # Output file
# N_PERSONAS = 6  # Number of personas per subtopic
# ROUGE_THRESHOLD = 0.6  # Relaxed threshold for filtering similar personas
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure API key is set

# # Verify seed data directory and files
# if not os.path.exists(SEED_DATA_DIR):
#     raise FileNotFoundError(f"Seed data directory not found at {SEED_DATA_DIR}.")
# if not os.path.exists(SUBTOPICS_FILE):
#     raise FileNotFoundError(f"Subtopics file not found at {SUBTOPICS_FILE}.")
# if not os.path.exists(PERSONA_SRC_FILE):
#     raise FileNotFoundError(f"Source persona file not found at {PERSONA_SRC_FILE}.")

# # Initialize ROUGE scorer for filtering duplicates
# scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

# # Tailored prompt for persona generation
# generate_personas_prompt = """
# You are an expert persona generator specializing in fraud psychology and trauma-informed policing.
# You will be given a subtopic related to scams, fraud reporting, victim support, or scam prevention education.
# Your task is to generate {n_personas} diverse personas who are most likely to have a conversation about the given subtopic.
# Personas should include specific demographics (e.g., age, gender, occupation) and reflect victim vulnerabilities (e.g., impulsivity, trust in authority, emotional distress, low tech literacy) or officer characteristics (e.g., empathetic and supportive, task-focused and procedural) relevant to the subtopic.
# Ensure personas are not repetitive and align with fraud psychology (e.g., Self-Control Theory, Cialdini’s persuasion principles) or trauma-informed policing principles (e.g., empathy, procedural efficiency).
# Return the responses as a Python list, with each persona enclosed in double quotes.
# DO NOT FORGET to close the list with ']'.

# Use the following examples as a reference:

# ## Example 1
# Subtopic: Fake online reviews and testimonials
# n_personas: 4
# Personas: ["A 65-year-old retiree who trusts online reviews and skips reading terms due to low tech literacy", "A 40-year-old woman who believes in the goodness of online sellers and feels confident in her judgment", "A compassionate female officer who reassures scam victims with clear, supportive communication", "A task-focused detective who collects precise scam details to build strong cases"]

# ## Example 2
# Subtopic: Emotional counseling for scam victims
# n_personas: 4
# Personas: ["A 52-year-old woman who shops online to cope with anxiety, seeking emotional support after a scam", "A 30-year-old man who responds to urgent phishing emails out of fear, needing counseling", "A supportive officer who provides victims with recovery resources and emotional guidance", "A procedural officer who ensures victims feel safe during reporting"]

# Your response should start with '[' and end with ']'.
# """

# def load_subtopics(subtopics_file: str) -> Dict[str, List[str]]:
#     """Load subtopics from a JSON file."""
#     try:
#         with open(subtopics_file, 'r') as f:
#             return json.load(f)
#     except Exception as e:
#         raise FileNotFoundError(f"Error loading subtopics from {subtopics_file}: {e}")

# def load_existing_personas(persona_file: str) -> List[str]:
#     """Load existing personas from a JSONL file."""
#     personas = []
#     if os.path.exists(persona_file):
#         with open(persona_file, 'r') as f:
#             for line in f:
#                 try:
#                     persona_data = json.loads(line.strip())
#                     personas.append(persona_data.get("persona", ""))
#                 except json.JSONDecodeError:
#                     print(f"Skipping invalid JSON line in {persona_file}")
#     return personas

# def map_existing_personas(subtopic: str, existing_personas: List[str]) -> List[str]:
#     """Map relevant personas from persona.jsonl to the subtopic."""
#     relevant_personas = []
#     for persona in existing_personas:
#         if any(keyword in persona.lower() for keyword in subtopic.lower().split()):
#             relevant_personas.append(persona)
#     return relevant_personas[:N_PERSONAS]  # Return up to N_PERSONAS

# def filter_repeated_personas(new_personas: List[str], existing_personas: List[str]) -> List[str]:
#     """Filter out personas that are too similar to existing or other new personas using ROUGE-L."""
#     unique_personas = []
#     filtered_out = []
#     for persona in new_personas:
#         if all(scorer.score(persona, existing)['rougeL'].fmeasure < ROUGE_THRESHOLD for existing in existing_personas + unique_personas):
#             unique_personas.append(persona)
#         else:
#             filtered_out.append(persona)
#     if filtered_out:
#         print(f"Filtered out due to ROUGE-L > {ROUGE_THRESHOLD}: {filtered_out}")
#     return unique_personas

# def validate_personas(personas: List[str], subtopic: str) -> List[str]:
#     """Validate personas for relevance to fraud psychology or policing principles."""
#     scam_keywords = ["scam", "fraud", "impersonation", "phishing", "urgency", "trust", "emotional", "authority", "deception", "manipulation", "vulnerable", "anxiety", "impulsive", "literacy"]
#     policing_keywords = ["officer", "detective", "evidence", "support", "empathy", "counseling", "recovery", "education", "reporting", "investigation", "prevention", "victim", "procedural"]
#     validated = []
#     for p in personas:
#         if any(kw in p.lower() for kw in scam_keywords + policing_keywords + subtopic.lower().split()):
#             validated.append(p)
#         else:
#             print(f"Persona filtered out for {subtopic} (no relevant keywords): {p}")
#     if len(validated) < N_PERSONAS:
#         print(f"Warning: Only {len(validated)} valid personas for {subtopic}, adding fallbacks")
#         fallbacks = [f"A person affected by {subtopic} with relevant concerns" for _ in range(N_PERSONAS - len(validated))]
#         validated.extend(fallbacks)
#     return validated[:N_PERSONAS]

# def generate_personas(subtopic: str, n_personas: int, existing_personas: List[str]) -> List[str]:
#     """Generate new personas for a given subtopic using gpt-4o-mini."""
#     if not OPENAI_API_KEY:
#         raise ValueError("OPENAI_API_KEY not found in environment")
#     llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.3, max_tokens=4096)
#     prompt_template = ChatPromptTemplate.from_messages([
#         ("system", generate_personas_prompt),
#         ("user", "Subtopic: {subtopic}")
#     ])
#     chain = prompt_template | llm
#     try:
#         output = chain.invoke({"n_personas": n_personas, "subtopic": subtopic}).content
#         start_idx, end_idx = output.find('['), output.find(']')
#         if start_idx == -1 or end_idx == -1:
#             raise ValueError("Invalid list format in LLM output")
#         personas = ast.literal_eval(output[start_idx:end_idx+1])
#         personas = [p for p in personas if isinstance(p, str)]
#         filtered_personas = filter_repeated_personas(personas, existing_personas)
#         validated_personas = validate_personas(filtered_personas, subtopic)
#         return validated_personas[:n_personas]
#     except Exception as e:
#         print(f"Error generating personas for {subtopic}: {e}, retrying with higher temperature")
#         try:
#             llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.5, max_tokens=4096)
#             chain = prompt_template | llm
#             output = chain.invoke({"n_personas": n_personas, "subtopic": subtopic}).content
#             start_idx, end_idx = output.find('['), output.find(']')
#             personas = ast.literal_eval(output[start_idx:end_idx+1])
#             personas = [p for p in personas if isinstance(p, str)]
#             filtered_personas = filter_repeated_personas(personas, existing_personas)
#             validated_personas = validate_personas(filtered_personas, subtopic)
#             return validated_personas[:n_personas]
#         except Exception as e:
#             print(f"Retry failed for {subtopic}: {e}, using fallbacks")
#             return [f"A person affected by {subtopic} with relevant concerns" for _ in range(n_personas)]

# def main():
#     # Load subtopics and existing personas
#     subtopics = load_subtopics(SUBTOPICS_FILE)
#     existing_personas = load_existing_personas(PERSONA_SRC_FILE)
    
#     # Map existing personas to subtopics
#     subtopic_personas = {subtopic: [] for topic in subtopics for subtopic in subtopics[topic]}
#     for subtopic in subtopic_personas:
#         mapped_personas = map_existing_personas(subtopic, existing_personas)
#         subtopic_personas[subtopic].extend(mapped_personas)
    
#     # Generate new personas for each subtopic
#     with open(PERSONAS_FILE, 'w') as f:
#         for topic, subtopic_list in subtopics.items():
#             for subtopic in subtopic_list:
#                 print(f"Generating personas for subtopic: {subtopic} (topic: {topic})")
#                 # Use existing personas if enough, otherwise generate new ones
#                 current_personas = subtopic_personas.get(subtopic, [])
#                 remaining = N_PERSONAS - len(current_personas)
#                 if remaining > 0:
#                     new_personas = generate_personas(subtopic, remaining, existing_personas + current_personas)
#                     subtopic_personas[subtopic].extend(new_personas)
#                 # Write all personas for this subtopic
#                 for persona in subtopic_personas[subtopic][:N_PERSONAS]:
#                     f.write(json.dumps({"subtopic": subtopic, "persona": persona}) + "\n")
#                 print(f"Personas for {subtopic}: {subtopic_personas[subtopic][:N_PERSONAS]}")
#                 existing_personas.extend(subtopic_personas[subtopic][:N_PERSONAS])  # Update to avoid duplicates
    
#     print(f"Personas saved to {PERSONAS_FILE}")

# if __name__ == "__main__":
#     main()


import json
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict
import ast
from rouge_score import rouge_scorer
import random

# Configuration
MODEL_NAME = "gpt-4o-mini"  # Use ChatGPT-4-mini
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # Get the script's directory
SEED_DATA_DIR = os.path.join(SCRIPT_DIR, "seed_data")
SUBTOPICS_FILE = os.path.join(SEED_DATA_DIR, "subtopics_updated.json")
PERSONA_SRC_FILE = os.path.join(SEED_DATA_DIR, "persona.jsonl")  # Source personas
PERSONAS_FILE = os.path.join(SEED_DATA_DIR, "personas_updated.jsonl")  # Output file
N_PERSONAS_PER_ROLE = 3  # Number of victim and officer personas per subtopic
ROUGE_THRESHOLD = 0.6  # Threshold for filtering similar personas
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure API key is set

# Verify seed data directory and files
if not os.path.exists(SEED_DATA_DIR):
    raise FileNotFoundError(f"Seed data directory not found at {SEED_DATA_DIR}.")
if not os.path.exists(SUBTOPICS_FILE):
    raise FileNotFoundError(f"Subtopics file not found at {SUBTOPICS_FILE}.")
if not os.path.exists(PERSONA_SRC_FILE):
    raise FileNotFoundError(f"Source persona file not found at {PERSONA_SRC_FILE}.")

# Initialize ROUGE scorer for filtering duplicates
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def load_subtopics(subtopics_file: str) -> Dict[str, List[str]]:
    """Load subtopics from a JSON file."""
    try:
        with open(subtopics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        raise FileNotFoundError(f"Error loading subtopics from {subtopics_file}: {e}")

def load_existing_personas(persona_file: str) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Load existing personas, splitting into victims and officers."""
    victim_personas = []
    officer_personas = []
    if os.path.exists(persona_file):
        with open(persona_file, 'r') as f:
            for line in f:
                try:
                    persona_data = json.loads(line.strip())
                    persona_entry = {
                        "persona": persona_data.get("persona", ""),
                        "topic": persona_data.get("topic", ""),
                        "role": persona_data.get("role", "")
                    }
                    if persona_data.get("role") == "Officer":
                        officer_personas.append(persona_entry)
                    elif persona_data.get("role") == "Victim":
                        victim_personas.append(persona_entry)
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {persona_file}")
    return victim_personas, officer_personas

def select_seed_personas(subtopic: str, topic: str, personas: List[Dict[str, str]], role: str, used_personas: set, max_examples: int = 3) -> List[str]:
    """Select up to max_examples personas based on role, topic, and excluding used personas."""
    topic_map = {
        "ecommerce scam": "Ecommerce",
        "government impersonation scam": "GOIS",
        "phishing scam": "Phishing",
        "romance scam": "Romance",
        "fraud reporting": "All",
        "victim support": "All",
        "scam prevention education": "All"
    }
    target_topic = topic_map.get(topic, "All")
    subtopic_keywords = subtopic.lower().split()
    relevant_personas = []
    
    # Shuffle personas to ensure varied selection
    shuffled_personas = random.sample(personas, len(personas))
    for persona_data in shuffled_personas:
        persona = persona_data["persona"]
        if persona in used_personas:
            continue
        persona_topic = persona_data["topic"]
        persona_role = persona_data["role"]
        if role == "officer" and persona_role == "Officer" and persona_topic == "All":
            relevant_personas.append(persona)
        elif role == "victim" and persona_role == "Victim" and persona_topic == target_topic:
            relevant_personas.append(persona)
        elif role == persona_role.lower() and target_topic == "All" and any(keyword in persona.lower() for keyword in subtopic_keywords + ["scam", "fraud", "phishing", "victim", "officer", "reporting"]):
            relevant_personas.append(persona)
        if len(relevant_personas) >= max_examples:
            break
    
    return relevant_personas[:max_examples]

def generate_victim_personas_prompt(subtopic: str, n_personas: int, seed_personas: List[str]) -> str:
    """Generate a prompt for victim personas focused on fraud reporting."""
    base_prompt = """
You are an expert persona generator specializing in fraud psychology.
You will be given a subtopic related to scams, fraud reporting, victim support, or scam prevention education.
Your task is to generate {n_personas} diverse victim personas who are likely to report a fraud related to the given subtopic.
Personas should include specific demographics (e.g., age, sex, occupation) and reflect victim vulnerabilities informed by fraud psychology, such as:
- Dispositional factors: impulsivity, high trust, emotional instability, deference to authority (Self-Control Theory, Five-Factor Model).
- Experiential factors: frequent online shopping, high social media usage, low IT literacy, or lack of scam awareness (Routine Activities Theory).
- Situational factors: emotional distress, urgency, or trust in authority figures during the scam.
No names are required.
Ensure personas are not repetitive and are likely to engage in fraud reporting. Each persona should only be described in one concise sentence.
Return the responses as a Python list, with each persona enclosed in double quotes.
DO NOT FORGET to close the list with ']'.

Use the following examples from an existing persona dataset as a reference:
"""
    examples = ""
    for i, persona in enumerate(seed_personas, 1):
        examples += f"""
## Example {i}
Subtopic: {subtopic}
n_personas: 1
Personas: ["{persona}"]
"""
    examples += """
Your response should start with '[' and end with ']'.
"""
    return base_prompt.format(n_personas=n_personas) + examples

def generate_officer_personas_prompt(subtopic: str, n_personas: int, seed_personas: List[str]) -> str:
    """Generate a prompt for officer personas focused on trauma-informed policing."""
    base_prompt = """
You are an expert persona generator specializing in trauma-informed policing.
You will be given a subtopic related to scams, fraud reporting, victim support, or scam prevention education.
Your task is to generate {n_personas} diverse police officer personas who are likely to assist victims in reporting a fraud related to the given subtopic.
Personas should include specific demographics (e.g., age, sex, experience level) and reflect trauma-informed policing or victim support characteristics, such as empathy, clear communication, procedural efficiency, or providing emotional and practical support.
No names are required.
Ensure personas are not repetitive and are suited to handle fraud reporting. Each persona should only be described in one concise sentence.
Return the responses as a Python list, with each persona enclosed in double quotes.
DO NOT FORGET to close the list with ']'.

Use the following examples from an existing persona dataset as a reference:
"""
    examples = ""
    for i, persona in enumerate(seed_personas, 1):
        examples += f"""
## Example {i}
Subtopic: {subtopic}
n_personas: 1
Personas: ["{persona}"]
"""
    examples += """
Your response should start with '[' and end with ']'.
"""
    return base_prompt.format(n_personas=n_personas) + examples

def map_existing_personas(subtopic: str, topic: str, personas: List[Dict[str, str]], role: str, used_personas: set) -> List[str]:
    """Map relevant personas from persona.jsonl to the subtopic based on role, topic, and excluding used personas."""
    topic_map = {
        "ecommerce scam": "Ecommerce",
        "government impersonation scam": "GOIS",
        "phishing scam": "Phishing",
        "romance scam": "Romance",
        "fraud reporting": "All",
        "victim support": "All",
        "scam prevention education": "All"
    }
    target_topic = topic_map.get(topic, "All")
    subtopic_keywords = subtopic.lower().split()
    relevant_personas = []
    
    # Shuffle personas to ensure varied selection
    shuffled_personas = random.sample(personas, len(personas))
    for persona_data in shuffled_personas:
        persona = persona_data["persona"]
        if persona in used_personas:
            continue
        persona_topic = persona_data["topic"]
        persona_role = persona_data["role"]
        if role == "officer" and persona_role == "Officer" and persona_topic == "All":
            relevant_personas.append(persona)
        elif role == "victim" and persona_role == "Victim" and persona_topic == target_topic:
            relevant_personas.append(persona)
        elif role == persona_role.lower() and target_topic == "All" and any(keyword in persona.lower() for keyword in subtopic_keywords + ["scam", "fraud", "phishing", "victim", "reporting"]):
            relevant_personas.append(persona)
        if len(relevant_personas) >= N_PERSONAS_PER_ROLE:
            break
    
    return relevant_personas[:N_PERSONAS_PER_ROLE]

def filter_repeated_personas(new_personas: List[str], existing_personas: List[str], used_personas: set) -> List[str]:
    """Filter out personas that are too similar to existing or used personas using ROUGE-L."""
    unique_personas = []
    filtered_out = []
    for persona in new_personas:
        if persona in used_personas:
            filtered_out.append(persona)
            continue
        if all(scorer.score(persona, existing)['rougeL'].fmeasure < ROUGE_THRESHOLD for existing in existing_personas + unique_personas):
            unique_personas.append(persona)
        else:
            filtered_out.append(persona)
    if filtered_out:
        print(f"Filtered out due to ROUGE-L > {ROUGE_THRESHOLD} or already used: {filtered_out}")
    return unique_personas

def generate_personas(subtopic: str, topic: str, n_personas: int, existing_personas: List[str], seed_personas: List[str], role: str, used_personas: set) -> List[str]:
    """Generate new personas for a given subtopic and role using gpt-4o-mini."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not found in environment")
    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.4, max_tokens=4096)
    prompt = generate_victim_personas_prompt(subtopic, n_personas, seed_personas) if role == "victim" else generate_officer_personas_prompt(subtopic, n_personas, seed_personas)
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", prompt),
        ("user", "Subtopic: {subtopic}")
    ])
    chain = prompt_template | llm
    try:
        output = chain.invoke({"subtopic": subtopic}).content
        start_idx, end_idx = output.find('['), output.find(']')
        if start_idx == -1 or end_idx == -1:
            raise ValueError("Invalid list format in LLM output")
        personas = ast.literal_eval(output[start_idx:end_idx+1])
        personas = [p for p in personas if isinstance(p, str)]
        filtered_personas = filter_repeated_personas(personas, existing_personas, used_personas)
        if len(filtered_personas) < n_personas:
            print(f"Warning: Only {len(filtered_personas)} {role} personas after filtering for {subtopic}, adding fallbacks")
            fallbacks = [f"A {random.randint(20, 70)}-year-old {role} affected by {subtopic} with relevant concerns" for _ in range(n_personas - len(filtered_personas))]
            filtered_personas.extend(fallbacks)
        return filtered_personas[:n_personas]
    except Exception as e:
        print(f"Error generating {role} personas for {subtopic}: {e}, retrying with higher temperature")
        try:
            llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.6, max_tokens=4096)
            chain = prompt_template | llm
            output = chain.invoke({"subtopic": subtopic}).content
            start_idx, end_idx = output.find('['), output.find(']')
            personas = ast.literal_eval(output[start_idx:end_idx+1])
            personas = [p for p in personas if isinstance(p, str)]
            filtered_personas = filter_repeated_personas(personas, existing_personas, used_personas)
            if len(filtered_personas) < n_personas:
                print(f"Warning: Only {len(filtered_personas)} {role} personas after retry for {subtopic}, adding fallbacks")
                fallbacks = [f"A {random.randint(20, 70)}-year-old {role} affected by {subtopic} with relevant concerns" for _ in range(n_personas - len(filtered_personas))]
                filtered_personas.extend(fallbacks)
            return filtered_personas[:n_personas]
        except Exception as e:
            print(f"Retry failed for {subtopic}: {e}, using fallbacks")
            return [f"A {random.randint(20, 70)}-year-old {role} affected by {subtopic} with relevant concerns" for _ in range(n_personas)]

def main():
    # Load subtopics and existing personas
    subtopics = load_subtopics(SUBTOPICS_FILE)
    victim_personas, officer_personas = load_existing_personas(PERSONA_SRC_FILE)
    
    # Track used personas to prevent repetition
    used_personas = set()
    
    # Map existing personas to subtopics
    subtopic_personas = {subtopic: {"victim": [], "officer": []} for topic in subtopics for subtopic in subtopics[topic]}
    for topic, subtopic_list in subtopics.items():
        for subtopic in subtopic_list:
            # Map victim personas
            mapped_victim_personas = map_existing_personas(subtopic, topic, victim_personas, "victim", used_personas)
            subtopic_personas[subtopic]["victim"].extend(mapped_victim_personas)
            used_personas.update(mapped_victim_personas)
            # Map officer personas
            mapped_officer_personas = map_existing_personas(subtopic, topic, officer_personas, "officer", used_personas)
            subtopic_personas[subtopic]["officer"].extend(mapped_officer_personas)
            used_personas.update(mapped_officer_personas)
    
    # Generate new personas for each subtopic
    with open(PERSONAS_FILE, 'w') as f:
        for topic, subtopic_list in subtopics.items():
            for subtopic in subtopic_list:
                print(f"Generating personas for subtopic: {subtopic} (topic: {topic})")
                # Select seed personas
                victim_seed_personas = select_seed_personas(subtopic, topic, victim_personas, "victim", used_personas, max_examples=3)
                officer_seed_personas = select_seed_personas(subtopic, topic, officer_personas, "officer", used_personas, max_examples=3)
                # Generate victim personas if needed
                current_victims = subtopic_personas[subtopic]["victim"]
                remaining_victims = N_PERSONAS_PER_ROLE - len(current_victims)
                if remaining_victims > 0:
                    new_victim_personas = generate_personas(subtopic, topic, remaining_victims, [p["persona"] for p in victim_personas + officer_personas], victim_seed_personas, "victim", used_personas)
                    subtopic_personas[subtopic]["victim"].extend(new_victim_personas)
                    used_personas.update(new_victim_personas)
                # Generate officer personas if needed
                current_officers = subtopic_personas[subtopic]["officer"]
                remaining_officers = N_PERSONAS_PER_ROLE - len(current_officers)
                if remaining_officers > 0:
                    new_officer_personas = generate_personas(subtopic, topic, remaining_officers, [p["persona"] for p in victim_personas + officer_personas], officer_seed_personas, "officer", used_personas)
                    subtopic_personas[subtopic]["officer"].extend(new_officer_personas)
                    used_personas.update(new_officer_personas)
                # Write all personas for this subtopic
                for persona in subtopic_personas[subtopic]["victim"][:N_PERSONAS_PER_ROLE]:
                    f.write(json.dumps({"topic": topic, "subtopic": subtopic, "persona": persona, "role": "victim"}) + "\n")
                for persona in subtopic_personas[subtopic]["officer"][:N_PERSONAS_PER_ROLE]:
                    f.write(json.dumps({"topic": topic, "subtopic": subtopic, "persona": persona, "role": "officer"}) + "\n")
                print(f"Victim personas for {subtopic}: {subtopic_personas[subtopic]['victim'][:N_PERSONAS_PER_ROLE]}")
                print(f"Officer personas for {subtopic}: {subtopic_personas[subtopic]['officer'][:N_PERSONAS_PER_ROLE]}")
    
    print(f"Personas saved to {PERSONAS_FILE}")

if __name__ == "__main__":
    main()
