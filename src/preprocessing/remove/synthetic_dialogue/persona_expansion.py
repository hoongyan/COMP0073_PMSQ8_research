# import sys
# import os
# import uuid
# import pandas as pd
# from langchain_openai import ChatOpenAI
# from sqlalchemy.orm import Session
# from typing import List, Dict
# import logging
# from pathlib import Path
# import argparse
# from sqlalchemy import text
# from database.database import SessionLocal
# from database.data_model import VictimPersonaAttribute, OfficerPersonaAttribute

# # Set up logging for debugging and monitoring
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# # Environment configuration
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # Initialize LLM client
# llm = ChatOpenAI(model_name="gpt-4", api_key=OPENAI_API_KEY, temperature=0.7)

# def reset_database(db: Session) -> None:
#     """
#     Clears all data from victim_persona_attributes and officer_persona_attributes tables if they exist.
#     """
#     try:
#         table_check = db.execute(text("""
#             SELECT EXISTS (
#                 SELECT FROM information_schema.tables 
#                 WHERE table_schema = 'public' 
#                 AND table_name = :table_name
#             )
#         """), {"table_name": "victim_persona_attributes"}).scalar()
#         if table_check:
#             db.execute(text("TRUNCATE TABLE victim_persona_attributes CASCADE"))
#             logger.info("Truncated victim_persona_attributes table")
#         else:
#             logger.info("victim_persona_attributes table does not exist, skipping truncate")

#         table_check = db.execute(text("""
#             SELECT EXISTS (
#                 SELECT FROM information_schema.tables 
#                 WHERE table_schema = 'public' 
#                 AND table_name = :table_name
#             )
#         """), {"table_name": "officer_persona_attributes"}).scalar()
#         if table_check:
#             db.execute(text("TRUNCATE TABLE officer_persona_attributes CASCADE"))
#             logger.info("Truncated officer_persona_attributes table")
#         else:
#             logger.info("officer_persona_attributes table does not exist, skipping truncate")

#         db.commit()
#         logger.info("Successfully completed database reset")
#     except Exception as e:
#         db.rollback()
#         logger.error(f"Error resetting database: {str(e)}")
#         raise

# def load_seed_personas(csv_path: str) -> Dict[str, Dict]:
#     """
#     Loads seed personas from CSV and organizes them by type (victim/officer) and scam type.
#     Maps 'Dimension' column to 'category' (Dispositional, Experiential, Situational).
#     Returns a dictionary with victim and officer personas, each grouped by scam type.
#     """
#     try:
#         df = pd.read_csv(
#             csv_path,
#             delimiter=',',
#             quotechar='"',
#             escapechar='\\',
#             encoding='utf-8',
#             on_bad_lines='warn',
#             names=['Type', 'Scam Type', 'Archetype', 'Dimension', 'Attribute'],
#             header=0,
#             dtype=str,
#             keep_default_na=True
#         )
#     except Exception as e:
#         logger.error(f"Error reading CSV {csv_path}: {str(e)}")
#         raise
    
#     # Verify columns
#     expected_columns = ['Type', 'Scam Type', 'Archetype', 'Dimension', 'Attribute']
#     if not all(col in df.columns for col in expected_columns):
#         missing = [col for col in expected_columns if col not in df.columns]
#         logger.error(f"Missing columns in CSV: {missing}")
#         raise ValueError(f"CSV missing required columns: {missing}")
    
#     personas = {
#         "victims": {
#             "ecommerce": [],
#             "gois": [],
#             "phishing": [],
#             "romance": []
#         },
#         "officers": {
#             "ecommerce": [],
#             "gois": [],
#             "phishing": [],
#             "romance": [],
#             "all": []
#         }
#     }
    
#     # Map CSV scam types to dictionary keys
#     scam_type_mapping = {
#         "ecommerce": "ecommerce",
#         "e-commerce": "ecommerce",
#         "gois": "gois",
#         "government_impersonation": "gois",
#         "phishing": "phishing",
#         "romance": "romance",
#         "all": "all"
#     }

#     for _, row in df.iterrows():
#         raw_scam_type = row["Scam Type"].lower().strip()  # Handle whitespace
#         scam_type = scam_type_mapping.get(raw_scam_type, raw_scam_type)
#         attribute = {
#             "id": str(uuid.uuid4()),
#             "attribute": row["Attribute"],
#             "category": row["Dimension"].lower(),
#             "archetype": row["Archetype"],
#             "scam_type": scam_type
#         }
#         logger.debug(f"Processing row with Scam Type: {raw_scam_type} -> mapped to: {scam_type}")
#         if row["Type"] == "Victim":
#             try:
#                 personas["victims"][scam_type].append(attribute)
#                 logger.debug(f"Added victim attribute for scam_type: {scam_type}")
#             except KeyError:
#                 logger.error(f"Invalid scam_type '{scam_type}' for victim persona")
#                 raise
#         else:  # Officer
#             try:
#                 personas["officers"][scam_type].append(attribute)
#                 logger.debug(f"Added officer attribute for scam_type: {scam_type}")
#             except KeyError:
#                 logger.error(f"Invalid scam_type '{scam_type}' for officer persona")
#                 raise

#     for scam_type, attrs in personas["victims"].items():
#         logger.info(f"Loaded {len(attrs)} victim attributes for {scam_type}")
#     for scam_type, attrs in personas["officers"].items():
#         logger.info(f"Loaded {len(attrs)} officer attributes for {scam_type}")
#     return personas

# def save_attributes_to_db(personas: Dict[str, Dict], db: Session, reset: bool = False) -> None:
#     """
#     Saves persona attributes to 'victim_persona_attributes' and 'officer_persona_attributes' tables using SQLAlchemy.
#     Optionally resets the database before saving. Checks for duplicates based on attribute and scam_type.
#     """
#     if reset:
#         reset_database(db)

#     # Victim attributes
#     for scam_type, attrs in personas["victims"].items():
#         for attr in attrs:
#             existing = db.query(VictimPersonaAttribute).filter(
#                 VictimPersonaAttribute.attribute == attr["attribute"],
#                 VictimPersonaAttribute.scam_type == scam_type
#             ).first()
#             if existing:
#                 logger.debug(f"Skipping duplicate victim attribute: {attr['attribute']} for {scam_type}")
#                 continue
#             db_attr = VictimPersonaAttribute(
#                 attribute_id=attr["id"],
#                 scam_type=scam_type,
#                 attribute=attr["attribute"],
#                 category=attr["category"],
#                 archetype=attr["archetype"]
#             )
#             try:
#                 db.add(db_attr)
#                 logger.info(f"Saved victim attribute {attr['id']} for {scam_type}")
#             except Exception as e:
#                 logger.error(f"Error saving victim attribute {attr['id']}: {str(e)}")

#     # Officer attributes
#     for scam_type, attrs in personas["officers"].items():
#         for attr in attrs:
#             existing = db.query(OfficerPersonaAttribute).filter(
#                 OfficerPersonaAttribute.attribute == attr["attribute"],
#                 OfficerPersonaAttribute.scam_type == scam_type
#             ).first()
#             if existing:
#                 logger.debug(f"Skipping duplicate officer attribute: {attr['attribute']} for {scam_type}")
#                 continue
#             db_attr = OfficerPersonaAttribute(
#                 attribute_id=attr["id"],
#                 scam_type=scam_type,
#                 attribute=attr["attribute"],
#                 category=attr["category"],
#                 archetype=attr["archetype"]
#             )
#             try:
#                 db.add(db_attr)
#                 logger.info(f"Saved officer attribute {attr['id']} for {scam_type}")
#             except Exception as e:
#                 logger.error(f"Error saving officer attribute {attr['id']}: {str(e)}")

#     try:
#         db.commit()
#     except Exception as e:
#         db.rollback()
#         logger.error(f"Error committing to database: {str(e)}")
#         raise

# def query_induction(seed_attributes: List[Dict], persona_type: str, scam_type: str) -> List[str]:
#     """
#     Performs query induction to generate persona attributes for fraud psychology.
#     Uses predefined dimensions (Dispositional, Experiential, Situational) and type-specific prompts (victim/officer).
#     Returns a list of up to 30 queries tailored to the persona type and scam type.
#     """
#     # Group attributes by dimension
#     dimensions = ["dispositional", "experiential", "situational"]
#     dim_to_attrs = {dim: [] for dim in dimensions}
#     for attr in seed_attributes:
#         if attr["category"] in dimensions:
#             dim_to_attrs[attr["category"]].append(attr["attribute"])

#     # Define type- and scam-specific prompt templates
#     prompt_templates = {
#         "victim": {
#             "ecommerce": {
#                 "dispositional": "Given the following e-commerce scam victim attributes (category: {dim}): {attrs}. Generate a query that captures psychological or demographic traits influencing trust or impulsivity in online shopping (e.g., 'How does your trust in website aesthetics affect your purchasing decisions?').",
#                 "experiential": "Given the following e-commerce scam victim attributes (category: {dim}): {attrs}. Generate a query that captures past experiences leading to vulnerability in e-commerce scams (e.g., 'What past online shopping experiences make you skip verifying seller legitimacy?').",
#                 "situational": "Given the following e-commerce scam victim attributes (category: {dim}): {attrs}. Generate a query that captures situational triggers for e-commerce scam vulnerability (e.g., 'What urgency in a flash sale prompted you to act quickly?')."
#             },
#             "gois": {
#                 "dispositional": "Given the following government impersonation scam victim attributes (category: {dim}): {attrs}. Generate a query that captures psychological or demographic traits influencing trust in authority (e.g., 'How does your respect for government officials affect your response to official-sounding communications?').",
#                 "experiential": "Given the following government impersonation scam victim attributes (category: {dim}): {attrs}. Generate a query that captures past experiences leading to vulnerability in GOIS scams (e.g., 'What lack of experience with legal systems makes you trust callers claiming to be officials?').",
#                 "situational": "Given the following government impersonation scam victim attributes (category: {dim}): {attrs}. Generate a query that captures situational triggers for GOIS scam vulnerability (e.g., 'What specific threats, like arrest or fines, made you comply with a scammer’s demands?')."
#             },
#             "phishing": {
#                 "dispositional": "Given the following phishing scam victim attributes (category: {dim}): {attrs}. Generate a query that captures psychological or demographic traits influencing trust in digital communications (e.g., 'How does your anxiety about technology affect your response to urgent security alerts?').",
#                 "experiential": "Given the following phishing scam victim attributes (category: {dim}): {attrs}. Generate a query that captures past experiences leading to vulnerability in phishing scams (e.g., 'What past experiences with emails make you likely to click links without verifying?').",
#                 "situational": "Given the following phishing scam victim attributes (category: {dim}): {attrs}. Generate a query that captures situational triggers for phishing scam vulnerability (e.g., 'What elements of an email, like logo or sender name, convinced you it was legitimate?')."
#             },
#             "romance": {
#                 "dispositional": "Given the following romance scam victim attributes (category: {dim}): {attrs}. Generate a query that captures psychological or demographic traits influencing trust in online relationships (e.g., 'How does your desire for emotional connection influence your trust in online partners?').",
#                 "experiential": "Given the following romance scam victim attributes (category: {dim}): {attrs}. Generate a query that captures past experiences leading to vulnerability in romance scams (e.g., 'What past relationship experiences make you likely to send money to someone you met online?').",
#                 "situational": "Given the following romance scam victim attributes (category: {dim}): {attrs}. Generate a query that captures situational triggers for romance scam vulnerability (e.g., 'What specific story from an online partner prompted you to provide financial help?')."
#             }
#         },
#         "officer": {
#             "ecommerce": {
#                 "dispositional": "Given the following officer attributes for e-commerce fraud cases (category: {dim}): {attrs}. Generate a query that captures psychological or professional traits influencing fraud case handling (e.g., 'How does your analytical mindset shape your approach to e-commerce scam investigations?').",
#                 "experiential": "Given the following officer attributes for e-commerce fraud cases (category: {dim}): {attrs}. Generate a query that captures past experiences shaping e-commerce fraud case handling (e.g., 'What past e-commerce fraud cases taught you to prioritize specific transaction details?').",
#                 "situational": "Given the following officer attributes for e-commerce fraud cases (category: {dim}): {attrs}. Generate a query that captures situational approaches to e-commerce fraud case handling (e.g., 'What specific victim behaviors do you address to encourage reporting in e-commerce scams?')."
#             },
#             "gois": {
#                 "dispositional": "Given the following officer attributes for government impersonation fraud cases (category: {dim}): {attrs}. Generate a query that captures psychological or professional traits influencing fraud case handling (e.g., 'How does your authoritative demeanor affect your approach to GOIS scam victims?').",
#                 "experiential": "Given the following officer attributes for government impersonation fraud cases (category: {dim}): {attrs}. Generate a query that captures past experiences shaping GOIS fraud case handling (e.g., 'What past GOIS fraud cases taught you to focus on victim fear responses?').",
#                 "situational": "Given the following officer attributes for government impersonation fraud cases (category: {dim}): {attrs}. Generate a query that captures situational approaches to GOIS fraud case handling (e.g., 'What specific victim fears do you address to build trust in GOIS scam investigations?')."
#             },
#             "phishing": {
#                 "dispositional": "Given the following officer attributes for phishing fraud cases (category: {dim}): {attrs}. Generate a query that captures psychological or professional traits influencing fraud case handling (e.g., 'How does your technical expertise shape your approach to phishing scam investigations?').",
#                 "experiential": "Given the following officer attributes for phishing fraud cases (category: {dim}): {attrs}. Generate a query that captures past experiences shaping phishing fraud case handling (e.g., 'What past phishing fraud cases taught you to prioritize email security details?').",
#                 "situational": "Given the following officer attributes for phishing fraud cases (category: {dim}): {attrs}. Generate a query that captures situational approaches to phishing fraud case handling (e.g., 'What specific victim actions do you guide to secure accounts in phishing scams?')."
#             },
#             "romance": {
#                 "dispositional": "Given the following officer attributes for romance fraud cases (category: {dim}): {attrs}. Generate a query that captures psychological or professional traits influencing fraud case handling (e.g., 'How does your empathetic nature affect your approach to romance scam victims?').",
#                 "experiential": "Given the following officer attributes for romance fraud cases (category: {dim}): {attrs}. Generate a query that captures past experiences shaping romance fraud case handling (e.g., 'What past romance fraud cases taught you to address victim emotional vulnerabilities?').",
#                 "situational": "Given the following officer attributes for romance fraud cases (category: {dim}): {attrs}. Generate a query that captures situational approaches to romance fraud case handling (e.g., 'What specific emotional triggers do you address to encourage reporting in romance scams?')."
#             },
#             "all": {
#                 "dispositional": "Given the following officer attributes for fraud cases (category: {dim}): {attrs}. Generate a query that captures psychological or professional traits influencing fraud case handling (e.g., 'How does your empathetic nature affect your approach to victim interviews?').",
#                 "experiential": "Given the following officer attributes for fraud cases (category: {dim}): {attrs}. Generate a query that captures past experiences shaping fraud case handling (e.g., 'What past fraud cases taught you to prioritize certain victim details?').",
#                 "situational": "Given the following officer attributes for fraud cases (category: {dim}): {attrs}. Generate a query that captures situational approaches to fraud case handling (e.g., 'What specific victim emotions do you address to encourage reporting?')."
#             }
#         }
#     }

#     # Validate scam_type
#     valid_scam_types = prompt_templates[persona_type].keys()
#     if scam_type not in valid_scam_types:
#         if persona_type == "victim":
#             logger.error(f"Invalid scam_type '{scam_type}' for victim persona; expected one of {list(valid_scam_types)}")
#             raise ValueError(f"Invalid scam_type '{scam_type}' for victim persona")
#         else:  # Officer
#             scam_type = "all"  # Fallback to 'all' for officers
#             logger.debug(f"Using fallback scam_type 'all' for officer persona with invalid scam_type '{scam_type}'")

#     # Generate initial queries for each dimension
#     queries = []
#     for dim in dimensions:
#         if dim_to_attrs[dim]:  # Only generate queries if attributes exist
#             sample_attrs = dim_to_attrs[dim][:3]  # Take up to 3 sample attributes
#             prompt_template = prompt_templates[persona_type][scam_type][dim]
#             prompt = prompt_template.format(dim=dim, attrs='; '.join(sample_attrs))
#             try:
#                 response = llm.invoke(prompt)
#                 query = response.content.strip()
#                 queries.append(query)
#                 logger.info(f"Generated query for {persona_type} {scam_type} ({dim}): {query}")
#             except Exception as e:
#                 logger.error(f"Error generating query for {persona_type} {scam_type} ({dim}): {str(e)}")
#                 continue

#     # Expand queries for diversity
#     expanded_queries = queries.copy()
#     for _ in range(2):  # Two iterations for modest expansion
#         prompt = f"""Given the following queries related to {persona_type} for {scam_type} scams:
# {'; '.join(expanded_queries)}
# Generate 3 new queries that cover unobserved persona aspects, focusing on dispositional, experiential, or situational dimensions relevant to fraud psychology for {scam_type} scams. For victims, elicit scam-specific vulnerabilities (e.g., trust in authority for GOIS, impulsivity for e-commerce). For officers, elicit case-handling strategies (e.g., empathy for romance, technical focus for phishing). Example: 'What specific threats made you comply with a scammer’s demands?' (victim) or 'What victim emotions do you address to encourage reporting?' (officer)."""
#         try:
#             response = llm.invoke(prompt)
#             new_queries = response.content.strip().split('\n')[:3]
#             new_queries = [q.strip() for q in new_queries if q.strip()]
#             expanded_queries.extend(new_queries)
#             logger.info(f"Expanded queries for {persona_type} {scam_type}: {new_queries}")
#         except Exception as e:
#             logger.error(f"Error expanding queries for {persona_type} {scam_type}: {str(e)}")
#             continue

#     return expanded_queries[:30]  # Limit to 30 queries

# def main(reset: bool = False):
#     logger.info("Starting persona expansion")
    
#     # Load seed personas
#     csv_path = Path(__file__).parent / "seed.csv"
#     personas = load_seed_personas(str(csv_path))
    
#     # Save to database
#     with SessionLocal() as db:
#         save_attributes_to_db(personas, db, reset=reset)
    
#     # Perform query induction separately for victims and officers
#     all_queries = {"victims": {}, "officers": {}}
#     for scam_type, attrs in personas["victims"].items():
#         logger.debug(f"Query induction for victim scam_type: {scam_type}")
#         if attrs:
#             all_queries["victims"][scam_type] = query_induction(attrs, "victim", scam_type)
#     for scam_type, attrs in personas["officers"].items():
#         logger.debug(f"Query induction for officer scam_type: {scam_type}")
#         if attrs:
#             all_queries["officers"][scam_type] = query_induction(attrs, "officer", scam_type)

#     logger.info("Query induction completed")
#     return personas, all_queries

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run persona expansion with optional database reset")
#     parser.add_argument("--reset", action="store_true", help="Reset the database before saving attributes")
#     args = parser.parse_args()
#     personas, queries = main(reset=args.reset)

import sys
import os
import uuid
import pandas as pd
import numpy as np
# from langchain_openai import ChatOpenAI
# from langchain_ollama.llms import OllamaLLM
from sentence_transformers import SentenceTransformer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sqlalchemy.orm import Session
from typing import List, Dict
import logging
from pathlib import Path
import argparse
from sqlalchemy import text
from src.database.remove.database import SessionLocal
from src.models.data_model import VictimPersonaAttribute, OfficerPersonaAttribute
from src.agents.remove.llm_providers import get_llm, SUPPORTED_MODELS

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize clients
provider = "Ollama"
model = "llama3.2"
if model not in SUPPORTED_MODELS[provider]:
    raise ValueError(f"Model '{model}' not supported for provider '{provider}'. Must be one of {SUPPORTED_MODELS[provider]}")

# llm = ChatOpenAI(model_name="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0.7)
llm = get_llm(provider=provider, model=model)
embedder = SentenceTransformer('all-MiniLM-L6-v2')
nli_model = T5ForConditionalGeneration.from_pretrained('t5-base')
nli_tokenizer = T5Tokenizer.from_pretrained('t5-base', model_max_length=512)

def reset_database(db: Session) -> None:
    """
    Clears all data from victim_persona_attributes and officer_persona_attributes tables if they exist.
    """
    try:
        table_check = db.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = :table_name
            )
        """), {"table_name": "victim_persona_attributes"}).scalar()
        if table_check:
            db.execute(text("TRUNCATE TABLE victim_persona_attributes CASCADE"))
            logger.info("Truncated victim_persona_attributes table")
        else:
            logger.info("victim_persona_attributes table does not exist, skipping truncate")

        table_check = db.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = :table_name
            )
        """), {"table_name": "officer_persona_attributes"}).scalar()
        if table_check:
            db.execute(text("TRUNCATE TABLE officer_persona_attributes CASCADE"))
            logger.info("Truncated officer_persona_attributes table")
        else:
            logger.info("officer_persona_attributes table does not exist, skipping truncate")

        db.commit()
        logger.info("Successfully completed database reset")
    except Exception as e:
        db.rollback()
        logger.error(f"Error resetting database: {str(e)}")
        raise

def load_seed_personas(csv_path: str) -> Dict[str, Dict]:
    """
    Loads seed personas from CSV and organizes them by type (victim/officer) and scam type.
    Maps 'Dimension' column to 'category' (Dispositional, Experiential, Situational).
    Returns a dictionary with victim and officer personas, each grouped by scam type.
    """
    try:
        df = pd.read_csv(
            csv_path,
            delimiter=',',
            quotechar='"',
            escapechar='\\',
            encoding='utf-8',
            on_bad_lines='warn',
            names=['Type', 'Scam Type', 'Archetype', 'Dimension', 'Attribute'],
            header=0,
            dtype=str,
            keep_default_na=True
        )
    except Exception as e:
        logger.error(f"Error reading CSV {csv_path}: {str(e)}")
        raise
    
    # Verify columns
    expected_columns = ['Type', 'Scam Type', 'Archetype', 'Dimension', 'Attribute']
    if not all(col in df.columns for col in expected_columns):
        missing = [col for col in expected_columns if col not in df.columns]
        logger.error(f"Missing columns in CSV: {missing}")
        raise ValueError(f"CSV missing required columns: {missing}")
    
    personas = {
        "victims": {
            "ecommerce": [],
            "gois": [],
            "phishing": [],
            "romance": []
        },
        "officers": {
            "ecommerce": [],
            "gois": [],
            "phishing": [],
            "romance": [],
            "all": []
        }
    }
    
    # Map CSV scam types to dictionary keys
    scam_type_mapping = {
        "ecommerce": "ecommerce",
        "e-commerce": "ecommerce",
        "gois": "gois",
        "government_impersonation": "gois",
        "phishing": "phishing",
        "romance": "romance",
        "all": "all"
    }

    for _, row in df.iterrows():
        raw_scam_type = row["Scam Type"].lower().strip()
        scam_type = scam_type_mapping.get(raw_scam_type, raw_scam_type)
        attribute = {
            "id": str(uuid.uuid4()),
            "attribute": row["Attribute"],
            "category": row["Dimension"].lower(),
            "archetype": row["Archetype"],
            "scam_type": scam_type
        }
        logger.debug(f"Processing row with Scam Type: {raw_scam_type} -> mapped to: {scam_type}")
        if row["Type"] == "Victim":
            try:
                personas["victims"][scam_type].append(attribute)
                logger.debug(f"Added victim attribute for scam_type: {scam_type}")
            except KeyError:
                logger.error(f"Invalid scam_type '{scam_type}' for victim persona")
                raise
        else:
            try:
                personas["officers"][scam_type].append(attribute)
                logger.debug(f"Added officer attribute for scam_type: {scam_type}")
            except KeyError:
                logger.error(f"Invalid scam_type '{scam_type}' for officer persona")
                raise

    for scam_type, attrs in personas["victims"].items():
        logger.info(f"Loaded {len(attrs)} victim attributes for {scam_type}")
    for scam_type, attrs in personas["officers"].items():
        logger.info(f"Loaded {len(attrs)} officer attributes for {scam_type}")
    return personas

def save_attributes_to_db(personas: Dict[str, Dict], db: Session, reset: bool = False) -> None:
    """
    Saves persona attributes to 'victim_persona_attributes' and 'officer_persona_attributes' tables using SQLAlchemy.
    Optionally resets the database before saving. Checks for duplicates based on attribute and scam_type.
    """
    if reset:
        reset_database(db)

    # Victim attributes
    for scam_type, attrs in personas["victims"].items():
        for attr in attrs:
            existing = db.query(VictimPersonaAttribute).filter(
                VictimPersonaAttribute.attribute == attr["attribute"],
                VictimPersonaAttribute.scam_type == scam_type
            ).first()
            if existing:
                logger.debug(f"Skipping duplicate victim attribute: {attr['attribute']} for {scam_type}")
                continue
            db_attr = VictimPersonaAttribute(
                attribute_id=attr["id"],
                scam_type=scam_type,
                attribute=attr["attribute"],
                category=attr["category"],
                archetype=attr["archetype"]
            )
            try:
                db.add(db_attr)
                logger.info(f"Saved victim attribute {attr['id']} for {scam_type}")
            except Exception as e:
                logger.error(f"Error saving victim attribute {attr['id']}: {str(e)}")

    # Officer attributes
    for scam_type, attrs in personas["officers"].items():
        for attr in attrs:
            existing = db.query(OfficerPersonaAttribute).filter(
                OfficerPersonaAttribute.attribute == attr["attribute"],
                OfficerPersonaAttribute.scam_type == scam_type
            ).first()
            if existing:
                logger.debug(f"Skipping duplicate officer attribute: {attr['attribute']} for {scam_type}")
                continue
            db_attr = OfficerPersonaAttribute(
                attribute_id=attr["id"],
                scam_type=scam_type,
                attribute=attr["attribute"],
                category=attr["category"],
                archetype=attr["archetype"]
            )
            try:
                db.add(db_attr)
                logger.info(f"Saved officer attribute {attr['id']} for {scam_type}")
            except Exception as e:
                logger.error(f"Error saving officer attribute {attr['id']}: {str(e)}")

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error committing to database: {str(e)}")
        raise

def query_induction(seed_attributes: List[Dict], persona_type: str, scam_type: str) -> List[str]:
    """
    Performs query induction to generate persona attributes for fraud psychology.
    Uses predefined dimensions (Dispositional, Experiential, Situational) and type-specific prompts (victim/officer).
    Returns a list of up to 30 queries tailored to the persona type and scam type.
    """
    # Group attributes by dimension
    dimensions = ["dispositional", "experiential", "situational"]
    dim_to_attrs = {dim: [] for dim in dimensions}
    for attr in seed_attributes:
        if attr["category"] in dimensions:
            dim_to_attrs[attr["category"]].append(attr["attribute"])

    # Define type- and scam-specific prompt templates
    prompt_templates = {
        "victim": {
            "ecommerce": {
                "dispositional": "Given the following e-commerce scam victim attributes (category: {dim}): {attrs}. Generate a query that captures psychological or demographic traits influencing trust or impulsivity in online shopping (e.g., 'How does your trust in website aesthetics affect your purchasing decisions?').",
                "experiential": "Given the following e-commerce scam victim attributes (category: {dim}): {attrs}. Generate a query that captures past experiences leading to vulnerability in e-commerce scams (e.g., 'What past online shopping experiences make you skip verifying seller legitimacy?').",
                "situational": "Given the following e-commerce scam victim attributes (category: {dim}): {attrs}. Generate a query that captures situational triggers for e-commerce scam vulnerability (e.g., 'What urgency in a flash sale prompted you to act quickly?')."
            },
            "gois": {
                "dispositional": "Given the following government impersonation scam victim attributes (category: {dim}): {attrs}. Generate a query that captures psychological or demographic traits influencing trust in authority (e.g., 'How does your respect for government officials affect your response to official-sounding communications?').",
                "experiential": "Given the following government impersonation scam victim attributes (category: {dim}): {attrs}. Generate a query that captures past experiences leading to vulnerability in GOIS scams (e.g., 'What lack of experience with legal systems makes you trust callers claiming to be officials?').",
                "situational": "Given the following government impersonation scam victim attributes (category: {dim}): {attrs}. Generate a query that captures situational triggers for GOIS scam vulnerability (e.g., 'What specific threats, like arrest or fines, made you comply with a scammer’s demands?')."
            },
            "phishing": {
                "dispositional": "Given the following phishing scam victim attributes (category: {dim}): {attrs}. Generate a query that captures psychological or demographic traits influencing trust in digital communications (e.g., 'How does your anxiety about technology affect your response to urgent security alerts?').",
                "experiential": "Given the following phishing scam victim attributes (category: {dim}): {attrs}. Generate a query that captures past experiences leading to vulnerability in phishing scams (e.g., 'What past experiences with emails make you likely to click links without verifying?').",
                "situational": "Given the following phishing scam victim attributes (category: {dim}): {attrs}. Generate a query that captures situational triggers for phishing scam vulnerability (e.g., 'What elements of an email, like logo or sender name, convinced you it was legitimate?')."
            },
            "romance": {
                "dispositional": "Given the following romance scam victim attributes (category: {dim}): {attrs}. Generate a query that captures psychological or demographic traits influencing trust in online relationships (e.g., 'How does your desire for emotional connection influence your trust in online partners?').",
                "experiential": "Given the following romance scam victim attributes (category: {dim}): {attrs}. Generate a query that captures past experiences leading to vulnerability in romance scams (e.g., 'What past relationship experiences make you likely to send money to someone you met online?').",
                "situational": "Given the following romance scam victim attributes (category: {dim}): {attrs}. Generate a query that captures situational triggers for romance scam vulnerability (e.g., 'What specific story from an online partner prompted you to provide financial help?')."
            }
        },
        "officer": {
            "ecommerce": {
                "dispositional": "Given the following officer attributes for e-commerce fraud cases (category: {dim}): {attrs}. Generate a query that captures psychological or professional traits influencing fraud case handling (e.g., 'How does your analytical mindset shape your approach to e-commerce scam investigations?').",
                "experiential": "Given the following officer attributes for e-commerce fraud cases (category: {dim}): {attrs}. Generate a query that captures past experiences shaping e-commerce fraud case handling (e.g., 'What past e-commerce fraud cases taught you to prioritize specific transaction details?').",
                "situational": "Given the following officer attributes for e-commerce fraud cases (category: {dim}): {attrs}. Generate a query that captures situational approaches to e-commerce fraud case handling (e.g., 'What specific victim behaviors do you address to encourage reporting in e-commerce scams?')."
            },
            "gois": {
                "dispositional": "Given the following officer attributes for government impersonation fraud cases (category: {dim}): {attrs}. Generate a query that captures psychological or professional traits influencing fraud case handling (e.g., 'How does your authoritative demeanor affect your approach to GOIS scam victims?').",
                "experiential": "Given the following officer attributes for government impersonation fraud cases (category: {dim}): {attrs}. Generate a query that captures past experiences shaping GOIS fraud case handling (e.g., 'What past GOIS fraud cases taught you to focus on victim fear responses?').",
                "situational": "Given the following officer attributes for government impersonation fraud cases (category: {dim}): {attrs}. Generate a query that captures situational approaches to GOIS fraud case handling (e.g., 'What specific victim fears do you address to build trust in GOIS scam investigations?')."
            },
            "phishing": {
                "dispositional": "Given the following officer attributes for phishing fraud cases (category: {dim}): {attrs}. Generate a query that captures psychological or professional traits influencing fraud case handling (e.g., 'How does your technical expertise shape your approach to phishing scam investigations?').",
                "experiential": "Given the following officer attributes for phishing fraud cases (category: {dim}): {attrs}. Generate a query that captures past experiences shaping phishing fraud case handling (e.g., 'What past phishing fraud cases taught you to prioritize email security details?').",
                "situational": "Given the following officer attributes for phishing fraud cases (category: {dim}): {attrs}. Generate a query that captures situational approaches to phishing fraud case handling (e.g., 'What specific victim actions do you guide to secure accounts in phishing scams?')."
            },
            "romance": {
                "dispositional": "Given the following officer attributes for romance fraud cases (category: {dim}): {attrs}. Generate a query that captures psychological or professional traits influencing fraud case handling (e.g., 'How does your empathetic nature affect your approach to romance scam victims?').",
                "experiential": "Given the following officer attributes for romance fraud cases (category: {dim}): {attrs}. Generate a query that captures past experiences shaping romance fraud case handling (e.g., 'What past romance fraud cases taught you to address victim emotional vulnerabilities?').",
                "situational": "Given the following officer attributes for romance fraud cases (category: {dim}): {attrs}. Generate a query that captures situational approaches to romance fraud case handling (e.g., 'What specific emotional triggers do you address to encourage reporting in romance scams?')."
            },
            "all": {
                "dispositional": "Given the following officer attributes for fraud cases (category: {dim}): {attrs}. Generate a query that captures psychological or professional traits influencing fraud case handling (e.g., 'How does your empathetic nature affect your approach to victim interviews?').",
                "experiential": "Given the following officer attributes for fraud cases (category: {dim}): {attrs}. Generate a query that captures past experiences shaping fraud case handling (e.g., 'What past fraud cases taught you to prioritize certain victim details?').",
                "situational": "Given the following officer attributes for fraud cases (category: {dim}): {attrs}. Generate a query that captures situational approaches to fraud case handling (e.g., 'What specific victim emotions do you address to encourage reporting?')."
            }
        }
    }

    # Validate scam_type
    valid_scam_types = prompt_templates[persona_type].keys()
    if scam_type not in valid_scam_types:
        if persona_type == "victim":
            logger.error(f"Invalid scam_type '{scam_type}' for victim persona; expected one of {list(valid_scam_types)}")
            raise ValueError(f"Invalid scam_type '{scam_type}' for victim persona")
        else:  # Officer
            scam_type = "all"
            logger.debug(f"Using fallback scam_type 'all' for officer persona with invalid scam_type '{scam_type}'")

    # Generate initial queries for each dimension
    queries = []
    for dim in dimensions:
        if dim_to_attrs[dim]:  # Only generate queries if attributes exist
            sample_attrs = dim_to_attrs[dim][:3]  # Take up to 3 sample attributes
            prompt_template = prompt_templates[persona_type][scam_type][dim]
            prompt = prompt_template.format(dim=dim, attrs='; '.join(sample_attrs))
            try:
                response = llm.invoke(prompt)
                query = response.content.strip()
                queries.append(query)
                logger.info(f"Generated query for {persona_type} {scam_type} ({dim}): {query}")
            except Exception as e:
                logger.error(f"Error generating query for {persona_type} {scam_type} ({dim}): {str(e)}")
                continue

    # Expand queries for diversity
    expanded_queries = queries.copy()
    for _ in range(3):  # Two iterations for modest expansion
        prompt = f"""Given the following queries related to {persona_type} for {scam_type} scams:
{'; '.join(expanded_queries)}
Generate 3 new queries that cover unobserved persona aspects, focusing on dispositional, experiential, or situational dimensions relevant to fraud psychology for {scam_type} scams. For victims, elicit scam-specific vulnerabilities (e.g., trust in authority for GOIS, impulsivity for e-commerce). For officers, elicit case-handling strategies (e.g., empathy for romance, technical focus for phishing). Example: 'What specific threats made you comply with a scammer’s demands?' (victim) or 'What victim emotions do you address to encourage reporting?' (officer)."""
        try:
            response = llm.invoke(prompt)
            new_queries = response.content.strip().split('\n')[:3]
            new_queries = [q.strip() for q in new_queries if q.strip()]
            expanded_queries.extend(new_queries)
            logger.info(f"Expanded queries for {persona_type} {scam_type}: {new_queries}")
        except Exception as e:
            logger.error(f"Error expanding queries for {persona_type} {scam_type}: {str(e)}")
            continue

    return expanded_queries[:30]  # Limit to 30 queries

def persona_bootstrapping(
    queries: Dict[str, Dict[str, List[str]]],
    seed_personas: Dict[str, Dict],
    db: Session,
    target_count: int = 50
) -> None:
    """
    Performs persona bootstrapping to generate new attributes using queries and seed attributes.
    Ensures diversity (via Sentence Transformers) and consistency (via T5-NLI).
    Saves new attributes to 'victim_persona_attributes' and 'officer_persona_attributes'.
    """
    # Load existing attributes from database for diversity and consistency checks
    existing_victim_attrs = {
        scam_type: [attr.attribute for attr in db.query(VictimPersonaAttribute).filter(VictimPersonaAttribute.scam_type == scam_type).all()]
        for scam_type in seed_personas["victims"].keys()
    }
    existing_officer_attrs = {
        scam_type: [attr.attribute for attr in db.query(OfficerPersonaAttribute).filter(OfficerPersonaAttribute.scam_type == scam_type).all()]
        for scam_type in seed_personas["officers"].keys()
    }

    # Target ~200-250 attributes per scam type for victims, ~200 for officers
    target_per_scam_type = {
        "victims": target_count // (len(seed_personas["victims"]) + 1),  # ~200 per scam type
        "officers": target_count // (len(seed_personas["victims"]) + 1)   # ~200 total
    }

    # Bootstrapping prompt templates
    bootstrap_templates = {
        "victim": {
            "ecommerce": {
                "dispositional": "Given the query '{query}' and these e-commerce scam victim attributes (category: {dim}): {attrs}, generate a new dispositional attribute capturing psychological or demographic traits influencing trust or impulsivity in online shopping.",
                "experiential": "Given the query '{query}' and these e-commerce scam victim attributes (category: {dim}): {attrs}, generate a new experiential attribute capturing past experiences leading to vulnerability in e-commerce scams.",
                "situational": "Given the query '{query}' and these e-commerce scam victim attributes (category: {dim}): {attrs}, generate a new situational attribute capturing triggers for e-commerce scam vulnerability."
            },
            "gois": {
                "dispositional": "Given the query '{query}' and these government impersonation scam victim attributes (category: {dim}): {attrs}, generate a new dispositional attribute capturing psychological or demographic traits influencing trust in authority.",
                "experiential": "Given the query '{query}' and these government impersonation scam victim attributes (category: {dim}): {attrs}, generate a new experiential attribute capturing past experiences leading to vulnerability in GOIS scams.",
                "situational": "Given the query '{query}' and these government impersonation scam victim attributes (category: {dim}): {attrs}, generate a new situational attribute capturing triggers for GOIS scam vulnerability."
            },
            "phishing": {
                "dispositional": "Given the query '{query}' and these phishing scam victim attributes (category: {dim}): {attrs}, generate a new dispositional attribute capturing psychological or demographic traits influencing trust in digital communications.",
                "experiential": "Given the query '{query}' and these phishing scam victim attributes (category: {dim}): {attrs}, generate a new experiential attribute capturing past experiences leading to vulnerability in phishing scams.",
                "situational": "Given the query '{query}' and these phishing scam victim attributes (category: {dim}): {attrs}, generate a new situational attribute capturing triggers for phishing scam vulnerability."
            },
            "romance": {
                "dispositional": "Given the query '{query}' and these romance scam victim attributes (category: {dim}): {attrs}, generate a new dispositional attribute capturing psychological or demographic traits influencing trust in online relationships.",
                "experiential": "Given the query '{query}' and these romance scam victim attributes (category: {dim}): {attrs}, generate a new experiential attribute capturing past experiences leading to vulnerability in romance scams.",
                "situational": "Given the query '{query}' and these romance scam victim attributes (category: {dim}): {attrs}, generate a new situational attribute capturing triggers for romance scam vulnerability."
            }
        },
        "officer": {
            "ecommerce": {
                "dispositional": "Given the query '{query}' and these officer attributes for e-commerce fraud cases (category: {dim}): {attrs}, generate a new dispositional attribute capturing psychological or professional traits influencing fraud case handling.",
                "experiential": "Given the query '{query}' and these officer attributes for e-commerce fraud cases (category: {dim}): {attrs}, generate a new experiential attribute capturing past experiences shaping e-commerce fraud case handling.",
                "situational": "Given the query '{query}' and these officer attributes for e-commerce fraud cases (category: {dim}): {attrs}, generate a new situational attribute capturing approaches to e-commerce fraud case handling."
            },
            "gois": {
                "dispositional": "Given the query '{query}' and these officer attributes for government impersonation fraud cases (category: {dim}): {attrs}, generate a new dispositional attribute capturing psychological or professional traits influencing fraud case handling.",
                "experiential": "Given the query '{query}' and these officer attributes for government impersonation fraud cases (category: {dim}): {attrs}, generate a new experiential attribute capturing past experiences shaping GOIS fraud case handling.",
                "situational": "Given the query '{query}' and these officer attributes for government impersonation fraud cases (category: {dim}): {attrs}, generate a new situational attribute capturing approaches to GOIS fraud case handling."
            },
            "phishing": {
                "dispositional": "Given the query '{query}' and these officer attributes for phishing fraud cases (category: {dim}): {attrs}, generate a new dispositional attribute capturing psychological or professional traits influencing fraud case handling.",
                "experiential": "Given the query '{query}' and these officer attributes for phishing fraud cases (category: {dim}): {attrs}, generate a new experiential attribute capturing past experiences shaping phishing fraud case handling.",
                "situational": "Given the query '{query}' and these officer attributes for phishing fraud cases (category: {dim}): {attrs}, generate a new situational attribute capturing approaches to phishing fraud case handling."
            },
            "romance": {
                "dispositional": "Given the query '{query}' and these officer attributes for romance fraud cases (category: {dim}): {attrs}, generate a new dispositional attribute capturing psychological or professional traits influencing fraud case handling.",
                "experiential": "Given the query '{query}' and these officer attributes for romance fraud cases (category: {dim}): {attrs}, generate a new experiential attribute capturing past experiences shaping romance fraud case handling.",
                "situational": "Given the query '{query}' and these officer attributes for romance fraud cases (category: {dim}): {attrs}, generate a new situational attribute capturing approaches to romance fraud case handling."
            },
            "all": {
                "dispositional": "Given the query '{query}' and these officer attributes for fraud cases (category: {dim}): {attrs}, generate a new dispositional attribute capturing psychological or professional traits influencing fraud case handling.",
                "experiential": "Given the query '{query}' and these officer attributes for fraud cases (category: {dim}): {attrs}, generate a new experiential attribute capturing past experiences shaping fraud case handling.",
                "situational": "Given the query '{query}' and these officer attributes for fraud cases (category: {dim}): {attrs}, generate a new situational attribute capturing approaches to fraud case handling."
            }
        }
    }

    # Bootstrapping for victims
    for scam_type, q_list in queries["victims"].items():
        current_count = len(existing_victim_attrs[scam_type])
        target = target_per_scam_type["victims"]
        seed_attrs = seed_personas["victims"][scam_type]
        dim_to_attrs = {dim: [a["attribute"] for a in seed_attrs if a["category"] == dim] for dim in ["dispositional", "experiential", "situational"]}
        archetype_map = {a["attribute"]: a["archetype"] for a in seed_attrs}

        while current_count < target and q_list:
            for query in q_list:
                if current_count >= target:
                    break
                # Select dimension and seed attributes
                dim = np.random.choice(["dispositional", "experiential", "situational"])
                sample_attrs = dim_to_attrs.get(dim, [])[:3]
                if not sample_attrs:
                    continue
                prompt = bootstrap_templates["victim"][scam_type][dim].format(query=query, dim=dim, attrs='; '.join(sample_attrs))
                try:
                    response = llm.invoke(prompt)
                    new_attr = response.content.strip()
                    # Diversity check
                    embeddings = embedder.encode([new_attr] + existing_victim_attrs[scam_type])
                    similarities = np.dot(embeddings[1:], embeddings[0]) / (
                        np.linalg.norm(embeddings[1:], axis=1) * np.linalg.norm(embeddings[0])
                    )
                    if np.max(similarities) > 0.9:
                        logger.debug(f"Skipping similar victim attribute: {new_attr} for {scam_type}")
                        continue
                    # Consistency check
                    consistent = True
                    for existing_attr in existing_victim_attrs[scam_type]:
                        input_text = f"premise: {existing_attr} hypothesis: {new_attr}"
                        inputs = nli_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                        outputs = nli_model.generate(inputs.input_ids, max_length=32)
                        prediction = nli_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        if prediction == "contradiction":
                            consistent = False
                            logger.debug(f"Skipping contradictory victim attribute: {new_attr} contradicts {existing_attr}")
                            break
                    if not consistent:
                        continue
                    # Infer archetype from most similar seed attribute
                    seed_embeddings = embedder.encode([a["attribute"] for a in seed_attrs])
                    new_embedding = embedder.encode([new_attr])[0]
                    similarities = np.dot(seed_embeddings, new_embedding) / (
                        np.linalg.norm(seed_embeddings, axis=1) * np.linalg.norm(new_embedding)
                    )
                    closest_idx = np.argmax(similarities)
                    archetype = archetype_map[seed_attrs[closest_idx]["attribute"]]
                    # Save new attribute
                    db_attr = VictimPersonaAttribute(
                        attribute_id=str(uuid.uuid4()),
                        scam_type=scam_type,
                        attribute=new_attr,
                        category=dim,
                        archetype=archetype
                    )
                    db.add(db_attr)
                    existing_victim_attrs[scam_type].append(new_attr)
                    current_count += 1
                    logger.info(f"Added new victim attribute {new_attr} for {scam_type} ({dim}, {archetype})")
                except Exception as e:
                    logger.error(f"Error generating victim attribute for {scam_type}: {str(e)}")
                    continue

    # Bootstrapping for officers
    for scam_type, q_list in queries["officers"].items():
        current_count = len(existing_officer_attrs[scam_type])
        target = target_per_scam_type["officers"] if scam_type == "all" else target_per_scam_type["officers"] // 4
        seed_attrs = seed_personas["officers"][scam_type]
        dim_to_attrs = {dim: [a["attribute"] for a in seed_attrs if a["category"] == dim] for dim in ["dispositional", "experiential", "situational"]}
        archetype_map = {a["attribute"]: a["archetype"] for a in seed_attrs}

        while current_count < target and q_list:
            for query in q_list:
                if current_count >= target:
                    break
                dim = np.random.choice(["dispositional", "experiential", "situational"])
                sample_attrs = dim_to_attrs.get(dim, [])[:3]
                if not sample_attrs:
                    continue
                prompt = bootstrap_templates["officer"][scam_type][dim].format(query=query, dim=dim, attrs='; '.join(sample_attrs))
                try:
                    response = llm.invoke(prompt)
                    new_attr = response.content.strip()
                    embeddings = embedder.encode([new_attr] + existing_officer_attrs[scam_type])
                    similarities = np.dot(embeddings[1:], embeddings[0]) / (
                        np.linalg.norm(embeddings[1:], axis=1) * np.linalg.norm(embeddings[0])
                    )
                    if np.max(similarities) > 0.9:
                        logger.debug(f"Skipping similar officer attribute: {new_attr} for {scam_type}")
                        continue
                    consistent = True
                    for existing_attr in existing_officer_attrs[scam_type]:
                        input_text = f"premise: {existing_attr} hypothesis: {new_attr}"
                        inputs = nli_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
                        outputs = nli_model.generate(inputs.input_ids, max_length=32)
                        prediction = nli_tokenizer.decode(outputs[0], skip_special_tokens=True)
                        if prediction == "contradiction":
                            consistent = False
                            logger.debug(f"Skipping contradictory officer attribute: {new_attr} contradicts {existing_attr}")
                            break
                    if not consistent:
                        continue
                    seed_embeddings = embedder.encode([a["attribute"] for a in seed_attrs])
                    new_embedding = embedder.encode([new_attr])[0]
                    similarities = np.dot(seed_embeddings, new_embedding) / (
                        np.linalg.norm(seed_embeddings, axis=1) * np.linalg.norm(new_embedding)
                    )
                    closest_idx = np.argmax(similarities)
                    archetype = archetype_map[seed_attrs[closest_idx]["attribute"]]
                    db_attr = OfficerPersonaAttribute(
                        attribute_id=str(uuid.uuid4()),
                        scam_type=scam_type,
                        attribute=new_attr,
                        category=dim,
                        archetype=archetype
                    )
                    db.add(db_attr)
                    existing_officer_attrs[scam_type].append(new_attr)
                    current_count += 1
                    logger.info(f"Added new officer attribute {new_attr} for {scam_type} ({dim}, {archetype})")
                except Exception as e:
                    logger.error(f"Error generating officer attribute for {scam_type}: {str(e)}")
                    continue

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Error committing to database: {str(e)}")
        raise

def main(reset: bool = False):
    logger.info("Starting persona expansion")
    
    # Load seed personas
    csv_path = Path(__file__).parent / "seed.csv"
    personas = load_seed_personas(str(csv_path))
    
    # Save to database and perform query induction
    with SessionLocal() as db:
        save_attributes_to_db(personas, db, reset=reset)
        # Perform query induction separately for victims and officers
        all_queries = {"victims": {}, "officers": {}}
        for scam_type, attrs in personas["victims"].items():
            logger.debug(f"Query induction for victim scam_type: {scam_type}")
            if attrs:
                all_queries["victims"][scam_type] = query_induction(attrs, "victim", scam_type)
        for scam_type, attrs in personas["officers"].items():
            logger.debug(f"Query induction for officer scam_type: {scam_type}")
            if attrs:
                all_queries["officers"][scam_type] = query_induction(attrs, "officer", scam_type)
        
        # Perform persona bootstrapping
        logger.info("Starting persona bootstrapping")
        persona_bootstrapping(all_queries, personas, db, target_count=50)
    
    logger.info("Persona expansion completed")
    return personas, all_queries

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run persona expansion with optional database reset")
    parser.add_argument("--reset", action="store_true", help="Reset the database before saving attributes")
    args = parser.parse_args()
    personas, queries = main(reset=args.reset)