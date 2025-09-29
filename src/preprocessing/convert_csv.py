import pandas as pd

# Function to create the dictionary string from the four columns
def create_rating_dict(row):
    """To transform the three rating columns for human evaluated simulations results into a single conlumn."""
    lang = row.get('language_proficiency_appropriateness (1 - 5)', None)
    tech = row.get('tech_literacy_appropriateness (1-5)', None)
    emo = row.get('emotional_state_appropriateness (1-5)', None)
    
    if pd.isna(lang) and pd.isna(tech) and pd.isna(emo):
        return "{}"
    
    try:
        lang = int(lang) if not pd.isna(lang) else None
    except ValueError:
        lang = None
    try:
        tech = int(tech) if not pd.isna(tech) else None
    except ValueError:
        tech = None
    try:
        emo = int(emo) if not pd.isna(emo) else None
    except ValueError:
        emo = None
    
    rating_dict = {
        "language_proficiency": lang,
        "tech_literacy": tech,
        "emotional_state": emo,
    }

    return str(rating_dict).replace("None", "null") 

# Define the file path 
file_path = "simulations/final_results/phase_1/profile_rag_ie_kb/autonomous_conversation_history_profile_rag_ie_kb_evaluated.csv" 

# Read the CSV
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: File not found. Please check the path in the script and try again.")
    exit()
except pd.errors.ParserError:
    print("Error: Could not parse the CSV. Make sure it's a valid CSV file.")
    exit()


df.columns = [col.strip() for col in df.columns]

# Add the new column
df['communication_appropriateness_rating'] = df.apply(create_rating_dict, axis=1)

columns_to_drop = [
    'language_proficiency_appropriateness (1 - 5)',
    'tech_literacy_appropriateness (1-5)',
    'emotional_state_appropriateness (1-5)'
]

unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
columns_to_drop.extend(unnamed_cols)

df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])  # Drop only if they exist


cols = list(df.columns)

if 'upserted_strategy' in cols:
    upserted_idx = cols.index('upserted_strategy')
    if 'communication_appropriateness_rating' in cols:
        cols.remove('communication_appropriateness_rating')
    cols.insert(upserted_idx + 1, 'communication_appropriateness_rating')

victim_col = 'victim_persona_appropriateness (1-5)'
if victim_col in cols:
    cols.remove(victim_col)
    cols.append(victim_col)

df = df[cols]


df.to_csv(file_path, index=False)
print("CSV file updated successfully! New column 'communication_appropriateness_rating' added after 'upserted_strategy', specified rating columns removed (except 'victim_persona_appropriateness (1-5)' which is now last), and any unnamed columns removed.")