import pandas as pd

# Function to create the dictionary string from the four columns
def create_rating_dict(row):
    lang = row.get('language_proficiency_appropriateness (1 - 5)', None)
    tech = row.get('tech_literacy_appropriateness (1-5)', None)
    emo = row.get('emotional_state_appropriateness (1-5)', None)
    
    # If all are missing, return empty string or "{}"
    if pd.isna(lang) and pd.isna(tech) and pd.isna(emo):
        return "{}"
    
    # Convert to int if possible, else None
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
    
    # Build the dict
    rating_dict = {
        "language_proficiency": lang,
        "tech_literacy": tech,
        "emotional_state": emo,
    }
    # Convert to string like "{'language_proficiency': 4, 'tech_literacy': 3, 'emotional_state': 2, 'victim_persona': 5}"
    return str(rating_dict).replace("None", "null")  # Use "null" for missing values

# Define the file path directly here (change this to your actual file path)
file_path = "simulations/phase_1/test_simulation/use/updated/autonomous_conversation_history_profile_rag_ie_kb_evaluated.csv"  # Update to the correct path if needed

# Read the CSV
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print("Error: File not found. Please check the path in the script and try again.")
    exit()
except pd.errors.ParserError:
    print("Error: Could not parse the CSV. Make sure it's a valid CSV file.")
    exit()

# Strip any leading/trailing whitespace from column names to handle potential mismatches
df.columns = [col.strip() for col in df.columns]

# Add the new column
df['communication_appropriateness_rating'] = df.apply(create_rating_dict, axis=1)

# Drop the unnecessary columns: the three specified original rating columns and any unnamed column if present
columns_to_drop = [
    'language_proficiency_appropriateness (1 - 5)',
    'tech_literacy_appropriateness (1-5)',
    'emotional_state_appropriateness (1-5)'
]
# Add any unnamed columns (e.g., 'Unnamed: 22') if they exist
unnamed_cols = [col for col in df.columns if col.startswith('Unnamed:')]
columns_to_drop.extend(unnamed_cols)

df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])  # Drop only if they exist

# Reorder columns to place 'communication_appropriateness_rating' right after 'upserted_strategy'
# Get current columns
cols = list(df.columns)

# Find the index of 'upserted_strategy'
if 'upserted_strategy' in cols:
    upserted_idx = cols.index('upserted_strategy')
    # Remove 'communication_appropriateness_rating' from its current position if needed
    if 'communication_appropriateness_rating' in cols:
        cols.remove('communication_appropriateness_rating')
    # Insert it after 'upserted_strategy'
    cols.insert(upserted_idx + 1, 'communication_appropriateness_rating')

# Now, ensure 'victim_persona_appropriateness (1-5)' is the last column
victim_col = 'victim_persona_appropriateness (1-5)'
if victim_col in cols:
    cols.remove(victim_col)
    cols.append(victim_col)

# Reorder the DataFrame
df = df[cols]

# Save back to the same file (overwrites original)
df.to_csv(file_path, index=False)
print("CSV file updated successfully! New column 'communication_appropriateness_rating' added after 'upserted_strategy', specified rating columns removed (except 'victim_persona_appropriateness (1-5)' which is now last), and any unnamed columns removed.")