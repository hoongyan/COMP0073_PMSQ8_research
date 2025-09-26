import pandas as pd
import numpy as np

rag_csv = "simulations/phase_2/rag_ie/autonomous_rag_invocation.csv"
victim_detail_json = "data/victim_profile/victim_details.json"

# Load the CSV into a DataFrame
df_rag = pd.read_csv(rag_csv,index_col=False)

# Load the JSON into a DataFrame
# Option 1: If it's a list of dictionaries (common for profiles), use this:
df_victim = pd.read_json(victim_detail_json, orient='records')

# Option 2: If it's a dictionary with keys as indices (e.g., victim IDs), uncomment this instead:
# df_victim = pd.read_json(victim_detail_json, orient='index').reset_index(names='victim_id')

# Print the individual DataFrames to inspect
print("RAG Invocation DataFrame (from CSV):")
print(df_rag['conversation_id'].head(20))
print("\nShape:", df_rag.shape)
print("\nColumns:", df_rag.columns.tolist())

print("\nVictim Details DataFrame (from JSON):")
print(df_victim.head())
print("\nShape:", df_victim.shape)
print("\nColumns:", df_victim.columns.tolist())

