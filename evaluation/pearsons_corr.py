import numpy as np
from scipy.stats import pearsonr

# Original arrays (as before)
auto_llm = np.array([0.5217, 0.8421, 0.5767, 0.8736, 0.6097, 0.8257, 0.4231, 0.8049, 0.6577, 0.8148, 0.7947, 0.8542, 0.6917, 0.8210, 0.5965, 0.7794])
nonauto_llm = np.array([0.4995, 0.8167, 0.5227, 0.8063, 0.5445, 0.8518, 0.4965, 0.7471, 0.5672, 0.7896, 0.6481, 0.8319, 0.7018, 0.8125, 0.5495, 0.7972])
nonauto_human = np.array([0.4995, 0.6333, 0.5796, 0.6428, 0.6280, 0.6043, 0.6019, 0.6425, 0.6734, 0.7046, 0.6687, 0.6951, 0.6848, 0.6422, 0.6933, 0.6985])
auto_human = np.array([0.5698, 0.6428, 0.7091, 0.6407, 0.6520, 0.6106, 0.5444, 0.5528, 0.7798, 0.7111, 0.8711, 0.7533, 0.6866, 0.6442, 0.7658, 0.7269])

# Slice for low_low_distressed
low_auto_llm = auto_llm[::2]
low_nonauto_llm = nonauto_llm[::2]
low_nonauto_human = nonauto_human[::2]
low_auto_human = auto_human[::2]

# Slice for high_high_neutral 
high_auto_llm = auto_llm[1::2]
high_nonauto_llm = nonauto_llm[1::2]
high_nonauto_human = nonauto_human[1::2]
high_auto_human = auto_human[1::2]

# Correlations for low_low_distressed (8 points each)
r_low_auto_nonauto_llm, p_low_auto_nonauto_llm = pearsonr(low_auto_llm, low_nonauto_llm)
r_low_nonauto_llm_human, p_low_nonauto_llm_human = pearsonr(low_nonauto_llm, low_nonauto_human)
r_low_auto_llm_human, p_low_auto_llm_human = pearsonr(low_auto_llm, low_auto_human)

# Correlations for high_high_neutral (8 points each)
r_high_auto_nonauto_llm, p_high_auto_nonauto_llm = pearsonr(high_auto_llm, high_nonauto_llm)
r_high_nonauto_llm_human, p_high_nonauto_llm_human = pearsonr(high_nonauto_llm, high_nonauto_human)
r_high_auto_llm_human, p_high_auto_llm_human = pearsonr(high_auto_llm, high_auto_human)

# Correlations for auto vs non-auto via Human Evaluator 
r_auto_nonauto_human, p_auto_nonauto_human = pearsonr(auto_human, nonauto_human)
r_low_auto_nonauto_human, p_low_auto_nonauto_human = pearsonr(low_auto_human, low_nonauto_human)
r_high_auto_nonauto_human, p_high_auto_nonauto_human = pearsonr(high_auto_human, high_nonauto_human)

# Correlations for auto vs non-auto via LLM evaluator 
r_auto_nonauto_llm, p_auto_nonauto_llm = pearsonr(auto_llm, nonauto_llm)
r_low_auto_nonauto_llm, p_low_auto_nonauto_llm = pearsonr(low_auto_llm, low_nonauto_llm)
r_high_auto_nonauto_llm, p_high_auto_nonauto_llm = pearsonr(high_auto_llm, high_nonauto_llm)

# Print new results for auto vs non-auto human (overall and segments)
print("Overall - Auto vs Non-Auto Human: r=", r_auto_nonauto_human, "p=", p_auto_nonauto_human)
print("Low Segment - Auto vs Non-Auto Human: r=", r_low_auto_nonauto_human, "p=", p_low_auto_nonauto_human)
print("High Segment - Auto vs Non-Auto Human: r=", r_high_auto_nonauto_human, "p=", p_high_auto_nonauto_human)

# Printresults for auto vs non-auto human (overall and segments)
print("Overall - Auto vs Non-Auto LLM: r=", r_auto_nonauto_llm, "p=", p_auto_nonauto_llm)
print("Low Segment - Auto vs Non-Auto LLM: r=", r_low_auto_nonauto_llm, "p=", p_low_auto_nonauto_llm)
print("High Segment - Auto vs Non-Auto LLM: r=", r_high_auto_nonauto_llm, "p=", p_high_auto_nonauto_llm)


# Print results for low
print("Low Segment - Auto vs Non-Auto LLM: r=", r_low_auto_nonauto_llm, "p=", p_low_auto_nonauto_llm)
print("Low Segment - Non-Auto LLM vs Human: r=", r_low_nonauto_llm_human, "p=", p_low_nonauto_llm_human)
print("Low Segment - Auto LLM vs Human: r=", r_low_auto_llm_human, "p=", p_low_auto_llm_human)

# Print results for high
print("High Segment - Auto vs Non-Auto LLM: r=", r_high_auto_nonauto_llm, "p=", p_high_auto_nonauto_llm)
print("High Segment - Non-Auto LLM vs Human: r=", r_high_nonauto_llm_human, "p=", p_high_nonauto_llm_human)
print("High Segment - Auto LLM vs Human: r=", r_high_auto_llm_human, "p=", p_high_auto_llm_human)