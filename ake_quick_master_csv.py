import pandas as pd

# Load your 100 clean actives
actives_df = pd.read_csv("mpro_actives_clean.txt", sep=None, engine="python")

# Ensure columns align
print(f"Loaded {len(actives_df)} active compounds.")

# For the hackathon evaluation set, we can format your master set:
# We'll label your 100 actives as 'active' and create your 5-column master format.
rows = []

# Add actives
for i, row in actives_df.iterrows():
    smiles = row.iloc[0] # first column is smiles
    rows.append({
        "id": f"ACTIVE_{i+1:04d}",
        "name": f"Active_{i+1}",
        "smiles": smiles,
        "label": "active",
        "is_reference": "True" if i == 0 else "False" # Set first as reference/Nirmatrelvir or similar
    })

master_df = pd.DataFrame(rows)

# Save as your master benchmark set ready for docking & AI
master_df.to_csv("data/processed/mpro_benchmark_set.csv", index=False)
print("Success! Created data/processed/mpro_benchmark_set.csv with your actives.")
print("Now you can proceed straight to running your AutoDock Vina baseline script!")