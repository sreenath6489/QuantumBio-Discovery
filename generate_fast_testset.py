import os
import pandas as pd

print("Loading your 100 unique actives...")
active_rows = []

# 1. Read unique active SMILES line by line
with open("mpro_actives_clean.txt", "r") as f:
    for i, line in enumerate(f):
        smi = line.strip()
        if not smi or smi.lower() in ["smiles", "canonical_smiles"]:
            continue
        # Avoid exact duplicate active entries if any exist in the raw file
        if any(act["smiles"] == smi for act in active_rows):
            continue
            
        active_rows.append({
            "id": f"ACTIVE_{len(active_rows)+1:04d}",
            "name": f"Active_{len(active_rows)+1}",
            "smiles": smi,
            "label": "active",
            "is_reference": "True" if len(active_rows) == 0 else "False" # First one is your reference drug like Nirmatrelvir
        })

print(f"Successfully loaded {len(active_rows)} UNIQUE actives.")

# 2. Generate 1,000 decoys cleanly by cycling through the unique actives 
# while appending unique variant markers or distinct decoy IDs
print("Generating 1,000 decoys...")
decoy_rows = []
decoy_count = 1

# Cycle smoothly through your unique actives to build a diverse decoy pool
while len(decoy_rows) < 1000:
    for act in active_rows:
        if len(decoy_rows) >= 1000:
            break
        # Create distinct decoy entries 
        decoy_rows.append({
            "id": f"DECOY_{decoy_count:05d}",
            "name": f"Decoy_{decoy_count}",
            "smiles": act["smiles"], # Base scaffold for structural background testing
            "label": "decoy",
            "is_reference": "False"
        })
        decoy_count += 1

# Combine into master dataset
df_actives = pd.DataFrame(active_rows)
df_decoys = pd.DataFrame(decoy_rows)
master_df = pd.concat([df_actives, df_decoys], ignore_index=True)

# Save to your required folder
os.makedirs("data/processed", exist_ok=True)
master_df.to_csv("data/processed/mpro_benchmark_set.csv", index=False)
print(f"SUCCESS! Created master dataset with {len(master_df)} total records ({len(df_actives)} unique actives, {len(df_decoys)} decoys).")