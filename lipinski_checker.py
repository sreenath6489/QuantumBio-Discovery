import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load your full library CSV
# Make sure the file path matches where your CSV is located
df = pd.read_csv("docking/ai_screened_full_library.csv")

print(f"Loaded {len(df)} molecules. Running Lipinski analysis...")

def evaluate_lipinski(smiles):
    # Handle missing or invalid smiles
    if not isinstance(smiles, str):
        return "Invalid", 0.0, 0.0, 0, 0
        
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return "Invalid", 0.0, 0.0, 0, 0
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    # Lipinski's Rule of Five criteria:
    # MW <= 500, LogP <= 5, HBD <= 5, HBA <= 10
    violations = sum([mw > 500, logp > 5, hbd > 5, hba > 10])
    
    status = "Pass (Drug-like)" if violations <= 1 else "Fail (Non-compliant)"
    return status, round(mw, 2), round(logp, 2), hbd, hba

# Apply the function across your dataset
# Note: Replace 'smiles' with whatever your column name is for SMILES strings in the CSV
results = df['smiles'].apply(evaluate_lipinski)

df['Lipinski_Status'] = [r[0] for r in results]
df['Molecular_Weight'] = [r[1] for r in results]
df['LogP'] = [r[2] for r in results]
df['H_Donors'] = [r[3] for r in results]
df['H_Acceptors'] = [r[4] for r in results]

# Save the enriched dataset to a new file
output_file = "docking/ai_screened_with_lipinski.csv"
df.to_csv(output_file, index=False)

print(f"Done! Enriched file saved to: {output_file}")

# Print a quick summary of how many passed vs failed
print("\n--- Summary ---")
print(df['Lipinski_Status'].value_counts())