import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from sklearn.ensemble import RandomForestRegressor

def smiles_to_fp(smiles, radius=2, n_bits=2048):
    """Convert a SMILES string into an ECFP4 bit vector numpy array."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr
    except Exception:
        return None

def main():
    print("--- STEP 1: Loading Scored Docking Data (Training Set) ---")
    scored_path = "docking/pilot_scores.csv"
    if not os.path.exists(scored_path):
        print(f"Error: Could not find {scored_path}. Make sure your docking CSV is in the docking folder!")
        return
        
    scored_df = pd.read_csv(scored_path)
    scored_df = scored_df[scored_df["vina_score"].notna()].reset_index(drop=True)
    print(f"Loaded {len(scored_df)} successfully docked records for training.")

    print("\n--- STEP 2: Extracting Chemical Fingerprints ---")
    X_train = []
    y_train = []
    for _, row in scored_df.iterrows():
        fp = smiles_to_fp(row["smiles"])
        if fp is not None:
            X_train.append(fp)
            y_train.append(float(row["vina_score"]))
            
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    print(f"Successfully vectorized {len(X_train)} training molecules.")

    print("\n--- STEP 3: Training 'The Guesser' AI Model (Random Forest Regressor) ---")
    # n_estimators=300 builds 300 decision trees to learn chemical-to-score relationships
    guesser = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    guesser.fit(X_train, y_train)
    print("AI training completed successfully!")

    print("\n--- STEP 4: Screening Master Dataset (1,000+ Records) ---")
    master_path = "testset/mpro_testset.csv"
    if not os.path.exists(master_path):
        print(f"Error: Could not find master file {master_path}.")
        return
        
    master_df = pd.read_csv(master_path)
    print(f"Loaded master library of {len(master_df)} molecules. Predicting scores instantly...")

    predictions = []
    for _, row in master_df.iterrows():
        fp = smiles_to_fp(row["smiles"])
        if fp is not None:
            pred_score = guesser.predict([fp])[0]
        else:
            pred_score = np.nan
        predictions.append(pred_score)

    master_df["ai_predicted_vina_score"] = predictions
    
    # Sort by predicted score: more negative score = stronger predicted binding affinity
    df_sorted = master_df.sort_values(by="ai_predicted_vina_score", ascending=True).reset_index(drop=True)

    print("\n--- STEP 5: Analyzing Results & Reference Drug Rank ---")
    ref_rows = df_sorted[df_sorted["is_reference"].astype(str).str.lower().isin(["yes", "true"])]
    if len(ref_rows) > 0:
        ref_rank = ref_rows.index[0] + 1
        print(f"🌟 Nirmatrelvir (Reference Drug) AI-Predicted Rank: {ref_rank} out of {len(df_sorted)}")
    else:
        print("Reference drug marker not found, skipping rank check.")

    # Save outputs
    os.makedirs("docking", exist_ok=True)
    out_csv = "docking/ai_screened_full_library.csv"
    df_sorted.to_csv(out_csv, index=False)
    
    # Export top 30 shortlist for scientists
    top_30 = df_sorted.head(30)
    top_30_csv = "docking/top_30_scientist_shortlist.csv"
    top_30.to_csv(top_30_csv, index=False)

    print(f"\nSaved full AI-screened library to: {out_csv}")
    print(f"Saved elite Top 30 scientist shortlist to: {top_30_csv}")
    
    print("\nTop 5 Predicted Top Hit Candidates:")
    print(df_sorted[["id", "label", "ai_predicted_vina_score"]].head(5))

if __name__ == "__main__":
    main()