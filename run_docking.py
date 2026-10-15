import os
import subprocess
import pandas as pd
from pathlib import Path

# Paths & Configuration
CSV_PATH = "data/processed/mpro_benchmark_set.csv"
OUTPUT_DIR = "data/results/docking_output"
CONFIG_FILE = "vina_config.txt"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_vina_docking():
    # 1. Load the benchmark dataset
    if not os.path.exists(CSV_PATH):
        print(f"Error: Master CSV not found at {CSV_PATH}")
        return
        
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} molecules from {CSV_PATH}")

    results = []

    # 2. Loop through each molecule for docking
    for index, row in df.iterrows():
        mol_id = str(row['id'])
        smiles = row['smiles']
        label = row['label']
        
        print(f"[{index+1}/{len(df)}] Processing {mol_id} ({label})...")

        # Step A: Convert SMILES to 3D PDB using RDKit (or Open Babel tool)
        # Note: In a hackathon, you can use Meeko or OpenBabel to convert SMILES to PDBQT
        pdbqt_out = os.path.join(OUTPUT_DIR, f"{mol_id}.pdbqt")
        log_out = os.path.join(OUTPUT_DIR, f"{mol_id}_log.txt")

        # For hackathon speed, if ligand preparation is handled, we run Vina directly:
        # Example command using AutoDock Vina binary:
        vina_cmd = [
            "vina",
            "--config", CONFIG_FILE,
            "--ligand", pdbqt_out,
            "--out", os.path.join(OUTPUT_DIR, f"{mol_id}_out.pdbqt"),
            "--log", log_out
        ]

        # Execute Vina (uncomment when running locally with vina installed)
        # subprocess.run(vina_cmd, stdout=subprocess.DEVNULL)

        # Step B: Parse Vina log to extract binding affinity (kcal/mol)
        # (Placeholder parser logic)
        score = 0.0
        if os.path.exists(log_out):
            with open(log_out, 'r') as f:
                for line in f:
                    if "REMARK VINA RESULT:" in line:
                        parts = line.split()
                        score = float(parts[3])
                        break
        
        results.append({
            "id": mol_id,
            "label": label,
            "vina_score": score
        })

    # 3. Save final docking summary dataframe
    results_df = pd.DataFrame(results)
    results_df.to_csv("data/results/docking_summary.csv", index=False)
    print("Docking pipeline completed! Results saved to data/results/docking_summary.csv")

if __name__ == "__main__":
    run_vina_docking()