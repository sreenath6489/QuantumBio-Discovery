#!/usr/bin/env python3
"""
check_fairness.py - can simple molecule properties ALONE tell actives from decoys?

For each property (weight, H-bond donors, ...), we rank all molecules by that property
and compute an AUC score:
    0.50 = the property tells you nothing (perfectly fair test)
    0.50-0.60 = slight bias, fine for a hackathon
    0.65+ = size/shape is giving too much away; build a better-matched decoy set
(0.40 or lower just means the bias goes the other way, decoys score higher.)

Usage (from your project folder):
    python check_fairness.py
    python check_fairness.py --testset testset/mpro_testset.csv
"""

import argparse

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski
from sklearn.metrics import roc_auc_score

RDLogger.DisableLog("rdApp.*")

PROPERTIES = [
    ("Molecular weight", Descriptors.MolWt),
    ("logP (greasiness)", Crippen.MolLogP),
    ("H-bond donors", Lipinski.NumHDonors),
    ("H-bond acceptors", Lipinski.NumHAcceptors),
    ("Rotatable bonds", Lipinski.NumRotatableBonds),
    ("Heavy atoms", lambda m: m.GetNumHeavyAtoms()),
]


def verdict(auc):
    if 0.40 <= auc <= 0.60:
        return "fair"
    if 0.35 <= auc <= 0.65:
        return "slight bias"
    return "BIASED"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="testset/mpro_testset.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.testset)
    mols = [Chem.MolFromSmiles(s) for s in df["smiles"]]
    bad = sum(m is None for m in mols)
    if bad:
        print(f"Skipping {bad} rows with unreadable SMILES")
    keep = [i for i, m in enumerate(mols) if m is not None]
    df = df.iloc[keep].reset_index(drop=True)
    mols = [mols[i] for i in keep]

    y = (df["label"] == "active").astype(int)
    print(f"{int(y.sum())} actives, {int((1 - y).sum())} decoys\n")
    print(f"{'property':<22}{'AUC':>6}   verdict")
    for name, func in PROPERTIES:
        auc = roc_auc_score(y, [func(m) for m in mols])
        print(f"{name:<22}{auc:>6.2f}   {verdict(auc)}")

    dup = df["smiles"].duplicated().sum()
    print(f"\nDuplicate SMILES in the file: {dup} (should be 0)")


if __name__ == "__main__":
    main()